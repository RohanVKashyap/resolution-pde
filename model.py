import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm
from einops import rearrange
import h5py
import os
import copy
import math
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import sys

################################################################
#  1D Fourier Layer
################################################################

class SpectralConv1d(nn.Module):
  def __init__(self, in_channels, out_channels, modes1):
    super(SpectralConv1d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.modes1 = modes1

    self.scale = (1 / (in_channels*out_channels))
    self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

  def compl_mul1d(self, input, weights):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", input, weights)

  def forward(self, x):
    batchsize = x.shape[0]
    #Compute Fourier coeffcients up to factor of e^(- something constant)
    x_ft = torch.fft.rfft(x)

    # Multiply relevant Fourier modes
    out_ft = torch.zeros(
        batchsize,
        self.out_channels, 
        x.size(-1)//2 + 1,  
        device=x.device, 
        dtype=torch.cfloat)
    out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], 
    self.weights1)

    #Return to physical space
    x = torch.fft.irfft(out_ft, n=x.size(-1))
    return x

class FNOBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, activation=F.relu):
        super(FNOBlock1d, self).__init__()
        self.spectral_conv = SpectralConv1d(in_channels, out_channels, modes)
        self.bypass_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.spectral_conv(x) + self.bypass_conv(x))
    
class MLP1d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP1d, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class LinearMLP1d(nn.Module):
    def __init__(self, in_features, out_features, mid_features):
        super(LinearMLP1d, self).__init__()
        self.fc1 = nn.Linear(in_features, mid_features)
        self.fc2 = nn.Linear(mid_features, out_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x      

class FNO1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, width, grid=None, activation=F.relu, n_blocks=4):
        super(FNO1d, self).__init__()
        self.width = width
        self.grid = grid

        self.lifting = nn.Conv1d(in_channels + 1, width, 1)      # 1 for the grid coordinates

        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(width, width, modes, activation) for _ in range(n_blocks)])

        self.projection = MLP1d(self.width, out_channels, self.width * 4)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        
        if self.grid is not None:
            # Use provided grid coordinates
            x_coordinate = self.grid
            
            if not isinstance(x_coordinate, torch.Tensor):
                x_coordinate = torch.tensor(x_coordinate, dtype=torch.float)
            
            # Reshape and repeat for the batch dimension
            gridx = x_coordinate.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        else:
            # Create normalized grid coordinates from 0 to 1
            gridx = torch.tensor(np.linspace(0, 2 * np.pi, size_x), dtype=torch.float)     # [0, 2 * pi]
            gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        
        return gridx.to(device) 

    def forward(self, x):
        # Input shape: (batch, channels, length)
        batch_size, _, size_x = x.shape
        
        # Create grid and reshape to match expected dimensions
        grid = self.get_grid((batch_size, size_x), x.device)  # (batch, size_x, 1)
        grid = grid.permute(0, 2, 1)  # (batch, 1, size_x)
        
        # Concatenate input and grid along channel dimension
        x = torch.cat((x, grid), dim=1)  # (batch, channels+1, size_x)
        
        # Standard FNO forward flow
        x = self.lifting(x)
        
        # Pass through FNO blocks
        for fno_block in self.fno_blocks:
            x = fno_block(x)
        
        # Project back to output dimension
        x = self.projection(x)
        return x       

################################################################
#  2D Fourier Layer
################################################################    

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def complex_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Apply 2D FFT
        x_ft = torch.fft.rfft2(x)

        # Initialize output to zeros
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
            device=x.device, dtype=torch.cfloat)
        
        # Multiply low frequency Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.complex_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # Multiply high frequency Fourier modes  
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.complex_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            
        # Apply inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNOBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, activation=F.gelu):
        super(FNOBlock2d, self).__init__()
        self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.spectral_conv(x) + self.bypass_conv(x))

class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x    

class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width, grid=None, activation=F.gelu, n_blocks=4):
        super(FNO2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.grid = grid

        # Lifting layer to increase channel dimension
        self.lifting = nn.Conv2d(in_channels + 2, width, 1)  # 2 for the grid coordinates

        # FNO blocks with spectral convolutions
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(width, width, modes1, modes2, activation) for _ in range(n_blocks)
        ])

        # Projection layer to map back to output channels
        self.projection = MLP2d(self.width, self.out_channels, self.width * 4)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        
        if self.grid is not None:
            # Use provided grid coordinates
            x_coordinate = self.grid[0]
            y_coordinate = self.grid[1]
            
            if not isinstance(x_coordinate, torch.Tensor):
                x_coordinate = torch.tensor(x_coordinate, dtype=torch.float)
            if not isinstance(y_coordinate, torch.Tensor):
                y_coordinate = torch.tensor(y_coordinate, dtype=torch.float)
            
            # Reshape and repeat for the batch dimension and other dimension
            gridx = x_coordinate.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
            gridy = y_coordinate.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        else:
            # Create normalized grid coordinates from 0 to 1
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])

            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        
        # Concatenate x and y coordinates
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    def forward(self, x):
        # Input shape: (batch, channels, height, width)
        batch_size, _, size_x, size_y = x.shape

        # Create grid and reshape to match expected dimensions
        grid = self.get_grid((batch_size, size_x, size_y), x.device)  # (batch, size_x, size_y, 2)
        grid = grid.permute(0, 3, 1, 2)  # (batch, 2, size_x, size_y)

        # Concatenate input and grid along channel dimension
        x = torch.cat((x, grid), dim=1)  # (batch, channels+2, size_x, size_y)

        # Standard FNO forward flow
        x = self.lifting(x)

        # Pass through FNO blocks
        for fno_block in self.fno_blocks:
            x = fno_block(x)

        # Project back to output dimension
        x = self.projection(x)
        return x   

################################################################
#  FFNO
################################################################      
    
class FeedForward(nn.Module):
    '''Adapted from https://github.com/alasdairtran/fourierflow'''
    def __init__(self, dim, factor, n_layers = 2, ff_weight_norm = False, layer_norm = False, dropout = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(dropout),
                nn.GELU() if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


act_registry = {"gelu" : F.gelu,
                "identity": nn.Identity(),
                "relu": F.relu}

################################################################
#  1D FFNO
################################################################  

class FSpectralConv1d(nn.Module):
    '''Adapted from https://github.com/alasdairtran/fourierflow'''
    def __init__(self, d_model, modes, forecast_ff = None, backcast_ff = None,
                 fourier_weight = None, factor = 4, ff_weight_norm = False,
                 n_ff_layers = 2, layer_norm = False, use_fork = False, dropout = 0.0, mode = 'full', activation = "identity",
                  fft_norm = "ortho", **kwargs):
        super().__init__()
        self.in_dim = d_model
        self.out_dim = d_model
        self.n_modes = modes
        self.mode = mode
        self.use_fork = use_fork
        self.fft_norm = fft_norm

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(1): # only x dimension
                weight = torch.FloatTensor(d_model, d_model, modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        # if use_fork:
        #     self.forecast_ff = forecast_ff
        #     if not self.forecast_ff:
        #         self.forecast_ff = FeedForward(
        #             out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                d_model, factor, ff_weight_norm = ff_weight_norm, n_layers = n_ff_layers, layer_norm = layer_norm, dropout = dropout)
        
        self.act = act_registry[activation]

    def forward(self, x, batch_dt = None):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        # f = self.forecast_ff(x) if self.use_fork else None

        # return b, f
        b = self.act(b)
        # framework to pass out, state
        return b, None

    def forward_fourier(self, x):
        x = rearrange(x, 'b x h -> b h x')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, H, Sx = x.shape

        # # # Dimesion X # # #
        x_ft = torch.fft.rfft(x, dim=-1, norm=self.fft_norm)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_ft.new_zeros(B, H, Sx // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        if self.mode == 'full':
            out_ft[:, :, :self.n_modes] = torch.einsum(
                "bix,iox->box",
                x_ft[:, :, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))
                # self.fourier_weight[0])
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.n_modes] = x_ft[:, :, :self.n_modes]
        else: 
            raise ValueError(f"Mode {self.mode} not recognized")

        out = torch.fft.irfft(out_ft, n=Sx, dim=-1, norm=self.fft_norm)
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        out = rearrange(out, 'b h x -> b x h')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return out

################################################################
#  2D FFNO
################################################################  
    
class FSpectralConv2d(nn.Module):
    '''Adapted from https://github.com/alasdairtran/fourierflow'''
    def __init__(self, d_model, modes, forecast_ff=None, backcast_ff=None,
                 fourier_weight=None, factor=4, ff_weight_norm=False,
                 n_ff_layers=2, layer_norm=False, use_fork=False, dropout=0.0, mode='full'):
        super().__init__()
        self.in_dim = d_model
        self.out_dim = d_model
        self.n_modes = modes
        self.mode = mode
        self.use_fork = use_fork
        self.fourier_weight = fourier_weight
        
        # Initialize Fourier weights if not provided
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):  # x and y dimensions
                weight = torch.FloatTensor(d_model, d_model, modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)
        
        # Feed-forward network for backcast
        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                d_model, factor, ff_weight_norm=ff_weight_norm, 
                n_layers=n_ff_layers, layer_norm=layer_norm, dropout=dropout
            )
        
        # Feed-forward network for forecast (optional)
        self.forecast_ff = forecast_ff
        if use_fork and not self.forecast_ff:
            self.forecast_ff = FeedForward(
                d_model, factor, ff_weight_norm=ff_weight_norm,
                n_layers=n_ff_layers, layer_norm=layer_norm, dropout=dropout
            )
            
    def forward(self, x, batch_dt=None):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)
        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f    

def forward_fourier(self, x):
    # Rearrange from [batch, x, y, channels] to [batch, channels, x, y] for 2D FFT operations
    x = rearrange(x, 'b x y h -> b h x y')
    # x.shape == [batch_size, in_dim, grid_size, grid_size]
    B, I, M, N = x.shape
    
    # ===============================================
    # Process Y dimension - equivalent to FFTy in the diagram
    # ===============================================
    x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')  # FFT along y dimension
    # x_fty.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]
    out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
    
    if self.mode == 'full':
        # Apply learned weights in Fourier space (equivalent to "Linear map in frequency domain")
        out_ft[:, :, :, :self.n_modes] = torch.einsum(
            "bixy,ioy->boxy",
            x_fty[:, :, :, :self.n_modes],
            torch.view_as_complex(self.fourier_weight[0])
        )
    elif self.mode == 'low-pass':
        out_ft[:, :, :, :self.n_modes] = x_fty[:, :, :, :self.n_modes]
        
    # Inverse FFT to get back to spatial domain (equivalent to IFFTy in the diagram)
    xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
    
    # ===============================================
    # Process X dimension - equivalent to FFTx in the diagram
    # ===============================================
    x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')  # FFT along x dimension
    # x_ftx.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]
    out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
    
    if self.mode == 'full':
        # Apply learned weights in Fourier space (equivalent to "Linear map in frequency domain")
        out_ft[:, :, :self.n_modes, :] = torch.einsum(
            "bixy,iox->boxy",
            x_ftx[:, :, :self.n_modes, :],
            torch.view_as_complex(self.fourier_weight[1])
        )
    elif self.mode == 'low-pass':
        out_ft[:, :, :self.n_modes, :] = x_ftx[:, :, :self.n_modes, :]
        
    # Inverse FFT to get back to spatial domain (equivalent to IFFTx in the diagram)
    xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
    
    # ===============================================
    # Combine the results from both dimensions (merging in physical space)
    # ===============================================
    x = xx + xy  # Element-wise addition of the results
    
    # Rearrange back to original format
    x = rearrange(x, 'b i m n -> b m n i')
    # x.shape == [batch_size, grid_size, grid_size, out_dim]
    return x

################################################################
#  2D FFNO Block
################################################################ 

class WNLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, wnorm=False):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)
        if wnorm:
            weight_norm(self)

        self._fix_weight_norm_deepcopy()

    def _fix_weight_norm_deepcopy(self):
        # Fix bug where deepcopy doesn't work with weightnorm.
        # Taken from https://github.com/pytorch/pytorch/issues/28594#issuecomment-679534348
        orig_deepcopy = getattr(self, '__deepcopy__', None)

        def __deepcopy__(self, memo):
            # save and delete all weightnorm weights on self
            weights = {}
            for hook in self._forward_pre_hooks.values():
                if isinstance(hook, WeightNorm):
                    weights[hook.name] = getattr(self, hook.name)
                    delattr(self, hook.name)
            # remove this deepcopy method, restoring the object's original one if necessary
            __deepcopy__ = self.__deepcopy__
            if orig_deepcopy:
                self.__deepcopy__ = orig_deepcopy
            else:
                del self.__deepcopy__
            # actually do the copy
            result = copy.deepcopy(self)
            # restore weights and method on self
            for name, value in weights.items():
                setattr(self, name, value)
            self.__deepcopy__ = __deepcopy__
            return result
        # bind __deepcopy__ to the weightnorm'd layer
        self.__deepcopy__ = __deepcopy__.__get__(self, self.__class__)

class FeedForward(nn.Module):
    def __init__(self, dim, factor, ff_weight_norm, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                WNLinear(in_dim, out_dim, wnorm=ff_weight_norm),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FFNO1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=64,
        n_layers=4,
        n_modes=16,
        factor=4,
        ff_weight_norm=False,
        n_ff_layers=2,
        layer_norm=False,
        dropout=0.0,
        mode='full',
        fft_norm="ortho",
        activation="identity"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_modes = n_modes
        
        # Input projection with WNLinear
        self.in_proj = WNLinear(self.in_channels, hidden_dim, wnorm=ff_weight_norm)
        
        # F-FNO layers
        self.fourier_layers = nn.ModuleList([])
        for i in range(n_layers):
            self.fourier_layers.append(
                FSpectralConv1d(
                    hidden_dim, n_modes, 
                    factor=factor,
                    ff_weight_norm=ff_weight_norm,
                    n_ff_layers=n_ff_layers,
                    layer_norm=layer_norm,
                    dropout=dropout,
                    mode=mode,
                    fft_norm=fft_norm,
                    activation=activation
                )
            )
        
        # Output projection with WNLinear
        self.out_proj = WNLinear(hidden_dim, self.out_channels, wnorm=ff_weight_norm)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        
        if self.grid is not None:
            # Use provided grid coordinates
            x_coordinate = self.grid
            
            if not isinstance(x_coordinate, torch.Tensor):
                x_coordinate = torch.tensor(x_coordinate, dtype=torch.float)
            
            # Reshape and repeat for the batch dimension
            gridx = x_coordinate.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        else:
            # Create normalized grid coordinates from 0 to 1
            gridx = torch.tensor(np.linspace(0, 2 * np.pi, size_x), dtype=torch.float)     # [0, 2 * pi]
            gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        
        return gridx.to(device)

    def forward(self, x):
        # Input shape: (batch, channels, seq_length)
        batch_size, _, seq_length = x.shape
        
        # Create 1D grid
        grid = self.get_grid_1d(batch_size, seq_length, x.device)  # [batch_size, seq_length, 1]
        grid = grid.permute(0, 2, 1)  # [batch_size, 1, seq_length]
        
        # Concatenate input and grid along channel dimension
        x = torch.cat((x, grid), dim=1)  # [batch_size, channels+1, seq_length]
        
        # Convert to channels-last format for processing
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, channels+1]
        
        # Input projection
        x = self.in_proj(x)
        
        # Apply Fourier layers
        for i, layer in enumerate(self.fourier_layers):
            x_new, _ = layer(x)
            x = x + x_new  # Residual connection
        
        # Output projection
        x = self.out_proj(x)
        
        x = x.permute(0, 2, 1)  # [batch_size, out_channels, seq_length]
        
        return x     


    # def forward(self, x):
    #     # x.shape == [batch_size, seq_length, in_channels]
    #     batch_size, seq_length, _ = x.shape
        
    #     # Create 1D grid
    #     grid = self.get_grid(batch_size, seq_length, x.device)  # [batch_size, seq_length, 1]
        
    #     # Concatenate input and grid along channel dimension
    #     x = torch.cat((x, grid), dim=-1)  # [batch_size, seq_length, in_channels+1]
        
    #     # Input projection
    #     x = self.in_proj(x)
        
    #     # Apply Fourier layers
    #     for i, layer in enumerate(self.fourier_layers):
    #         x_new, _ = layer(x)
    #         x = x + x_new  # Residual connection
        
    #     # Output projection
    #     x = self.out_proj(x)
    
    #     return x    
    
    # def forward(self, x):
    #     # x.shape == [batch_size, seq_length, in_channels]
        
    #     # Input projection
    #     x = self.in_proj(x)
        
    #     # Apply Fourier layers
    #     for i, layer in enumerate(self.fourier_layers):
    #         x_new, _ = layer(x)
    #         x = x + x_new  # Residual connection
        
    #     # Output projection
    #     x = self.out_proj(x)
        
    #     return x
    

class FFNO2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=64,
        n_layers=4,
        n_modes=16,
        factor=4,
        ff_weight_norm=False,
        n_ff_layers=2,
        layer_norm=False,
        dropout=0.0,
        mode='full'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_modes = n_modes
        self.ff_weight_norm = ff_weight_norm

        if self.ff_weight_norm: 
            self.in_proj = WNLinear(self.in_channels, hidden_dim, wnorm=ff_weight_norm)
        else:
            self.in_proj = nn.Linear(self.in_channels, hidden_dim)

        
        # F-FNO layers
        self.fourier_layers = nn.ModuleList([])
        for i in range(n_layers):
            self.fourier_layers.append(
                FSpectralConv2d(
                    hidden_dim, n_modes, 
                    factor=factor,
                    ff_weight_norm=ff_weight_norm,
                    n_ff_layers=n_ff_layers,
                    layer_norm=layer_norm,
                    dropout=dropout,
                    mode=mode))
        
        if self.ff_weight_norm:     
             self.out_proj = WNLinear(hidden_dim, self.out_channels, wnorm=ff_weight_norm)
        else:
            self.out_proj = nn.Linear(hidden_dim, self.out_channels)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        
        if self.grid is not None:
            # Use provided grid coordinates
            x_coordinate = self.grid[0]
            y_coordinate = self.grid[1]
            
            if not isinstance(x_coordinate, torch.Tensor):
                x_coordinate = torch.tensor(x_coordinate, dtype=torch.float)
            if not isinstance(y_coordinate, torch.Tensor):
                y_coordinate = torch.tensor(y_coordinate, dtype=torch.float)
            
            # Reshape and repeat for the batch dimension and other dimension
            gridx = x_coordinate.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
            gridy = y_coordinate.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        else:
            # Create normalized grid coordinates from 0 to 1
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])

            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        
        # Concatenate x and y coordinates
        return torch.cat((gridx, gridy), dim=-1).to(device)  

    def forward(self, x):
        # Input shape: (batch, channels, height, width)
        batch_size, _, size_x, size_y = x.shape

        # Create grid in channels-last format
        grid = self.get_grid((batch_size, size_x, size_y), x.device)       # (batch, size_x, size_y, 2)
        grid = grid.permute(0, 3, 1, 2)                                    # (batch, 2, size_x, size_y)
        
        # Concatenate input and grid along channel dimension
        x = torch.cat((x, grid), dim = 1)                                   # (batch, channels + 2, size_x, size_y)

        x = x.permute(0, 2, 3, 1)                                           # (batch, size_x, size_y, channels + 2)
        
        # Input projection
        x = self.in_proj(x)
        
        # Apply Fourier layers
        for i, layer in enumerate(self.fourier_layers):
            x_new, _ = layer(x)
            x = x + x_new
        
        # Output projection
        x = self.out_proj(x)

        x = x.permute(0, 3, 1, 2)

        return x     

        
    # def forward(self, x):
    #     # x.shape == [batch_size, grid_size, grid_size, in_dim]
    #     grid = self.get_grid(x.shape, x.device)

    #     # Input projection
    #     x = self.in_proj(x)
        
    #     # Apply Fourier layers
    #     for i, layer in enumerate(self.fourier_layers):
    #         x_new, _ = layer(x)
    #         x = x + x_new
        
    #     x = self.out_proj(x)
            
    #     return x    
    

# class FNOFactorized2DBlock(nn.Module):
#     def __init__(self, modes, width, input_dim=12, dropout=0.0, in_dropout=0.0,
#                  n_layers=4, share_weight: bool = False,
#                  share_fork=False, factor=2,
#                  ff_weight_norm=False, n_ff_layers=2,
#                  gain=1, layer_norm=False, use_fork=False, mode='full'):
#         super().__init__()
#         self.modes = modes
#         self.width = width
#         self.input_dim = input_dim
#         self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
#         self.drop = nn.Dropout(in_dropout)
#         self.n_layers = n_layers
#         self.use_fork = use_fork

#         self.forecast_ff = self.backcast_ff = None
#         if share_fork:
#             if use_fork:
#                 self.forecast_ff = FeedForward(
#                     width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)
#             self.backcast_ff = FeedForward(
#                 width, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

#         self.fourier_weight = None
#         if share_weight:
#             self.fourier_weight = nn.ParameterList([])
#             for _ in range(2):
#                 weight = torch.FloatTensor(width, width, modes, 2)
#                 param = nn.Parameter(weight)
#                 nn.init.xavier_normal_(param, gain=gain)
#                 self.fourier_weight.append(param)

#         self.spectral_layers = nn.ModuleList([])
#         for _ in range(n_layers):
#             self.spectral_layers.append(SpectralConv2d(in_dim=width,
#                                                        out_dim=width,
#                                                        n_modes=modes,
#                                                        forecast_ff=self.forecast_ff,
#                                                        backcast_ff=self.backcast_ff,
#                                                        fourier_weight=self.fourier_weight,
#                                                        factor=factor,
#                                                        ff_weight_norm=ff_weight_norm,
#                                                        n_ff_layers=n_ff_layers,
#                                                        layer_norm=layer_norm,
#                                                        use_fork=use_fork,
#                                                        dropout=dropout,
#                                                        mode=mode))

#         self.out = nn.Sequential(
#             WNLinear(self.width, 128, wnorm=ff_weight_norm),
#             WNLinear(128, 1, wnorm=ff_weight_norm))

#     def forward(self, x, **kwargs):
#         # x.shape == [n_batches, *dim_sizes, input_size]
#         forecast = 0
#         x = self.in_proj(x)
#         x = self.drop(x)
#         forecast_list = []
#         for i in range(self.n_layers):
#             layer = self.spectral_layers[i]
#             b, f = layer(x)

#             if self.use_fork:
#                 f_out = self.out(f)
#                 forecast = forecast + f_out
#                 forecast_list.append(f_out)

#             x = x + b

#         if not self.use_fork:
#             forecast = self.out(b)

#         return {
#             'forecast': forecast,
#             'forecast_list': forecast_list,
#         }
