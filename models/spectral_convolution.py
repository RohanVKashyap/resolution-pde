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

from models.custom_layer import FeedForward

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

################################################################
#  1D FFNO
################################################################  
    
act_registry = {"gelu" : F.gelu,
                "identity": nn.Identity(),
                "relu": F.relu}    

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

        # if self.mode == 'full':
        #     out_ft[:, :, :self.n_modes] = torch.einsum(
        #         "bix,iox->box",
        #         x_ft[:, :, :self.n_modes],
        #         torch.view_as_complex(self.fourier_weight[0]))
        #         # self.fourier_weight[0])
        # elif self.mode == 'low-pass':
        #     out_ft[:, :, :self.n_modes] = x_ft[:, :, :self.n_modes]
        # else: 
        #     raise ValueError(f"Mode {self.mode} not recognized")

       # Use [f /2] modes at each resolution (UPDATE)
        available_modes = Sx // 2 + 1
        effective_modes = min(self.n_modes, available_modes)
        if self.mode == 'full':
            # Use effective_modes instead of self.n_modes
            out_ft[:, :, :effective_modes] = torch.einsum(
                "bix,iox->box",
                x_ft[:, :, :effective_modes],
                torch.view_as_complex(self.fourier_weight[0][:, :, :effective_modes]))
                
        elif self.mode == 'low-pass':
            # Use effective_modes instead of self.n_modes
            out_ft[:, :, :effective_modes] = x_ft[:, :, :effective_modes]
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
        
        # Use available modes logic for Y dimension
        available_modes_y = N // 2 + 1
        effective_modes_y = min(self.n_modes, available_modes_y)
        
        if self.mode == 'full':
            # Apply learned weights in Fourier space (equivalent to "Linear map in frequency domain")
            out_ft[:, :, :, :effective_modes_y] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :effective_modes_y],
                torch.view_as_complex(self.fourier_weight[0][:, :, :effective_modes_y])
            )
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :effective_modes_y] = x_fty[:, :, :, :effective_modes_y]
            
        # Inverse FFT to get back to spatial domain (equivalent to IFFTy in the diagram)
        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        
        # ===============================================
        # Process X dimension - equivalent to FFTx in the diagram
        # ===============================================
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')  # FFT along x dimension
        # x_ftx.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]
        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        
        # Use available modes logic for X dimension
        available_modes_x = M // 2 + 1
        effective_modes_x = min(self.n_modes, available_modes_x)
        
        if self.mode == 'full':
            # Apply learned weights in Fourier space (equivalent to "Linear map in frequency domain")
            out_ft[:, :, :effective_modes_x, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :effective_modes_x, :],
                torch.view_as_complex(self.fourier_weight[1][:, :, :effective_modes_x])
            )
        elif self.mode == 'low-pass':
            out_ft[:, :, :effective_modes_x, :] = x_ftx[:, :, :effective_modes_x, :]
            
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
    
# class FSpectralConv2d(nn.Module):
#     '''Adapted from https://github.com/alasdairtran/fourierflow'''
#     def __init__(self, d_model, modes, forecast_ff=None, backcast_ff=None,
#                  fourier_weight=None, factor=4, ff_weight_norm=False,
#                  n_ff_layers=2, layer_norm=False, use_fork=False, dropout=0.0, mode='full'):
#         super().__init__()
#         self.in_dim = d_model
#         self.out_dim = d_model
#         self.n_modes = modes
#         self.mode = mode
#         self.use_fork = use_fork
#         self.fourier_weight = fourier_weight
        
#         # Initialize Fourier weights if not provided
#         if not self.fourier_weight:
#             self.fourier_weight = nn.ParameterList([])
#             for _ in range(2):  # x and y dimensions
#                 weight = torch.FloatTensor(d_model, d_model, modes, 2)
#                 param = nn.Parameter(weight)
#                 nn.init.xavier_normal_(param)
#                 self.fourier_weight.append(param)
        
#         # Feed-forward network for backcast
#         self.backcast_ff = backcast_ff
#         if not self.backcast_ff:
#             self.backcast_ff = FeedForward(
#                 d_model, factor, ff_weight_norm=ff_weight_norm, 
#                 n_layers=n_ff_layers, layer_norm=layer_norm, dropout=dropout
#             )
        
#         # Feed-forward network for forecast (optional)
#         self.forecast_ff = forecast_ff
#         if use_fork and not self.forecast_ff:
#             self.forecast_ff = FeedForward(
#                 d_model, factor, ff_weight_norm=ff_weight_norm,
#                 n_layers=n_ff_layers, layer_norm=layer_norm, dropout=dropout
#             )
            
#     def forward(self, x, batch_dt=None):
#         # x.shape == [batch_size, grid_size, grid_size, in_dim]
#         if self.mode != 'no-fourier':
#             x = self.forward_fourier(x)
#         b = self.backcast_ff(x)
#         f = self.forecast_ff(x) if self.use_fork else None
#         return b, f    

#     def forward_fourier(self, x):
#         # Rearrange from [batch, x, y, channels] to [batch, channels, x, y] for 2D FFT operations
#         x = rearrange(x, 'b x y h -> b h x y')
#         # x.shape == [batch_size, in_dim, grid_size, grid_size]
#         B, I, M, N = x.shape
        
#         # ===============================================
#         # Process Y dimension - equivalent to FFTy in the diagram
#         # ===============================================
#         x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')  # FFT along y dimension
#         # x_fty.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]
#         out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        
#         if self.mode == 'full':
#             # Apply learned weights in Fourier space (equivalent to "Linear map in frequency domain")
#             out_ft[:, :, :, :self.n_modes] = torch.einsum(
#                 "bixy,ioy->boxy",
#                 x_fty[:, :, :, :self.n_modes],
#                 torch.view_as_complex(self.fourier_weight[0])
#             )
#         elif self.mode == 'low-pass':
#             out_ft[:, :, :, :self.n_modes] = x_fty[:, :, :, :self.n_modes]
            
#         # Inverse FFT to get back to spatial domain (equivalent to IFFTy in the diagram)
#         xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        
#         # ===============================================
#         # Process X dimension - equivalent to FFTx in the diagram
#         # ===============================================
#         x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')  # FFT along x dimension
#         # x_ftx.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]
#         out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        
#         if self.mode == 'full':
#             # Apply learned weights in Fourier space (equivalent to "Linear map in frequency domain")
#             out_ft[:, :, :self.n_modes, :] = torch.einsum(
#                 "bixy,iox->boxy",
#                 x_ftx[:, :, :self.n_modes, :],
#                 torch.view_as_complex(self.fourier_weight[1])
#             )
#         elif self.mode == 'low-pass':
#             out_ft[:, :, :self.n_modes, :] = x_ftx[:, :, :self.n_modes, :]
            
#         # Inverse FFT to get back to spatial domain (equivalent to IFFTx in the diagram)
#         xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        
#         # ===============================================
#         # Combine the results from both dimensions (merging in physical space)
#         # ===============================================
#         x = xx + xy  # Element-wise addition of the results
        
#         # Rearrange back to original format
#         x = rearrange(x, 'b i m n -> b m n i')
#         # x.shape == [batch_size, grid_size, grid_size, out_dim]
#         return x        