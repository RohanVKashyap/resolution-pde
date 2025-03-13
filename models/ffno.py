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

from models.custom_layer import WNLinear
from models.spectral_convolution import FSpectralConv1d, FSpectralConv2d

################################################################
#  1D FFNO
################################################################

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
        grid=None,
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
        self.grid = grid
        
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
        grid = self.get_grid(batch_size, seq_length, x.device)  # [batch_size, seq_length, 1]
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
    
################################################################
#  2D FFNO
################################################################ 

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
        grid=None,
        dropout=0.0,
        mode='full'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_modes = n_modes
        self.ff_weight_norm = ff_weight_norm
        self.grid = grid

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