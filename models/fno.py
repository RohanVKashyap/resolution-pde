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

from models.fno_blocks import FNOBlock1d, LinearMLP1d, MLP1d, FNOBlock2d, MLP2d

################################################################
#  1D Fourier Layer
################################################################

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