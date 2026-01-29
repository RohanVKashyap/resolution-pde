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

from models.spectral_convolution import SpectralConv1d, SpectralConv2d


################################################################
#  1D Fourier Layer
################################################################

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

################################################################
#  2D Fourier Layer
################################################################      

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