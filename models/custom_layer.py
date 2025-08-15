import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from torch.nn.utils.weight_norm import WeightNorm
from einops import repeat, rearrange
import h5py
import os
import copy
import math
from abc import ABC, abstractmethod
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import sys

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()
        # x shape: (batch, H, W, channels)
        self.mean = torch.mean(x, 0)     # Shape: (H, W, channels)
        self.std = torch.std(x, 0)       # Shape: (H, W, channels)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, device='cuda:0'):
        std = self.std + self.eps
        mean = self.mean

        std = std.to(device)
        mean = mean.to(device)
        
        x = (x * std) + mean
        return x
    
    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

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


################################################################
#  1D Fourier Layer
################################################################
        
"""Source code: https://github.com/RohanVKashyap/invariance-pde/blob/main/models/custom_layers.py"""


class ZeroLayer(nn.Module):
    def __init__(self):
        super(ZeroLayer, self).__init__()
    
    def forward(self, x):
        return x * 0.0
    
def get_residual_layer(residual_type, d_model):
    # from models.custom_layers import ZeroLayer
    registry = {"weighted": nn.Linear(d_model, d_model),
                "identity": nn.Identity(),
                "zero": ZeroLayer()}
    return registry[residual_type]


def get_norm_layer(norm_type, d_model):
    registry = {"LayerNorm": nn.LayerNorm(d_model),
                "identity": nn.Identity()}
    return registry[norm_type]

def get_ffn_layer(ffn_type, d_model, factor = 4):
    registry = {"ffn": FeedForward(d_model, factor = factor, n_layers=2),
                "zero": ZeroLayer(),
    }
    return registry[ffn_type]

class GridInputProcessor(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, grid):
        '''How to  process input x and grid.
        :param x: generally (B,S,1) or (B,S,D)
        :param grid: (B,S)'''
        pass

class InputProcessor(nn.Module, ABC):
    skip_step = True
    @abstractmethod
    def forward(self, x):
        '''How to  process input x.
        :param x: generally (B,S,1) or (B,S,D)
        '''
        pass

class OutputProcessor(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, input_shape):
        '''How to  process input x and grid.
        :param x: generally (B,S,1) or (B,S,D)
        :param grid: (B,S)'''
        pass

class Id(OutputProcessor):
    def forward(self, x, input_shape):
        '''Does nothing
        x: (B,S,D)'''
        return x

class Squeeze(OutputProcessor):
    def forward(self, x, input_shape):
        ''' Squeezes the last dimension
        x: (B,S,H,1) -> (B,S,H)'''
        assert x.shape[-1] == 1
        return x.squeeze(-1)

class Unsqueeze(OutputProcessor):
    def forward(self, x, input_shape):
        ''' Squeezes the last dimension
        x: (B,S,H) -> (B,S,H,1)'''
        return x.unsqueeze(-1)

class UnflatTrans(OutputProcessor):
    def forward(self, x, input_shape):
        ''' Transposes the temporal and spatial dimensions
        x: (B,T,H) -> (B,S,T,H)'''
        B, T, H = x.shape
        D = input_shape[-1]
        S = H // D
        x = rearrange(x, 'b t (s d) -> b s t d', s=S, d=D)
        return x

class Trans(OutputProcessor):
    def forward(self, x, input_shape):
        ''' Transposes5 the temporal and spatial dimensions
        x: (B,T,Sx,[Sy],[Sz],H) -> (B,Sx,[Sy],[Sz],T,H)'''
        return rearrange(x, 'b t ... h -> b ... t h')


class GridIO(): 
    def __init__(self, input_processor : GridInputProcessor, output_processor : OutputProcessor):
        '''Processing input of architecture and output'''
        self.input_processor = grid_input_registry[input_processor]()
        self.output_processor = output_registry[output_processor]()
        self.input_shape = None

    def process_input(self, x, grid):
        self.input_shape = x.shape
        return self.input_processor(x, grid)

    def process_output(self, x):
        assert self.input_shape is not None, "Input shape is not set. Please call process_input first."
        return self.output_processor(x, self.input_shape)

    def __repr__(self):
        return f"Input: {self.input_processor} -> Output: {self.output_processor}"

class Concat(GridInputProcessor):
    def forward(self, x, grid):
        '''Concatenates grid to the last dimension of x
         x: (B, Sx, [Sy], [Sz], H)
        grid: (B, Sx, [Sy], [Sz], 1)'''
        return torch.cat((x, grid), dim=-1)

class ConcatND(GridInputProcessor):
    def forward(self, x, grid):
        '''Adds one extra dimension to x and concatenates grid to it
        x: (B,S,H)
        grid: (B,S,1)'''
        B, S, H = x.shape
        x = x.unsqueeze(-1) #(B, S, D) -> (B, S, D, 1)
        grid = repeat(grid, 'b s h -> b s c h', c=H)
        return torch.cat((x, grid), dim=-1)

class ConcatTransSqueeze1D(GridInputProcessor):
    def forward(self, x, grid):
        '''Transposes the spatial dimensions and concatenates grid to it
        x: (B,S,1)
        grid: (B,S,1)
        :return: (B, 2S)'''
        x = torch.cat((x.squeeze(-1), grid.squeeze(-1)), dim=-1)
        return x

class ConcatFlatTrans(GridInputProcessor):
    def forward(self, x, grid):
        '''Flattens grid and concatenates it to x
        :param x: (B,S,T,H)
        :param grid: (B,S,1)
        :return: (B,T,(S*H+1))'''
        B, S, T, H = x.shape
        # Tranpose and flatten spatial dimensions
        x = rearrange(x, 'b s t h -> b t (s h)')

        # Reshape the grid to have dimensions [b, 1, (s*h)] (temporal dimension repeated)
        grid = repeat( rearrange(grid, 'b s h -> b (s h)'), 'b h -> b t h', t=T)

        return torch.cat((x, grid), dim=-1)

class ConcatTrans(GridInputProcessor):
    def forward(self, x, grid):
        '''Transposes the spatial and temporal dimensions in x and concatenates grid to it
        :param x: (B, Sx, [Sy], [Sz], T, H)
        :param grid: (B, Sx, [Sy], [Sz], 1)
        :return: (B, T, Sx, [Sy], [Sz], H+1)'''
        T, H = x.shape[-2:]
        # Tranposepatial dimensions
        x = rearrange(x, "b ... t h -> b t ... h")

        # Reshape the grid to have dimensions [b, 1, (s*d)] (temporal dimension repeated)
        grid = repeat(grid, "b ... h -> b t ... h", t=T)

        return torch.cat((x, grid), dim=-1)


class InputId(InputProcessor):
    def forward(self, x):
        '''Does nothing
        x: (B,T,Sx,[Sy],[Sz],H) '''
        return x      
    
class BatchTime(InputProcessor):
    def forward(self, x):
        '''Reshapes x to (B,T,Sx,[Sy],[Sz],H) -> ((B,T),Sx,[Sy],[Sz],H)
        :param x: (B,T,Sx,[Sy],[Sz],H)'''
        return rearrange(x, 'b t ... h -> (b t) ... h')


class UnbatchTime(OutputProcessor):
    def forward(self, x, input_shape):
        '''Reshapes x from ((B,T),Sx,[Sy],[Sz],h) -> (B,T,Sx,[Sy],[Sz],H)'''
        B = input_shape[0]
        T = input_shape[1]
        return rearrange(x, '(b t) ... h -> b t ... h', b=B, t=T)    

class UnbatchSpace(OutputProcessor):
    def forward(self, x, input_shape):
        '''Reshapes x to ((B,Sx,[Sy],[Sz]),T,H) -> (B,T,Sx,[Sy],[Sz],H)'''
        B, T, H = input_shape[0], input_shape[1], input_shape[-1]
        other_dims = input_shape[2:-1]  # This will capture Sx, Sy, Sz if they exist
        dim_letters = ['sx', 'sy', 'sz']
        pattern = ' '.join([f'{dim_letters[i]}' for i in range(len(other_dims))])
        rearrange_pattern = f'(b {pattern}) t h -> b t {pattern} h'

        return rearrange(x, rearrange_pattern, b=B, t=T, **{dim_letters[i]: other_dims[i] for i in range(len(other_dims))})
        # n_dims = len(other_dims)
        # if n_dims == 1:
        #     return rearrange(x, '(b s) t h -> b t s h', b=B, t=T, s=other_dims[0])
        # elif n_dims == 2:
        #     return rearrange(x, '(b sx sy) t h -> b t sx sy h', b=B, t=T, sx=other_dims[0], sy=other_dims[1])
        # elif n_dims == 3:
        #     return rearrange(x, '(b sx sy sz) t h -> b t sx sy sz h', b=B, t=T, sx=other_dims[0], sy=other_dims[1], sz=other_dims[2])

    def step(self, x, input_shape):
        '''Reshapes x to ((B,Sx,[Sy],[Sz]),H) -> (B,Sx,[Sy],[Sz],H)'''
        B = input_shape[0]
        other_dims = input_shape[1:-1]  # This will capture Sx, Sy, Sz if they exist
        dim_letters = ['sx', 'sy', 'sz']
        pattern = ' '.join([f'{dim_letters[i]}' for i in range(len(other_dims))])
        rearrange_pattern = f'(b {pattern}) h -> b {pattern} h'

        return rearrange(x, rearrange_pattern, b=B, **{dim_letters[i]: other_dims[i] for i in range(len(other_dims))})

class SpaceToHidden(InputProcessor):
    skip_step = False
    def forward(self, x):
        '''Reshapes x to (B,T,S,H) -> (B,T,(S*H))'''
        B, T, S, H = x.shape
        return rearrange(x, 'b t s h -> b t (s h)')
    
    def step(self, x):
        '''Reshapes x to (B,S,H) -> (B,S*D)'''
        B, S, H = x.shape
        return rearrange(x, 'b s h -> b (s h)')  

class SpaceFromHidden(OutputProcessor):
    def forward(self, x, input_shape):
        '''Reshapes x to (B,T,(S*H)) -> (B,T,S,H)'''
        B, T, S, H = input_shape
        return rearrange(x, 'b t (s h) -> b t s h', b=B, t=T, s=S, h=H)

    def step(self, x, input_shape):
        '''Reshapes x to (B,S*H) -> (B,S,H)'''
        B, S, H = input_shape
        return rearrange(x, 'b (s h) -> b s h', b=B, s=S, h=H)

class IO(nn.Module): 
    def __init__(self, input_processor : InputProcessor, output_processor : OutputProcessor):
        '''Processing input of architecture and output'''
        super().__init__()
        self.input_processor = input_registry[input_processor]()
        self.output_processor = output_registry[output_processor]()
        self.input_shape = None

    def process_input(self, x):
        self.input_shape = x.shape
        return self.input_processor(x)
    
    def step_input(self, x):
        self.input_shape = x.shape
        return self.input_processor.step(x)

    def process_output(self, x):
        assert self.input_shape is not None, "Input shape is not set. Please call process_input first."
        return self.output_processor(x, self.input_shape)
    
    def step_output(self, x):
        assert self.input_shape is not None, "Input shape is not set. Please call process_input first."
        return self.output_processor.step(x, self.input_shape)
    
    def __repr__(self):
        return f"Input: {self.input_processor} -> Output: {self.output_processor}"  

class BatchSpace(InputProcessor):
    skip_step = False
    def forward(self, x):
        '''Reshapes x to (B,T,Sx,[Sy],[Sz],H) -> (B*Sx[*Sy][*Sz],T,H)'''
        return rearrange(x, 'b t ... h -> (b ...) t h')
    
    def step(self, x):
        '''Reshapes x from (B,Sx,[Sy],[Sz],H) -> (B*Sx[*Sy][*Sz],H)'''
        return rearrange(x, 'b ... h -> (b ...) h')

class BatchSpaceConv(InputProcessor):
    '''Applies a convolution to the spatial dimensions and then reshapes the tensor'''
    skip_step = False
    def __init__(self, d_model = 128, kernel_size = 3, stride = 1, padding = "same", dim=1):
        super().__init__()
        if dim == 1:
            self.conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size, stride, padding, device = device),
                # nn.GELU(),
                # nn.Conv1d( d_model, d_model, kernel_size, stride, padding, device = device),
            )
        elif dim == 2: 
            self.conv = nn.Conv2d(d_model, d_model, kernel_size, stride, padding, device = device)
        else: 
            raise ValueError("Only 1D and 2D convolutions are supported")
    
    def forward(self, x):
        '''Reshapes x to (B,T,Sx,[Sy],[Sz],H) -> (B*Sx[*Sy][*Sz],T,H)'''
        B, T, *spatial_shape, H = x.shape
        x = rearrange(x, 'b t ... h -> (b t) h ...')
        x = self.conv(x)
        return rearrange(x, '(b t)  h ... -> (b ...) t h', b=B, t=T) 
          
act_registry = {"gelu" : F.gelu,
                "identity": nn.Identity(),
                "relu": F.relu}

grid_input_registry = {"Concat": Concat,
                        "ConcatND": ConcatND,
                        "ConcatFlatTrans": ConcatFlatTrans,
                        "ConcatTrans": ConcatTrans,
                        "ConcatTransSqueeze1D": ConcatTransSqueeze1D}

input_registry = {"identity": InputId,
                  "BatchTime": BatchTime,
                #   "BatchSpaceFourier": BatchSpaceFourier,
                  "BatchSpace": BatchSpace,
                  "BatchSpaceConv": BatchSpaceConv,
                  "SpaceToHidden": SpaceToHidden}


output_registry = {"identity": Id,
                   "Squeeze": Squeeze,
                   "Unsqueeze": Unsqueeze,
                   "UnflatTrans": UnflatTrans,
                   "UnbatchTime": UnbatchTime,
                   "UnbatchSpace": UnbatchSpace,
                    #  "UnbatchSpaceFourier": UnbatchSpaceFourier,
                   "SpaceFromHidden": SpaceFromHidden,
                   "Trans": Trans}    

