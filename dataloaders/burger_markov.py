import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import os
import glob
import h5py
import numpy as np
import math as mt
from einops import rearrange

class H5pyMarkovDataset(Dataset):
    def __init__(self, 
                 filename, 
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 test_ratio=0.1,
                 if_test=False,
                 **kwargs,):
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        with h5py.File(root_path, 'r') as f:
            print(f'Loading from file: {root_path}')
            keys = list(f.keys())
            keys.sort()
            _data = np.array(f['tensor'], dtype=np.float32)
            print(_data.shape)
            _data = _data[::reduced_batch, 
                          ::reduced_resolution_t, 
                          ::reduced_resolution]
            self.grid = np.array(f["x-coordinate"], dtype=np.float32)
            self.grid = torch.tensor(
                self.grid[::reduced_resolution],
                 dtype=torch.float).unsqueeze(-1)

            # print(num_samples_max)
            if num_samples_max>0:
                num_samples_max  = min(num_samples_max, _data.shape[0])
            else:
                num_samples_max = _data.shape[0]

            test_idx = int(num_samples_max * test_ratio)
            if if_test:
                self.data = _data[:test_idx]
            else:
                self.data = _data[test_idx:num_samples_max]
            print(self.data.shape)

        x = self.data[:, 1:-1, :]
        self.x = rearrange(x, 'b t m -> (b t) m 1 1')
        y = self.data[:, 2:, :]
        self.y = rearrange(y, 'b t m -> (b t) m 1 1')
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"Input shape: {self.x.shape}")
        print(f"Output shape: {self.y.shape}")
        print(f"Grid shape: {self.grid.shape}")
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.grid