import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np

def get_start_end(N, if_test, test_ratio, num_samples_max):
    if if_test: 
        # when testing, ignore num_samples_max
        start = int(N * (1-test_ratio))
        end = N
    elif num_samples_max > 0:
        if num_samples_max > int(N * (1-test_ratio)):
            raise ValueError(f"num_samples_max={num_samples_max} can't be larger than N * (1-test_ratio)={int(N * (1-test_ratio))}")
        start = 0
        end = num_samples_max
    else: 
        start = 0
        end = int(N * (1-test_ratio))
    return start, end

class H5DarcyDataset(Dataset):
    def __init__(self, 
                 filename,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max=-1):
        """
        Dataset for Darcy Flow problems, using permeability field (nu) as input
        and solution field (tensor) as output.
        
        Args:
            filename: Name of the h5 file
            saved_folder: Path to the folder containing the h5 file
            reduced_resolution: Factor by which to reduce spatial resolution
            reduced_batch: Factor by which to reduce batch size
            if_test: Whether to use test set
            test_ratio: Fraction of data to use for testing
            num_samples_max: Maximum number of samples to use (-1 for all)
        """
        # Define path to file
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        
        with h5py.File(root_path, 'r') as f:
            print(f'Loading from file: {root_path}')
            keys = list(f.keys())
            keys.sort()
            print('keys:', keys)
            print([(key, f[key].shape) for key in keys])
            
            # Get split indices
            N = f['nu'].shape[0]
            start, end = get_start_end(N, if_test, test_ratio, num_samples_max)
            
            # Load permeability field (input)
            nu_data = np.array(f['nu'], dtype=np.float32)[start:end:reduced_batch]
            nu_data = nu_data[:, ::reduced_resolution, ::reduced_resolution]
            
            # Load solution field (output)
            tensor_data = np.array(f['tensor'], dtype=np.float32)[start:end:reduced_batch]
            # Squeeze out time dimension if it's 1
            if tensor_data.shape[1] == 1:
                tensor_data = np.squeeze(tensor_data, axis=1)
            else:
                tensor_data = tensor_data[:, 0]  # Use only first time step
            
            tensor_data = tensor_data[:, ::reduced_resolution, ::reduced_resolution]
            
            # Setup grid
            x = np.array(f["x-coordinate"], dtype=np.float32)[::reduced_resolution]
            y = np.array(f["y-coordinate"], dtype=np.float32)[::reduced_resolution]
            
            # Create meshgrid
            x_grid = torch.tensor(x, dtype=torch.float)
            y_grid = torch.tensor(y, dtype=torch.float)
            X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
            self.grid = torch.stack((X, Y), axis=-1)
            
            # Convert data to tensors
            self.nu = torch.tensor(nu_data, dtype=torch.float)
            self.tensor = torch.tensor(tensor_data, dtype=torch.float)
            
            # Add channel dimension for CNN compatibility
            if len(self.nu.shape) == 3:  # [batch, x, y]
                self.nu = self.nu.unsqueeze(-1)  # -> [batch, x, y, 1]
            if len(self.tensor.shape) == 3:  # [batch, x, y]
                self.tensor = self.tensor.unsqueeze(-1)  # -> [batch, x, y, 1]
            
            print(f"Input shape: {self.nu.shape}")
            print(f"Output shape: {self.tensor.shape}")
            print(f"Grid shape: {self.grid.shape}")

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.nu)
    
    def __getitem__(self, idx):
        """Return input, output, and grid for the given index"""
        return self.nu[idx], self.tensor[idx], self.grid