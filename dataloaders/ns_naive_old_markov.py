import torch
from torch.utils.data import Dataset, DataLoader, random_split
from models.custom_layer import UnitGaussianNormalizer
import os
import glob
import h5py
import numpy as np
import math as mt
from einops import rearrange

class NavierStokesMarkovDataset(Dataset):
    def __init__(self, 
                 filename, 
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 **kwargs):
        
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        with h5py.File(root_path, 'r') as f:
            print(f'Loading from file: {root_path}')
            keys = list(f.keys())
            print(f"Available keys: {keys}")
            
            # Load velocity data
            velocity_data = np.array(f['velocity'], dtype=np.float32)
            print(f"Velocity data shape: {velocity_data.shape}")
            # Shape: (num_simulations, timesteps, height, width, 2)
            
            # Load particle data
            particles_data = np.array(f['particles'], dtype=np.float32)
            print(f"Particles data shape: {particles_data.shape}")
            # Shape: (num_simulations, timesteps, height, width, 1)
            
            # Apply reductions
            velocity_data = velocity_data[::reduced_batch, 
                                         ::reduced_resolution_t, 
                                         ::reduced_resolution, 
                                         ::reduced_resolution, :]
            
            particles_data = particles_data[::reduced_batch, 
                                           ::reduced_resolution_t, 
                                           ::reduced_resolution, 
                                           ::reduced_resolution, :]
            
            # Load time data if available
            if 't' in keys:
                self.time = np.array(f['t'], dtype=np.float32)
                self.time = self.time[::reduced_batch, ::reduced_resolution_t]
                # print(f"Time data shape: {self.time.shape}")
            
            # Load grid data if available (assuming uniform grid)
            self.grid = None
            if 'x-coordinate' in keys and 'y-coordinate' in keys:
                x_coords = np.array(f['x-coordinate'], dtype=np.float32)
                y_coords = np.array(f['y-coordinate'], dtype=np.float32)
                x_coords = x_coords[::reduced_resolution]
                y_coords = y_coords[::reduced_resolution]
                xx, yy = np.meshgrid(x_coords, y_coords)
                self.grid = np.stack([xx, yy], axis=-1)
                self.grid = torch.tensor(self.grid, dtype=torch.float)
                print(f"Grid shape: {self.grid.shape}")

            # Apply sample limit
            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, velocity_data.shape[0])
            else:
                num_samples_max = velocity_data.shape[0]

            velocity_data = velocity_data[:num_samples_max]
            particles_data = particles_data[:num_samples_max]
            
            # print(f"Velocity data shape after reduction: {velocity_data.shape}")
            # print(f"Particles data shape after reduction: {particles_data.shape}")
            
            # Combine particles and velocity into a single data tensor
            # Combine the last dimensions: particles (1) + velocity (2) = 3 channels
            self.data = np.concatenate([particles_data, velocity_data], axis=-1)
            print(f"Combined data shape: {self.data.shape}")
            # Shape should be (num_simulations, timesteps, height, width, 3)
        
        # Extract the input sequence (all timesteps except first and last)
        x = self.data[:, 1:-1]
        # Extract the output sequence (all timesteps except first two)
        y = self.data[:, 2:]
        
        # Flatten batch and time dimensions together
        batch_size, time_steps = x.shape[0], x.shape[1]
        height, width, channels = x.shape[2], x.shape[3], x.shape[4]
        
        # Reshape to (batch*time, channels, height, width) format for FFNO2D
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        
        self.x = rearrange(self.x, 'b t h w c -> (b t) c h w')
        self.y = rearrange(self.y, 'b t h w c -> (b t) c h w')
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"x shape: {self.x.shape}")
        print(f"y shape: {self.y.shape}")
        if self.grid is not None:
            print(f"grid shape: {self.grid.shape}")
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.grid is not None:
            return self.x[idx], self.y[idx]       # self.grid
        else:
            return self.x[idx], self.y[idx]       # None


def ns_markov_dataset(filename, saved_folder, data_normalizer=True, **kwargs):
    """
    Returns train, validation, and test datasets for Navier-Stokes with particles with 0.8/0.1/0.1 ratio.
    
    Args:
        filename: H5 file name
        saved_folder: Path to folder containing the file
        data_normalizer: Whether to apply normalization
        **kwargs: Additional arguments to pass to NavierStokesMarkovDataset
        
    Returns:
        train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
    """
    # Create the full dataset
    full_dataset = NavierStokesMarkovDataset(filename, saved_folder, **kwargs)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    x_normalizer = None
    y_normalizer = None
    
    if data_normalizer:
        print('---------Using data normalizer---------------')
        temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        
        # Collect all training data for computing normalization
        x_train_all = []
        y_train_all = []
        for batch in temp_loader:
            if len(batch) == 3:  # If grid is included
                x_batch, y_batch, _ = batch
            else:
                x_batch, y_batch = batch
                
            x_train_all.append(x_batch)
            y_train_all.append(y_batch)
        
        x_train_tensor = torch.cat(x_train_all, dim=0)
        y_train_tensor = torch.cat(y_train_all, dim=0)
        
        # Initialize normalizers using training data
        x_normalizer = UnitGaussianNormalizer(x_train_tensor)
        y_normalizer = UnitGaussianNormalizer(y_train_tensor)
        
        # Create a wrapper dataset class that applies normalization
        class NormalizedDataset(Dataset):
            def __init__(self, dataset, x_normalizer, y_normalizer):
                self.dataset = dataset
                self.x_normalizer = x_normalizer
                self.y_normalizer = y_normalizer
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                x, y = self.dataset[idx]
                
                # if len(item) == 3:  # If grid is included
                #     x, y, grid = item
                #     return self.x_normalizer.encode(x), self.y_normalizer.encode(y)   # grid
                # else:
                #     x, y = item
                return self.x_normalizer.encode(x), self.y_normalizer.encode(y)
        
        # Apply normalization to each dataset
        train_dataset = NormalizedDataset(train_dataset, x_normalizer, y_normalizer)
        val_dataset = NormalizedDataset(val_dataset, x_normalizer, y_normalizer)
        test_dataset = NormalizedDataset(test_dataset, x_normalizer, y_normalizer)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer