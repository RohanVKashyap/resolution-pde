import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import glob
import h5py
import numpy as np
from einops import rearrange

from utils.res_utils import downsample, resize

class ActiveMatterMarkovDataset(Dataset):
    def __init__(self, 
                 filename, 
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 s=None,  # Target spatial resolution
                 **kwargs):
        
        assert reduced_resolution == 1, "reduced_resolution must be 1 when using parameter 's' for downsampling. Use 's' parameter instead of reduced_resolution for spatial downsampling."
        
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        print(f'Loading from file: {root_path}')
        
        # Load HDF5 data
        self._load_hdf5_data(root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max)
        
        # Apply resolution change if s is specified and different from current resolution
        current_spatial_size = self.data.shape[2]  # Assuming square grid
        print('Current Spatial Size:', current_spatial_size)

        if s is not None and s != current_spatial_size:
            print(f"Resizing from {current_spatial_size}x{current_spatial_size} to {s}x{s}")
            
            # Reshape for resizing: (batch, time, height, width, channels) 
            # -> (batch*time*channels, 1, height, width)
            batch_size, time_steps, height, width, channels = self.data.shape
            data_reshaped = self.data.transpose(0, 1, 4, 2, 3)  # (batch, time, channels, height, width)
            data_reshaped = data_reshaped.reshape(batch_size * time_steps * channels, 1, height, width)
            
            # Choose appropriate resizing method based on target size
            if s < current_spatial_size:
                # Downsampling: use downsample function
                print(f"Downsampling using downsample function")
                data_resized = downsample(data_reshaped, s)
            else:
                # Upsampling: use resize function
                print(f"Upsampling using resize function")
                # Convert to torch tensor for resize function
                data_torch = torch.tensor(data_reshaped, dtype=torch.float32)
                data_resized = resize(data_torch, (s, s))
                # Convert back to numpy
                data_resized = data_resized.numpy()
            
            # Reshape back: (batch*time*channels, 1, s, s) -> (batch, time, s, s, channels)
            data_resized = data_resized.reshape(batch_size, time_steps, channels, s, s)
            data_resized = data_resized.transpose(0, 1, 3, 4, 2)  # (batch, time, height, width, channels)
            
            self.data = data_resized
            print(f"Data shape after resizing: {self.data.shape}")
            
            # Update grid for new resolution
            if hasattr(self, 'grid') and self.grid is not None:
                x_coords = np.linspace(0, 1, s)
                y_coords = np.linspace(0, 1, s)
                xx, yy = np.meshgrid(x_coords, y_coords)
                self.grid = np.stack([xx, yy], axis=-1)
                self.grid = torch.tensor(self.grid, dtype=torch.float)
        
        # Extract the input sequence (all timesteps except last)
        x = self.data[:, :-1]  # (batch, time-1, height, width, channels)
        # Extract the output sequence (all timesteps except first)
        y = self.data[:, 1:]   # (batch, time-1, height, width, channels)
        
        # Flatten batch and time dimensions together
        batch_size, time_steps = x.shape[0], x.shape[1]
        height, width, channels = x.shape[2], x.shape[3], x.shape[4]
        
        # Reshape to (batch*time, channels, height, width) format
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        
        self.x = rearrange(self.x, 'b t h w c -> (b t) c h w')
        self.y = rearrange(self.y, 'b t h w c -> (b t) c h w')
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"x shape: {self.x.shape}")
        print(f"y shape: {self.y.shape}")
        if hasattr(self, 'grid') and self.grid is not None:
            print(f"grid shape: {self.grid.shape}")
    
    def _load_hdf5_data(self, root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max):
        """Load data from HDF5 file format for active matter dataset"""
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            print(f"Available keys: {keys}")
            
            # Load concentration data: shape (1, 81, 256, 256)
            concentration_data = np.array(f['t0_fields/concentration'], dtype=np.float32)
            print(f"Concentration data shape: {concentration_data.shape}")
            
            # Load velocity data: shape (1, 81, 256, 256, 2)
            velocity_data = np.array(f['t1_fields/velocity'], dtype=np.float32)
            print(f"Velocity data shape: {velocity_data.shape}")
            
            # Load scalar parameters
            if 'scalars/alpha' in f:
                self.alpha = f['scalars/alpha'][()]
                print(f"Alpha: {self.alpha}")
            if 'scalars/zeta' in f:
                self.zeta = f['scalars/zeta'][()]
                print(f"Zeta: {self.zeta}")
            
            # Expand concentration to have channel dimension: (1, 81, 256, 256) -> (1, 81, 256, 256, 1)
            concentration_data = np.expand_dims(concentration_data, axis=-1)
            print(f"Concentration data shape after expanding: {concentration_data.shape}")
            
            # Apply reductions
            concentration_data = concentration_data[::reduced_batch, 
                                                  ::reduced_resolution_t, 
                                                  ::reduced_resolution, 
                                                  ::reduced_resolution, :]
            
            velocity_data = velocity_data[::reduced_batch, 
                                         ::reduced_resolution_t, 
                                         ::reduced_resolution, 
                                         ::reduced_resolution, :]
            
            # Concatenate concentration and velocity: (1, 81, 256, 256, 1) + (1, 81, 256, 256, 2) -> (1, 81, 256, 256, 3)
            self.data = np.concatenate([concentration_data, velocity_data], axis=-1)
            print(f"Combined data shape: {self.data.shape}")
            print(f"Channels: [concentration, vx, vy]")
            
            # Create grid (assuming uniform grid from 0 to 1)
            self.grid = None
            current_height, current_width = self.data.shape[2], self.data.shape[3]
            x_coords = np.linspace(0, 1, current_width)
            y_coords = np.linspace(0, 1, current_height)
            xx, yy = np.meshgrid(x_coords, y_coords)
            self.grid = np.stack([xx, yy], axis=-1)
            self.grid = torch.tensor(self.grid, dtype=torch.float)
            print(f"Grid shape: {self.grid.shape}")

            # Apply sample limit
            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, self.data.shape[0])
            else:
                num_samples_max = self.data.shape[0]

            self.data = self.data[:num_samples_max]
            
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if hasattr(self, 'grid') and self.grid is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx], self.y[idx]


def active_matter_markov_dataset(filename, saved_folder, data_normalizer=True, s=None, **kwargs):
    """
    Returns train, validation, and test datasets for Active Matter with 0.8/0.1/0.1 ratio.
    Uses min-max normalization to [0, 1] range.
    
    Args:
        filename: HDF5 file name (.hdf5 extension)
        saved_folder: Path to folder containing the file
        data_normalizer: Whether to apply normalization
        s: Target spatial resolution for resizing (None to keep original)
        **kwargs: Additional arguments to pass to ActiveMatterMarkovDataset
        
    Returns:
        train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
    """
    # Create the full dataset
    full_dataset = ActiveMatterMarkovDataset(filename, saved_folder, s=s, **kwargs)
    
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
    
    min_data = None
    max_data = None
    min_model = None
    max_model = None
    
    if data_normalizer:
        print('---------Computing min-max normalization statistics---------------')
        temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        
        # Collect all training data for computing min-max statistics
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
        
        # Compute min-max statistics
        min_data = float(x_train_tensor.min())
        max_data = float(x_train_tensor.max())
        min_model = float(y_train_tensor.min())
        max_model = float(y_train_tensor.max())
        
        print(f"Input data range: [{min_data:.6f}, {max_data:.6f}]")
        print(f"Output data range: [{min_model:.6f}, {max_model:.6f}]")
        
        # Create a wrapper dataset class that applies min-max normalization
        class MinMaxNormalizedDataset(Dataset):
            def __init__(self, dataset, min_data, max_data, min_model, max_model):
                self.dataset = dataset
                self.min_data = min_data
                self.max_data = max_data
                self.min_model = min_model
                self.max_model = max_model
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                x, y = self.dataset[idx]
                
                # Apply min-max normalization to [0, 1]
                x_normalized = (x - self.min_data) / (self.max_data - self.min_data)
                y_normalized = (y - self.min_model) / (self.max_model - self.min_model)
                
                return x_normalized, y_normalized
        
        # Apply normalization to each dataset
        train_dataset = MinMaxNormalizedDataset(train_dataset, min_data, max_data, min_model, max_model)
        val_dataset = MinMaxNormalizedDataset(val_dataset, min_data, max_data, min_model, max_model)
        test_dataset = MinMaxNormalizedDataset(test_dataset, min_data, max_data, min_model, max_model)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model