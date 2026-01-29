import torch
from torch.utils.data import Dataset, DataLoader, random_split
from models.custom_layer import UnitGaussianNormalizer
import os
import glob
import h5py
import numpy as np
import math as mt
from einops import rearrange

class NavierStokesWindowDataset(Dataset):
    def __init__(self, 
                 filename, 
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 window_size=10,  # Number of previous timesteps to use for prediction
                 flatten_window=True,  # Whether to flatten the window dimension
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
                print(f"Time data shape: {self.time.shape}")
            
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
            
            # Combine particles and velocity into a single data tensor
            # Combine the last dimensions: particles (1) + velocity (2) = 3 channels
            self.data = np.concatenate([particles_data, velocity_data], axis=-1)
            print(f"Combined data shape: {self.data.shape}")
            # Shape should be (num_simulations, timesteps, height, width, 3)
            
        # Check if we have enough timesteps
        if self.data.shape[1] < window_size + 1:
            raise ValueError(f"Dataset has only {self.data.shape[1]} timesteps, need at least {window_size + 1}")
        
        # Store parameters for dynamic computation
        self.window_size = window_size
        self.flatten_window = flatten_window
        
        # Pre-compute valid sample indices instead of pre-processing all samples
        self.sample_indices = []
        for sim_idx in range(self.data.shape[0]):
            for t in range(self.data.shape[1] - window_size):
                self.sample_indices.append((sim_idx, t))
        
        print(f"Dataset initialized with {len(self.sample_indices)} samples")
        print(f"Raw data shape: {self.data.shape}")
        print(f"Window size: {window_size}, Flatten window: {flatten_window}")
        
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        # Dynamically compute the sample on-demand
        sim_idx, t_start = self.sample_indices[idx]
        
        # Extract the simulation data
        sim_data = self.data[sim_idx]  # shape: [timesteps, height, width, 3]
        
        # Create window-target pair for this specific index
        # Window: window_size consecutive timesteps
        window = sim_data[t_start:t_start+self.window_size]  # shape: [window_size, height, width, 3]
        # Target: the timestep after the window
        target = sim_data[t_start+self.window_size]  # shape: [height, width, 3]
        
        # Convert to PyTorch tensors
        x = torch.tensor(window, dtype=torch.float)
        y = torch.tensor(target, dtype=torch.float)
        
        # Reshape data for model input based on flatten_window option
        if self.flatten_window:
            # Flatten window dimension with channels
            # [window_size, height, width, 3] -> [window_size*3, height, width]
            x = rearrange(x, 't h w c -> (t c) h w')
        else:
            # Keep window dimension separate but move channels to proper position for convolution
            # [window_size, height, width, 3] -> [3, window_size, height, width]
            # This format preserves the window structure while being suitable for 3D convolutions
            x = rearrange(x, 't h w c -> c t h w')
        
        # Move channel dimension to proper position for target
        # [height, width, 3] -> [3, height, width]
        y = rearrange(y, 'h w c -> c h w')
        
        if self.grid is not None:
            return x, y, self.grid
        else:
            return x, y

# def ns_window_dataset(filename, saved_folder, data_normalizer=True, window_size=10, flatten_window=False, **kwargs):
#     full_dataset = NavierStokesWindowDataset(
#         filename, 
#         saved_folder, 
#         window_size=window_size, 
#         flatten_window=flatten_window, 
#         **kwargs)
    
#     dataset_size = len(full_dataset)
#     train_size = int(0.8 * dataset_size)
#     val_size = int(0.1 * dataset_size)
#     test_size = dataset_size - train_size - val_size
    
#     train_dataset, val_dataset, test_dataset = random_split(
#         full_dataset, 
#         [train_size, val_size, test_size],
#         generator=torch.Generator().manual_seed(42)
#     )
    
#     x_normalizer = None
#     y_normalizer = None
    
#     if data_normalizer:
#         x_normalizer, y_normalizer = compute_normalizers_incrementally(train_dataset, batch_size=4)
        
#         class NormalizedDataset(Dataset):
#             def __init__(self, dataset, x_normalizer, y_normalizer):
#                 self.dataset = dataset
#                 self.x_normalizer = x_normalizer
#                 self.y_normalizer = y_normalizer
            
#             def __len__(self):
#                 return len(self.dataset)
            
#             def __getitem__(self, idx):
#                 item = self.dataset[idx]
                
#                 if len(item) == 3:  # If grid is included
#                     x, y, grid = item
#                     return self.x_normalizer.encode(x), self.y_normalizer.encode(y), grid
#                 else:
#                     x, y = item
#                     return self.x_normalizer.encode(x), self.y_normalizer.encode(y)
        
#         train_dataset = NormalizedDataset(train_dataset, x_normalizer, y_normalizer)
#         val_dataset = NormalizedDataset(val_dataset, x_normalizer, y_normalizer)
#         test_dataset = NormalizedDataset(test_dataset, x_normalizer, y_normalizer)
    
#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
#     print(f"Test dataset size: {len(test_dataset)}")
    
#     return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer        


def ns_window_dataset(filename, saved_folder, data_normalizer=True, window_size=10, flatten_window=False, **kwargs):
    """
    Returns train, validation, and test datasets for Navier-Stokes with particles using window-based prediction with 0.8/0.1/0.1 ratio.
    
    Args:
        filename: H5 file name
        saved_folder: Path to folder containing the file
        data_normalizer: Whether to apply normalization
        window_size: Number of previous timesteps to use for prediction
        flatten_window: Whether to flatten the window dimension
        **kwargs: Additional arguments to pass to NavierStokesWindowDataset
        
    Returns:
        train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
    """
    # Create the full dataset
    full_dataset = NavierStokesWindowDataset(
        filename, 
        saved_folder, 
        window_size=window_size, 
        flatten_window=flatten_window, 
        **kwargs
    )
    
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
        temp_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        
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
                item = self.dataset[idx]
                
                if len(item) == 3:  # If grid is included
                    x, y, grid = item
                    return self.x_normalizer.encode(x), self.y_normalizer.encode(y), grid
                else:
                    x, y = item
                    return self.x_normalizer.encode(x), self.y_normalizer.encode(y)
        
        # Apply normalization to each dataset
        train_dataset = NormalizedDataset(train_dataset, x_normalizer, y_normalizer)
        val_dataset = NormalizedDataset(val_dataset, x_normalizer, y_normalizer)
        test_dataset = NormalizedDataset(test_dataset, x_normalizer, y_normalizer)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer