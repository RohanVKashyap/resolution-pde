import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
from einops import rearrange

from utils.res_utils import downsample_1d, resize_1d

class KSMarkovDataset(Dataset):
    def __init__(self, 
                 filename, 
                 saved_folder=None, 
                 reduced_batch=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 s=None,  # Target spatial resolution
                 **kwargs):
        
        # Handle full path or folder + filename
        if saved_folder is not None:
            root_path = os.path.join(os.path.abspath(saved_folder), filename)
        else:
            root_path = filename  # Assume filename is full path
            
        print(f'Loading from file: {root_path}')
        
        # Load KS data using torch.load
        _data = torch.load(root_path, map_location=torch.device('cpu'))
        
        # Convert to numpy for consistency with other processing
        if isinstance(_data, torch.Tensor):
            _data = _data.numpy().astype(np.float32)
        
        print(f"Original data shape: {_data.shape}")
        # Expected shape: (num_trajectories, time_steps, spatial_points)
        
        # Apply reductions for batch and time
        _data = _data[::reduced_batch, 
                      ::reduced_resolution_t, 
                      :]  # Don't apply spatial reduction here since we'll use s parameter
        
        # Create spatial grid (assuming domain is [0, 6π] based on your settings)
        num_spatial_points = _data.shape[2]
        domain_length = 6 * np.pi  # 2 * pi * half_period = 2 * pi * 3
        self.grid = np.linspace(0, domain_length, num_spatial_points, dtype=np.float32)
        self.grid = torch.tensor(self.grid, dtype=torch.float).unsqueeze(-1)
        print(f"Original grid shape: {self.grid.shape}")

        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, _data.shape[0])
        else:
            num_samples_max = _data.shape[0]

        self.data = _data[:num_samples_max]
        print(f"Data shape after batch/time reduction: {self.data.shape}")

        # Apply resolution change if s is specified and different from current resolution
        current_spatial_size = self.data.shape[2]  # Spatial dimension
        print(f'Current Spatial Size: {current_spatial_size}')

        if s is not None and s != current_spatial_size:
            print(f"Resizing from {current_spatial_size} to {s}")
            
            # Reshape for resizing: (batch, time, spatial) -> (batch*time, spatial)
            batch_size, time_steps, spatial_size = self.data.shape
            data_reshaped = self.data.reshape(batch_size * time_steps, spatial_size)
            
            # Choose appropriate resizing method based on target size
            if s < current_spatial_size:
                # Downsampling: use FFT-based downsample function
                print(f"Downsampling using FFT-based downsample_1d function")
                data_resized = downsample_1d(data_reshaped, s)
            else:
                # Upsampling: use FFT-based resize function
                print(f"Upsampling using FFT-based resize_1d function")
                # Convert to torch tensor for resize function
                data_torch = torch.tensor(data_reshaped, dtype=torch.float32)
                data_resized = resize_1d(data_torch, s)
                # Convert back to numpy
                data_resized = data_resized.numpy()
            
            # Reshape back: (batch*time, s) -> (batch, time, s)
            data_resized = data_resized.reshape(batch_size, time_steps, s)
            
            self.data = data_resized
            print(f"Data shape after resizing: {self.data.shape}")
            
            # Update grid coordinates for new resolution
            grid_start = 0.0
            grid_end = domain_length
            self.grid = torch.linspace(grid_start, grid_end, s, dtype=torch.float).unsqueeze(-1)
            print(f"Updated grid shape: {self.grid.shape}")

        # Create input-output pairs for Markov property
        # u(t) -> u(t + dt_save)
        x = self.data[:, :-1, :]  # Input: all timesteps except last
        self.x = rearrange(x, 'b t m -> (b t) 1 m')
        y = self.data[:, 1:, :]   # Output: all timesteps except first
        self.y = rearrange(y, 'b t m -> (b t) 1 m')
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"x shape: {self.x.shape}")
        print(f"y shape: {self.y.shape}")
        print(f"grid shape: {self.grid.shape}")
        print(f"Time step between input-output pairs: dt_save from your simulation")
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # return self.x[idx], self.y[idx], self.grid
        return self.x[idx], self.y[idx]


def ks_pino_markov_dataset(filename, saved_folder=None, data_normalizer=True, s=None, **kwargs):
    """
    Returns train, validation, and test datasets for KS equation with 0.8/0.1/0.1 ratio.
    Uses min-max normalization to [0, 1] range.
    
    Args:
        filename: .pt file name or full path
        saved_folder: Path to folder containing the file (None if filename is full path)
        data_normalizer: Whether to apply normalization
        s: Target spatial resolution for resizing (None to keep original)
        **kwargs: Additional arguments to pass to KSMarkovDataset
        
    Returns:
        train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
    """
    # Create the full dataset with optional resizing
    full_dataset = KSMarkovDataset(filename, saved_folder, s=s, **kwargs)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
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
                items = self.dataset[idx]
                
                if len(items) == 3:  # x, y, grid
                    x, y, grid = items
                    # Ensure x and y are PyTorch tensors, not NumPy arrays
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x).float()
                    if isinstance(y, np.ndarray):
                        y = torch.from_numpy(y).float()
                    
                    # Apply min-max normalization to [0, 1]
                    x_normalized = (x - self.min_data) / (self.max_data - self.min_data)
                    y_normalized = (y - self.min_model) / (self.max_model - self.min_model)
                    
                    return x_normalized, y_normalized, grid
                else:  # x, y
                    x, y = items
                    # Ensure x and y are PyTorch tensors, not NumPy arrays
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x).float()
                    if isinstance(y, np.ndarray):
                        y = torch.from_numpy(y).float()
                    
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


# Example usage:
# if __name__ == "__main__":
#     # Test with your KS data
#     train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model = ks_markov_dataset(
#         filename='/home/rvk/T=20,niu=0.01,N=1024,dt=0.001,6pi,dtsave=0.1,sample=51_ut.pt',
#         saved_folder=None,  # filename is full path
#         s=512,  # Resize to 512 spatial points (test resolution invariance)
#         data_normalizer=True,
#         reduced_batch=1,
#         reduced_resolution_t=1
#     )
    
#     print("Dataset created successfully!")
#     print(f"Train samples: {len(train_dataset)}")
#     print(f"Val samples: {len(val_dataset)}")
#     print(f"Test samples: {len(test_dataset)}")
    
#     # Test loading a batch
#     loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     for batch in loader:
#         if len(batch) == 3:
#             x, y, grid = batch
#             print(f"Batch x shape: {x.shape}, y shape: {y.shape}, grid shape: {grid.shape}")
#         else:
#             x, y = batch
#             print(f"Batch x shape: {x.shape}, y shape: {y.shape}")
#         break