import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import h5py
import numpy as np
from einops import rearrange

# Import your 1D resize functions
from utils.res_utils import downsample_1d, resize_1d

class KSMarkovDataset(Dataset):
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
        
        # Determine which split from filename
        if 'train' in filename.lower():
            self.split = 'train'
        elif 'valid' in filename.lower():
            self.split = 'valid'
        elif 'test' in filename.lower():
            self.split = 'test'
        else:
            # Fallback: assume train split
            self.split = 'train'
            print(f"Warning: Could not determine split from filename {filename}, assuming 'train'")
        
        self._load_ks_data(root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max)
        
        # Apply resolution change if s is specified and different from current resolution
        current_spatial_size = self.data.shape[2]  # Spatial dimension
        print('Current Spatial Size:', current_spatial_size)

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
            
            # Update spatial coordinates for new resolution
            if hasattr(self, 'x_coords') and self.x_coords is not None:
                # Create new spatial coordinates
                if hasattr(self, 'original_domain_length'):
                    domain_length = self.original_domain_length
                else:
                    domain_length = self.x_coords[-1] - self.x_coords[0]
                
                self.x_coords = np.linspace(self.x_coords[0], self.x_coords[0] + domain_length, s)
                print(f"Updated spatial coordinates shape: {self.x_coords.shape}")
        
        # Extract the input sequence (all timesteps except last)
        x = self.data[:, :-1]  # (batch, time-1, spatial)
        # Extract the output sequence (all timesteps except first)
        y = self.data[:, 1:]   # (batch, time-1, spatial)
        
        # Flatten batch and time dimensions together and add channel dimension
        batch_size, time_steps = x.shape[0], x.shape[1]
        spatial_size = x.shape[2]
        
        # Reshape to (batch*time, channels=1, spatial) format for consistency with 2D version
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        
        # Add channel dimension: (batch, time, spatial) -> (batch*time, 1, spatial)
        self.x = rearrange(self.x, 'b t s -> (b t) 1 s')
        self.y = rearrange(self.y, 'b t s -> (b t) 1 s')
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"x shape: {self.x.shape}")
        print(f"y shape: {self.y.shape}")
        if hasattr(self, 'x_coords') and self.x_coords is not None:
            print(f"spatial coordinates shape: {self.x_coords.shape}")
    
    def _load_ks_data(self, root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max):
        """Load data from KS HDF5 file format"""
        with h5py.File(root_path, 'r') as f:
            # Navigate to the correct split
            if self.split in f:
                group = f[self.split]
            else:
                # If split not found, try to find data directly
                keys = list(f.keys())
                print(f"Available keys: {keys}")
                if len(keys) == 1:
                    group = f[keys[0]]
                    print(f"Using group: {keys[0]}")
                else:
                    raise ValueError(f"Could not find split '{self.split}' in file. Available keys: {keys}")
            
            data_keys = list(group.keys())
            print(f"Available data keys in '{self.split}': {data_keys}")
            
            # Find the main PDE data key
            pde_key = None
            for key in data_keys:
                if 'pde' in key.lower() and '-' in key:
                    pde_key = key
                    break
            
            if pde_key is None:
                raise ValueError(f"Could not find PDE data key in {data_keys}")
            
            # Load main PDE data: (batch, time, spatial)
            pde_data = np.array(group[pde_key], dtype=np.float32)
            print(f"PDE data shape: {pde_data.shape}")
            
            # Load time data if available
            if 't' in data_keys:
                self.time = np.array(group['t'], dtype=np.float32)
                print(f"Time data shape: {self.time.shape}")
            else:
                self.time = None
            
            # Load spatial coordinates if available
            if 'x' in data_keys:
                x_coords = np.array(group['x'], dtype=np.float32)
                # x might be (batch, spatial) or just (spatial)
                if len(x_coords.shape) == 2:
                    self.x_coords = x_coords[0]  # Assume all batches have same spatial coords
                else:
                    self.x_coords = x_coords
                print(f"Spatial coordinates shape: {self.x_coords.shape}")
                # Store original domain length for resizing
                self.original_domain_length = self.x_coords[-1] - self.x_coords[0]
            else:
                self.x_coords = None
            
            # Load dx and dt if available
            if 'dx' in data_keys:
                self.dx = np.array(group['dx'], dtype=np.float32)
                print(f"dx shape: {self.dx.shape}")
            else:
                self.dx = None
                
            if 'dt' in data_keys:
                self.dt = np.array(group['dt'], dtype=np.float32)
                print(f"dt shape: {self.dt.shape}")
            else:
                self.dt = None

            pde_data = pde_data[::reduced_batch, 
                               ::reduced_resolution_t, 
                               :]                           # Use resize_1d function for resolution             
            
            # Apply sample limit
            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, pde_data.shape[0])
            else:
                num_samples_max = pde_data.shape[0]

            self.data = pde_data[:num_samples_max]
            print(f"Final data shape: {self.data.shape}")
            
            # Also reduce time and coordinate arrays if they exist
            if self.time is not None:
                if len(self.time.shape) == 2:  # (batch, time)
                    self.time = self.time[:num_samples_max, ::reduced_resolution_t]
                else:  # (time,)
                    self.time = self.time[::reduced_resolution_t]
                    
            if self.x_coords is not None and reduced_resolution != 1:
                self.x_coords = self.x_coords[::reduced_resolution]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # if hasattr(self, 'x_coords') and self.x_coords is not None:
        #     # Include spatial coordinates as additional info
        #     return self.x[idx], self.y[idx], torch.tensor(self.x_coords, dtype=torch.float)
        # else:
        return self.x[idx], self.y[idx]

def ks_markov_dataset(filename, saved_folder, data_normalizer=True, 
                      normalization_type="simple",  # "simple" or "minmax"
                      val_filename="KS_valid.h5", test_filename="KS_test.h5", **kwargs):
    """
    Returns train, validation, and test datasets for Kuramoto-Sivashinsky using separate files.
    Supports both simple Gaussian normalization and min-max normalization.
    
    Args:
        filename: Training file name (e.g., "KS_train_2048.h5")
        saved_folder: Path to folder containing the files
        data_normalizer: Whether to apply normalization
        normalization_type: Type of normalization - "simple" (UnitGaussian-like) or "minmax"
        val_filename: Validation file name (default: "KS_valid.h5")
        test_filename: Test file name (default: "KS_test.h5")
        **kwargs: Additional arguments to pass to KSMarkovDataset
        
    Returns:
        For simple normalization: train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
        For minmax normalization: train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
    """
    # Create separate datasets for train, validation, and test
    train_dataset = KSMarkovDataset(filename, saved_folder, **kwargs)
    val_dataset = KSMarkovDataset(val_filename, saved_folder, **kwargs)
    test_dataset = KSMarkovDataset(test_filename, saved_folder, **kwargs)
    
    # Initialize return values
    x_normalizer = None
    y_normalizer = None
    min_data = None
    max_data = None
    min_model = None
    max_model = None

    if data_normalizer:
        if normalization_type == "simple":
            print('---------Using simple global normalization---------------')
            
            # Collect all values as flat lists
            all_x_values = []
            all_y_values = []
            
            for x, y in train_dataset:
                all_x_values.extend(x.flatten().tolist())
                all_y_values.extend(y.flatten().tolist())
            
            # Compute global statistics
            x_tensor = torch.tensor(all_x_values)
            y_tensor = torch.tensor(all_y_values)
            
            x_mean, x_std = x_tensor.mean(), x_tensor.std()
            y_mean, y_std = y_tensor.mean(), y_tensor.std()
            
            print(f"Global statistics: X(mean={x_mean:.6f}, std={x_std:.6f}), Y(mean={y_mean:.6f}, std={y_std:.6f})")
            print(f"Computed from {len(train_dataset)} samples")
            
            # Simple normalizer class
            class SimpleNormalizer:
                def __init__(self, mean, std, eps=1e-8):
                    self.mean = float(mean)
                    self.std = float(std)
                    self.eps = eps
                
                def encode(self, x):
                    return (x - self.mean) / (self.std + self.eps)
                
                def decode(self, x, device='cuda'):
                    return x * (self.std + self.eps) + self.mean
                
                def cuda(self):
                    return self
                
                def cpu(self):
                    return self
            
            x_normalizer = SimpleNormalizer(x_mean, x_std)
            y_normalizer = SimpleNormalizer(y_mean, y_std)
            
            # Simple wrapper dataset
            class SimpleNormalizedDataset(Dataset):
                def __init__(self, dataset, x_normalizer, y_normalizer):
                    self.dataset = dataset
                    self.x_normalizer = x_normalizer
                    self.y_normalizer = y_normalizer
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    x, y = self.dataset[idx]
                    return self.x_normalizer.encode(x), self.y_normalizer.encode(y)
            
            # Apply normalization to each dataset
            train_dataset = SimpleNormalizedDataset(train_dataset, x_normalizer, y_normalizer)
            val_dataset = SimpleNormalizedDataset(val_dataset, x_normalizer, y_normalizer)
            test_dataset = SimpleNormalizedDataset(test_dataset, x_normalizer, y_normalizer)
            
        elif normalization_type == "minmax":
            print('---------Computing min-max normalization statistics---------------')
            temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
            
            # Collect all training data for computing min-max statistics
            x_train_all = []
            y_train_all = []
            for batch in temp_loader:
                if len(batch) == 3:  # If spatial coordinates are included
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
                    if len(items) == 3:  # x, y, spatial_coords
                        x, y, coords = items
                        # Apply min-max normalization to [0, 1]
                        x_normalized = (x - self.min_data) / (self.max_data - self.min_data)
                        y_normalized = (y - self.min_model) / (self.max_model - self.min_model)
                        return x_normalized, y_normalized, coords
                    else:  # x, y
                        x, y = items
                        # Apply min-max normalization to [0, 1]
                        x_normalized = (x - self.min_data) / (self.max_data - self.min_data)
                        y_normalized = (y - self.min_model) / (self.max_model - self.min_model)
                        return x_normalized, y_normalized
            
            # Apply normalization to each dataset
            train_dataset = MinMaxNormalizedDataset(train_dataset, min_data, max_data, min_model, max_model)
            val_dataset = MinMaxNormalizedDataset(val_dataset, min_data, max_data, min_model, max_model)
            test_dataset = MinMaxNormalizedDataset(test_dataset, min_data, max_data, min_model, max_model)
            
        else:
            raise ValueError(f"Invalid normalization_type: {normalization_type}. Must be 'simple' or 'minmax'")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Return appropriate values based on normalization type
    if normalization_type == "simple":
        return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
    else:  # minmax
        return train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
  

# This is correct does only min-max normalization
# def ks_markov_dataset(filename, saved_folder, data_normalizer=True, s=None, 
#                       val_filename="KS_valid.h5", test_filename="KS_test.h5", **kwargs):
#     """
#     Returns train, validation, and test datasets for Kuramoto-Sivashinsky using separate files.
#     Uses min-max normalization to [0, 1] range.
    
#     Args:
#         filename: Training file name (e.g., "KS_train_2048.h5")
#         saved_folder: Path to folder containing the files
#         data_normalizer: Whether to apply normalization
#         s: Target spatial resolution for resizing (None to keep original)
#         val_filename: Validation file name (default: "KS_valid.h5")
#         test_filename: Test file name (default: "KS_test.h5")
#         **kwargs: Additional arguments to pass to KSMarkovDataset
        
#     Returns:
#         train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
#     """
#     # Create separate datasets for train, validation, and test
#     train_dataset = KSMarkovDataset(filename, saved_folder, s=s, **kwargs)
#     val_dataset = KSMarkovDataset(val_filename, saved_folder, s=s, **kwargs)
#     test_dataset = KSMarkovDataset(test_filename, saved_folder, s=s, **kwargs)
    
#     min_data = None
#     max_data = None
#     min_model = None
#     max_model = None
    
#     if data_normalizer:
#         print('---------Computing min-max normalization statistics---------------')
#         temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        
#         # Collect all training data for computing min-max statistics
#         x_train_all = []
#         y_train_all = []
#         for batch in temp_loader:
#             if len(batch) == 3:  # If spatial coordinates are included
#                 x_batch, y_batch, _ = batch
#             else:
#                 x_batch, y_batch = batch
                
#             x_train_all.append(x_batch)
#             y_train_all.append(y_batch)
        
#         x_train_tensor = torch.cat(x_train_all, dim=0)
#         y_train_tensor = torch.cat(y_train_all, dim=0)
        
#         # Compute min-max statistics
#         min_data = float(x_train_tensor.min())
#         max_data = float(x_train_tensor.max())
#         min_model = float(y_train_tensor.min())
#         max_model = float(y_train_tensor.max())
        
#         print(f"Input data range: [{min_data:.6f}, {max_data:.6f}]")
#         print(f"Output data range: [{min_model:.6f}, {max_model:.6f}]")
        
#         # Create a wrapper dataset class that applies min-max normalization
#         class MinMaxNormalizedDataset(Dataset):
#             def __init__(self, dataset, min_data, max_data, min_model, max_model):
#                 self.dataset = dataset
#                 self.min_data = min_data
#                 self.max_data = max_data
#                 self.min_model = min_model
#                 self.max_model = max_model
            
#             def __len__(self):
#                 return len(self.dataset)
            
#             def __getitem__(self, idx):
#                 items = self.dataset[idx]
#                 if len(items) == 3:  # x, y, spatial_coords
#                     x, y, coords = items
#                     # Apply min-max normalization to [0, 1]
#                     x_normalized = (x - self.min_data) / (self.max_data - self.min_data)
#                     y_normalized = (y - self.min_model) / (self.max_model - self.min_model)
#                     return x_normalized, y_normalized, coords
#                 else:  # x, y
#                     x, y = items
#                     # Apply min-max normalization to [0, 1]
#                     x_normalized = (x - self.min_data) / (self.max_data - self.min_data)
#                     y_normalized = (y - self.min_model) / (self.max_model - self.min_model)
#                     return x_normalized, y_normalized
        
#         # Apply normalization to each dataset
#         train_dataset = MinMaxNormalizedDataset(train_dataset, min_data, max_data, min_model, max_model)
#         val_dataset = MinMaxNormalizedDataset(val_dataset, min_data, max_data, min_model, max_model)
#         test_dataset = MinMaxNormalizedDataset(test_dataset, min_data, max_data, min_model, max_model)
    
#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
#     print(f"Test dataset size: {len(test_dataset)}")
    
#     return train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
    

# # Example usage:
# """
# # For simple normalization (UnitGaussian-like)
# train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer = ks_markov_dataset(
#     filename="KS_train_2048.h5",
#     saved_folder="/data/user_data/rvk/ks/visc_0.1/",
#     normalization_type="simple",
#     data_normalizer=True
# )

# # For min-max normalization
# train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model = ks_markov_dataset(
#     filename="KS_train_2048.h5",
#     saved_folder="/data/user_data/rvk/ks/visc_0.1/",
#     normalization_type="minmax", 
#     data_normalizer=True
# )

# # Without normalization
# train_dataset, val_dataset, test_dataset, _, _ = ks_markov_dataset(
#     filename="KS_train_2048.h5",
#     saved_folder="/data/user_data/rvk/ks/visc_0.1/",
#     data_normalizer=False
# )
# """      


# Example usage:
# if __name__ == "__main__":
#     # Test with your KS data using separate files
#     train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model = ks_markov_dataset(
#         filename="KS_train_2048.h5",
#         saved_folder="/data/user_data/rvk/ks/visc_0.1/",
#         val_filename="KS_valid.h5",
#         test_filename="KS_test.h5",
#         s=256,  # Resize from 512 to 256
#         data_normalizer=True
#     )
    
#     print("Dataset created successfully!")
#     print(f"Train samples: {len(train_dataset)}")
#     print(f"Val samples: {len(val_dataset)}")
#     print(f"Test samples: {len(test_dataset)}")
    
#     # Test loading a batch from each dataset
#     for name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
#         loader = DataLoader(dataset, batch_size=32, shuffle=True)
#         for batch in loader:
#             if len(batch) == 3:
#                 x, y, coords = batch
#                 print(f"{name} - Batch x shape: {x.shape}, y shape: {y.shape}, coords shape: {coords.shape}")
#             else:
#                 x, y = batch
#                 print(f"{name} - Batch x shape: {x.shape}, y shape: {y.shape}")
#             break