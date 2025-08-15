import torch
from torch.utils.data import Dataset, DataLoader, random_split
from models.custom_layer import UnitGaussianNormalizer
import os
import glob
import h5py
import numpy as np
import math as mt
from einops import rearrange

# Channel First DataLoader
# S4 model: self.x = rearrange(x, 'b t m -> (b t) m 1 1') and self.y = rearrange(y, 'b t m -> (b t) m 1 1')
class H5pyMarkovDataset(Dataset):
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
            keys.sort()
            _data = np.array(f['tensor'], dtype=np.float32)
            print(_data.shape)
            _data = _data[::reduced_batch, 
                          ::reduced_resolution_t, 
                          ::reduced_resolution]
            print('After downsampling data shape:', _data.shape)
            self.grid = np.array(f["x-coordinate"], dtype=np.float32)
            self.grid = torch.tensor(
                self.grid[::reduced_resolution],
                 dtype=torch.float).unsqueeze(-1)

            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, _data.shape[0])
            else:
                num_samples_max = _data.shape[0]

            self.data = _data[:num_samples_max]
            print(f"Total data shape: {self.data.shape}")

        x = self.data[:, 1:-1, :]
        self.x = rearrange(x, 'b t m -> (b t) 1 m')
        y = self.data[:, 2:, :]
        self.y = rearrange(y, 'b t m -> (b t) 1 m')
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"x shape: {self.x.shape}")
        print(f"y shape: {self.y.shape}")
        print(f"grid shape: {self.grid.shape}")
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # return self.x[idx], self.y[idx], self.grid
        return self.x[idx], self.y[idx]

def burger_markov_dataset(filename, saved_folder, data_normalizer=True, 
                          normalization_type="minmax",  # "simple" or "minmax"
                          **kwargs):
    """
    Returns train, validation, and test datasets for Burger's equation with 0.8/0.1/0.1 ratio.
    Supports both simple Gaussian normalization and min-max normalization.
    
    Args:
        filename: H5 file name
        saved_folder: Path to folder containing the file
        data_normalizer: Whether to apply normalization
        normalization_type: Type of normalization - "simple" (UnitGaussian-like) or "minmax"
        s: Target spatial resolution for resizing (None to keep original)
        **kwargs: Additional arguments to pass to H5pyMarkovDataset
        
    Returns:
        For simple normalization: train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
        For minmax normalization: train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
    """
    # Create the full dataset with optional resizing
    full_dataset = H5pyMarkovDataset(filename, saved_folder, **kwargs)
    
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
                    items = self.dataset[idx]
                    
                    if len(items) == 3:  # x, y, grid
                        x, y, grid = items
                        # Ensure x and y are PyTorch tensors, not NumPy arrays
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x).float()
                        if isinstance(y, np.ndarray):
                            y = torch.from_numpy(y).float()
                        
                        return self.x_normalizer.encode(x), self.y_normalizer.encode(y), grid
                    else:  # x, y
                        x, y = items
                        # Ensure x and y are PyTorch tensors, not NumPy arrays
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x).float()
                        if isinstance(y, np.ndarray):
                            y = torch.from_numpy(y).float()
                        
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

# This is correct but only does minmax normalization
# def burger_markov_dataset(filename, saved_folder, data_normalizer=True, **kwargs):
#     """
#     Returns train, validation, and test datasets with 0.8/0.1/0.1 ratio.
    
#     Args:
#         filename: H5 file name
#         saved_folder: Path to folder containing the file
#         data_normalizer: Whether to apply normalization
#         **kwargs: Additional arguments to pass to H5pyMarkovDataset
        
#     Returns:
#         train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
#     """
#     # Create the full dataset
#     full_dataset = H5pyMarkovDataset(filename, saved_folder, **kwargs)
    
#     # Calculate split sizes
#     dataset_size = len(full_dataset)
#     train_size = int(0.8 * dataset_size)
#     val_size = int(0.1 * dataset_size)
#     test_size = dataset_size - train_size - val_size
    
#     # Split dataset
#     train_dataset, val_dataset, test_dataset = random_split(
#         full_dataset, 
#         [train_size, val_size, test_size],
#         generator=torch.Generator().manual_seed(42)  # For reproducibility
#     )
    
#     x_normalizer = None
#     y_normalizer = None
    
#     if data_normalizer:
#         print('---------Using data normalizer---------------')
#         temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        
#         # Collect all training data for computing normalization
#         x_train_all = []
#         y_train_all = []
#         for x_batch, y_batch, _ in temp_loader:
#             x_train_all.append(x_batch)
#             y_train_all.append(y_batch)
        
#         x_train_tensor = torch.cat(x_train_all, dim=0)
#         y_train_tensor = torch.cat(y_train_all, dim=0)
        
#         # Initialize normalizers using training data
#         x_normalizer = UnitGaussianNormalizer(x_train_tensor)
#         y_normalizer = UnitGaussianNormalizer(y_train_tensor)
        
#         # Create a wrapper dataset class that applies normalization
#         class NormalizedDataset(Dataset):
#             def __init__(self, dataset, x_normalizer, y_normalizer):
#                 self.dataset = dataset
#                 self.x_normalizer = x_normalizer
#                 self.y_normalizer = y_normalizer
            
#             def __len__(self):
#                 return len(self.dataset)
            
#             def __getitem__(self, idx):
#                 x, y, grid = self.dataset[idx]

#                 # Ensure x and y are PyTorch tensors, not NumPy arrays
#                 if isinstance(x, np.ndarray):
#                     x = torch.from_numpy(x).float()
#                 if isinstance(y, np.ndarray):
#                     y = torch.from_numpy(y).float()

#                 return self.x_normalizer.encode(x), self.y_normalizer.encode(y)    # Can also return grid
        
#         # Apply normalization to each dataset
#         train_dataset = NormalizedDataset(train_dataset, x_normalizer, y_normalizer)
#         val_dataset = NormalizedDataset(val_dataset, x_normalizer, y_normalizer)
#         test_dataset = NormalizedDataset(test_dataset, x_normalizer, y_normalizer)
    
#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
#     print(f"Test dataset size: {len(test_dataset)}")
    
#     return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer