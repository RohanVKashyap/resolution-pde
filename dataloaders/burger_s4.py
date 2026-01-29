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
class H5pyWindowDataset(Dataset):
    def __init__(self, 
                 filename, 
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 window_size=10,  # Number of previous timesteps to use for prediction
                 flatten_window=False,  # Whether to flatten the window dimension
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

        # Check if we have enough timesteps
        if self.data.shape[1] < window_size + 1:
            raise ValueError(f"Dataset has only {self.data.shape[1]} timesteps, need at least {window_size + 1}")
        
        x_all = []
        y_all = []
        
        # Process each batch separately
        for b_idx in range(self.data.shape[0]):
            batch_data = self.data[b_idx]  # shape: [t, m]
            
            # Create window-target pairs for this batch
            for t in range(batch_data.shape[0] - window_size):
                # Window: window_size consecutive timesteps
                window = batch_data[t:t+window_size]  # shape: [window_size, m]
                # Target: the timestep after the window
                target = batch_data[t+window_size]  # shape: [m]
                
                x_all.append(window)
                y_all.append(target)
        
        # Convert to numpy arrays
        x_all = np.array(x_all)  # shape: [(b*(t-window_size)), window_size, m]
        y_all = np.array(y_all)  # shape: [(b*(t-window_size)), m]
        
        # Reshape x based on whether to flatten the window dimension
        if flatten_window:
            # Flatten window and spatial dimensions
            self.x = x_all.reshape(x_all.shape[0], 1, -1)  # shape: [(b*(t-window_size)), 1, window_size*m]
        else:
            # Keep window dimension separate
            self.x = x_all  # shape: [(b*(t-window_size)), window_size, m]
        
        # Add channel dimension to y
        self.y = np.expand_dims(y_all, axis=1)  # shape: [(b*(t-window_size)), 1, m]
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"x shape: {self.x.shape}")
        print(f"y shape: {self.y.shape}")
        print(f"grid shape: {self.grid.shape}")
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.grid


def burger_window_dataset(filename, saved_folder, data_normalizer=True, window_size=10, flatten_window=False, **kwargs):
    """
    Returns train, validation, and test datasets with 0.8/0.1/0.1 ratio.
    
    Args:
        filename: H5 file name
        saved_folder: Path to folder containing the file
        data_normalizer: Whether to apply normalization
        window_size: Number of previous timesteps to use for prediction
        flatten_window: Whether to flatten the window dimension
        **kwargs: Additional arguments to pass to H5pyMarkovDataset
        
    Returns:
        train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
    """
    # Create the full dataset
    full_dataset = H5pyWindowDataset(filename, saved_folder, window_size=window_size, flatten_window=flatten_window, **kwargs)
    
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
        for x_batch, y_batch, _ in temp_loader:
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
                x, y, grid = self.dataset[idx]

                # Ensure x and y are PyTorch tensors, not NumPy arrays
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float()
                if isinstance(y, np.ndarray):
                    y = torch.from_numpy(y).float()

                return self.x_normalizer.encode(x), self.y_normalizer.encode(y)    # Can also return grid
        
        # Apply normalization to each dataset
        train_dataset = NormalizedDataset(train_dataset, x_normalizer, y_normalizer)
        val_dataset = NormalizedDataset(val_dataset, x_normalizer, y_normalizer)
        test_dataset = NormalizedDataset(test_dataset, x_normalizer, y_normalizer)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer