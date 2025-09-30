import torch
from torch.utils.data import Dataset, DataLoader, random_split
from models.custom_layer import UnitGaussianNormalizer
import os
import numpy as np
from einops import rearrange
from scipy.io import loadmat
import h5py

class NSMarkovDataset(Dataset):
    def __init__(self, 
                 filename,  # e.g., "ns_32_1e-3.mat" or "ns_256_1e-03.h5"
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 **kwargs):
        
        # Construct full file path
        file_path = os.path.join(saved_folder, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f'Loading from file: {file_path}')
        
        # Determine file type from extension
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.mat':
            raw_data = self._load_mat_file(file_path)
        elif file_ext == '.h5':
            raw_data = self._load_h5_file(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}. Supported extensions: .mat, .h5")
        
        print(f"Raw data shape: {raw_data.shape}")
        
        if len(raw_data.shape) != 4:
            raise ValueError(f"Expected 4D array, got {raw_data.shape}")
        
        # For .h5 files, data is already in (batch, time, height, width) format
        # For .mat files, we need to transpose from (batch, height, width, time)
        if file_ext == '.mat':
            # Transpose from (batch, height, width, time) to (batch, time, height, width)
            combined_data = np.transpose(raw_data, (0, 3, 1, 2))
        else:  # .h5 files
            # Data is already in (batch, time, height, width) format
            combined_data = raw_data
        
        # Add channel dimension: (batch, time, height, width, 1)
        combined_data = np.expand_dims(combined_data, axis=-1)
        print(f"Converted to shape: {combined_data.shape}")
        
        # Apply reductions
        # combined_data shape: (batch, time, height, width, channels)
        combined_data = combined_data[::reduced_batch,          # batch dimension
                                    ::reduced_resolution_t,    # time dimension  
                                    ::reduced_resolution,       # height dimension
                                    ::reduced_resolution,       # width dimension
                                    :]                          # all channels
        
        print(f"After reduction: {combined_data.shape}")
        
        # Apply sample limit
        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, combined_data.shape[0])
            combined_data = combined_data[:num_samples_max]
            print(f"After sample limit: {combined_data.shape}")
        
        self.data = combined_data
        
        # Extract the input sequence (all timesteps except first and last)
        x = self.data[:, 1:-1]
        # Extract the output sequence (all timesteps except first two)
        y = self.data[:, 2:]
        
        print(f"Markov pairs - x: {x.shape}, y: {y.shape}")
        
        # Reshape to (batch*time, channels, height, width) format for neural networks
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        
        self.x = rearrange(self.x, 'b t h w c -> (b t) c h w')
        self.y = rearrange(self.y, 'b t h w c -> (b t) c h w')
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"Final x shape: {self.x.shape}")
        print(f"Final y shape: {self.y.shape}")
    
    def _load_mat_file(self, file_path):
        """Load data from .mat file"""
        print("Loading .mat file...")
        mat_data = loadmat(file_path)
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        print(f"Available keys: {keys}")
        
        # Load 'u' key for vorticity time series data
        if 'u' not in mat_data:
            raise KeyError(f"'u' key not found in {file_path}. Available keys: {keys}")
        
        raw_data = np.array(mat_data['u'], dtype=np.float32)
        return raw_data
    
    def _load_h5_file(self, file_path):
        """Load data from .h5 file"""
        print("Loading .h5 file...")
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            print(f"Available keys: {keys}")
            
            # Load 'u' key for vorticity time series data
            if 'u' not in f:
                raise KeyError(f"'u' key not found in {file_path}. Available keys: {keys}")
            
            # Load the data into memory
            raw_data = np.array(f['u'], dtype=np.float32)
            
            # For h5 files, the format is typically (batch, time, height, width)
            # But let's check and transpose if needed
            print(f"H5 data shape before any processing: {raw_data.shape}")
            
            # If the shape suggests it's (batch, height, width, time), transpose it
            if len(raw_data.shape) == 4:
                # Heuristic: if last dimension is smallest and reasonable for time steps
                if raw_data.shape[-1] < 100 and raw_data.shape[-1] < min(raw_data.shape[1], raw_data.shape[2]):
                    print("Detected (batch, height, width, time) format, transposing to (batch, time, height, width)")
                    raw_data = np.transpose(raw_data, (0, 3, 1, 2))
                else:
                    print("Assuming data is already in (batch, time, height, width) format")
            
            return raw_data
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def ns_markov_dataset(filename, 
                            saved_folder, 
                            data_normalizer=True,
                            normalization_type="unit_gaussian",
                            **kwargs):
    """
    Returns train, validation, and test datasets for single resolution Navier-Stokes 
    with 0.8/0.1/0.1 ratio.
    
    Args:
        filename: .mat or .h5 file name (e.g., "ns_32_1e-3.mat", "ns_256_1e-03.h5")
        saved_folder: Path to folder containing the file
        data_normalizer: Whether to apply normalization
        normalization_type: "simple" or "unit_gaussian"
        **kwargs: Additional arguments to pass to NSMarkovDataset
        
    Returns:
        train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
    """
    # Create the full dataset
    full_dataset = NSMarkovDataset(filename, saved_folder, **kwargs)
    
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
    
    x_normalizer = None
    y_normalizer = None
    
    if data_normalizer:
        if normalization_type == "simple":
            print('---------Using simple global normalization---------------')
            
            # Collect all values from training set
            all_x_values = []
            all_y_values = []
            
            for x, y in train_dataset:
                all_x_values.extend(x.flatten().tolist())
                all_y_values.extend(y.flatten().tolist())
            
            x_tensor = torch.tensor(all_x_values)
            y_tensor = torch.tensor(all_y_values)
            
            x_mean, x_std = x_tensor.mean(), x_tensor.std()
            y_mean, y_std = y_tensor.mean(), y_tensor.std()
            
            print(f"Global statistics: X(mean={x_mean:.6f}, std={x_std:.6f}), Y(mean={y_mean:.6f}, std={y_std:.6f})")
            
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
            
            train_dataset = SimpleNormalizedDataset(train_dataset, x_normalizer, y_normalizer)
            val_dataset = SimpleNormalizedDataset(val_dataset, x_normalizer, y_normalizer)
            test_dataset = SimpleNormalizedDataset(test_dataset, x_normalizer, y_normalizer)
            
        elif normalization_type == "unit_gaussian":
            print('---------Using UnitGaussianNormalizer---------------')
            temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
            
            # Collect all training data for computing normalization
            x_train_all = []
            y_train_all = []
            for batch in temp_loader:
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
                    return self.x_normalizer.encode(x), self.y_normalizer.encode(y)
            
            # Apply normalization to each dataset
            train_dataset = NormalizedDataset(train_dataset, x_normalizer, y_normalizer)
            val_dataset = NormalizedDataset(val_dataset, x_normalizer, y_normalizer)
            test_dataset = NormalizedDataset(test_dataset, x_normalizer, y_normalizer)
            
        else:
            raise ValueError(f"Invalid normalization_type: {normalization_type}. Must be 'simple' or 'unit_gaussian'")
    
    print(f"\nFinal dataset sizes:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer