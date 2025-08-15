import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import h5py
import numpy as np
from einops import rearrange

class H5pyMultiResMarkovDataset(Dataset):
    def __init__(self, 
                 filename, 
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 add_res=None,  # Additional resolutions for multi-resolution training
                 num_add_res_samples=0,  # Number of samples at each additional resolution
                 split_ratio=None,  # For splitting additional samples: [train%, val%, test%]
                 random_seed=42,  # Seed for multi-resolution sampling only
                 split='train',  # Which split this dataset represents
                 **kwargs):
        
        # Store seed for later use in multi-resolution generation only
        self.random_seed = random_seed
        self.split = split  # Store which split this is
        
        # Set default split ratio if not provided
        if split_ratio is None:
            split_ratio = [0.8, 0.1, 0.1]  # 80% train, 10% val, 10% test
        
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        with h5py.File(root_path, 'r') as f:
            print(f'Loading from file: {root_path}')
            keys = list(f.keys())
            keys.sort()
            _data = np.array(f['tensor'], dtype=np.float32)
            print(f"Original data shape: {_data.shape}")
            
            # ✅ Store original data BEFORE any downsampling for multi-resolution generation
            original_data = _data.copy()
            original_spatial_size = _data.shape[2]  # Spatial dimension
            print('Original Spatial Size:', original_spatial_size)
            
            # Apply reductions (original behavior)
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
            print(f"Data shape after reductions: {self.data.shape}")

        # Generate additional resolution data if specified
        self.multi_res_data_list = []
        if add_res is not None and num_add_res_samples > 0:
            print(f"Generating multi-resolution data with additional resolutions: {add_res}")
            multi_res_data_list = self._generate_multi_resolution_data(
                original_data, original_spatial_size, add_res, num_add_res_samples, 
                split_ratio, reduced_batch, reduced_resolution_t
            )
            
            if multi_res_data_list is not None:
                self.multi_res_data_list = multi_res_data_list
                print(f"Generated {len(self.multi_res_data_list)} additional resolution datasets")

        # Process main data - Create input-output pairs for Markov property (same as original)
        x = self.data[:, 1:-1, :]
        y = self.data[:, 2:, :]
        
        # Process main data - Create input-output pairs for Markov property (same as original)
        x = self.data[:, 1:-1, :]
        y = self.data[:, 2:, :]
        
        # Convert to tensors first, then use rearrange (same as original)
        x_tensor = torch.tensor(x, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.float)
        
        x_main = rearrange(x_tensor, 'b t m -> (b t) 1 m')
        y_main = rearrange(y_tensor, 'b t m -> (b t) 1 m')
        
        # Store as lists to handle different spatial dimensions
        self.x = []
        self.y = []
        
        # Add main data to lists
        for i in range(len(x_main)):
            self.x.append(x_main[i])
            self.y.append(y_main[i])
        
        # Store original length for main data
        main_data_length = len(self.x)
        
        # Process additional multi-resolution data and append to main data
        if hasattr(self, 'multi_res_data_list') and self.multi_res_data_list:
            for multi_res_data in self.multi_res_data_list:
                # Create input-output pairs for this resolution
                x_multi = multi_res_data[:, 1:-1, :]  # Input: skip first and last timestep
                y_multi = multi_res_data[:, 2:, :]    # Output: skip first two timesteps
                
                # Convert using rearrange
                x_multi_tensor = torch.tensor(x_multi, dtype=torch.float)
                y_multi_tensor = torch.tensor(y_multi, dtype=torch.float)
                
                x_multi_reshaped = rearrange(x_multi_tensor, 'b t m -> (b t) 1 m')
                y_multi_reshaped = rearrange(y_multi_tensor, 'b t m -> (b t) 1 m')
                
                # Add to lists (not concatenate - different spatial dimensions)
                for i in range(len(x_multi_reshaped)):
                    self.x.append(x_multi_reshaped[i])
                    self.y.append(y_multi_reshaped[i])
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"Total samples: {len(self.x)} (main: {main_data_length}, additional: {len(self.x) - main_data_length})")
        if len(self.x) > 0:
            print(f"Sample x shapes: {[tuple(x.shape) for x in self.x[:min(3, len(self.x))]]}")  # Show first 3 sample shapes
            print(f"Sample y shapes: {[tuple(y.shape) for y in self.y[:min(3, len(self.y))]]}")  # Show first 3 sample shapes
        print(f"grid shape: {self.grid.shape}")
    
    def _generate_multi_resolution_data(self, original_data, original_spatial_size, add_res, 
                                       num_add_res_samples, split_ratio, reduced_batch, reduced_resolution_t):
        """Generate additional data at different resolutions using downsampling"""
        if not isinstance(add_res, (list, tuple)):
            add_res = [add_res]
        
        # Calculate how many samples for current split
        split_idx = {'train': 0, 'valid': 1, 'val': 1, 'test': 2}.get(self.split, 0)
        samples_for_this_split = int(num_add_res_samples * split_ratio[split_idx])
        
        if samples_for_this_split == 0:
            print(f"No additional samples allocated for {self.split} split")
            return None
        
        print(f"Generating {samples_for_this_split} additional samples at resolutions {add_res} for {self.split} split")
        
        multi_res_data_list = []
        
        # Use deterministic random seed based on split AND dataset parameters for full reproducibility
        seed_string = f"{self.split}_{str(add_res)}_{num_add_res_samples}_{original_data.shape[0]}_{self.random_seed}"
        seed_base = sum(ord(c) for c in seed_string) % (2**31)
        
        # Create a local random state to avoid affecting global numpy random state
        rng = np.random.RandomState(seed_base + split_idx)
        
        for target_res in add_res:
            print(f"Generating data at resolution {target_res}")
            
            # Calculate downsampling factor needed to get target_res from original_spatial_size
            if target_res > original_spatial_size:
                print(f"Warning: Target resolution {target_res} is larger than original {original_spatial_size}. Skipping.")
                continue
            elif target_res == original_spatial_size:
                # No downsampling needed
                downsample_factor = 1
            else:
                # Calculate downsampling factor
                if original_spatial_size % target_res != 0:
                    print(f"Warning: Original resolution {original_spatial_size} is not divisible by target {target_res}. Using closest factor.")
                    downsample_factor = round(original_spatial_size / target_res)
                else:
                    downsample_factor = original_spatial_size // target_res
            
            print(f"Using downsampling factor: {downsample_factor} (original: {original_spatial_size} -> target: {target_res})")
            
            # Sample deterministic indices from original data using local random state
            num_original_samples = original_data.shape[0]
            sample_indices = rng.choice(num_original_samples, samples_for_this_split, replace=True)
            sampled_data = original_data[sample_indices]
            
            # Apply downsampling (same as original dataset but with different factors)
            final_data = sampled_data[::1,  # No batch reduction for sampled data
                                     ::reduced_resolution_t,  # Same time reduction as main
                                     ::downsample_factor]     # Spatial downsampling to get target resolution
            
            multi_res_data_list.append(final_data)
            print(f"Generated {final_data.shape[0]} samples at resolution {target_res}, final shape: {final_data.shape}")
        
        if multi_res_data_list:
            print(f"Total additional multi-resolution datasets: {len(multi_res_data_list)}")
            return multi_res_data_list
        
        return None
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # return self.x[idx], self.y[idx], self.grid
        return self.x[idx], self.y[idx]


def burger_multires_markov_dataset(filename, saved_folder, data_normalizer=True, 
                                   normalization_type="minmax",  # "simple" or "minmax"
                                   add_res=None, num_add_res_samples=0, random_seed=42,
                                   **kwargs):
    """
    Burger multi-resolution dataset using downsampling for additional resolutions.
    Supports both simple Gaussian normalization and min-max normalization.
    
    Args:
        filename: H5 file name
        saved_folder: Path to folder containing the file
        data_normalizer: Whether to apply normalization
        normalization_type: Type of normalization - "simple" (UnitGaussian-like) or "minmax"
        add_res: Additional resolutions for multi-resolution training (achieved via downsampling)
        num_add_res_samples: Number of additional samples per resolution
        random_seed: Random seed for reproducibility
        **kwargs: Additional arguments to pass to H5pyMultiResMarkovDataset (including reduced_resolution)
        
    Returns:
        For simple normalization: train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
        For minmax normalization: train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
    """
    split_ratio = [0.8, 0.1, 0.1]
    
    # Create the full dataset with optional multi-resolution
    full_dataset = H5pyMultiResMarkovDataset(
        filename, saved_folder, 
        add_res=add_res, 
        num_add_res_samples=num_add_res_samples,
        split_ratio=split_ratio, 
        random_seed=random_seed,
        split='train',  # This will be overridden during split
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
    
    # Update split information for each dataset (for debugging purposes)
    train_dataset.dataset.split = 'train' if hasattr(train_dataset, 'dataset') else 'train'
    val_dataset.dataset.split = 'val' if hasattr(val_dataset, 'dataset') else 'val'
    test_dataset.dataset.split = 'test' if hasattr(test_dataset, 'dataset') else 'test'
    
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
    
    if add_res is not None and num_add_res_samples > 0:
        print(f"Multi-resolution training enabled with additional resolutions: {add_res}")
        print(f"Additional samples per resolution: {num_add_res_samples}")
        print(f"Distribution: Train={int(num_add_res_samples * 0.8)}, "
              f"Val={int(num_add_res_samples * 0.1)}, Test={int(num_add_res_samples * 0.1)}")
    
    # Return appropriate values based on normalization type
    if normalization_type == "simple":
        return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
    else:  # minmax
        return train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model


# Example usage:
# """
# # Original resolution 512, want main data at 64 and additional at 128, 256, 512
# train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model = burger_multires_markov_dataset(
#     filename="burgers_data_R10.h5",
#     saved_folder="/path/to/data/",
#     normalization_type="minmax",
#     data_normalizer=True,
#     reduced_resolution=8,     # Main data: 512/8 = 64 resolution 
#     add_res=[128, 256, 512],  # Additional resolutions via downsampling
#     num_add_res_samples=1000  # Extra samples per resolution
# )

# # For original resolution 512, no main downsampling, add lower resolutions
# train_dataset, val_dataset, test_dataset, x_norm, y_norm = burger_multires_markov_dataset(
#     filename="burgers_data_R10.h5",
#     saved_folder="/path/to/data/",
#     normalization_type="simple",
#     data_normalizer=True,
#     reduced_resolution=1,     # Main data: no downsampling (512)
#     add_res=[64, 128, 256],   # Additional resolutions via downsampling 
#     num_add_res_samples=1000
# )
# """