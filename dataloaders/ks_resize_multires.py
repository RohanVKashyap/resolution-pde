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
                 add_res=None,  # Additional resolutions for multi-resolution training
                 num_add_res_samples=0,  # Number of samples at each additional resolution
                 split_ratio=None,  # For splitting additional samples: [train%, val%, test%]
                 random_seed=42,  # Seed for multi-resolution sampling only
                 **kwargs):
        
        assert reduced_resolution == 1, "reduced_resolution must be 1 when using parameter 's' for downsampling. Use 's' parameter instead of reduced_resolution for spatial downsampling."
        
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        print(f'Loading from file: {root_path}')
        
        # Store seed for later use in multi-resolution generation only
        self.random_seed = random_seed
        
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
        
        # Set default split ratio if not provided
        if split_ratio is None:
            split_ratio = [0.8, 0.1, 0.1]  # 80% train, 10% val, 10% test
        
        self._load_ks_data(root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max)
        
        # ✅ Store original data BEFORE any resizing for multi-resolution generation
        original_data = self.data.copy()
        original_spatial_size = self.data.shape[2]  # Spatial dimension
        print('Original Spatial Size:', original_spatial_size)
        
        # Apply resolution change if s is specified and different from current resolution
        current_spatial_size = self.data.shape[2]  # Spatial dimension
        
        if s is not None and s != current_spatial_size:
            print(f"Resizing from {current_spatial_size} to {s}")
            self.data = self._resize_data(self.data, current_spatial_size, s)
            print(f"Data shape after resizing: {self.data.shape}")
            
            # Update spatial coordinates for new resolution
            if hasattr(self, 'x_coords') and self.x_coords is not None:
                self._update_spatial_coords(s)
        
        target_resolution = s if s is not None else current_spatial_size
        
        # Generate additional resolution data if specified
        self.multi_res_data_list = []
        if add_res is not None and num_add_res_samples > 0:
            print(f"Generating multi-resolution data with additional resolutions: {add_res}")
            #multi_res_data_list = self._generate_multi_resolution_data(
                #original_data, target_resolution, add_res, num_add_res_samples, split_ratio
            #)
            multi_res_data_list = self._generate_multi_resolution_data(
               original_data, original_spatial_size, add_res, num_add_res_samples, split_ratio
            )
            
            if multi_res_data_list is not None:
                self.multi_res_data_list = multi_res_data_list
                print(f"Generated {len(self.multi_res_data_list)} additional resolution datasets")
        
        # Process main data
        # Extract the input sequence (all timesteps except last)
        x = self.data[:, :-1]  # (batch, time-1, spatial)
        # Extract the output sequence (all timesteps except first)
        y = self.data[:, 1:]   # (batch, time-1, spatial)
        
        # Convert to list of tensors and flatten time dimension
        self.x = []
        self.y = []
        
        # Process main data
        batch_size, time_steps = x.shape[0], x.shape[1]
        for i in range(batch_size):
            x_sample = torch.tensor(x[i], dtype=torch.float)  # (time, spatial)
            y_sample = torch.tensor(y[i], dtype=torch.float)  # (time, spatial)
            
            # Add channel dimension: (time, spatial) -> (time, 1, spatial)
            x_sample = rearrange(x_sample, 't s -> t 1 s')
            y_sample = rearrange(y_sample, 't s -> t 1 s')
            
            # Flatten time dimension: (time, 1, spatial) -> individual samples
            for t in range(time_steps):
                self.x.append(x_sample[t])  # (1, spatial)
                self.y.append(y_sample[t])  # (1, spatial)
        
        # Process additional multi-resolution data
        if hasattr(self, 'multi_res_data_list') and self.multi_res_data_list:
            for multi_res_data in self.multi_res_data_list:
                # Extract sequences for this resolution
                x_multi = multi_res_data[:, :-1]  # (batch, time-1, spatial)
                y_multi = multi_res_data[:, 1:]   # (batch, time-1, spatial)
                
                batch_size_multi, time_steps_multi = x_multi.shape[0], x_multi.shape[1]
                for i in range(batch_size_multi):
                    x_sample = torch.tensor(x_multi[i], dtype=torch.float)
                    y_sample = torch.tensor(y_multi[i], dtype=torch.float)
                    
                    # Add channel dimension
                    x_sample = rearrange(x_sample, 't s -> t 1 s')
                    y_sample = rearrange(y_sample, 't s -> t 1 s')
                    
                    # Flatten time dimension
                    for t in range(time_steps_multi):
                        self.x.append(x_sample[t])  # (1, spatial)
                        self.y.append(y_sample[t])  # (1, spatial)
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"Total samples: {len(self.x)}")
        if len(self.x) > 0:
            print(f"Sample spatial dimensions: {[tuple(x.shape) for x in self.x[:min(3, len(self.x))]]}")  # Show first 3 sample shapes
        if hasattr(self, 'x_coords') and self.x_coords is not None:
            print(f"spatial coordinates shape: {self.x_coords.shape}")
    
    def _resize_data(self, data, current_size, target_size):
        """Resize data from current_size to target_size"""
        # Reshape for resizing: (batch, time, spatial) -> (batch*time, spatial)
        batch_size, time_steps, spatial_size = data.shape
        data_reshaped = data.reshape(batch_size * time_steps, spatial_size)
        
        # Choose appropriate resizing method based on target size
        if target_size < current_size:
            # Downsampling: use FFT-based downsample function
            print(f"Downsampling using FFT-based downsample_1d function")
            data_resized = downsample_1d(data_reshaped, target_size)
        else:
            # Upsampling: use FFT-based resize function
            print(f"Upsampling using FFT-based resize_1d function")
            # Convert to torch tensor for resize function
            data_torch = torch.tensor(data_reshaped, dtype=torch.float32)
            data_resized = resize_1d(data_torch, target_size)
            # Convert back to numpy
            data_resized = data_resized.numpy()
        
        # Reshape back: (batch*time, target_size) -> (batch, time, target_size)
        data_resized = data_resized.reshape(batch_size, time_steps, target_size)
        return data_resized
    
    def _update_spatial_coords(self, new_size):
        """Update spatial coordinates for new resolution"""
        if hasattr(self, 'original_domain_length'):
            domain_length = self.original_domain_length
        else:
            domain_length = self.x_coords[-1] - self.x_coords[0]
        
        self.x_coords = np.linspace(self.x_coords[0], self.x_coords[0] + domain_length, new_size)
        print(f"Updated spatial coordinates shape: {self.x_coords.shape}")
    
    # def _generate_multi_resolution_data(self, original_data, target_resolution, add_res, num_add_res_samples, split_ratio):
    #     """Generate additional data at different resolutions for multi-resolution training"""
    #     if not isinstance(add_res, (list, tuple)):
    #         add_res = [add_res]
        
    def _generate_multi_resolution_data(self, original_data, original_spatial_size, add_res, num_add_res_samples, split_ratio):
        """Generate additional data at different resolutions for multi-resolution training"""
        # Handle all possible input types (int, list, tuple, ListConfig)
        if hasattr(add_res, '__iter__') and not isinstance(add_res, str):
            # It's iterable (list, tuple, ListConfig)
            add_res = [int(res) for res in add_res]
        else:
            # It's a single value
            add_res = [int(add_res)]
        
        # Calculate how many samples for current split
        split_idx = {'train': 0, 'valid': 1, 'test': 2}.get(self.split, 0)
        samples_for_this_split = int(num_add_res_samples * split_ratio[split_idx])
        
        if samples_for_this_split == 0:
            print(f"No additional samples allocated for {self.split} split")
            return None
        
        print(f"Generating {samples_for_this_split} additional samples at resolutions {add_res} for {self.split} split")
        
        multi_res_data_list = []
        
        # Use deterministic random seed based on split AND dataset parameters for full reproducibility
        # Use a more stable seed generation that doesn't rely on hash()
        seed_string = f"{self.split}_{str(add_res)}_{num_add_res_samples}_{original_data.shape[0]}_{self.random_seed}"
        seed_base = sum(ord(c) for c in seed_string) % (2**31)
        
        # Create a local random state to avoid affecting global numpy random state
        rng = np.random.RandomState(seed_base + split_idx)
        
        for res in add_res:
            print(f"Generating data at resolution {res}")
            
            # Sample deterministic indices from original data using local random state
            num_original_samples = original_data.shape[0]
            sample_indices = rng.choice(num_original_samples, samples_for_this_split, replace=True)
            sampled_data = original_data[sample_indices]
            
            # Only resize if add_res is different from current_spatial_size
            current_spatial_size = sampled_data.shape[2]
            if res != current_spatial_size:
                print(f"Resizing from {current_spatial_size} to {res}")
                # Resize to the add_res resolution and keep it at that resolution
                final_data = self._resize_data(sampled_data, current_spatial_size, res)
            else:
                print(f"Resolution {res} matches current spatial size {current_spatial_size}, no resize needed")
                final_data = sampled_data
            
            multi_res_data_list.append(final_data)
            print(f"Generated {final_data.shape[0]} samples at resolution {res}")
        
        if multi_res_data_list:
            print(f"Total additional multi-resolution datasets: {len(multi_res_data_list)}")
            return multi_res_data_list
        
        return None
    
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

            # Apply reductions
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
        return self.x[idx], self.y[idx]

def ks_multires_markov_dataset(filename, saved_folder, data_normalizer=True, 
                               normalization_type="simple",  # "simple" or "minmax"
                               add_res=None, num_add_res_samples=0, random_seed=42,
                               val_filename="KS_valid.h5", test_filename="KS_test.h5", **kwargs):
    """
    KS multi-resolution dataset with naive downsampling and flexible normalization.
    Supports both simple Gaussian normalization and min-max normalization.
    
    Args:
        filename: Training file name (e.g., "KS_train_2048.h5")
        saved_folder: Path to folder containing the files
        data_normalizer: Whether to apply normalization
        normalization_type: Type of normalization - "simple" (UnitGaussian-like) or "minmax"
        add_res: Additional resolutions for multi-resolution training
        num_add_res_samples: Number of additional samples per resolution
        random_seed: Random seed for reproducibility
        val_filename: Validation file name (default: "KS_valid.h5")
        test_filename: Test file name (default: "KS_test.h5")
        **kwargs: Additional arguments to pass to KSMarkovDataset
        
    Returns:
        For simple normalization: train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
        For minmax normalization: train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
    """
    split_ratio = [0.8, 0.1, 0.1]
    
    # Create datasets
    train_dataset = KSMarkovDataset(filename, saved_folder, 
                                   add_res=add_res, num_add_res_samples=num_add_res_samples,
                                   split_ratio=split_ratio, random_seed=random_seed, **kwargs)
    val_dataset = KSMarkovDataset(val_filename, saved_folder,
                                 add_res=add_res, num_add_res_samples=num_add_res_samples,
                                 split_ratio=split_ratio, random_seed=random_seed, **kwargs)
    test_dataset = KSMarkovDataset(test_filename, saved_folder,
                                  add_res=add_res, num_add_res_samples=num_add_res_samples,
                                  split_ratio=split_ratio, random_seed=random_seed, **kwargs)
    
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
            
            # Simple normalized dataset wrapper
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
            
            # Apply normalization
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

# This is correct does only min-max normalization
# def ks_multires_markov_dataset(filename, saved_folder, data_normalizer=True, s=None, 
#                       add_res=None, num_add_res_samples=0, random_seed=42,
#                       val_filename="KS_valid.h5", test_filename="KS_test.h5", **kwargs):
#     """
#     Returns train, validation, and test datasets for Kuramoto-Sivashinsky using separate files.
#     Uses min-max normalization to [0, 1] range.
#     Supports multi-resolution training by generating additional samples at different resolutions.
    
#     Args:
#         filename: Training file name (e.g., "KS_train_2048.h5")
#         saved_folder: Path to folder containing the files
#         data_normalizer: Whether to apply normalization
#         s: Target spatial resolution for resizing (None to keep original)
#         add_res: Additional resolutions for multi-resolution training (list or single value)
#         num_add_res_samples: Number of samples to generate at each additional resolution
#         val_filename: Validation file name (default: "KS_valid.h5")
#         test_filename: Test file name (default: "KS_test.h5")
#         **kwargs: Additional arguments to pass to KSMarkovDataset
        
#     Returns:
#         train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
#     """
#     # Define split ratios for additional samples: 80% train, 10% val, 10% test
#     split_ratio = [0.8, 0.1, 0.1]
    
#     # Create separate datasets for train, validation, and test
#     train_dataset = KSMarkovDataset(filename, saved_folder, s=s, 
#                                    add_res=add_res, num_add_res_samples=num_add_res_samples,
#                                    split_ratio=split_ratio, random_seed=random_seed, **kwargs)
#     val_dataset = KSMarkovDataset(val_filename, saved_folder, s=s,
#                                  add_res=add_res, num_add_res_samples=num_add_res_samples,
#                                  split_ratio=split_ratio, random_seed=random_seed, **kwargs)
#     test_dataset = KSMarkovDataset(test_filename, saved_folder, s=s,
#                                   add_res=add_res, num_add_res_samples=num_add_res_samples,
#                                   split_ratio=split_ratio, random_seed=random_seed, **kwargs)
    
#     min_data = None
#     max_data = None
#     min_model = None
#     max_model = None
    
#     if data_normalizer:
#         print('---------Computing min-max normalization statistics---------------')
#         temp_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)  # Use batch_size=1 for multi-res
        
#         # Collect all training data for computing min-max statistics
#         x_train_all = []
#         y_train_all = []
#         for batch in temp_loader:
#             if len(batch) == 3:  # If spatial coordinates are included
#                 x_batch, y_batch, _ = batch
#             else:
#                 x_batch, y_batch = batch
            
#             # Handle both single tensors and lists of tensors
#             if isinstance(x_batch, list):
#                 x_train_all.extend(x_batch)
#                 y_train_all.extend(y_batch)
#             else:
#                 x_train_all.append(x_batch)
#                 y_train_all.append(y_batch)
        
#         # Compute min-max statistics across all samples (regardless of spatial dimension)
#         all_x_values = []
#         all_y_values = []
        
#         for x_tensor in x_train_all:
#             all_x_values.append(x_tensor.flatten())
#         for y_tensor in y_train_all:
#             all_y_values.append(y_tensor.flatten())
        
#         # Concatenate flattened values (all 1D now, so no shape conflicts)
#         x_train_flat = torch.cat(all_x_values, dim=0)
#         y_train_flat = torch.cat(all_y_values, dim=0)
        
#         # Compute min-max statistics
#         min_data = float(x_train_flat.min())
#         max_data = float(x_train_flat.max())
#         min_model = float(y_train_flat.min())
#         max_model = float(y_train_flat.max())
        
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
    
#     if add_res is not None and num_add_res_samples > 0:
#         print(f"Multi-resolution training enabled with additional resolutions: {add_res}")
#         print(f"Additional samples per resolution: {num_add_res_samples}")
#         print(f"Distribution: Train={int(num_add_res_samples * 0.8)}, "
#               f"Val={int(num_add_res_samples * 0.1)}, Test={int(num_add_res_samples * 0.1)}")
    
#     return train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
    

# Example usage:
# """
# # For simple normalization (UnitGaussian-like) with multi-resolution
# train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer = ks_multires_markov_dataset(
#     filename="KS_train_2048.h5",
#     saved_folder="/data/user_data/rvk/ks/visc_0.1/",
#     normalization_type="simple",
#     data_normalizer=True,
#     add_res=[128, 256, 512],
#     num_add_res_samples=1000
# )

# # For min-max normalization with multi-resolution
# train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model = ks_multires_markov_dataset(
#     filename="KS_train_2048.h5",
#     saved_folder="/data/user_data/rvk/ks/visc_0.1/",
#     normalization_type="minmax", 
#     data_normalizer=True,
#     add_res=[128, 256, 512],
#     num_add_res_samples=1000
# )

# # Without normalization but with multi-resolution
# train_dataset, val_dataset, test_dataset, _, _ = ks_multires_markov_dataset(
#     filename="KS_train_2048.h5",
#     saved_folder="/data/user_data/rvk/ks/visc_0.1/",
#     data_normalizer=False,
#     add_res=[128, 256, 512],
#     num_add_res_samples=1000
# )
# """        