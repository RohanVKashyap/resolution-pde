import torch
from torch.utils.data import Dataset, DataLoader, random_split
from models.custom_layer import UnitGaussianNormalizer
import os
import h5py
import numpy as np
from einops import rearrange


import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np
from einops import rearrange

class KSMarkovDataset(Dataset):
    def __init__(self, 
                 filename, 
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 add_res=None,
                 num_add_res_samples=0,
                 split_ratio=None,
                 random_seed=42,
                 **kwargs):
        
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        print(f'Loading from file: {root_path}')
        
        self.random_seed = random_seed
        
        # Determine split from filename
        if 'train' in filename.lower():
            self.split = 'train'
        elif 'valid' in filename.lower():
            self.split = 'valid'
        elif 'test' in filename.lower():
            self.split = 'test'
        else:
            self.split = 'train'
            print(f"Warning: Could not determine split from filename {filename}, assuming 'train'")
        
        if split_ratio is None:
            split_ratio = [0.8, 0.1, 0.1]
        
        self._load_ks_data(root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max)
        
        # Use original full-resolution data for multi-resolution generation
        if hasattr(self, 'original_data_full'):
            original_data = self.original_data_full.copy()
            original_spatial_size = original_data.shape[2]
            print(f'True Original Spatial Size: {original_spatial_size}')
        else:
            original_data = self.data.copy()
            original_spatial_size = self.data.shape[2]
            print(f'Warning: Using processed data spatial size: {original_spatial_size}')
        
        # Generate additional resolution data
        self.multi_res_data_list = []
        if add_res is not None and num_add_res_samples > 0:
            print(f"Generating multi-resolution data with additional resolutions: {add_res}")
            print(f"Using original data with spatial size: {original_spatial_size}")
            # multi_res_data_list = self._generate_multi_resolution_data(
            #     original_data, original_spatial_size, add_res, num_add_res_samples, split_ratio
            # )
            multi_res_data_list = self._generate_multi_resolution_data(
                    original_data, original_spatial_size, add_res, num_add_res_samples, split_ratio
            )
            
            if multi_res_data_list is not None:
                self.multi_res_data_list = multi_res_data_list
                print(f"Generated {len(self.multi_res_data_list)} additional resolution datasets")
        
        # Process all data into x, y pairs
        self.x = []
        self.y = []
        
        # Process main data
        self._process_data(self.data)
        
        # Process additional multi-resolution data
        if hasattr(self, 'multi_res_data_list') and self.multi_res_data_list:
            for multi_res_data in self.multi_res_data_list:
                self._process_data(multi_res_data)
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"Total samples: {len(self.x)}")
        if len(self.x) > 0:
            print(f"Sample spatial dimensions: {[tuple(x.shape) for x in self.x[:min(3, len(self.x))]]}")
    
    def _process_data(self, data):
        """Convert data to individual x,y pairs"""
        x = data[:, :-1]  # (batch, time-1, spatial)
        y = data[:, 1:]   # (batch, time-1, spatial)
        
        batch_size, time_steps = x.shape[0], x.shape[1]
        for i in range(batch_size):
            x_sample = torch.tensor(x[i], dtype=torch.float)
            y_sample = torch.tensor(y[i], dtype=torch.float)
            
            # Add channel dimension and flatten time
            x_sample = rearrange(x_sample, 't s -> t 1 s')
            y_sample = rearrange(y_sample, 't s -> t 1 s')
            
            for t in range(time_steps):
                self.x.append(x_sample[t])
                self.y.append(y_sample[t])
    
    def _naive_downsample_data(self, data, data_resolution, target_resolution):
        """Naive downsampling using array slicing"""
        if target_resolution >= data_resolution:
            print(f"Target resolution {target_resolution} >= data resolution {data_resolution}, no downsampling needed")
            return data
            
        downsample_factor = data_resolution // target_resolution
        print(f"Naive downsampling with factor {downsample_factor}: {data_resolution} -> {target_resolution}")
        
        downsampled_data = data[:, :, ::downsample_factor]
        
        # Ensure exact target resolution
        actual_resolution = downsampled_data.shape[2]
        if actual_resolution != target_resolution:
            print(f"Adjusting from {actual_resolution} to exact target {target_resolution}")
            downsampled_data = downsampled_data[:, :, :target_resolution]
        
        print(f"Data shape after naive downsampling: {downsampled_data.shape}")
        return downsampled_data
    
    # def _generate_multi_resolution_data(self, original_data, original_spatial_size, add_res, num_add_res_samples, split_ratio):
    #     """Generate additional data at different resolutions"""
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
        
        split_idx = {'train': 0, 'valid': 1, 'test': 2}.get(self.split, 0)
        samples_for_this_split = int(num_add_res_samples * split_ratio[split_idx])
        
        if samples_for_this_split == 0:
            print(f"No additional samples allocated for {self.split} split")
            return None
        
        print(f"Generating {samples_for_this_split} additional samples at resolutions {add_res} for {self.split} split")
        
        multi_res_data_list = []
        seed_string = f"{self.split}_{str(add_res)}_{num_add_res_samples}_{original_data.shape[0]}_{self.random_seed}"
        seed_base = sum(ord(c) for c in seed_string) % (2**31)
        rng = np.random.RandomState(seed_base + split_idx)
        
        for res in add_res:
            print(f"Generating data at resolution {res}")
            
            num_original_samples = original_data.shape[0]
            sample_indices = rng.choice(num_original_samples, samples_for_this_split, replace=True)
            sampled_data = original_data[sample_indices]
            
            if res != original_spatial_size:
                if res > original_spatial_size:
                    print(f"Warning: Target resolution {res} > original resolution {original_spatial_size}. Skipping upsampling.")
                    continue
                final_data = self._naive_downsample_data(sampled_data, original_spatial_size, res)
            else:
                print(f"Resolution {res} matches original spatial size {original_spatial_size}, no downsampling needed")
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
            if self.split in f:
                group = f[self.split]
            else:
                keys = list(f.keys())
                print(f"Available keys: {keys}")
                if len(keys) == 1:
                    group = f[keys[0]]
                    print(f"Using group: {keys[0]}")
                else:
                    raise ValueError(f"Could not find split '{self.split}' in file. Available keys: {keys}")
            
            data_keys = list(group.keys())
            print(f"Available data keys in '{self.split}': {data_keys}")
            
            # Find PDE data key
            pde_key = None
            for key in data_keys:
                if 'pde' in key.lower() and '-' in key:
                    pde_key = key
                    break
            
            if pde_key is None:
                raise ValueError(f"Could not find PDE data key in {data_keys}")
            
            # Load data
            pde_data = np.array(group[pde_key], dtype=np.float32)
            print(f"Original PDE data shape: {pde_data.shape}")
            
            # Store original data BEFORE any processing
            self.original_data_full = pde_data.copy()
            print(f"Stored original full-resolution data with shape: {self.original_data_full.shape}")
            
            # Apply reductions
            pde_data = pde_data[::reduced_batch, ::reduced_resolution_t, ::reduced_resolution]
            print(f"Data shape after downsampling: {pde_data.shape}")
            
            # Apply sample limit
            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, pde_data.shape[0])
            else:
                num_samples_max = pde_data.shape[0]

            self.data = pde_data[:num_samples_max]
            print(f"Final data shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def ks_multires_markov_dataset(filename, saved_folder, data_normalizer=True, 
                      add_res=None, num_add_res_samples=0, random_seed=42,
                      val_filename="KS_valid.h5", test_filename="KS_test.h5", **kwargs):
    """
    KS multi-resolution dataset with naive downsampling and simple global normalization.
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
    
    x_normalizer = None
    y_normalizer = None
    
    if data_normalizer:
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
        
        # Simple normalizer
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
        
        # Normalized dataset wrapper
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
        
        # Apply normalization
        train_dataset = NormalizedDataset(train_dataset, x_normalizer, y_normalizer)
        val_dataset = NormalizedDataset(val_dataset, x_normalizer, y_normalizer)
        test_dataset = NormalizedDataset(test_dataset, x_normalizer, y_normalizer)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    if add_res is not None and num_add_res_samples > 0:
        print(f"Multi-resolution training enabled with additional resolutions: {add_res}")
        print(f"Additional samples per resolution: {num_add_res_samples}")
        print(f"Distribution: Train={int(num_add_res_samples * 0.8)}, "
              f"Val={int(num_add_res_samples * 0.1)}, Test={int(num_add_res_samples * 0.1)}")
    
    return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer