import torch
from torch.utils.data import Dataset, DataLoader, random_split
from models.custom_layer import UnitGaussianNormalizer
import os
import h5py
import numpy as np
from einops import rearrange

class KSTrajectoryDatasetFromFile(Dataset):
    """
    Dataset that returns full KS trajectories from a single file for rollout evaluation.
    """
    def __init__(self, filename, saved_folder, reduced_batch=1, reduced_resolution=1, 
                 reduced_resolution_t=1, num_samples_max=-1, **kwargs):
        
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        print(f'Loading trajectories from file: {root_path}')
        
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
        
        self.trajectories = []
        self.trajectory_info = []
        
        self._load_ks_trajectories(root_path, reduced_batch, reduced_resolution, 
                                 reduced_resolution_t, num_samples_max)
    
    def _load_ks_trajectories(self, root_path, reduced_batch, reduced_resolution, 
                            reduced_resolution_t, num_samples_max):
        """Load full trajectories from KS HDF5 file format"""
        with h5py.File(root_path, 'r') as f:
            # Navigate to the correct split
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
            print(f"Original PDE data shape: {pde_data.shape}")
            
            # Apply reductions
            pde_data = pde_data[::reduced_batch, 
                               ::reduced_resolution_t, 
                               ::reduced_resolution]
            print(f"Data shape after downsampling: {pde_data.shape}")
            
            # Apply sample limit
            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, pde_data.shape[0])
            else:
                num_samples_max = pde_data.shape[0]

            final_data = pde_data[:num_samples_max]
            print(f"Final data shape: {final_data.shape}")
            
            # Store each trajectory individually (preserve full temporal structure)
            for i in range(final_data.shape[0]):
                trajectory = torch.tensor(final_data[i], dtype=torch.float)  # Shape: (time_steps, spatial_points)
                self.trajectories.append(trajectory)
                self.trajectory_info.append({
                    'original_index': i,
                    'source': f'{self.split}_file',
                    'filename': os.path.basename(root_path)
                })
            
            print(f"Loaded {len(self.trajectories)} complete trajectories from {self.split} split")
            if len(self.trajectories) > 0:
                print(f"Trajectory shape: {self.trajectories[0].shape}")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx]  # Returns shape: (time_steps, spatial_points)
    
    def get_trajectory_info(self, idx):
        return self.trajectory_info[idx]
    
    def get_all_info(self):
        return self.trajectory_info

class KSMarkovDataset(Dataset):
    def __init__(self, 
                 filename, 
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 **kwargs):
        
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
        
        # Extract the input sequence (all timesteps except last)
        x = self.data[:, :-1]  # (batch, time-1, spatial)
        # Extract the output sequence (all timesteps except first)
        y = self.data[:, 1:]   # (batch, time-1, spatial)
        
        # Flatten batch and time dimensions together and add channel dimension
        batch_size, time_steps = x.shape[0], x.shape[1]
        spatial_size = x.shape[2]
        
        # Reshape to (batch*time, channels=1, spatial) format for consistency with burger dataset
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
            print(f"Original PDE data shape: {pde_data.shape}")
            
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
                print(f"Original spatial coordinates shape: {self.x_coords.shape}")
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

            # Naive downsampling
            pde_data = pde_data[::reduced_batch, 
                               ::reduced_resolution_t, 
                               ::reduced_resolution]
            print(f"Data shape after downsampling: {pde_data.shape}")
            
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
                    
            if self.x_coords is not None:
                self.x_coords = self.x_coords[::reduced_resolution]
                if hasattr(self, 'x_coords'):
                    self.grid = torch.tensor(self.x_coords, dtype=torch.float).unsqueeze(-1)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def ks_markov_dataset(filename, saved_folder, data_normalizer=True, 
                      val_filename="KS_valid.h5", test_filename="KS_test.h5", **kwargs):
    """
    Returns train, validation, test datasets AND rollout test dataset for Kuramoto-Sivashinsky.
    
    Args:
        filename: Training file name (e.g., "KS_train_2048.h5")
        saved_folder: Path to folder containing the files
        data_normalizer: Whether to apply normalization
        val_filename: Validation file name (default: "KS_valid.h5")
        test_filename: Test file name (default: "KS_test.h5")
        **kwargs: Additional arguments to pass to KSMarkovDataset
        
    Returns:
        train_dataset, val_dataset, test_dataset, rollout_test_dataset, x_normalizer, y_normalizer
    """
    
    # Create Markov datasets for train, validation, and test (for teacher-forcing)
    train_dataset = KSMarkovDataset(filename, saved_folder, **kwargs)
    val_dataset = KSMarkovDataset(val_filename, saved_folder, **kwargs)
    test_dataset = KSMarkovDataset(test_filename, saved_folder, **kwargs)
    
    # NEW: Create rollout test dataset (full trajectories from test file)
    print("\n" + "="*50)
    print("CREATING ROLLOUT TEST DATASET")
    print("="*50)
    
    rollout_test_dataset = KSTrajectoryDatasetFromFile(
        filename=test_filename,
        saved_folder=saved_folder,
        **kwargs
    )
    
    print(f"Rollout test dataset created with {len(rollout_test_dataset)} trajectories")
    
    x_normalizer = None
    y_normalizer = None

    if data_normalizer:
        print('---------Using simple global normalization---------------')
        
        # Collect all values as flat lists from training data
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
        
        # Simple wrapper dataset for normalized data
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
        
        # Apply normalization to Markov datasets
        train_dataset = NormalizedDataset(train_dataset, x_normalizer, y_normalizer)
        val_dataset = NormalizedDataset(val_dataset, x_normalizer, y_normalizer)
        test_dataset = NormalizedDataset(test_dataset, x_normalizer, y_normalizer)
        
        # Note: rollout_test_dataset is NOT normalized here - normalization applied during evaluation
    
    print(f"\nFinal dataset sizes:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Rollout test dataset size: {len(rollout_test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, rollout_test_dataset, x_normalizer, y_normalizer


# Example usage:
# """
# # Load datasets with rollout capability
# train_dataset, val_dataset, test_dataset, rollout_test_dataset, x_norm, y_norm = ks_markov_dataset(
#     filename="KS_train_2048.h5",
#     saved_folder="/data/user_data/rvk/ks/res_512/visc_0.075_L64.0_lmax8_et5.0_nte51_nt51/",
#     val_filename="KS_valid.h5",
#     test_filename="KS_test.h5",
#     reduced_batch=1,
#     reduced_resolution=1,
#     reduced_resolution_t=1,
#     num_samples_max=-1,
#     data_normalizer=True
# )

# # Use train_dataset, val_dataset, test_dataset for teacher-forcing training/evaluation
# # Use rollout_test_dataset for rollout evaluation

# # Example of accessing rollout data:
# sample_trajectory = rollout_test_dataset[0]  # Shape: (time_steps, spatial_points)
# print(f"Sample trajectory shape: {sample_trajectory.shape}")

# # Use with rollout evaluation function:
# rollout_results = evaluate_1d_rollout_all_resolution_with_dataset(
#     model=your_model,
#     rollout_test_dataset=rollout_test_dataset,
#     current_res=512,
#     test_resolutions=[128, 256, 512],
#     data_resolution=512,
#     pde='ks',
#     x_normalizer=x_norm,
#     y_normalizer=y_norm,
#     rollout_steps=10
# )
# """

# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from models.custom_layer import UnitGaussianNormalizer
# import os
# import h5py
# import numpy as np
# from einops import rearrange

# class KSMarkovDataset(Dataset):
#     def __init__(self, 
#                  filename, 
#                  saved_folder, 
#                  reduced_batch=1, 
#                  reduced_resolution=1, 
#                  reduced_resolution_t=1, 
#                  num_samples_max=-1,
#                  **kwargs):
        
#         root_path = os.path.join(os.path.abspath(saved_folder), filename)
#         print(f'Loading from file: {root_path}')
        
#         # Determine which split from filename
#         if 'train' in filename.lower():
#             self.split = 'train'
#         elif 'valid' in filename.lower():
#             self.split = 'valid'
#         elif 'test' in filename.lower():
#             self.split = 'test'
#         else:
#             # Fallback: assume train split
#             self.split = 'train'
#             print(f"Warning: Could not determine split from filename {filename}, assuming 'train'")
        
#         self._load_ks_data(root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max)
        
#         # Extract the input sequence (all timesteps except last)
#         x = self.data[:, :-1]  # (batch, time-1, spatial)
#         # Extract the output sequence (all timesteps except first)
#         y = self.data[:, 1:]   # (batch, time-1, spatial)
        
#         # Flatten batch and time dimensions together and add channel dimension
#         batch_size, time_steps = x.shape[0], x.shape[1]
#         spatial_size = x.shape[2]
        
#         # Reshape to (batch*time, channels=1, spatial) format for consistency with burger dataset
#         self.x = torch.tensor(x, dtype=torch.float)
#         self.y = torch.tensor(y, dtype=torch.float)
        
#         # Add channel dimension: (batch, time, spatial) -> (batch*time, 1, spatial)
#         self.x = rearrange(self.x, 'b t s -> (b t) 1 s')
#         self.y = rearrange(self.y, 'b t s -> (b t) 1 s')
        
#         assert len(self.x) == len(self.y), "Invalid input output pairs"
#         print(f"x shape: {self.x.shape}")
#         print(f"y shape: {self.y.shape}")
#         if hasattr(self, 'x_coords') and self.x_coords is not None:
#             print(f"spatial coordinates shape: {self.x_coords.shape}")
    
#     def _load_ks_data(self, root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max):
#         """Load data from KS HDF5 file format"""
#         with h5py.File(root_path, 'r') as f:
#             # Navigate to the correct split
#             if self.split in f:
#                 group = f[self.split]
#             else:
#                 # If split not found, try to find data directly
#                 keys = list(f.keys())
#                 print(f"Available keys: {keys}")
#                 if len(keys) == 1:
#                     group = f[keys[0]]
#                     print(f"Using group: {keys[0]}")
#                 else:
#                     raise ValueError(f"Could not find split '{self.split}' in file. Available keys: {keys}")
            
#             data_keys = list(group.keys())
#             print(f"Available data keys in '{self.split}': {data_keys}")
            
#             # Find the main PDE data key
#             pde_key = None
#             for key in data_keys:
#                 if 'pde' in key.lower() and '-' in key:
#                     pde_key = key
#                     break
            
#             if pde_key is None:
#                 raise ValueError(f"Could not find PDE data key in {data_keys}")
            
#             # Load main PDE data: (batch, time, spatial)
#             pde_data = np.array(group[pde_key], dtype=np.float32)
#             print(f"Original PDE data shape: {pde_data.shape}")
            
#             # Load time data if available
#             if 't' in data_keys:
#                 self.time = np.array(group['t'], dtype=np.float32)
#                 print(f"Time data shape: {self.time.shape}")
#             else:
#                 self.time = None
            
#             # Load spatial coordinates if available
#             if 'x' in data_keys:
#                 x_coords = np.array(group['x'], dtype=np.float32)
#                 # x might be (batch, spatial) or just (spatial)
#                 if len(x_coords.shape) == 2:
#                     self.x_coords = x_coords[0]  # Assume all batches have same spatial coords
#                 else:
#                     self.x_coords = x_coords
#                 print(f"Original spatial coordinates shape: {self.x_coords.shape}")
#             else:
#                 self.x_coords = None
            
#             # Load dx and dt if available
#             if 'dx' in data_keys:
#                 self.dx = np.array(group['dx'], dtype=np.float32)
#                 print(f"dx shape: {self.dx.shape}")
#             else:
#                 self.dx = None
                
#             if 'dt' in data_keys:
#                 self.dt = np.array(group['dt'], dtype=np.float32)
#                 print(f"dt shape: {self.dt.shape}")
#             else:
#                 self.dt = None

#             # Naive downsampling
#             pde_data = pde_data[::reduced_batch, 
#                                ::reduced_resolution_t, 
#                                ::reduced_resolution]
#             print(f"Data shape after downsampling: {pde_data.shape}")
            
#             # Apply sample limit
#             if num_samples_max > 0:
#                 num_samples_max = min(num_samples_max, pde_data.shape[0])
#             else:
#                 num_samples_max = pde_data.shape[0]

#             self.data = pde_data[:num_samples_max]
#             print(f"Final data shape: {self.data.shape}")
            
#             # Also reduce time and coordinate arrays if they exist
#             if self.time is not None:
#                 if len(self.time.shape) == 2:  # (batch, time)
#                     self.time = self.time[:num_samples_max, ::reduced_resolution_t]
#                 else:  # (time,)
#                     self.time = self.time[::reduced_resolution_t]
                    
#             if self.x_coords is not None:
#                 self.x_coords = self.x_coords[::reduced_resolution]
#                 if hasattr(self, 'x_coords'):
#                     self.grid = torch.tensor(self.x_coords, dtype=torch.float).unsqueeze(-1)
    
#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]


# def ks_markov_dataset(filename, saved_folder, data_normalizer=True, 
#                       val_filename="KS_valid.h5", test_filename="KS_test.h5", **kwargs):
#     """
#     Returns train, validation, and test datasets for Kuramoto-Sivashinsky using separate files.
#     Uses UnitGaussianNormalizer similar to burger dataset.
    
#     Args:
#         filename: Training file name (e.g., "KS_train_2048.h5")
#         saved_folder: Path to folder containing the files
#         data_normalizer: Whether to apply normalization
#         val_filename: Validation file name (default: "KS_valid.h5")
#         test_filename: Test file name (default: "KS_test.h5")
#         **kwargs: Additional arguments to pass to KSMarkovDataset
        
#     Returns:
#         train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
#     """
#     # Create separate datasets for train, validation, and test
#     train_dataset = KSMarkovDataset(filename, saved_folder, **kwargs)
#     val_dataset = KSMarkovDataset(val_filename, saved_folder, **kwargs)
#     test_dataset = KSMarkovDataset(test_filename, saved_folder, **kwargs)
    
#     x_normalizer = None
#     y_normalizer = None

#     if data_normalizer:
#         print('---------Using simple global normalization---------------')
        
#         # Collect all values as flat lists
#         all_x_values = []
#         all_y_values = []
        
#         for x, y in train_dataset:
#             all_x_values.extend(x.flatten().tolist())
#             all_y_values.extend(y.flatten().tolist())
        
#         # Compute global statistics
#         x_tensor = torch.tensor(all_x_values)
#         y_tensor = torch.tensor(all_y_values)
        
#         x_mean, x_std = x_tensor.mean(), x_tensor.std()
#         y_mean, y_std = y_tensor.mean(), y_tensor.std()
        
#         print(f"Global statistics: X(mean={x_mean:.6f}, std={x_std:.6f}), Y(mean={y_mean:.6f}, std={y_std:.6f})")
#         print(f"Computed from {len(train_dataset)} samples")
        
#         # Simple normalizer
#         class SimpleNormalizer:
#             def __init__(self, mean, std, eps=1e-8):
#                 self.mean = float(mean)
#                 self.std = float(std)
#                 self.eps = eps
            
#             def encode(self, x):
#                 return (x - self.mean) / (self.std + self.eps)
            
#             def decode(self, x, device='cuda'):
#                 return x * (self.std + self.eps) + self.mean
            
#             def cuda(self):
#                 return self
            
#             def cpu(self):
#                 return self
        
#         x_normalizer = SimpleNormalizer(x_mean, x_std)
#         y_normalizer = SimpleNormalizer(y_mean, y_std)
        
#         # Simple wrapper dataset
#         class NormalizedDataset(Dataset):
#             def __init__(self, dataset, x_normalizer, y_normalizer):
#                 self.dataset = dataset
#                 self.x_normalizer = x_normalizer
#                 self.y_normalizer = y_normalizer
            
#             def __len__(self):
#                 return len(self.dataset)
            
#             def __getitem__(self, idx):
#                 x, y = self.dataset[idx]
#                 return self.x_normalizer.encode(x), self.y_normalizer.encode(y)
    
#     # if data_normalizer:
#     #     print('---------Using data normalizer---------------')
#     #     temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        
#     #     # Collect all training data for computing normalization
#     #     x_train_all = []
#     #     y_train_all = []
#     #     for x_batch, y_batch in temp_loader:
#     #         x_train_all.append(x_batch)
#     #         y_train_all.append(y_batch)
        
#     #     x_train_tensor = torch.cat(x_train_all, dim=0)
#     #     y_train_tensor = torch.cat(y_train_all, dim=0)
        
#     #     # Initialize normalizers using training data
#     #     x_normalizer = UnitGaussianNormalizer(x_train_tensor)
#     #     y_normalizer = UnitGaussianNormalizer(y_train_tensor)
        
#     #     # Create a wrapper dataset class that applies normalization
#     #     class NormalizedDataset(Dataset):
#     #         def __init__(self, dataset, x_normalizer, y_normalizer):
#     #             self.dataset = dataset
#     #             self.x_normalizer = x_normalizer
#     #             self.y_normalizer = y_normalizer
            
#     #         def __len__(self):
#     #             return len(self.dataset)
            
#     #         def __getitem__(self, idx):
#     #             x, y = self.dataset[idx]

#     #             # Ensure x and y are PyTorch tensors, not NumPy arrays
#     #             if isinstance(x, np.ndarray):
#     #                 x = torch.from_numpy(x).float()
#     #             if isinstance(y, np.ndarray):
#     #                 y = torch.from_numpy(y).float()

#     #             return self.x_normalizer.encode(x), self.y_normalizer.encode(y)
        
#         # Apply normalization to each dataset
#         train_dataset = NormalizedDataset(train_dataset, x_normalizer, y_normalizer)
#         val_dataset = NormalizedDataset(val_dataset, x_normalizer, y_normalizer)
#         test_dataset = NormalizedDataset(test_dataset, x_normalizer, y_normalizer)
    
#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
#     print(f"Test dataset size: {len(test_dataset)}")
    
#     return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer