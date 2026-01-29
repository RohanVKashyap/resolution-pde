import torch
from torch.utils.data import Dataset, DataLoader, random_split
from models.custom_layer import UnitGaussianNormalizer
import os
import numpy as np
from einops import rearrange
from scipy.io import loadmat
import h5py
from utils.low_pass_filter import lowpass_filter_2d


class NSTrajectoryDatasetFromExtracted(Dataset):
    """
    Simple dataset wrapper for extracted NS trajectories for rollout evaluation.
    """
    def __init__(self, trajectories, trajectory_info):
        self.trajectories = trajectories
        self.trajectory_info = trajectory_info
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx]  # Returns shape: (time_steps, height, width)
    
    def get_trajectory_info(self, idx):
        return self.trajectory_info[idx]
    
    def get_all_info(self):
        return self.trajectory_info


def extract_ns_test_trajectories_for_rollout_single(filename,
                                                     saved_folder,
                                                     reduced_batch=1,
                                                     reduced_resolution=1,
                                                     reduced_resolution_t=1,
                                                     use_low_pass_filter=False,
                                                     lowpass_cutoff_ratio=1.0,
                                                     num_samples_max=-1,
                                                     split_ratio=None,
                                                     random_seed=42):
    """
    Extract complete test trajectories from single-resolution NS dataset BEFORE Markov pair creation.
    """
    if split_ratio is None:
        split_ratio = [0.8, 0.1, 0.1]
    
    print(f"Extracting NS test trajectories for rollout evaluation")
    print(f"File: {filename}")
    print(f"Using low-pass filter: {use_low_pass_filter}")
    
    # Construct full file path
    file_path = os.path.join(saved_folder, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f'Loading from file: {file_path}')
    
    # Determine file type from extension
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext == '.mat':
        # Load .mat file
        mat_data = loadmat(file_path)
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        print(f"Available keys: {keys}")
        
        if 'u' not in mat_data:
            raise KeyError(f"'u' key not found in {file_path}")
        
        raw_data = np.array(mat_data['u'], dtype=np.float32)
        print(f"Original data shape: {raw_data.shape}")
        
        # Transpose from (batch, height, width, time) to (batch, time, height, width)
        _data = np.transpose(raw_data, (0, 3, 1, 2))
        
    elif file_ext == '.h5':
        # Load .h5 file
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            print(f"Available keys: {keys}")
            
            if 'u' not in f:
                raise KeyError(f"'u' key not found in {file_path}")
            
            raw_data = np.array(f['u'], dtype=np.float32)
            print(f"Original data shape: {raw_data.shape}")
            
            # Check if we need to transpose
            if raw_data.shape[-1] < 100 and raw_data.shape[-1] < min(raw_data.shape[1], raw_data.shape[2]):
                print("Detected (batch, height, width, time) format, transposing to (batch, time, height, width)")
                _data = np.transpose(raw_data, (0, 3, 1, 2))
            else:
                print("Assuming data is already in (batch, time, height, width) format")
                _data = raw_data
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")
    
    print(f"After format conversion: {_data.shape}")
    
    # Apply batch and time reductions first
    _data = _data[::reduced_batch, ::reduced_resolution_t, :, :]
    print(f"After batch/time downsampling: {_data.shape}")
    
    # Apply spatial downsampling (with or without low-pass filter)
    if reduced_resolution > 1:
        if use_low_pass_filter:
            print(f"Using low-pass filter for spatial resolution reduction (factor={reduced_resolution})")
            
            # Convert to tensor
            # Current shape: (batch, time, height, width)
            data_tensor = torch.from_numpy(_data).float()
            
            # Calculate cutoff ratio
            adaptive_cutoff = (1.0 / reduced_resolution) * lowpass_cutoff_ratio
            print(f"  Cutoff ratio: {adaptive_cutoff:.4f}")
            
            # Apply low-pass filter (generates lower resolution data directly)
            _data = lowpass_filter_2d(data_tensor, cutoff_ratio=adaptive_cutoff).numpy()
        else:
            print(f"Using naive spatial downsampling (factor={reduced_resolution})")
            # Naive downsampling: take every Nth pixel
            _data = _data[:, :, ::reduced_resolution, ::reduced_resolution]

    print(f"After spatial processing: {_data.shape}")
    
    # Apply num_samples_max limit if specified
    if num_samples_max > 0:
        num_samples_max = min(num_samples_max, _data.shape[0])
    else:
        num_samples_max = _data.shape[0]
    
    data = _data[:num_samples_max]
    print(f"Total data shape after sampling: {data.shape}")
    
    # Split the data into train/val/test using the same split ratio
    total_samples = data.shape[0]
    train_end = int(total_samples * split_ratio[0])
    val_end = train_end + int(total_samples * split_ratio[1])
    
    # Extract test split
    test_data = data[val_end:]
    print(f"Test split data shape: {test_data.shape}")
    
    all_trajectories = []
    trajectory_info = []
    
    # Store each trajectory individually
    for i in range(test_data.shape[0]):
        trajectory = torch.tensor(test_data[i], dtype=torch.float)  # Shape: (time_steps, height, width)
        all_trajectories.append(trajectory)
        trajectory_info.append({
            'original_index': i,
            'source': 'single_resolution_file'
        })
    
    print(f"Extracted {len(all_trajectories)} complete trajectories")
    if len(all_trajectories) > 0:
        print(f"Trajectory shapes: {[traj.shape for traj in all_trajectories[:3]]}")
    
    return all_trajectories, trajectory_info


class NSMarkovDataset(Dataset):
    def __init__(self, 
                 filename,
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1,
                 use_low_pass_filter=False,
                 lowpass_cutoff_ratio=1.0,
                 num_samples_max=-1,
                 **kwargs):
        
        self.use_low_pass_filter = use_low_pass_filter
        self.lowpass_cutoff_ratio = lowpass_cutoff_ratio
        
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
        
        # Apply batch and time reductions first
        combined_data = combined_data[::reduced_batch, ::reduced_resolution_t, :, :, :]
        print(f"After batch/time reduction: {combined_data.shape}")
        
        # Apply spatial downsampling (with or without low-pass filter)
        if reduced_resolution > 1:
            if use_low_pass_filter:
                print(f"Using low-pass filter for spatial resolution reduction (factor={reduced_resolution})")
                
                # Convert to tensor for low-pass filtering
                # Current shape: (batch, time, height, width, channels)
                data_tensor = torch.from_numpy(combined_data).float()
                
                # Calculate cutoff ratio
                adaptive_cutoff = (1.0 / reduced_resolution) * lowpass_cutoff_ratio
                print(f"  Cutoff ratio: {adaptive_cutoff:.4f}")
                
                # Rearrange to: (batch, time, channels, height, width)
                data_tensor = rearrange(data_tensor, 'b t h w c -> b t c h w')
                
                # Apply low-pass filter (generates lower resolution data directly)
                combined_data = lowpass_filter_2d(data_tensor, cutoff_ratio=adaptive_cutoff)
                
                # Rearrange back to: (batch, time, height, width, channels)
                combined_data = rearrange(combined_data, 'b t c h w -> b t h w c').numpy()
            else:
                print(f"Using naive spatial downsampling (factor={reduced_resolution})")
                # Naive downsampling: take every Nth pixel
                combined_data = combined_data[:, :, ::reduced_resolution, ::reduced_resolution, :]

        print(f"After spatial processing: {combined_data.shape}")
        
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
                      use_low_pass_filter=False,
                      lowpass_cutoff_ratio=1.0,
                      data_normalizer=True,
                      normalization_type="unit_gaussian",
                      **kwargs):
    """
    Returns train, validation, and test datasets for single resolution Navier-Stokes 
    with 0.8/0.1/0.1 ratio.
    NOW INCLUDES ROLLOUT TEST DATASET FOR EVALUATION.
    
    Args:
        filename: .mat or .h5 file name (e.g., "ns_32_1e-3.mat", "ns_256_1e-03.h5")
        saved_folder: Path to folder containing the file
        use_low_pass_filter: If True, use low-pass filter for spatial downsampling
        lowpass_cutoff_ratio: Cutoff ratio for low-pass filter (default: 1.0)
        data_normalizer: Whether to apply normalization
        normalization_type: "simple" or "unit_gaussian"
        **kwargs: Additional arguments to pass to NSMarkovDataset
        
    Returns:
        train_dataset, val_dataset, test_dataset, rollout_test_dataset, x_normalizer, y_normalizer
    """
    print(f"Creating NS dataset with use_low_pass_filter={use_low_pass_filter}")
    if use_low_pass_filter:
        print(f"Low-pass filter cutoff ratio: {lowpass_cutoff_ratio}")
    
    # Create the full dataset
    full_dataset = NSMarkovDataset(
        filename, 
        saved_folder,
        use_low_pass_filter=use_low_pass_filter,
        lowpass_cutoff_ratio=lowpass_cutoff_ratio,
        **kwargs
    )
    
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
    
    # NEW: Create rollout test dataset (full trajectories)
    print("\n" + "="*50)
    print("CREATING ROLLOUT TEST DATASET")
    print("="*50)
    
    # Filter kwargs to only include parameters that the rollout extraction function accepts
    rollout_kwargs = {k: v for k, v in kwargs.items() 
                     if k in ['reduced_batch', 'reduced_resolution', 'reduced_resolution_t', 'num_samples_max']}
    
    rollout_trajectories, rollout_trajectory_info = extract_ns_test_trajectories_for_rollout_single(
        filename=filename,
        saved_folder=saved_folder,
        use_low_pass_filter=use_low_pass_filter,
        lowpass_cutoff_ratio=lowpass_cutoff_ratio,
        split_ratio=[0.8, 0.1, 0.1],  # Use same split ratio
        random_seed=42,
        **rollout_kwargs
    )
    
    rollout_test_dataset = NSTrajectoryDatasetFromExtracted(rollout_trajectories, rollout_trajectory_info)
    
    print(f"Rollout test dataset created with {len(rollout_test_dataset)} trajectories")
    
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
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x).float()
                    if isinstance(y, np.ndarray):
                        y = torch.from_numpy(y).float()
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
    print(f"Rollout test dataset size: {len(rollout_test_dataset)}")
    
    # return train_dataset, val_dataset, test_dataset, rollout_test_dataset, x_normalizer, y_normalizer
    return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer

# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from models.custom_layer import UnitGaussianNormalizer
# import os
# import numpy as np
# from einops import rearrange
# from scipy.io import loadmat
# import h5py

# class NSMarkovDataset(Dataset):
#     def __init__(self, 
#                  filename,  # e.g., "ns_32_1e-3.mat" or "ns_256_1e-03.h5"
#                  saved_folder, 
#                  reduced_batch=1, 
#                  reduced_resolution=1, 
#                  reduced_resolution_t=1, 
#                  num_samples_max=-1,
#                  **kwargs):
        
#         # Construct full file path
#         file_path = os.path.join(saved_folder, filename)
        
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
        
#         print(f'Loading from file: {file_path}')
        
#         # Determine file type from extension
#         file_ext = os.path.splitext(filename)[1].lower()
        
#         if file_ext == '.mat':
#             raw_data = self._load_mat_file(file_path)
#         elif file_ext == '.h5':
#             raw_data = self._load_h5_file(file_path)
#         else:
#             raise ValueError(f"Unsupported file extension: {file_ext}. Supported extensions: .mat, .h5")
        
#         print(f"Raw data shape: {raw_data.shape}")
        
#         if len(raw_data.shape) != 4:
#             raise ValueError(f"Expected 4D array, got {raw_data.shape}")
        
#         # For .h5 files, data is already in (batch, time, height, width) format
#         # For .mat files, we need to transpose from (batch, height, width, time)
#         if file_ext == '.mat':
#             # Transpose from (batch, height, width, time) to (batch, time, height, width)
#             combined_data = np.transpose(raw_data, (0, 3, 1, 2))
#         else:  # .h5 files
#             # Data is already in (batch, time, height, width) format
#             combined_data = raw_data
        
#         # Add channel dimension: (batch, time, height, width, 1)
#         combined_data = np.expand_dims(combined_data, axis=-1)
#         print(f"Converted to shape: {combined_data.shape}")
        
#         # Apply reductions
#         # combined_data shape: (batch, time, height, width, channels)
#         combined_data = combined_data[::reduced_batch,          # batch dimension
#                                     ::reduced_resolution_t,    # time dimension  
#                                     ::reduced_resolution,       # height dimension
#                                     ::reduced_resolution,       # width dimension
#                                     :]                          # all channels
        
#         print(f"After reduction: {combined_data.shape}")
        
#         # Apply sample limit
#         if num_samples_max > 0:
#             num_samples_max = min(num_samples_max, combined_data.shape[0])
#             combined_data = combined_data[:num_samples_max]
#             print(f"After sample limit: {combined_data.shape}")
        
#         self.data = combined_data
        
#         # Extract the input sequence (all timesteps except first and last)
#         x = self.data[:, 1:-1]
#         # Extract the output sequence (all timesteps except first two)
#         y = self.data[:, 2:]
        
#         print(f"Markov pairs - x: {x.shape}, y: {y.shape}")
        
#         # Reshape to (batch*time, channels, height, width) format for neural networks
#         self.x = torch.tensor(x, dtype=torch.float)
#         self.y = torch.tensor(y, dtype=torch.float)
        
#         self.x = rearrange(self.x, 'b t h w c -> (b t) c h w')
#         self.y = rearrange(self.y, 'b t h w c -> (b t) c h w')
        
#         assert len(self.x) == len(self.y), "Invalid input output pairs"
#         print(f"Final x shape: {self.x.shape}")
#         print(f"Final y shape: {self.y.shape}")
    
#     def _load_mat_file(self, file_path):
#         """Load data from .mat file"""
#         print("Loading .mat file...")
#         mat_data = loadmat(file_path)
#         keys = [k for k in mat_data.keys() if not k.startswith('__')]
#         print(f"Available keys: {keys}")
        
#         # Load 'u' key for vorticity time series data
#         if 'u' not in mat_data:
#             raise KeyError(f"'u' key not found in {file_path}. Available keys: {keys}")
        
#         raw_data = np.array(mat_data['u'], dtype=np.float32)
#         return raw_data
    
#     def _load_h5_file(self, file_path):
#         """Load data from .h5 file"""
#         print("Loading .h5 file...")
#         with h5py.File(file_path, 'r') as f:
#             keys = list(f.keys())
#             print(f"Available keys: {keys}")
            
#             # Load 'u' key for vorticity time series data
#             if 'u' not in f:
#                 raise KeyError(f"'u' key not found in {file_path}. Available keys: {keys}")
            
#             # Load the data into memory
#             raw_data = np.array(f['u'], dtype=np.float32)
            
#             # For h5 files, the format is typically (batch, time, height, width)
#             # But let's check and transpose if needed
#             print(f"H5 data shape before any processing: {raw_data.shape}")
            
#             # If the shape suggests it's (batch, height, width, time), transpose it
#             if len(raw_data.shape) == 4:
#                 # Heuristic: if last dimension is smallest and reasonable for time steps
#                 if raw_data.shape[-1] < 100 and raw_data.shape[-1] < min(raw_data.shape[1], raw_data.shape[2]):
#                     print("Detected (batch, height, width, time) format, transposing to (batch, time, height, width)")
#                     raw_data = np.transpose(raw_data, (0, 3, 1, 2))
#                 else:
#                     print("Assuming data is already in (batch, time, height, width) format")
            
#             return raw_data
        
#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]


# def ns_markov_dataset(filename, 
#                             saved_folder, 
#                             data_normalizer=True,
#                             normalization_type="unit_gaussian",
#                             **kwargs):
#     """
#     Returns train, validation, and test datasets for single resolution Navier-Stokes 
#     with 0.8/0.1/0.1 ratio.
    
#     Args:
#         filename: .mat or .h5 file name (e.g., "ns_32_1e-3.mat", "ns_256_1e-03.h5")
#         saved_folder: Path to folder containing the file
#         data_normalizer: Whether to apply normalization
#         normalization_type: "simple" or "unit_gaussian"
#         **kwargs: Additional arguments to pass to NSMarkovDataset
        
#     Returns:
#         train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
#     """
#     # Create the full dataset
#     full_dataset = NSMarkovDataset(filename, saved_folder, **kwargs)
    
#     # Calculate split sizes
#     dataset_size = len(full_dataset)
#     train_size = int(0.8 * dataset_size)
#     val_size = int(0.1 * dataset_size)
#     test_size = dataset_size - train_size - val_size
    
#     print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
#     # Split dataset
#     train_dataset, val_dataset, test_dataset = random_split(
#         full_dataset, 
#         [train_size, val_size, test_size],
#         generator=torch.Generator().manual_seed(42)  # For reproducibility
#     )
    
#     x_normalizer = None
#     y_normalizer = None
    
#     if data_normalizer:
#         if normalization_type == "simple":
#             print('---------Using simple global normalization---------------')
            
#             # Collect all values from training set
#             all_x_values = []
#             all_y_values = []
            
#             for x, y in train_dataset:
#                 all_x_values.extend(x.flatten().tolist())
#                 all_y_values.extend(y.flatten().tolist())
            
#             x_tensor = torch.tensor(all_x_values)
#             y_tensor = torch.tensor(all_y_values)
            
#             x_mean, x_std = x_tensor.mean(), x_tensor.std()
#             y_mean, y_std = y_tensor.mean(), y_tensor.std()
            
#             print(f"Global statistics: X(mean={x_mean:.6f}, std={x_std:.6f}), Y(mean={y_mean:.6f}, std={y_std:.6f})")
            
#             class SimpleNormalizer:
#                 def __init__(self, mean, std, eps=1e-8):
#                     self.mean = float(mean)
#                     self.std = float(std)
#                     self.eps = eps
                
#                 def encode(self, x):
#                     return (x - self.mean) / (self.std + self.eps)
                
#                 def decode(self, x, device='cuda'):
#                     return x * (self.std + self.eps) + self.mean
                
#                 def cuda(self):
#                     return self
                
#                 def cpu(self):
#                     return self
            
#             x_normalizer = SimpleNormalizer(x_mean, x_std)
#             y_normalizer = SimpleNormalizer(y_mean, y_std)
            
#             class SimpleNormalizedDataset(Dataset):
#                 def __init__(self, dataset, x_normalizer, y_normalizer):
#                     self.dataset = dataset
#                     self.x_normalizer = x_normalizer
#                     self.y_normalizer = y_normalizer
                
#                 def __len__(self):
#                     return len(self.dataset)
                
#                 def __getitem__(self, idx):
#                     x, y = self.dataset[idx]
#                     return self.x_normalizer.encode(x), self.y_normalizer.encode(y)
            
#             train_dataset = SimpleNormalizedDataset(train_dataset, x_normalizer, y_normalizer)
#             val_dataset = SimpleNormalizedDataset(val_dataset, x_normalizer, y_normalizer)
#             test_dataset = SimpleNormalizedDataset(test_dataset, x_normalizer, y_normalizer)
            
#         elif normalization_type == "unit_gaussian":
#             print('---------Using UnitGaussianNormalizer---------------')
#             temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
            
#             # Collect all training data for computing normalization
#             x_train_all = []
#             y_train_all = []
#             for batch in temp_loader:
#                 x_batch, y_batch = batch
#                 x_train_all.append(x_batch)
#                 y_train_all.append(y_batch)
            
#             x_train_tensor = torch.cat(x_train_all, dim=0)
#             y_train_tensor = torch.cat(y_train_all, dim=0)
            
#             # Initialize normalizers using training data
#             x_normalizer = UnitGaussianNormalizer(x_train_tensor)
#             y_normalizer = UnitGaussianNormalizer(y_train_tensor)
            
#             # Create a wrapper dataset class that applies normalization
#             class NormalizedDataset(Dataset):
#                 def __init__(self, dataset, x_normalizer, y_normalizer):
#                     self.dataset = dataset
#                     self.x_normalizer = x_normalizer
#                     self.y_normalizer = y_normalizer
                
#                 def __len__(self):
#                     return len(self.dataset)
                
#                 def __getitem__(self, idx):
#                     x, y = self.dataset[idx]
#                     return self.x_normalizer.encode(x), self.y_normalizer.encode(y)
            
#             # Apply normalization to each dataset
#             train_dataset = NormalizedDataset(train_dataset, x_normalizer, y_normalizer)
#             val_dataset = NormalizedDataset(val_dataset, x_normalizer, y_normalizer)
#             test_dataset = NormalizedDataset(test_dataset, x_normalizer, y_normalizer)
            
#         else:
#             raise ValueError(f"Invalid normalization_type: {normalization_type}. Must be 'simple' or 'unit_gaussian'")
    
#     print(f"\nFinal dataset sizes:")
#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
#     print(f"Test dataset size: {len(test_dataset)}")
    
#     return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer