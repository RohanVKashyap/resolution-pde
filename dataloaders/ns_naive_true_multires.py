import torch
from torch.utils.data import Dataset, DataLoader, random_split
from models.custom_layer import UnitGaussianNormalizer
import os
import glob
import h5py
import numpy as np
import math as mt
from einops import rearrange
from scipy.io import loadmat

class NSVTrueMultiResMarkovDataset(Dataset):
    def __init__(self, 
                 saved_folder,  # Folder containing .mat or .h5 files
                 viscosity="1e-3",  # Viscosity parameter for filename
                 file_extension=".h5",  # File extension: ".mat" or ".h5"
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 data_mres_size=None,  # Dict: {resolution: num_samples} from actual files
                 add_res=None,  # Additional resolutions for downsampling
                 add_res_samples=None,  # Dict: {resolution: num_samples} for downsampled data
                 downsample_from_res=None,  # Which resolution to use for downsampling (default: highest)
                 split_ratio=None,  # For splitting: [train%, val%, test%]
                 random_seed=42,
                 split='train',
                 **kwargs):
        
        self.random_seed = random_seed
        self.split = split
        self.viscosity = viscosity
        self.file_extension = file_extension.lower()
        self.downsample_from_res = downsample_from_res
        
        # Validate file extension
        if self.file_extension not in ['.mat', '.h5']:
            raise ValueError(f"Unsupported file extension: {self.file_extension}. Supported: .mat, .h5")
        
        # Set default split ratio if not provided
        if split_ratio is None:
            split_ratio = [0.8, 0.1, 0.1]  # 80% train, 10% val, 10% test
        
        # Set default data_mres_size if not provided
        if data_mres_size is None:
            data_mres_size = {512: 0, 64: 0, 32: 0}
        
        # Set downsample_from_res to highest available resolution if not specified
        if downsample_from_res is None and data_mres_size:
            # Only consider resolutions with non-zero samples
            available_resolutions = [res for res, samples in data_mres_size.items() if samples > 0]
            if available_resolutions:
                downsample_from_res = max(available_resolutions)
                print(f"Auto-selected downsample_from_res: {downsample_from_res} (highest available)")
            else:
                downsample_from_res = max(data_mres_size.keys()) if data_mres_size else None
                print(f"Auto-selected downsample_from_res: {downsample_from_res} (highest in config)")
        
        self.downsample_from_res = downsample_from_res
        
        # Set default add_res_samples if not provided
        if add_res_samples is None:
            add_res_samples = {64: 0, 32: 0}  # Default for downsampled resolutions
        
        print(f"Loading Navier-Stokes multi-resolution data from {saved_folder}")
        print(f"Viscosity: {viscosity}, File extension: {self.file_extension}")
        print(f"Target samples per resolution (from files): {data_mres_size}")
        if add_res:
            print(f"Additional downsampled resolutions: {add_res}")
            print(f"Target samples per downsampled resolution: {add_res_samples}")
            print(f"Will downsample from resolution: {downsample_from_res}")
        
        # Store data from all resolutions
        self.x = []
        self.y = []
        self.resolution_info = []  # Track which resolution each sample came from
        
        # Load data from each resolution (from actual files)
        for resolution, target_samples in data_mres_size.items():
            if target_samples == 0:
                continue
                
            # Construct path to file
            filename = f"ns_{resolution}_{viscosity}{self.file_extension}"
            res_file_path = os.path.join(saved_folder, filename)
            
            if not os.path.exists(res_file_path):
                print(f"Warning: File {res_file_path} does not exist. Skipping resolution {resolution}")
                continue
            
            print(f"Loading resolution {resolution} from {res_file_path}")
            
            # Load data based on file extension
            if self.file_extension == '.mat':
                raw_data = self._load_mat_file(res_file_path)
            else:  # .h5
                raw_data = self._load_h5_file(res_file_path)
            
            if raw_data is None:
                continue
                
            print(f"  Raw data shape for {resolution}: {raw_data.shape}")
            
            # Handle data format based on file type
            if self.file_extension == '.mat':
                # .mat files: (N_sims, height, width, time_steps)
                # Convert to: (N_sims, time_steps, height, width, channels)
                if len(raw_data.shape) != 4:
                    print(f"  Warning: Expected 4D array, got {raw_data.shape}. Skipping resolution {resolution}")
                    continue
                
                # Transpose from (batch, height, width, time) to (batch, time, height, width)
                combined_data = np.transpose(raw_data, (0, 3, 1, 2))
            else:  # .h5
                # .h5 files: typically (batch, time, height, width) already
                if len(raw_data.shape) != 4:
                    print(f"  Warning: Expected 4D array, got {raw_data.shape}. Skipping resolution {resolution}")
                    continue
                
                # Check if we need to transpose (heuristic based on dimensions)
                if raw_data.shape[-1] < 100 and raw_data.shape[-1] < min(raw_data.shape[1], raw_data.shape[2]):
                    print(f"  Detected (batch, height, width, time) format, transposing to (batch, time, height, width)")
                    combined_data = np.transpose(raw_data, (0, 3, 1, 2))
                else:
                    print(f"  Assuming data is already in (batch, time, height, width) format")
                    combined_data = raw_data
            
            # Add channel dimension: (batch, time, height, width, 1)
            combined_data = np.expand_dims(combined_data, axis=-1)
            print(f"  Converted to shape: {combined_data.shape}")
            
            # Apply reductions
            combined_data = combined_data[::reduced_batch, 
                                        ::reduced_resolution_t, 
                                        ::reduced_resolution, 
                                        ::reduced_resolution, :]
            
            print(f"  After reduction: {combined_data.shape}")
            
            # Split the data into train/val/test based on split_ratio
            total_samples = combined_data.shape[0]
            
            # Calculate split boundaries
            train_end = int(total_samples * split_ratio[0])
            val_end = train_end + int(total_samples * split_ratio[1])
            
            # Determine which samples to use based on split
            if self.split == 'train':
                split_data = combined_data[:train_end]
            elif self.split in ['val', 'valid']:
                split_data = combined_data[train_end:val_end]
            elif self.split == 'test':
                split_data = combined_data[val_end:]
            else:
                raise ValueError(f"Invalid split: {self.split}")
            
            print(f"  Split '{self.split}' data shape: {split_data.shape}")
            
            # Further sample if target_samples is specified and less than available
            if target_samples > 0 and target_samples < split_data.shape[0]:
                # Calculate samples for current split
                split_idx = {'train': 0, 'valid': 1, 'val': 1, 'test': 2}.get(self.split, 0)
                samples_for_split = int(target_samples * split_ratio[split_idx])
                
                if samples_for_split > 0:
                    # Sample deterministically based on split and resolution
                    np.random.seed(self.random_seed + resolution + split_idx)
                    sample_indices = np.random.choice(split_data.shape[0], samples_for_split, replace=False)
                    sampled_data = split_data[sample_indices]
                    print(f"  Subsampled to {sampled_data.shape[0]} samples for {self.split} split")
                else:
                    print(f"  No samples allocated for {self.split} split at resolution {resolution}")
                    continue
            else:
                sampled_data = split_data
                print(f"  Using all {sampled_data.shape[0]} samples for {self.split} split")
            
            # Create input-output pairs for Markov property (same as original NS dataset)
            x_res = sampled_data[:, 1:-1]  # Skip first and last timestep
            y_res = sampled_data[:, 2:]    # Skip first two timesteps
            
            # Convert to tensors and rearrange to (batch*time, channels, height, width)
            x_tensor = torch.tensor(x_res, dtype=torch.float)
            y_tensor = torch.tensor(y_res, dtype=torch.float)
            
            x_reshaped = rearrange(x_tensor, 'b t h w c -> (b t) c h w')
            y_reshaped = rearrange(y_tensor, 'b t h w c -> (b t) c h w')
            
            # Add to main lists
            for i in range(len(x_reshaped)):
                self.x.append(x_reshaped[i])
                self.y.append(y_reshaped[i])
                self.resolution_info.append(f"{resolution}_file")
            
            print(f"  Added {len(x_reshaped)} sample pairs from resolution {resolution}")
        
        # Handle add_res parameter for downsampling
        if add_res is not None and add_res_samples is not None:
            print(f"Adding downsampled resolutions: {add_res}")
            # Use the specified resolution for downsampling
            if self.downsample_from_res is not None:
                self._add_downsampled_data(saved_folder, self.downsample_from_res, add_res, add_res_samples, 
                                         split_ratio, reduced_batch, reduced_resolution, reduced_resolution_t)
            else:
                print("Warning: No resolution specified for downsampling and no available resolutions found.")
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"Total samples loaded: {len(self.x)}")
        
        # Print resolution distribution
        if self.resolution_info:
            unique_res, counts = np.unique(self.resolution_info, return_counts=True)
            print("Resolution distribution:")
            for res, count in zip(unique_res, counts):
                print(f"  {res}: {count} samples")
        
        if len(self.x) > 0:
            print(f"Sample x shapes: {[tuple(x.shape) for x in self.x[:min(3, len(self.x))]]}")
            print(f"Sample y shapes: {[tuple(y.shape) for y in self.y[:min(3, len(self.y))]]}")
    
    def _load_mat_file(self, file_path):
        """Load data from .mat file"""
        try:
            print(f"  Loading .mat file...")
            mat_data = loadmat(file_path)
            
            # Print available keys to understand data structure
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            print(f"  Available keys: {keys}")
            
            # Use 'u' key for vorticity time series data
            if 'u' not in mat_data:
                print(f"  Warning: 'u' key not found in {file_path}. Skipping.")
                return None
            
            raw_data = np.array(mat_data['u'], dtype=np.float32)
            return raw_data
        except Exception as e:
            print(f"  Error loading .mat file {file_path}: {e}")
            return None
    
    def _load_h5_file(self, file_path):
        """Load data from .h5 file"""
        try:
            print(f"  Loading .h5 file...")
            with h5py.File(file_path, 'r') as f:
                keys = list(f.keys())
                print(f"  Available keys: {keys}")
                
                # Use 'u' key for vorticity time series data
                if 'u' not in f:
                    print(f"  Warning: 'u' key not found in {file_path}. Skipping.")
                    return None
                
                # Load the data into memory
                raw_data = np.array(f['u'], dtype=np.float32)
                return raw_data
        except Exception as e:
            print(f"  Error loading .h5 file {file_path}: {e}")
            return None
    
    def _add_downsampled_data(self, saved_folder, base_resolution, add_res, add_res_samples, 
                            split_ratio, reduced_batch, reduced_resolution, reduced_resolution_t):
        """Add downsampled data from base resolution"""
        # Construct path to base resolution file
        filename = f"ns_{base_resolution}_{self.viscosity}{self.file_extension}"
        base_file_path = os.path.join(saved_folder, filename)
        
        if not os.path.exists(base_file_path):
            print(f"Warning: Base file {base_file_path} does not exist. Cannot create downsampled data.")
            return
        
        print(f"Adding downsampled data from {base_file_path}")
        
        # Load original data based on file extension
        if self.file_extension == '.mat':
            raw_data = self._load_mat_file(base_file_path)
        else:  # .h5
            raw_data = self._load_h5_file(base_file_path)
        
        if raw_data is None:
            print(f"Warning: Could not load base file {base_file_path}. Cannot create downsampled data.")
            return
        
        # Handle data format based on file type
        if self.file_extension == '.mat':
            # Handle standard NS data: (N_sims, height, width, time_steps)
            # Convert to: (N_sims, time_steps, height, width, channels)
            if len(raw_data.shape) != 4:
                print(f"Warning: Expected 4D array, got {raw_data.shape}. Cannot create downsampled data.")
                return
            
            # Transpose from (batch, height, width, time) to (batch, time, height, width)
            original_data = np.transpose(raw_data, (0, 3, 1, 2))
        else:  # .h5
            if len(raw_data.shape) != 4:
                print(f"Warning: Expected 4D array, got {raw_data.shape}. Cannot create downsampled data.")
                return
            
            # Check if we need to transpose (same heuristic as in main loading)
            if raw_data.shape[-1] < 100 and raw_data.shape[-1] < min(raw_data.shape[1], raw_data.shape[2]):
                print(f"  Detected (batch, height, width, time) format, transposing to (batch, time, height, width)")
                original_data = np.transpose(raw_data, (0, 3, 1, 2))
            else:
                print(f"  Assuming data is already in (batch, time, height, width) format")
                original_data = raw_data
        
        # Add channel dimension: (batch, time, height, width, 1)
        original_data = np.expand_dims(original_data, axis=-1)
        
        # Split the original data first
        total_samples = original_data.shape[0]
        train_end = int(total_samples * split_ratio[0])
        val_end = train_end + int(total_samples * split_ratio[1])
        
        # Get data for current split
        if self.split == 'train':
            split_original_data = original_data[:train_end]
        elif self.split in ['val', 'valid']:
            split_original_data = original_data[train_end:val_end]
        elif self.split == 'test':
            split_original_data = original_data[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        original_spatial_size = split_original_data.shape[2]  # Height dimension
        
        split_idx = {'train': 0, 'valid': 1, 'val': 1, 'test': 2}.get(self.split, 0)
        
        for target_res in add_res:
            if target_res >= original_spatial_size:
                print(f"  Warning: Target resolution {target_res} >= original {original_spatial_size}. Skipping.")
                continue
            
            # Get number of samples for this resolution
            target_samples = add_res_samples.get(target_res, 100)  # Default 100 if not specified
            samples_for_split = int(target_samples * split_ratio[split_idx])
            
            if samples_for_split == 0:
                print(f"  No downsampled samples allocated for {self.split} split at resolution {target_res}")
                continue
            
            # Calculate downsampling factor (assuming square grids)
            downsample_factor = original_spatial_size // target_res
            
            # Sample and downsample
            np.random.seed(self.random_seed + target_res + split_idx + 10000)  # Different seed space
            sample_indices = np.random.choice(split_original_data.shape[0], samples_for_split, replace=True)
            sampled_data = split_original_data[sample_indices]
            
            # Apply downsampling (spatial only, keep time and channels)
            downsampled_data = sampled_data[::reduced_batch, 
                                          ::reduced_resolution_t, 
                                          ::downsample_factor, 
                                          ::downsample_factor, :]
            
            # Process and add to dataset (same Markov logic as main data)
            x_down = downsampled_data[:, 1:-1]  # Skip first and last timestep
            y_down = downsampled_data[:, 2:]    # Skip first two timesteps
            
            x_tensor = torch.tensor(x_down, dtype=torch.float)
            y_tensor = torch.tensor(y_down, dtype=torch.float)
            
            x_reshaped = rearrange(x_tensor, 'b t h w c -> (b t) c h w')
            y_reshaped = rearrange(y_tensor, 'b t h w c -> (b t) c h w')
            
            for i in range(len(x_reshaped)):
                self.x.append(x_reshaped[i])
                self.y.append(y_reshaped[i])
                self.resolution_info.append(f"{target_res}_downsampled")
            
            print(f"  Added {len(x_reshaped)} downsampled samples at resolution {target_res}")
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def get_resolution_info(self):
        """Get resolution information for each sample"""
        return self.resolution_info


def ns_true_multires_markov_dataset(saved_folder, 
                                   viscosity="1e-3",  # Viscosity parameter for filename
                                   file_extension=".mat",  # File extension: ".mat" or ".h5"
                                   data_mres_size=None,
                                   add_res=None,
                                   add_res_samples=None,
                                   downsample_from_res=None,  # Which resolution to use for downsampling
                                   data_normalizer=True, 
                                   normalization_type="simple",
                                   random_seed=42,
                                   **kwargs):
    """
    Create true multi-resolution Navier-Stokes dataset from multiple .mat or .h5 files.
    
    Expected data format: ns_{resolution}_{viscosity}.mat/.h5 with 'u' key containing
    vorticity data in shape:
    - .mat files: (N_simulations, height, width, time_steps)
    - .h5 files: (N_simulations, time_steps, height, width) or (N_simulations, height, width, time_steps)
    
    Args:
        saved_folder: Folder containing files 
        viscosity: Viscosity parameter for filename (e.g., "1e-3")
        file_extension: File extension (".mat" or ".h5")
        data_mres_size: Dict with resolution -> num_samples mapping 
        add_res: Additional resolutions for downsampling 
        add_res_samples: Dict with resolution -> num_samples mapping for downsampled data
        downsample_from_res: Which resolution to use as base for downsampling
        data_normalizer: Whether to apply normalization
        normalization_type: "simple" or "unit_gaussian"
        random_seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
    """
    
    # Set default data_mres_size if not provided
    if data_mres_size is None:
        data_mres_size = {512: 16, 64: 16, 32: 16}
    
    # Set default add_res_samples if not provided
    if add_res_samples is None:
        add_res_samples = {16: 12, 8: 8}
    
    split_ratio = [0.8, 0.1, 0.1]
    
    print(f"Creating true multi-resolution Navier-Stokes dataset from {saved_folder}")
    print(f"Viscosity: {viscosity}, File extension: {file_extension}")
    print(f"Target samples per resolution: {data_mres_size}")
    if add_res:
        print(f"Additional downsampled resolutions: {add_res}")
        print(f"Target samples per downsampled resolution: {add_res_samples}")
        print(f"Will downsample from resolution: {downsample_from_res if downsample_from_res else 'auto-select highest'}")
    
    # Create datasets for each split
    train_dataset = NSVTrueMultiResMarkovDataset(
        saved_folder=saved_folder,
        viscosity=viscosity,
        file_extension=file_extension,
        data_mres_size=data_mres_size,
        add_res=add_res,
        add_res_samples=add_res_samples,
        downsample_from_res=downsample_from_res,
        split_ratio=split_ratio,
        random_seed=random_seed,
        split='train',
        **kwargs
    )
    
    val_dataset = NSVTrueMultiResMarkovDataset(
        saved_folder=saved_folder,
        viscosity=viscosity,
        file_extension=file_extension,
        data_mres_size=data_mres_size,
        add_res=add_res,
        add_res_samples=add_res_samples,
        downsample_from_res=downsample_from_res,
        split_ratio=split_ratio,
        random_seed=random_seed,
        split='val',
        **kwargs
    )
    
    test_dataset = NSVTrueMultiResMarkovDataset(
        saved_folder=saved_folder,
        viscosity=viscosity,
        file_extension=file_extension,
        data_mres_size=data_mres_size,
        add_res=add_res,
        add_res_samples=add_res_samples,
        downsample_from_res=downsample_from_res,
        split_ratio=split_ratio,
        random_seed=random_seed,
        split='test',
        **kwargs
    )
    
    # Initialize return values
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
    
    return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer


# Example usage for both .mat and .h5 files:
# """
# # For .mat files (original format)
# train_dataset, val_dataset, test_dataset, x_norm, y_norm = ns_true_multires_markov_dataset(
#     saved_folder="ns_data/",
#     viscosity="1e-3",  # Matches your ns_*_1e-3.mat files
#     file_extension=".mat",
#     data_mres_size={512: 16, 64: 16, 32: 16},  # Use 16 out of 20 simulations per resolution
#     add_res=[16, 8],   # Create additional resolutions by downsampling
#     add_res_samples={16: 12, 8: 8},  # Samples for downsampled resolutions
#     downsample_from_res=512,  # Use 512 resolution as base for downsampling
#     normalization_type="unit_gaussian"
# )

# # For .h5 files (new format)
# train_dataset, val_dataset, test_dataset, x_norm, y_norm = ns_true_multires_markov_dataset(
#     saved_folder="ns_data/",
#     viscosity="1e-03",  # Matches your ns_*_1e-03.h5 files
#     file_extension=".h5",
#     data_mres_size={256: 800, 128: 400, 64: 200},  # Adjust based on your .h5 file sizes
#     add_res=[32, 16],   # Create additional resolutions by downsampling
#     add_res_samples={32: 150, 16: 100},  # Samples for downsampled resolutions
#     downsample_from_res=256,  # Use 256 resolution as base for downsampling
#     normalization_type="unit_gaussian"
# )

# # Check sample shapes
# print(f"Sample shapes: x={train_dataset[0][0].shape}, y={train_dataset[0][1].shape}")
# """


# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from models.custom_layer import UnitGaussianNormalizer
# import os
# import glob
# import h5py
# import numpy as np
# import math as mt
# from einops import rearrange
# from scipy.io import loadmat

# class NSVTrueMultiResMarkovDataset(Dataset):
#     def __init__(self, 
#                  saved_folder,  # Folder containing .mat files
#                  viscosity="1e-3",  # Viscosity parameter for filename
#                  reduced_batch=1, 
#                  reduced_resolution=1, 
#                  reduced_resolution_t=1, 
#                  data_mres_size=None,  # Dict: {resolution: num_samples} from actual files
#                  add_res=None,  # Additional resolutions for downsampling
#                  add_res_samples=None,  # Dict: {resolution: num_samples} for downsampled data
#                  downsample_from_res=None,  # Which resolution to use for downsampling (default: highest)
#                  split_ratio=None,  # For splitting: [train%, val%, test%]
#                  random_seed=42,
#                  split='train',
#                  **kwargs):
        
#         self.random_seed = random_seed
#         self.split = split
#         self.viscosity = viscosity
#         self.downsample_from_res = downsample_from_res
        
#         # Set default split ratio if not provided
#         if split_ratio is None:
#             split_ratio = [0.8, 0.1, 0.1]  # 80% train, 10% val, 10% test
        
#         # Set default data_mres_size if not provided
#         if data_mres_size is None:
#             data_mres_size = {512: 0, 64: 0, 32: 0}
        
#         # Set downsample_from_res to highest available resolution if not specified
#         if downsample_from_res is None and data_mres_size:
#             # Only consider resolutions with non-zero samples
#             available_resolutions = [res for res, samples in data_mres_size.items() if samples > 0]
#             if available_resolutions:
#                 downsample_from_res = max(available_resolutions)
#                 print(f"Auto-selected downsample_from_res: {downsample_from_res} (highest available)")
#             else:
#                 downsample_from_res = max(data_mres_size.keys()) if data_mres_size else None
#                 print(f"Auto-selected downsample_from_res: {downsample_from_res} (highest in config)")
        
#         self.downsample_from_res = downsample_from_res
        
#         # Set default add_res_samples if not provided
#         if add_res_samples is None:
#             add_res_samples = {64: 0, 32: 0}  # Default for downsampled resolutions
        
#         print(f"Loading Navier-Stokes multi-resolution data from {saved_folder}")
#         print(f"Viscosity: {viscosity}")
#         print(f"Target samples per resolution (from files): {data_mres_size}")
#         if add_res:
#             print(f"Additional downsampled resolutions: {add_res}")
#             print(f"Target samples per downsampled resolution: {add_res_samples}")
#             print(f"Will downsample from resolution: {downsample_from_res}")
        
#         # Store data from all resolutions
#         self.x = []
#         self.y = []
#         self.resolution_info = []  # Track which resolution each sample came from
        
#         # Load data from each resolution (from actual files)
#         for resolution, target_samples in data_mres_size.items():
#             if target_samples == 0:
#                 continue
                
#             # Construct path to .mat file
#             filename = f"ns_{resolution}_{viscosity}.mat"
#             res_file_path = os.path.join(saved_folder, filename)
            
#             if not os.path.exists(res_file_path):
#                 print(f"Warning: File {res_file_path} does not exist. Skipping resolution {resolution}")
#                 continue
            
#             print(f"Loading resolution {resolution} from {res_file_path}")
            
#             # Load data for this resolution using scipy
#             mat_data = loadmat(res_file_path)
            
#             # Print available keys to understand data structure
#             keys = [k for k in mat_data.keys() if not k.startswith('__')]
#             print(f"  Available keys: {keys}")
            
#             # Use 'u' key for vorticity time series data
#             if 'u' not in mat_data:
#                 print(f"  Warning: 'u' key not found in {res_file_path}. Skipping.")
#                 continue
            
#             raw_data = np.array(mat_data['u'], dtype=np.float32)
#             print(f"  Raw data shape for {resolution}: {raw_data.shape}")
            
#             # Handle standard NS data: (N_sims, height, width, time_steps)
#             # Convert to: (N_sims, time_steps, height, width, channels)
#             if len(raw_data.shape) != 4:
#                 print(f"  Warning: Expected 4D array, got {raw_data.shape}. Skipping resolution {resolution}")
#                 continue
            
#             # Transpose from (batch, height, width, time) to (batch, time, height, width)
#             combined_data = np.transpose(raw_data, (0, 3, 1, 2))
#             # Add channel dimension: (batch, time, height, width, 1)
#             combined_data = np.expand_dims(combined_data, axis=-1)
#             print(f"  Converted to shape: {combined_data.shape}")
            
#             # Apply reductions
#             combined_data = combined_data[::reduced_batch, 
#                                         ::reduced_resolution_t, 
#                                         ::reduced_resolution, 
#                                         ::reduced_resolution, :]
            
#             print(f"  After reduction: {combined_data.shape}")
            
#             # Split the data into train/val/test based on split_ratio
#             total_samples = combined_data.shape[0]
            
#             # Calculate split boundaries
#             train_end = int(total_samples * split_ratio[0])
#             val_end = train_end + int(total_samples * split_ratio[1])
            
#             # Determine which samples to use based on split
#             if self.split == 'train':
#                 split_data = combined_data[:train_end]
#             elif self.split in ['val', 'valid']:
#                 split_data = combined_data[train_end:val_end]
#             elif self.split == 'test':
#                 split_data = combined_data[val_end:]
#             else:
#                 raise ValueError(f"Invalid split: {self.split}")
            
#             print(f"  Split '{self.split}' data shape: {split_data.shape}")
            
#             # Further sample if target_samples is specified and less than available
#             if target_samples > 0 and target_samples < split_data.shape[0]:
#                 # Calculate samples for current split
#                 split_idx = {'train': 0, 'valid': 1, 'val': 1, 'test': 2}.get(self.split, 0)
#                 samples_for_split = int(target_samples * split_ratio[split_idx])
                
#                 if samples_for_split > 0:
#                     # Sample deterministically based on split and resolution
#                     np.random.seed(self.random_seed + resolution + split_idx)
#                     sample_indices = np.random.choice(split_data.shape[0], samples_for_split, replace=False)
#                     sampled_data = split_data[sample_indices]
#                     print(f"  Subsampled to {sampled_data.shape[0]} samples for {self.split} split")
#                 else:
#                     print(f"  No samples allocated for {self.split} split at resolution {resolution}")
#                     continue
#             else:
#                 sampled_data = split_data
#                 print(f"  Using all {sampled_data.shape[0]} samples for {self.split} split")
            
#             # Create input-output pairs for Markov property (same as original NS dataset)
#             x_res = sampled_data[:, 1:-1]  # Skip first and last timestep
#             y_res = sampled_data[:, 2:]    # Skip first two timesteps
            
#             # Convert to tensors and rearrange to (batch*time, channels, height, width)
#             x_tensor = torch.tensor(x_res, dtype=torch.float)
#             y_tensor = torch.tensor(y_res, dtype=torch.float)
            
#             x_reshaped = rearrange(x_tensor, 'b t h w c -> (b t) c h w')
#             y_reshaped = rearrange(y_tensor, 'b t h w c -> (b t) c h w')
            
#             # Add to main lists
#             for i in range(len(x_reshaped)):
#                 self.x.append(x_reshaped[i])
#                 self.y.append(y_reshaped[i])
#                 self.resolution_info.append(f"{resolution}_file")
            
#             print(f"  Added {len(x_reshaped)} sample pairs from resolution {resolution}")
        
#         # Handle add_res parameter for downsampling
#         if add_res is not None and add_res_samples is not None:
#             print(f"Adding downsampled resolutions: {add_res}")
#             # Use the specified resolution for downsampling
#             if self.downsample_from_res is not None:
#                 self._add_downsampled_data(saved_folder, self.downsample_from_res, add_res, add_res_samples, 
#                                          split_ratio, reduced_batch, reduced_resolution, reduced_resolution_t)
#             else:
#                 print("Warning: No resolution specified for downsampling and no available resolutions found.")
        
#         assert len(self.x) == len(self.y), "Invalid input output pairs"
#         print(f"Total samples loaded: {len(self.x)}")
        
#         # Print resolution distribution
#         if self.resolution_info:
#             unique_res, counts = np.unique(self.resolution_info, return_counts=True)
#             print("Resolution distribution:")
#             for res, count in zip(unique_res, counts):
#                 print(f"  {res}: {count} samples")
        
#         if len(self.x) > 0:
#             print(f"Sample x shapes: {[tuple(x.shape) for x in self.x[:min(3, len(self.x))]]}")
#             print(f"Sample y shapes: {[tuple(y.shape) for y in self.y[:min(3, len(self.y))]]}")
    
#     def _add_downsampled_data(self, saved_folder, base_resolution, add_res, add_res_samples, 
#                             split_ratio, reduced_batch, reduced_resolution, reduced_resolution_t):
#         """Add downsampled data from base resolution"""
#         # Construct path to base resolution file
#         filename = f"ns_{base_resolution}_{self.viscosity}.mat"
#         base_file_path = os.path.join(saved_folder, filename)
        
#         if not os.path.exists(base_file_path):
#             print(f"Warning: Base file {base_file_path} does not exist. Cannot create downsampled data.")
#             return
        
#         print(f"Adding downsampled data from {base_file_path}")
        
#         # Load original data using scipy
#         mat_data = loadmat(base_file_path)
        
#         if 'u' not in mat_data:
#             print(f"Warning: 'u' key not found in {base_file_path}. Cannot create downsampled data.")
#             return
        
#         raw_data = np.array(mat_data['u'], dtype=np.float32)
        
#         # Handle standard NS data: (N_sims, height, width, time_steps)
#         # Convert to: (N_sims, time_steps, height, width, channels)
#         if len(raw_data.shape) != 4:
#             print(f"Warning: Expected 4D array, got {raw_data.shape}. Cannot create downsampled data.")
#             return
        
#         # Transpose from (batch, height, width, time) to (batch, time, height, width)
#         original_data = np.transpose(raw_data, (0, 3, 1, 2))
#         # Add channel dimension: (batch, time, height, width, 1)
#         original_data = np.expand_dims(original_data, axis=-1)
        
#         # Split the original data first
#         total_samples = original_data.shape[0]
#         train_end = int(total_samples * split_ratio[0])
#         val_end = train_end + int(total_samples * split_ratio[1])
        
#         # Get data for current split
#         if self.split == 'train':
#             split_original_data = original_data[:train_end]
#         elif self.split in ['val', 'valid']:
#             split_original_data = original_data[train_end:val_end]
#         elif self.split == 'test':
#             split_original_data = original_data[val_end:]
#         else:
#             raise ValueError(f"Invalid split: {self.split}")
        
#         original_spatial_size = split_original_data.shape[2]  # Height dimension
        
#         split_idx = {'train': 0, 'valid': 1, 'val': 1, 'test': 2}.get(self.split, 0)
        
#         for target_res in add_res:
#             if target_res >= original_spatial_size:
#                 print(f"  Warning: Target resolution {target_res} >= original {original_spatial_size}. Skipping.")
#                 continue
            
#             # Get number of samples for this resolution
#             target_samples = add_res_samples.get(target_res, 100)  # Default 100 if not specified
#             samples_for_split = int(target_samples * split_ratio[split_idx])
            
#             if samples_for_split == 0:
#                 print(f"  No downsampled samples allocated for {self.split} split at resolution {target_res}")
#                 continue
            
#             # Calculate downsampling factor (assuming square grids)
#             downsample_factor = original_spatial_size // target_res
            
#             # Sample and downsample
#             np.random.seed(self.random_seed + target_res + split_idx + 10000)  # Different seed space
#             sample_indices = np.random.choice(split_original_data.shape[0], samples_for_split, replace=True)
#             sampled_data = split_original_data[sample_indices]
            
#             # Apply downsampling (spatial only, keep time and channels)
#             downsampled_data = sampled_data[::reduced_batch, 
#                                           ::reduced_resolution_t, 
#                                           ::downsample_factor, 
#                                           ::downsample_factor, :]
            
#             # Process and add to dataset (same Markov logic as main data)
#             x_down = downsampled_data[:, 1:-1]  # Skip first and last timestep
#             y_down = downsampled_data[:, 2:]    # Skip first two timesteps
            
#             x_tensor = torch.tensor(x_down, dtype=torch.float)
#             y_tensor = torch.tensor(y_down, dtype=torch.float)
            
#             x_reshaped = rearrange(x_tensor, 'b t h w c -> (b t) c h w')
#             y_reshaped = rearrange(y_tensor, 'b t h w c -> (b t) c h w')
            
#             for i in range(len(x_reshaped)):
#                 self.x.append(x_reshaped[i])
#                 self.y.append(y_reshaped[i])
#                 self.resolution_info.append(f"{target_res}_downsampled")
            
#             print(f"  Added {len(x_reshaped)} downsampled samples at resolution {target_res}")
    
#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]
    
#     def get_resolution_info(self):
#         """Get resolution information for each sample"""
#         return self.resolution_info


# def ns_true_multires_markov_dataset(saved_folder, 
#                                    viscosity="1e-3",  # Viscosity parameter for filename
#                                    data_mres_size=None,
#                                    add_res=None,
#                                    add_res_samples=None,
#                                    downsample_from_res=None,  # Which resolution to use for downsampling
#                                    data_normalizer=True, 
#                                    normalization_type="simple",
#                                    random_seed=42,
#                                    **kwargs):
#     """
#     Create true multi-resolution Navier-Stokes dataset from multiple .mat files.
    
#     Expected data format: ns_{resolution}_{viscosity}.mat with 'u' key containing
#     vorticity data in shape (N_simulations, height, width, time_steps).
    
#     Args:
#         saved_folder: Folder containing .mat files 
#         viscosity: Viscosity parameter for filename (e.g., "1e-3")
#         data_mres_size: Dict with resolution -> num_samples mapping 
#         add_res: Additional resolutions for downsampling 
#         add_res_samples: Dict with resolution -> num_samples mapping for downsampled data
#         downsample_from_res: Which resolution to use as base for downsampling
#         data_normalizer: Whether to apply normalization
#         normalization_type: "simple" or "unit_gaussian"
#         random_seed: Random seed for reproducibility
    
#     Returns:
#         train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
#     """
    
#     # Set default data_mres_size if not provided
#     if data_mres_size is None:
#         data_mres_size = {512: 16, 64: 16, 32: 16}
    
#     # Set default add_res_samples if not provided
#     if add_res_samples is None:
#         add_res_samples = {16: 12, 8: 8}
    
#     split_ratio = [0.8, 0.1, 0.1]
    
#     print(f"Creating true multi-resolution Navier-Stokes dataset from {saved_folder}")
#     print(f"Viscosity: {viscosity}")
#     print(f"Target samples per resolution: {data_mres_size}")
#     if add_res:
#         print(f"Additional downsampled resolutions: {add_res}")
#         print(f"Target samples per downsampled resolution: {add_res_samples}")
#         print(f"Will downsample from resolution: {downsample_from_res if downsample_from_res else 'auto-select highest'}")
    
#     # Create datasets for each split
#     train_dataset = NSVTrueMultiResMarkovDataset(
#         saved_folder=saved_folder,
#         viscosity=viscosity,
#         data_mres_size=data_mres_size,
#         add_res=add_res,
#         add_res_samples=add_res_samples,
#         downsample_from_res=downsample_from_res,
#         split_ratio=split_ratio,
#         random_seed=random_seed,
#         split='train',
#         **kwargs
#     )
    
#     val_dataset = NSVTrueMultiResMarkovDataset(
#         saved_folder=saved_folder,
#         viscosity=viscosity,
#         data_mres_size=data_mres_size,
#         add_res=add_res,
#         add_res_samples=add_res_samples,
#         downsample_from_res=downsample_from_res,
#         split_ratio=split_ratio,
#         random_seed=random_seed,
#         split='val',
#         **kwargs
#     )
    
#     test_dataset = NSVTrueMultiResMarkovDataset(
#         saved_folder=saved_folder,
#         viscosity=viscosity,
#         data_mres_size=data_mres_size,
#         add_res=add_res,
#         add_res_samples=add_res_samples,
#         downsample_from_res=downsample_from_res,
#         split_ratio=split_ratio,
#         random_seed=random_seed,
#         split='test',
#         **kwargs
#     )
    
#     # Initialize return values
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
#                     if isinstance(x, np.ndarray):
#                         x = torch.from_numpy(x).float()
#                     if isinstance(y, np.ndarray):
#                         y = torch.from_numpy(y).float()
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


# # Example usage for your data:
# # """
# # # Use your actual ns_data folder with .mat files
# # train_dataset, val_dataset, test_dataset, x_norm, y_norm = ns_true_multires_markov_dataset(
# #     saved_folder="ns_data/",
# #     viscosity="1e-3",  # Matches your ns_*_1e-3.mat files
# #     data_mres_size={512: 16, 64: 16, 32: 16},  # Use 16 out of 20 simulations per resolution
# #     add_res=[16, 8],   # Create additional resolutions by downsampling
# #     add_res_samples={16: 12, 8: 8},  # Samples for downsampled resolutions
# #     downsample_from_res=512,  # Use 512 resolution as base for downsampling
# #     normalization_type="unit_gaussian"
# # )

# # # Check sample shapes
# # print(f"Sample shapes: x={train_dataset[0][0].shape}, y={train_dataset[0][1].shape}")
# # """