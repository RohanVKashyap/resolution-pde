import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import h5py
import numpy as np
from einops import rearrange
import glob

class KSTrajectoryDatasetFromExtracted(Dataset):
    """
    Simple dataset wrapper for extracted KS trajectories for rollout evaluation.
    """
    def __init__(self, trajectories, trajectory_info):
        self.trajectories = trajectories
        self.trajectory_info = trajectory_info
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx]  # Returns shape: (time_steps, spatial_points)
    
    def get_trajectory_info(self, idx):
        return self.trajectory_info[idx]
    
    def get_all_info(self):
        return self.trajectory_info

def extract_ks_test_trajectories_for_rollout(saved_folder,
                                           viscosity=0.05,
                                           L=64.0,
                                           lmax=8,
                                           et=5.0,
                                           nte=51,
                                           nt=51,
                                           train_s=2048,
                                           data_mres_size=None,
                                           split_ratio=None,
                                           reduced_batch=1,
                                           reduced_resolution_t=1,
                                           random_seed=42):
    """
    Extract complete test trajectories from KS dataset BEFORE Markov pair creation.
    """
    if split_ratio is None:
        split_ratio = [0.8, 0.1, 0.1]
    
    if data_mres_size is None:
        data_mres_size = {800: 0, 512: 0, 400: 0}
    
    print(f"Extracting KS test trajectories for rollout evaluation")
    print(f"Parameters: visc_{viscosity}_L{L}_lmax{lmax}_et{et}_nte{nte}_nt{nt}")
    
    all_trajectories = []
    trajectory_info = []
    
    # Extract trajectories from each resolution
    for resolution, target_samples in data_mres_size.items():
        if target_samples == 0:
            continue
            
        # Construct path to resolution-specific directory (same as original dataset)
        dir_name = f"visc_{viscosity}_L{L}_lmax{lmax}_et{et}_nte{nte}_nt{nt}"
        res_folder = os.path.join(saved_folder, f"res_{resolution}", dir_name)
        filename = f"KS_train_{train_s}.h5"
        res_file_path = os.path.join(res_folder, filename)
        
        if not os.path.exists(res_file_path):
            print(f"Warning: File {res_file_path} does not exist. Skipping resolution {resolution}")
            continue
        
        print(f"Loading trajectories from resolution {resolution}: {res_file_path}")
        
        # Load data (same logic as original dataset)
        with h5py.File(res_file_path, 'r') as f:
            # Find the data group
            group = None
            if 'train' in f:
                group = f['train']
                group_name = 'train'
            else:
                keys = list(f.keys())
                if len(keys) == 1:
                    group = f[keys[0]]
                    group_name = keys[0]
                else:
                    for key in keys:
                        if key.lower() in ['data', 'pde', 'train'] or 'pde' in key.lower():
                            group = f[key]
                            group_name = key
                            break
                    if group is None:
                        raise ValueError(f"Could not find data group in file. Available keys: {keys}")
            
            data_keys = list(group.keys())
            
            # Find PDE data key
            pde_key = None
            for key in data_keys:
                if 'pde' in key.lower() and '-' in key:
                    pde_key = key
                    break
            
            if pde_key is None:
                raise ValueError(f"Could not find PDE data key in {data_keys}")
            
            # Load the full data
            _data = np.array(group[pde_key], dtype=np.float32)
            print(f"  Original data shape: {_data.shape}")  # (num_trajectories, time_steps, spatial_points)
            
            # Apply time reduction
            _data = _data[::reduced_batch, ::reduced_resolution_t, :]
            print(f"  After time reduction: {_data.shape}")
            
            # Split the data (same logic as original dataset)
            total_samples = _data.shape[0]
            train_end = int(total_samples * split_ratio[0])
            val_end = train_end + int(total_samples * split_ratio[1])
            
            # Extract test split
            test_data = _data[val_end:]
            print(f"  Test split data shape: {test_data.shape}")
            
            # Further subsample if target_samples is specified
            if target_samples > 0 and target_samples < test_data.shape[0]:
                split_idx = 2  # test split index
                samples_for_test = int(target_samples * split_ratio[split_idx])
                
                if samples_for_test > 0:
                    np.random.seed(random_seed + resolution + split_idx)
                    sample_indices = np.random.choice(test_data.shape[0], samples_for_test, replace=False)
                    sampled_test_data = test_data[sample_indices]
                    print(f"  Subsampled to {sampled_test_data.shape[0]} test trajectories")
                else:
                    print(f"  No test samples allocated for resolution {resolution}")
                    continue
            else:
                sampled_test_data = test_data
                print(f"  Using all {sampled_test_data.shape[0]} test trajectories")
            
            # Store each trajectory individually
            for i in range(sampled_test_data.shape[0]):
                trajectory = torch.tensor(sampled_test_data[i], dtype=torch.float)  # Shape: (time_steps, spatial_points)
                all_trajectories.append(trajectory)
                trajectory_info.append({
                    'resolution': resolution,
                    'original_index': i,
                    'source': f'res_{resolution}_file'
                })
            
            print(f"  Extracted {sampled_test_data.shape[0]} complete trajectories from resolution {resolution}")
    
    print(f"\nTotal extracted trajectories: {len(all_trajectories)}")
    if len(all_trajectories) > 0:
        print(f"Trajectory shapes: {[traj.shape for traj in all_trajectories[:3]]}")
        
        # Print resolution distribution
        resolution_counts = {}
        for info in trajectory_info:
            res = info['resolution']
            resolution_counts[res] = resolution_counts.get(res, 0) + 1
        
        print("Resolution distribution:")
        for res, count in resolution_counts.items():
            print(f"  Resolution {res}: {count} trajectories")
    
    return all_trajectories, trajectory_info

class KSTrueMultiResMarkovDataset(Dataset):
    def __init__(self, 
                 saved_folder,  # Base folder containing resolution directories
                 viscosity=0.05,  # Viscosity parameter (for directory naming)
                 L=64.0,  # Domain length parameter
                 lmax=8,  # lmax parameter
                 et=5.0,  # et parameter
                 nte=51,  # nte parameter
                 nt=51,   # nt parameter
                 train_s=2048,  # NEW: Training data suffix (e.g., 2048)
                 reduced_batch=1, 
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
        self.L = L
        self.lmax = lmax
        self.et = et
        self.nte = nte
        self.nt = nt
        self.train_s = train_s
        self.downsample_from_res = downsample_from_res
        
        # Set default split ratio if not provided
        if split_ratio is None:
            split_ratio = [0.8, 0.1, 0.1]  # 80% train, 10% val, 10% test
        
        # Set default data_mres_size if not provided
        if data_mres_size is None:
            data_mres_size = {800: 0, 512: 0, 400: 0}
        
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
            add_res_samples = {200: 0, 128: 0}  # Default for downsampled resolutions
        
        print(f"Loading KS multi-resolution data from {saved_folder}")
        print(f"Parameters: visc_{viscosity}_L{L}_lmax{lmax}_et{et}_nte{nte}_nt{nt}")
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
                
            # Construct path to resolution-specific directory
            dir_name = f"visc_{viscosity}_L{L}_lmax{lmax}_et{et}_nte{nte}_nt{nt}"
            res_folder = os.path.join(saved_folder, f"res_{resolution}", dir_name)
            
            # Use the train_s parameter for filename
            filename = f"KS_train_{train_s}.h5"
            res_file_path = os.path.join(res_folder, filename)
            
            if not os.path.exists(res_file_path):
                print(f"Warning: File {res_file_path} does not exist. Skipping resolution {resolution}")
                continue
            
            print(f"Loading resolution {resolution} from {res_file_path}")
            
            # Load data for this resolution
            with h5py.File(res_file_path, 'r') as f:
                # Since we only have train files, look for the main data group
                # Try common group names
                group = None
                if 'train' in f:
                    group = f['train']
                    group_name = 'train'
                else:
                    keys = list(f.keys())
                    print(f"Available keys: {keys}")
                    if len(keys) == 1:
                        group = f[keys[0]]
                        group_name = keys[0]
                        print(f"Using group: {keys[0]}")
                    else:
                        # Try to find the most likely data group
                        for key in keys:
                            if key.lower() in ['data', 'pde', 'train'] or 'pde' in key.lower():
                                group = f[key]
                                group_name = key
                                break
                        if group is None:
                            raise ValueError(f"Could not find data group in file. Available keys: {keys}")
                
                data_keys = list(group.keys())
                print(f"Available data keys in '{group_name}': {data_keys}")
                
                # Find PDE data key
                pde_key = None
                for key in data_keys:
                    if 'pde' in key.lower() and '-' in key:
                        pde_key = key
                        break
                
                if pde_key is None:
                    raise ValueError(f"Could not find PDE data key in {data_keys}")
                
                _data = np.array(group[pde_key], dtype=np.float32)
                print(f"  Original data shape for {resolution}: {_data.shape}")
                
                # Apply time reduction
                _data = _data[::reduced_batch, ::reduced_resolution_t, :]
                print(f"  After time reduction: {_data.shape}")
                
                # Split the data into train/val/test based on split_ratio
                total_samples = _data.shape[0]
                
                # Calculate split boundaries
                train_end = int(total_samples * split_ratio[0])
                val_end = train_end + int(total_samples * split_ratio[1])
                
                # Determine which samples to use based on split
                if self.split == 'train':
                    split_data = _data[:train_end]
                elif self.split in ['val', 'valid']:
                    split_data = _data[train_end:val_end]
                elif self.split == 'test':
                    split_data = _data[val_end:]
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
                
                # Create input-output pairs for Markov property
                x_res = sampled_data[:, :-1, :]  # Skip last timestep
                y_res = sampled_data[:, 1:, :]   # Skip first timestep
                
                # Convert to tensors and rearrange
                x_tensor = torch.tensor(x_res, dtype=torch.float)
                y_tensor = torch.tensor(y_res, dtype=torch.float)
                
                x_reshaped = rearrange(x_tensor, 'b t m -> (b t) 1 m')
                y_reshaped = rearrange(y_tensor, 'b t m -> (b t) 1 m')
                
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
                                         split_ratio, reduced_batch, reduced_resolution_t)
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
    
    def _add_downsampled_data(self, saved_folder, base_resolution, add_res, add_res_samples, 
                            split_ratio, reduced_batch, reduced_resolution_t):
        """Add downsampled data from base resolution"""
        # Construct path to base resolution file
        dir_name = f"visc_{self.viscosity}_L{self.L}_lmax{self.lmax}_et{self.et}_nte{self.nte}_nt{self.nt}"
        base_res_folder = os.path.join(saved_folder, f"res_{base_resolution}", dir_name)
        
        # Use the train_s parameter for filename
        filename = f"KS_train_{self.train_s}.h5"
        base_file_path = os.path.join(base_res_folder, filename)
        
        if not os.path.exists(base_file_path):
            print(f"Warning: Base file {base_file_path} does not exist. Cannot create downsampled data.")
            return
        
        print(f"Adding downsampled data from {base_file_path}")
        
        with h5py.File(base_file_path, 'r') as f:
            # Find the appropriate group and data key (same logic as above)
            group = None
            if 'train' in f:
                group = f['train']
                group_name = 'train'
            else:
                keys = list(f.keys())
                if len(keys) == 1:
                    group = f[keys[0]]
                    group_name = keys[0]
                else:
                    # Try to find the most likely data group
                    for key in keys:
                        if key.lower() in ['data', 'pde', 'train'] or 'pde' in key.lower():
                            group = f[key]
                            group_name = key
                            break
                    if group is None:
                        raise ValueError(f"Could not find data group in file. Available keys: {keys}")
            
            data_keys = list(group.keys())
            
            # Find PDE data key
            pde_key = None
            for key in data_keys:
                if 'pde' in key.lower() and '-' in key:
                    pde_key = key
                    break
            
            if pde_key is None:
                raise ValueError(f"Could not find PDE data key in {data_keys}")
            
            original_data = np.array(group[pde_key], dtype=np.float32)
            
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
            
            original_spatial_size = split_original_data.shape[2]
        
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
            
            # Calculate downsampling factor
            downsample_factor = original_spatial_size // target_res
            
            # Sample and downsample
            np.random.seed(self.random_seed + target_res + split_idx + 10000)  # Different seed space
            sample_indices = np.random.choice(split_original_data.shape[0], samples_for_split, replace=True)
            sampled_data = split_original_data[sample_indices]
            
            # Apply downsampling
            downsampled_data = sampled_data[::reduced_batch, ::reduced_resolution_t, ::downsample_factor]
            
            # Process and add to dataset
            x_down = downsampled_data[:, :-1, :]  # Skip last timestep
            y_down = downsampled_data[:, 1:, :]   # Skip first timestep
            
            x_tensor = torch.tensor(x_down, dtype=torch.float)
            y_tensor = torch.tensor(y_down, dtype=torch.float)
            
            x_reshaped = rearrange(x_tensor, 'b t m -> (b t) 1 m')
            y_reshaped = rearrange(y_tensor, 'b t m -> (b t) 1 m')
            
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


def ks_true_multires_markov_dataset(saved_folder, 
                                   viscosity=0.05,
                                   L=64.0,
                                   lmax=8,
                                   et=5.0,
                                   nte=51,
                                   nt=51,
                                   train_s=2048,  # NEW: Training data suffix
                                   data_mres_size=None,
                                   add_res=None,
                                   add_res_samples=None,
                                   downsample_from_res=None,
                                   data_normalizer=True, 
                                   normalization_type="simple",
                                   random_seed=42,
                                   **kwargs):
    """
    Create true multi-resolution KS dataset from multiple HDF5 files.
    NOW INCLUDES ROLLOUT TEST DATASET FOR EVALUATION.
    
    Args:
        saved_folder: Base folder containing resolution directories (e.g., "/data/user_data/rvk/ks/")
        viscosity: Viscosity parameter for directory naming (default: 0.05)
        L: Domain length parameter (default: 64.0)
        lmax: lmax parameter (default: 8)
        et: et parameter (default: 5.0)
        nte: nte parameter (default: 51)
        nt: nt parameter (default: 51)
        train_s: Training data suffix for filename (e.g., 2048 -> KS_train_2048.h5)
        data_mres_size: Dict with resolution -> num_samples mapping for actual files 
                       (e.g., {800: 200, 512: 100, 400: 1000})
        add_res: Additional resolutions for downsampling (e.g., [200, 128])
        add_res_samples: Dict with resolution -> num_samples mapping for downsampled data
                        (e.g., {200: 150, 128: 100})
        downsample_from_res: Which resolution to use as base for downsampling (e.g., 800).
                           If None, will use the highest resolution with non-zero samples.
        data_normalizer: Whether to apply normalization
        normalization_type: "simple" or "minmax"
        random_seed: Random seed for reproducibility
        **kwargs: Additional arguments
        
    Returns:
        For "simple" normalization:
            train_dataset, val_dataset, test_dataset, rollout_test_dataset, x_normalizer, y_normalizer
        For "minmax" normalization:
            train_dataset, val_dataset, test_dataset, rollout_test_dataset, min_data, max_data, min_model, max_model
    """
    
    # Set default data_mres_size if not provided
    if data_mres_size is None:
        data_mres_size = {800: 200, 512: 100, 400: 1000}
    
    # Set default add_res_samples if not provided
    if add_res_samples is None:
        add_res_samples = {200: 150, 128: 100}
    
    split_ratio = [0.8, 0.1, 0.1]
    
    print(f"Creating true multi-resolution KS dataset from {saved_folder}")
    print(f"Parameters: visc_{viscosity}_L{L}_lmax{lmax}_et{et}_nte{nte}_nt{nt}")
    print(f"Using filename pattern: KS_train_{train_s}.h5")
    print(f"Target samples per resolution (from files): {data_mres_size}")
    if add_res:
        print(f"Additional downsampled resolutions: {add_res}")
        print(f"Target samples per downsampled resolution: {add_res_samples}")
        print(f"Will downsample from resolution: {downsample_from_res if downsample_from_res else 'auto-select highest'}")
    
    # Create Markov datasets for each split
    train_dataset = KSTrueMultiResMarkovDataset(
        saved_folder=saved_folder,
        viscosity=viscosity,
        L=L,
        lmax=lmax,
        et=et,
        nte=nte,
        nt=nt,
        train_s=train_s,
        data_mres_size=data_mres_size,
        add_res=add_res,
        add_res_samples=add_res_samples,
        downsample_from_res=downsample_from_res,
        split_ratio=split_ratio,
        random_seed=random_seed,
        split='train',
        **kwargs
    )
    
    val_dataset = KSTrueMultiResMarkovDataset(
        saved_folder=saved_folder,
        viscosity=viscosity,
        L=L,
        lmax=lmax,
        et=et,
        nte=nte,
        nt=nt,
        train_s=train_s,
        data_mres_size=data_mres_size,
        add_res=add_res,
        add_res_samples=add_res_samples,
        downsample_from_res=downsample_from_res,
        split_ratio=split_ratio,
        random_seed=random_seed,
        split='val',
        **kwargs
    )
    
    test_dataset = KSTrueMultiResMarkovDataset(
        saved_folder=saved_folder,
        viscosity=viscosity,
        L=L,
        lmax=lmax,
        et=et,
        nte=nte,
        nt=nt,
        train_s=train_s,
        data_mres_size=data_mres_size,
        add_res=add_res,
        add_res_samples=add_res_samples,
        downsample_from_res=downsample_from_res,
        split_ratio=split_ratio,
        random_seed=random_seed,
        split='test',
        **kwargs
    )
    
    # NEW: Create rollout test dataset (full trajectories)
    print("\n" + "="*50)
    print("CREATING ROLLOUT TEST DATASET")
    print("="*50)
    
    rollout_trajectories, rollout_trajectory_info = extract_ks_test_trajectories_for_rollout(
        saved_folder=saved_folder,
        viscosity=viscosity,
        L=L,
        lmax=lmax,
        et=et,
        nte=nte,
        nt=nt,
        train_s=train_s,
        data_mres_size=data_mres_size,
        split_ratio=split_ratio,
        reduced_batch=1,  # Keep same as original
        reduced_resolution_t=1,  # Keep same as original
        random_seed=random_seed
    )
    
    rollout_test_dataset = KSTrajectoryDatasetFromExtracted(rollout_trajectories, rollout_trajectory_info)
    
    print(f"Rollout test dataset created with {len(rollout_test_dataset)} trajectories")
    
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
            
        elif normalization_type == "minmax":
            print('---------Computing min-max normalization statistics---------------')
            temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
            
            x_train_all = []
            y_train_all = []
            for batch in temp_loader:
                x_batch, y_batch = batch
                x_train_all.append(x_batch)
                y_train_all.append(y_batch)
            
            x_train_tensor = torch.cat(x_train_all, dim=0)
            y_train_tensor = torch.cat(y_train_all, dim=0)
            
            min_data = float(x_train_tensor.min())
            max_data = float(x_train_tensor.max())
            min_model = float(y_train_tensor.min())
            max_model = float(y_train_tensor.max())
            
            print(f"Input data range: [{min_data:.6f}, {max_data:.6f}]")
            print(f"Output data range: [{min_model:.6f}, {max_model:.6f}]")
            
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
                    x, y = self.dataset[idx]
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x).float()
                    if isinstance(y, np.ndarray):
                        y = torch.from_numpy(y).float()
                    
                    x_normalized = (x - self.min_data) / (self.max_data - self.min_data)
                    y_normalized = (y - self.min_model) / (self.max_model - self.min_model)
                    
                    return x_normalized, y_normalized
            
            train_dataset = MinMaxNormalizedDataset(train_dataset, min_data, max_data, min_model, max_model)
            val_dataset = MinMaxNormalizedDataset(val_dataset, min_data, max_data, min_model, max_model)
            test_dataset = MinMaxNormalizedDataset(test_dataset, min_data, max_data, min_model, max_model)
            
        else:
            raise ValueError(f"Invalid normalization_type: {normalization_type}. Must be 'simple' or 'minmax'")
    
    print(f"\nFinal dataset sizes:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Rollout test dataset size: {len(rollout_test_dataset)}")
    
    # Return appropriate values based on normalization type
    if normalization_type == "simple":
        return train_dataset, val_dataset, test_dataset, rollout_test_dataset, x_normalizer, y_normalizer
    else:  # minmax
        return train_dataset, val_dataset, test_dataset, rollout_test_dataset, min_data, max_data, min_model, max_model

# Example usage:
# """
# # Basic usage with rollout test dataset
# train_dataset, val_dataset, test_dataset, rollout_test_dataset, x_norm, y_norm = ks_true_multires_markov_dataset(
#     saved_folder="/data/user_data/rvk/ks/",
#     viscosity=0.05,
#     L=64.0,
#     lmax=8,
#     et=5.0,
#     nte=51,
#     nt=51,
#     train_s=2048,  # This will look for KS_train_2048.h5
#     data_mres_size={800: 0, 512: 2048, 400: 0},
#     add_res=[32, 64],
#     add_res_samples={64: 0, 32: 0},
#     downsample_from_res=400,
#     normalization_type="simple",
#     data_normalizer=True
# )

# # Now you have:
# # - train_dataset, val_dataset, test_dataset: For teacher-forcing training/evaluation
# # - rollout_test_dataset: For rollout evaluation (contains full trajectories)
# # - x_norm, y_norm: Normalizers for both types of evaluation
# """

# class KSTrueMultiResMarkovDataset(Dataset):
#     def __init__(self, 
#                  saved_folder,  # Base folder containing resolution directories
#                  viscosity=0.05,  # Viscosity parameter (for directory naming)
#                  L=64.0,  # Domain length parameter
#                  lmax=8,  # lmax parameter
#                  et=5.0,  # et parameter
#                  nte=51,  # nte parameter
#                  nt=51,   # nt parameter
#                  train_s=2048,  # NEW: Training data suffix (e.g., 2048)
#                  reduced_batch=1, 
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
#         self.L = L
#         self.lmax = lmax
#         self.et = et
#         self.nte = nte
#         self.nt = nt
#         self.train_s = train_s
#         self.downsample_from_res = downsample_from_res
        
#         # Set default split ratio if not provided
#         if split_ratio is None:
#             split_ratio = [0.8, 0.1, 0.1]  # 80% train, 10% val, 10% test
        
#         # Set default data_mres_size if not provided
#         if data_mres_size is None:
#             data_mres_size = {800: 0, 512: 0, 400: 0}
        
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
#             add_res_samples = {200: 0, 128: 0}  # Default for downsampled resolutions
        
#         print(f"Loading KS multi-resolution data from {saved_folder}")
#         print(f"Parameters: visc_{viscosity}_L{L}_lmax{lmax}_et{et}_nte{nte}_nt{nt}")
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
                
#             # Construct path to resolution-specific directory
#             dir_name = f"visc_{viscosity}_L{L}_lmax{lmax}_et{et}_nte{nte}_nt{nt}"
#             res_folder = os.path.join(saved_folder, f"res_{resolution}", dir_name)
            
#             # Use the train_s parameter for filename
#             filename = f"KS_train_{train_s}.h5"
#             res_file_path = os.path.join(res_folder, filename)
            
#             if not os.path.exists(res_file_path):
#                 print(f"Warning: File {res_file_path} does not exist. Skipping resolution {resolution}")
#                 continue
            
#             print(f"Loading resolution {resolution} from {res_file_path}")
            
#             # Load data for this resolution
#             with h5py.File(res_file_path, 'r') as f:
#                 # Since we only have train files, look for the main data group
#                 # Try common group names
#                 group = None
#                 if 'train' in f:
#                     group = f['train']
#                     group_name = 'train'
#                 else:
#                     keys = list(f.keys())
#                     print(f"Available keys: {keys}")
#                     if len(keys) == 1:
#                         group = f[keys[0]]
#                         group_name = keys[0]
#                         print(f"Using group: {keys[0]}")
#                     else:
#                         # Try to find the most likely data group
#                         for key in keys:
#                             if key.lower() in ['data', 'pde', 'train'] or 'pde' in key.lower():
#                                 group = f[key]
#                                 group_name = key
#                                 break
#                         if group is None:
#                             raise ValueError(f"Could not find data group in file. Available keys: {keys}")
                
#                 data_keys = list(group.keys())
#                 print(f"Available data keys in '{group_name}': {data_keys}")
                
#                 # Find PDE data key
#                 pde_key = None
#                 for key in data_keys:
#                     if 'pde' in key.lower() and '-' in key:
#                         pde_key = key
#                         break
                
#                 if pde_key is None:
#                     raise ValueError(f"Could not find PDE data key in {data_keys}")
                
#                 _data = np.array(group[pde_key], dtype=np.float32)
#                 print(f"  Original data shape for {resolution}: {_data.shape}")
                
#                 # Apply time reduction
#                 _data = _data[::reduced_batch, ::reduced_resolution_t, :]
#                 print(f"  After time reduction: {_data.shape}")
                
#                 # Split the data into train/val/test based on split_ratio
#                 total_samples = _data.shape[0]
                
#                 # Calculate split boundaries
#                 train_end = int(total_samples * split_ratio[0])
#                 val_end = train_end + int(total_samples * split_ratio[1])
                
#                 # Determine which samples to use based on split
#                 if self.split == 'train':
#                     split_data = _data[:train_end]
#                 elif self.split in ['val', 'valid']:
#                     split_data = _data[train_end:val_end]
#                 elif self.split == 'test':
#                     split_data = _data[val_end:]
#                 else:
#                     raise ValueError(f"Invalid split: {self.split}")
                
#                 print(f"  Split '{self.split}' data shape: {split_data.shape}")
                
#                 # Further sample if target_samples is specified and less than available
#                 if target_samples > 0 and target_samples < split_data.shape[0]:
#                     # Calculate samples for current split
#                     split_idx = {'train': 0, 'valid': 1, 'val': 1, 'test': 2}.get(self.split, 0)
#                     samples_for_split = int(target_samples * split_ratio[split_idx])
                    
#                     if samples_for_split > 0:
#                         # Sample deterministically based on split and resolution
#                         np.random.seed(self.random_seed + resolution + split_idx)
#                         sample_indices = np.random.choice(split_data.shape[0], samples_for_split, replace=False)
#                         sampled_data = split_data[sample_indices]
#                         print(f"  Subsampled to {sampled_data.shape[0]} samples for {self.split} split")
#                     else:
#                         print(f"  No samples allocated for {self.split} split at resolution {resolution}")
#                         continue
#                 else:
#                     sampled_data = split_data
#                     print(f"  Using all {sampled_data.shape[0]} samples for {self.split} split")
                
#                 # Create input-output pairs for Markov property
#                 x_res = sampled_data[:, :-1, :]  # Skip last timestep
#                 y_res = sampled_data[:, 1:, :]   # Skip first timestep
                
#                 # Convert to tensors and rearrange
#                 x_tensor = torch.tensor(x_res, dtype=torch.float)
#                 y_tensor = torch.tensor(y_res, dtype=torch.float)
                
#                 x_reshaped = rearrange(x_tensor, 'b t m -> (b t) 1 m')
#                 y_reshaped = rearrange(y_tensor, 'b t m -> (b t) 1 m')
                
#                 # Add to main lists
#                 for i in range(len(x_reshaped)):
#                     self.x.append(x_reshaped[i])
#                     self.y.append(y_reshaped[i])
#                     self.resolution_info.append(f"{resolution}_file")
                
#                 print(f"  Added {len(x_reshaped)} sample pairs from resolution {resolution}")
        
#         # Handle add_res parameter for downsampling
#         if add_res is not None and add_res_samples is not None:
#             print(f"Adding downsampled resolutions: {add_res}")
#             # Use the specified resolution for downsampling
#             if self.downsample_from_res is not None:
#                 self._add_downsampled_data(saved_folder, self.downsample_from_res, add_res, add_res_samples, 
#                                          split_ratio, reduced_batch, reduced_resolution_t)
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
#                             split_ratio, reduced_batch, reduced_resolution_t):
#         """Add downsampled data from base resolution"""
#         # Construct path to base resolution file
#         dir_name = f"visc_{self.viscosity}_L{self.L}_lmax{self.lmax}_et{self.et}_nte{self.nte}_nt{self.nt}"
#         base_res_folder = os.path.join(saved_folder, f"res_{base_resolution}", dir_name)
        
#         # Use the train_s parameter for filename
#         filename = f"KS_train_{self.train_s}.h5"
#         base_file_path = os.path.join(base_res_folder, filename)
        
#         if not os.path.exists(base_file_path):
#             print(f"Warning: Base file {base_file_path} does not exist. Cannot create downsampled data.")
#             return
        
#         print(f"Adding downsampled data from {base_file_path}")
        
#         with h5py.File(base_file_path, 'r') as f:
#             # Find the appropriate group and data key (same logic as above)
#             group = None
#             if 'train' in f:
#                 group = f['train']
#                 group_name = 'train'
#             else:
#                 keys = list(f.keys())
#                 if len(keys) == 1:
#                     group = f[keys[0]]
#                     group_name = keys[0]
#                 else:
#                     # Try to find the most likely data group
#                     for key in keys:
#                         if key.lower() in ['data', 'pde', 'train'] or 'pde' in key.lower():
#                             group = f[key]
#                             group_name = key
#                             break
#                     if group is None:
#                         raise ValueError(f"Could not find data group in file. Available keys: {keys}")
            
#             data_keys = list(group.keys())
            
#             # Find PDE data key
#             pde_key = None
#             for key in data_keys:
#                 if 'pde' in key.lower() and '-' in key:
#                     pde_key = key
#                     break
            
#             if pde_key is None:
#                 raise ValueError(f"Could not find PDE data key in {data_keys}")
            
#             original_data = np.array(group[pde_key], dtype=np.float32)
            
#             # Split the original data first
#             total_samples = original_data.shape[0]
#             train_end = int(total_samples * split_ratio[0])
#             val_end = train_end + int(total_samples * split_ratio[1])
            
#             # Get data for current split
#             if self.split == 'train':
#                 split_original_data = original_data[:train_end]
#             elif self.split in ['val', 'valid']:
#                 split_original_data = original_data[train_end:val_end]
#             elif self.split == 'test':
#                 split_original_data = original_data[val_end:]
#             else:
#                 raise ValueError(f"Invalid split: {self.split}")
            
#             original_spatial_size = split_original_data.shape[2]
        
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
            
#             # Calculate downsampling factor
#             downsample_factor = original_spatial_size // target_res
            
#             # Sample and downsample
#             np.random.seed(self.random_seed + target_res + split_idx + 10000)  # Different seed space
#             sample_indices = np.random.choice(split_original_data.shape[0], samples_for_split, replace=True)
#             sampled_data = split_original_data[sample_indices]
            
#             # Apply downsampling
#             downsampled_data = sampled_data[::reduced_batch, ::reduced_resolution_t, ::downsample_factor]
            
#             # Process and add to dataset
#             x_down = downsampled_data[:, :-1, :]  # Skip last timestep
#             y_down = downsampled_data[:, 1:, :]   # Skip first timestep
            
#             x_tensor = torch.tensor(x_down, dtype=torch.float)
#             y_tensor = torch.tensor(y_down, dtype=torch.float)
            
#             x_reshaped = rearrange(x_tensor, 'b t m -> (b t) 1 m')
#             y_reshaped = rearrange(y_tensor, 'b t m -> (b t) 1 m')
            
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


# def ks_true_multires_markov_dataset(saved_folder, 
#                                    viscosity=0.05,
#                                    L=64.0,
#                                    lmax=8,
#                                    et=5.0,
#                                    nte=51,
#                                    nt=51,
#                                    train_s=2048,  # NEW: Training data suffix
#                                    data_mres_size=None,
#                                    add_res=None,
#                                    add_res_samples=None,
#                                    downsample_from_res=None,
#                                    data_normalizer=True, 
#                                    normalization_type="simple",
#                                    random_seed=42,
#                                    **kwargs):
#     """
#     Create true multi-resolution KS dataset from multiple HDF5 files.
    
#     Args:
#         saved_folder: Base folder containing resolution directories (e.g., "/data/user_data/rvk/ks/")
#         viscosity: Viscosity parameter for directory naming (default: 0.05)
#         L: Domain length parameter (default: 64.0)
#         lmax: lmax parameter (default: 8)
#         et: et parameter (default: 5.0)
#         nte: nte parameter (default: 51)
#         nt: nt parameter (default: 51)
#         train_s: Training data suffix for filename (e.g., 2048 -> KS_train_2048.h5)
#         data_mres_size: Dict with resolution -> num_samples mapping for actual files 
#                        (e.g., {800: 200, 512: 100, 400: 1000})
#         add_res: Additional resolutions for downsampling (e.g., [200, 128])
#         add_res_samples: Dict with resolution -> num_samples mapping for downsampled data
#                         (e.g., {200: 150, 128: 100})
#         downsample_from_res: Which resolution to use as base for downsampling (e.g., 800).
#                            If None, will use the highest resolution with non-zero samples.
#         data_normalizer: Whether to apply normalization
#         normalization_type: "simple" or "minmax"
#         random_seed: Random seed for reproducibility
#         **kwargs: Additional arguments
        
#     Returns:
#         train_dataset, val_dataset, test_dataset, normalizer_info
#     """
    
#     # Set default data_mres_size if not provided
#     if data_mres_size is None:
#         data_mres_size = {800: 200, 512: 100, 400: 1000}
    
#     # Set default add_res_samples if not provided
#     if add_res_samples is None:
#         add_res_samples = {200: 150, 128: 100}
    
#     split_ratio = [0.8, 0.1, 0.1]
    
#     print(f"Creating true multi-resolution KS dataset from {saved_folder}")
#     print(f"Parameters: visc_{viscosity}_L{L}_lmax{lmax}_et{et}_nte{nte}_nt{nt}")
#     print(f"Using filename pattern: KS_train_{train_s}.h5")
#     print(f"Target samples per resolution (from files): {data_mres_size}")
#     if add_res:
#         print(f"Additional downsampled resolutions: {add_res}")
#         print(f"Target samples per downsampled resolution: {add_res_samples}")
#         print(f"Will downsample from resolution: {downsample_from_res if downsample_from_res else 'auto-select highest'}")
    
#     # Create datasets for each split
#     train_dataset = KSTrueMultiResMarkovDataset(
#         saved_folder=saved_folder,
#         viscosity=viscosity,
#         L=L,
#         lmax=lmax,
#         et=et,
#         nte=nte,
#         nt=nt,
#         train_s=train_s,
#         data_mres_size=data_mres_size,
#         add_res=add_res,
#         add_res_samples=add_res_samples,
#         downsample_from_res=downsample_from_res,
#         split_ratio=split_ratio,
#         random_seed=random_seed,
#         split='train',
#         **kwargs
#     )
    
#     val_dataset = KSTrueMultiResMarkovDataset(
#         saved_folder=saved_folder,
#         viscosity=viscosity,
#         L=L,
#         lmax=lmax,
#         et=et,
#         nte=nte,
#         nt=nt,
#         train_s=train_s,
#         data_mres_size=data_mres_size,
#         add_res=add_res,
#         add_res_samples=add_res_samples,
#         downsample_from_res=downsample_from_res,
#         split_ratio=split_ratio,
#         random_seed=random_seed,
#         split='val',
#         **kwargs
#     )
    
#     test_dataset = KSTrueMultiResMarkovDataset(
#         saved_folder=saved_folder,
#         viscosity=viscosity,
#         L=L,
#         lmax=lmax,
#         et=et,
#         nte=nte,
#         nt=nt,
#         train_s=train_s,
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
#     min_data = None
#     max_data = None
#     min_model = None
#     max_model = None
    
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
            
#         elif normalization_type == "minmax":
#             print('---------Computing min-max normalization statistics---------------')
#             temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
            
#             x_train_all = []
#             y_train_all = []
#             for batch in temp_loader:
#                 x_batch, y_batch = batch
#                 x_train_all.append(x_batch)
#                 y_train_all.append(y_batch)
            
#             x_train_tensor = torch.cat(x_train_all, dim=0)
#             y_train_tensor = torch.cat(y_train_all, dim=0)
            
#             min_data = float(x_train_tensor.min())
#             max_data = float(x_train_tensor.max())
#             min_model = float(y_train_tensor.min())
#             max_model = float(y_train_tensor.max())
            
#             print(f"Input data range: [{min_data:.6f}, {max_data:.6f}]")
#             print(f"Output data range: [{min_model:.6f}, {max_model:.6f}]")
            
#             class MinMaxNormalizedDataset(Dataset):
#                 def __init__(self, dataset, min_data, max_data, min_model, max_model):
#                     self.dataset = dataset
#                     self.min_data = min_data
#                     self.max_data = max_data
#                     self.min_model = min_model
#                     self.max_model = max_model
                
#                 def __len__(self):
#                     return len(self.dataset)
                
#                 def __getitem__(self, idx):
#                     x, y = self.dataset[idx]
#                     if isinstance(x, np.ndarray):
#                         x = torch.from_numpy(x).float()
#                     if isinstance(y, np.ndarray):
#                         y = torch.from_numpy(y).float()
                    
#                     x_normalized = (x - self.min_data) / (self.max_data - self.min_data)
#                     y_normalized = (y - self.min_model) / (self.max_model - self.min_model)
                    
#                     return x_normalized, y_normalized
            
#             train_dataset = MinMaxNormalizedDataset(train_dataset, min_data, max_data, min_model, max_model)
#             val_dataset = MinMaxNormalizedDataset(val_dataset, min_data, max_data, min_model, max_model)
#             test_dataset = MinMaxNormalizedDataset(test_dataset, min_data, max_data, min_model, max_model)
            
#         else:
#             raise ValueError(f"Invalid normalization_type: {normalization_type}. Must be 'simple' or 'minmax'")
    
#     print(f"\nFinal dataset sizes:")
#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
#     print(f"Test dataset size: {len(test_dataset)}")
    
#     # Return appropriate values based on normalization type
#     if normalization_type == "simple":
#         return train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer
#     else:  # minmax
#         return train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model


# Example usage:
# """
# # Basic usage with train_s parameter
# train_dataset, val_dataset, test_dataset, x_norm, y_norm = ks_true_multires_markov_dataset(
#     saved_folder="/data/user_data/rvk/ks/",
#     viscosity=0.05,
#     L=64.0,
#     lmax=8,
#     et=5.0,
#     nte=51,
#     nt=51,
#     train_s=2048,  # This will look for KS_train_2048.h5
#     data_mres_size={800: 0, 512: 2048, 400: 0},
#     add_res=[32, 64],
#     add_res_samples={64: 0, 32: 0},
#     downsample_from_res=400,
#     normalization_type="simple",
#     data_normalizer=True
# )
# """