import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import glob
import h5py
import numpy as np
from einops import rearrange
import re

from utils.res_utils import downsample, resize

class MultiFileActiveMatterMarkovDataset(Dataset):
    def __init__(self, 
                 file_pattern, 
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 s=None,  # Target spatial resolution
                 max_files=None,  # Limit number of files to load
                 **kwargs):
        
        assert reduced_resolution == 1, "reduced_resolution must be 1 when using parameter 's' for downsampling. Use 's' parameter instead of reduced_resolution for spatial downsampling."
        
        # Find all matching files
        search_pattern = os.path.join(os.path.abspath(saved_folder), file_pattern)
        self.file_paths = sorted(glob.glob(search_pattern))
        
        if not self.file_paths:
            raise ValueError(f"No files found matching pattern: {search_pattern}")
        
        # Limit number of files if specified
        if max_files is not None and max_files > 0:
            self.file_paths = self.file_paths[:max_files]
        
        print(f'Found {len(self.file_paths)} files matching pattern: {file_pattern}')
        for i, path in enumerate(self.file_paths[:5]):  # Show first 5 files
            print(f'  {i+1}: {os.path.basename(path)}')
        if len(self.file_paths) > 5:
            print(f'  ... and {len(self.file_paths) - 5} more files')
        
        # Load and concatenate data from all files
        self._load_multi_hdf5_data(reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max)
        
        # Apply resolution change if s is specified and different from current resolution
        current_spatial_size = self.data.shape[2]  # Assuming square grid
        print('Current Spatial Size:', current_spatial_size)

        if s is not None and s != current_spatial_size:
            print(f"Resizing from {current_spatial_size}x{current_spatial_size} to {s}x{s}")
            
            # Reshape for resizing: (batch, time, height, width, channels) 
            # -> (batch*time*channels, 1, height, width)
            batch_size, time_steps, height, width, channels = self.data.shape
            data_reshaped = self.data.transpose(0, 1, 4, 2, 3)  # (batch, time, channels, height, width)
            data_reshaped = data_reshaped.reshape(batch_size * time_steps * channels, 1, height, width)
            
            # Choose appropriate resizing method based on target size
            if s < current_spatial_size:
                # Downsampling: use downsample function
                print(f"Downsampling using downsample function")
                data_resized = downsample(data_reshaped, s)
            else:
                # Upsampling: use resize function
                print(f"Upsampling using resize function")
                # Convert to torch tensor for resize function
                data_torch = torch.tensor(data_reshaped, dtype=torch.float32)
                data_resized = resize(data_torch, (s, s))
                # Convert back to numpy
                data_resized = data_resized.numpy()
            
            # Reshape back: (batch*time*channels, 1, s, s) -> (batch, time, s, s, channels)
            data_resized = data_resized.reshape(batch_size, time_steps, channels, s, s)
            data_resized = data_resized.transpose(0, 1, 3, 4, 2)  # (batch, time, height, width, channels)
            
            self.data = data_resized
            print(f"Data shape after resizing: {self.data.shape}")
            
            # Update grid for new resolution
            if hasattr(self, 'grid') and self.grid is not None:
                x_coords = np.linspace(0, 1, s)
                y_coords = np.linspace(0, 1, s)
                xx, yy = np.meshgrid(x_coords, y_coords)
                self.grid = np.stack([xx, yy], axis=-1)
                self.grid = torch.tensor(self.grid, dtype=torch.float)
        
        # Extract the input sequence (all timesteps except last)
        x = self.data[:, :-1]  # (batch, time-1, height, width, channels)
        # Extract the output sequence (all timesteps except first)
        y = self.data[:, 1:]   # (batch, time-1, height, width, channels)
        
        # Flatten batch and time dimensions together
        batch_size, time_steps = x.shape[0], x.shape[1]
        height, width, channels = x.shape[2], x.shape[3], x.shape[4]
        
        # Reshape to (batch*time, channels, height, width) format
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        
        self.x = rearrange(self.x, 'b t h w c -> (b t) c h w')
        self.y = rearrange(self.y, 'b t h w c -> (b t) c h w')
        
        assert len(self.x) == len(self.y), "Invalid input output pairs"
        print(f"x shape: {self.x.shape}")
        print(f"y shape: {self.y.shape}")
        if hasattr(self, 'grid') and self.grid is not None:
            print(f"grid shape: {self.grid.shape}")
    
    def _extract_parameters_from_filename(self, filename):
        """Extract L, zeta, and alpha parameters from filename"""
        # Pattern: active_matter_L_10.0_zeta_17.0_alpha_-5.0.hdf5
        pattern = r'active_matter_L_([\d.]+)_zeta_([\d.]+)_alpha_([-\d.]+)\.hdf5'
        match = re.search(pattern, filename)
        
        if match:
            L = float(match.group(1))
            zeta = float(match.group(2))
            alpha = float(match.group(3))
            return L, zeta, alpha
        else:
            print(f"Warning: Could not extract parameters from filename: {filename}")
            return None, None, None
    
    def _load_multi_hdf5_data(self, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max):
        """Load data from multiple HDF5 files and concatenate them"""
        all_concentration_data = []
        all_velocity_data = []
        self.file_parameters = []  # Store parameters for each trajectory
        
        for file_path in self.file_paths:
            # print(f'Loading from file: {file_path}')
            
            # Extract parameters from filename
            filename = os.path.basename(file_path)
            L, zeta, alpha = self._extract_parameters_from_filename(filename)
            
            with h5py.File(file_path, 'r') as f:
                keys = list(f.keys())
                # print(f"Available keys: {keys}")
                
                # Load concentration data
                concentration_data = np.array(f['t0_fields/concentration'], dtype=np.float32)
                # print(f"Concentration data shape: {concentration_data.shape}")
                
                # Load velocity data
                velocity_data = np.array(f['t1_fields/velocity'], dtype=np.float32)
                # print(f"Velocity data shape: {velocity_data.shape}")
                
                # Handle different possible shapes for concentration data
                if len(concentration_data.shape) == 3:
                    # Shape: (time, height, width) - add batch dimension
                    concentration_data = concentration_data[np.newaxis, :, :, :]
                    print(f"Added batch dimension to concentration: {concentration_data.shape}")
                elif len(concentration_data.shape) == 4:
                    # Shape: (batch, time, height, width) - already correct
                    pass
                else:
                    raise ValueError(f"Unexpected concentration shape: {concentration_data.shape}")
                
                # Handle different possible shapes for velocity data
                if len(velocity_data.shape) == 4:
                    # Shape: (time, height, width, 2) - add batch dimension
                    velocity_data = velocity_data[np.newaxis, :, :, :, :]
                    print(f"Added batch dimension to velocity: {velocity_data.shape}")
                elif len(velocity_data.shape) == 5:
                    # Shape: (batch, time, height, width, 2) - already correct
                    pass
                else:
                    raise ValueError(f"Unexpected velocity shape: {velocity_data.shape}")
                
                # Ensure both have same batch and time dimensions
                if concentration_data.shape[0] != velocity_data.shape[0]:
                    print(f"Warning: Batch dimension mismatch - concentration: {concentration_data.shape[0]}, velocity: {velocity_data.shape[0]}")
                if concentration_data.shape[1] != velocity_data.shape[1]:
                    print(f"Warning: Time dimension mismatch - concentration: {concentration_data.shape[1]}, velocity: {velocity_data.shape[1]}")
                
                # Store scalar parameters for each trajectory in this file
                file_alpha = f['scalars/alpha'][()] if 'scalars/alpha' in f else alpha
                file_zeta = f['scalars/zeta'][()] if 'scalars/zeta' in f else zeta
                
                # Add parameters for each trajectory in this file
                num_trajectories = concentration_data.shape[0]
                for _ in range(num_trajectories):
                    self.file_parameters.append({
                        'L': L,
                        'zeta': file_zeta,
                        'alpha': file_alpha,
                        'filename': filename
                    })
                
                # print(f"File parameters - L: {L}, zeta: {file_zeta}, alpha: {file_alpha}")
                
                # Apply reductions - handle based on actual dimensions
                if len(concentration_data.shape) == 4:  # (batch, time, height, width)
                    concentration_data = concentration_data[::reduced_batch, 
                                                          ::reduced_resolution_t, 
                                                          ::reduced_resolution, 
                                                          ::reduced_resolution]
                    
                if len(velocity_data.shape) == 5:  # (batch, time, height, width, 2)
                    velocity_data = velocity_data[::reduced_batch, 
                                                 ::reduced_resolution_t, 
                                                 ::reduced_resolution, 
                                                 ::reduced_resolution, :]
                
                all_concentration_data.append(concentration_data)
                all_velocity_data.append(velocity_data)
        
        # Concatenate data from all files along batch dimension
        print("Concatenating data from all files...")
        combined_concentration = np.concatenate(all_concentration_data, axis=0)
        combined_velocity = np.concatenate(all_velocity_data, axis=0)
        
        print(f"Combined concentration shape: {combined_concentration.shape}")
        print(f"Combined velocity shape: {combined_velocity.shape}")
        
        # Expand concentration to have channel dimension
        combined_concentration = np.expand_dims(combined_concentration, axis=-1)
        # print(f"Concentration data shape after expanding: {combined_concentration.shape}")
        
        # Concatenate concentration and velocity: (..., 1) + (..., 2) -> (..., 3)
        self.data = np.concatenate([combined_concentration, combined_velocity], axis=-1)
        print(f"Final combined data shape: {self.data.shape}")
        print(f"Channels: [concentration, vx, vy]")
        print(f"Total trajectories: {self.data.shape[0]}")
        
        # Create grid (assuming uniform grid from 0 to 1)
        self.grid = None
        current_height, current_width = self.data.shape[2], self.data.shape[3]
        x_coords = np.linspace(0, 1, current_width)
        y_coords = np.linspace(0, 1, current_height)
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.grid = np.stack([xx, yy], axis=-1)
        self.grid = torch.tensor(self.grid, dtype=torch.float)
        print(f"Grid shape: {self.grid.shape}")

        # Apply sample limit
        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, self.data.shape[0])
            self.data = self.data[:num_samples_max]
            self.file_parameters = self.file_parameters[:num_samples_max]
        
        # Store parameter statistics
        self.parameter_stats = self._compute_parameter_stats()
    
    def _compute_parameter_stats(self):
        """Compute statistics of the parameters across all trajectories"""
        if not self.file_parameters:
            return {}
        
        alphas = [p['alpha'] for p in self.file_parameters]
        zetas = [p['zeta'] for p in self.file_parameters]
        Ls = [p['L'] for p in self.file_parameters]
        
        stats = {
            'alpha': {'min': min(alphas), 'max': max(alphas), 'unique': list(set(alphas))},
            'zeta': {'min': min(zetas), 'max': max(zetas), 'unique': list(set(zetas))},
            'L': {'min': min(Ls), 'max': max(Ls), 'unique': list(set(Ls))},
            'total_trajectories': len(self.file_parameters),
            'total_files': len(self.file_paths)
        }
        
        print("\nParameter Statistics:")
        print(f"Alpha range: {stats['alpha']['min']} to {stats['alpha']['max']}")
        print(f"Unique alpha values: {sorted(stats['alpha']['unique'])}")
        print(f"Zeta range: {stats['zeta']['min']} to {stats['zeta']['max']}")
        print(f"Unique zeta values: {sorted(stats['zeta']['unique'])}")
        print(f"L values: {sorted(stats['L']['unique'])}")
        print(f"Total files: {stats['total_files']}")
        print(f"Total trajectories: {stats['total_trajectories']}")
        
        return stats
            
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if hasattr(self, 'grid') and self.grid is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx], self.y[idx]


def multi_file_active_matter_markov_dataset(file_pattern, saved_folder, data_normalizer=True, s=None, max_files=None, **kwargs):
    """
    Returns train, validation, and test datasets for Active Matter from multiple files with 0.8/0.1/0.1 ratio.
    Uses min-max normalization to [0, 1] range.
    
    Args:
        file_pattern: Glob pattern for files (e.g., "active_matter_*.hdf5")
        saved_folder: Path to folder containing the files
        data_normalizer: Whether to apply normalization
        s: Target spatial resolution for resizing (None to keep original)
        max_files: Maximum number of files to load (None for all)
        **kwargs: Additional arguments to pass to MultiFileActiveMatterMarkovDataset
        
    Returns:
        train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
    """
    # Create the full dataset
    full_dataset = MultiFileActiveMatterMarkovDataset(file_pattern, saved_folder, s=s, max_files=max_files, **kwargs)
    
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
    
    min_data = None
    max_data = None
    min_model = None
    max_model = None
    
    if data_normalizer:
        print('---------Computing min-max normalization statistics---------------')
        temp_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        
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
                x, y = self.dataset[idx]
                
                # Apply min-max normalization to [0, 1]
                x_normalized = (x - self.min_data) / (self.max_data - self.min_data)
                y_normalized = (y - self.min_model) / (self.max_model - self.min_model)
                
                return x_normalized, y_normalized
        
        # Apply normalization to each dataset
        train_dataset = MinMaxNormalizedDataset(train_dataset, min_data, max_data, min_model, max_model)
        val_dataset = MinMaxNormalizedDataset(val_dataset, min_data, max_data, min_model, max_model)
        test_dataset = MinMaxNormalizedDataset(test_dataset, min_data, max_data, min_model, max_model)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model