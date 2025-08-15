import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import glob
import h5py
import scipy.io
import numpy as np
import math as mt
from einops import rearrange
import scipy.fft

from utils.res_utils import downsample, resize

class NavierStokesMarkovDataset(Dataset):
    def __init__(self, 
                 filename, 
                 saved_folder, 
                 reduced_batch=1, 
                 reduced_resolution=1, 
                 reduced_resolution_t=1, 
                 num_samples_max=-1,
                 s=None,  # Target spatial resolution
                 **kwargs):
        
        assert reduced_resolution == 1, "reduced_resolution must be 1 when using parameter 's' for downsampling. Use 's' parameter instead of reduced_resolution for spatial downsampling."
        
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        print(f'Loading from file: {root_path}')
        
        # Determine file type and load accordingly
        file_extension = os.path.splitext(filename.lower())[1]
        
        if file_extension in ['.hdf5', '.h5']:
            self._load_hdf5_data(root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max)
        elif file_extension == '.mat':
            self._load_mat_data(root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .hdf5, .h5, .mat")
        
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
                current_height, current_width = s, s
                if file_extension == '.mat':
                    # For MAT files, create uniform grid
                    x_coords = np.linspace(0, 1, s)
                    y_coords = np.linspace(0, 1, s)
                else:
                    # For HDF5 files, try to maintain original grid structure
                    x_coords = np.linspace(0, 1, s)  # Fallback to uniform if no original coords
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
    
    def _load_hdf5_data(self, root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max):
        """Load data from HDF5 file format"""
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            print(f"Available keys: {keys}")
            
            # Load velocity data
            velocity_data = np.array(f['velocity'], dtype=np.float32)
            print(f"Velocity data shape: {velocity_data.shape}")
            
            # Load particle data
            particles_data = np.array(f['particles'], dtype=np.float32)
            print(f"Particles data shape: {particles_data.shape}")
            
            # Apply reductions
            velocity_data = velocity_data[::reduced_batch, 
                                         ::reduced_resolution_t, 
                                         ::reduced_resolution, 
                                         ::reduced_resolution, :]
            
            particles_data = particles_data[::reduced_batch, 
                                           ::reduced_resolution_t, 
                                           ::reduced_resolution, 
                                           ::reduced_resolution, :]
            
            # Load time data if available
            if 't' in keys:
                self.time = np.array(f['t'], dtype=np.float32)
                self.time = self.time[::reduced_batch, ::reduced_resolution_t]
            
            # Load grid data if available
            self.grid = None
            if 'x-coordinate' in keys and 'y-coordinate' in keys:
                x_coords = np.array(f['x-coordinate'], dtype=np.float32)
                y_coords = np.array(f['y-coordinate'], dtype=np.float32)
                x_coords = x_coords[::reduced_resolution]
                y_coords = y_coords[::reduced_resolution]
                xx, yy = np.meshgrid(x_coords, y_coords)
                self.grid = np.stack([xx, yy], axis=-1)
                self.grid = torch.tensor(self.grid, dtype=torch.float)
                print(f"Grid shape: {self.grid.shape}")

            # Apply sample limit
            if num_samples_max > 0:
                num_samples_max = min(num_samples_max, velocity_data.shape[0])
            else:
                num_samples_max = velocity_data.shape[0]

            velocity_data = velocity_data[:num_samples_max]
            particles_data = particles_data[:num_samples_max]
            
            # Combine particles and velocity into a single data tensor
            self.data = np.concatenate([particles_data, velocity_data], axis=-1)
            print(f"Combined data shape: {self.data.shape}")
    
    def _load_mat_data(self, root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max):
        """Load data from MAT file format"""
        # Load .mat file
        mat_data = scipy.io.loadmat(root_path)
        
        # Extract a, u, t from mat file
        a = mat_data['a']  # Shape: (20, 512, 512)
        u = mat_data['u']  # Shape: (20, 512, 512, 200)
        t = mat_data['t']  # Shape: (1, 200)
        
        print(f"a shape: {a.shape}")
        print(f"u shape: {u.shape}")
        print(f"t shape: {t.shape}")
        
        # Expand a to have time dimension and concatenate with u
        # a: (20, 512, 512) -> (20, 512, 512, 1)
        a_expanded = np.expand_dims(a, axis=-1)
        
        # Concatenate a and u along the last dimension
        # Result: (20, 512, 512, 201) where first channel is 'a' and next 200 are 'u'
        combined_data = np.concatenate([a_expanded, u], axis=-1)
        print(f"Combined data shape: {combined_data.shape}")
        
        # Convert to format: (batch, time, height, width, channels=1)
        # Each of the 201 channels becomes a time step
        batch_size, height, width, time_channels = combined_data.shape
        
        # Reshape to (batch, time, height, width, 1)
        self.data = combined_data.reshape(batch_size, height, width, time_channels, 1)
        self.data = self.data.transpose(0, 3, 1, 2, 4)  # (batch, time, height, width, channels)
        
        print(f"Reshaped data shape: {self.data.shape}")
        
        # Apply reductions
        self.data = self.data[::reduced_batch, 
                             ::reduced_resolution_t, 
                             ::reduced_resolution, 
                             ::reduced_resolution, :]
        
        # Store time data
        self.time = t.flatten()  # Convert (1, 200) to (200,)
        # Add time 0 for the 'a' channel
        self.time = np.concatenate([[0], self.time])  # Now (201,)
        self.time = self.time[::reduced_resolution_t]
        
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
        else:
            num_samples_max = self.data.shape[0]

        self.data = self.data[:num_samples_max]
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if hasattr(self, 'grid') and self.grid is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx], self.y[idx]


def cno_ns_markov_dataset(filename, saved_folder, data_normalizer=True, s=None, **kwargs):
    """
    Returns train, validation, and test datasets for Navier-Stokes with 0.8/0.1/0.1 ratio.
    Uses min-max normalization to [0, 1] range.
    Supports both HDF5 (.h5, .hdf5) and MAT (.mat) file formats.
    
    Args:
        filename: File name (supports .h5, .hdf5, .mat extensions)
        saved_folder: Path to folder containing the file
        data_normalizer: Whether to apply normalization
        s: Target spatial resolution for resizing (None to keep original)
        **kwargs: Additional arguments to pass to NavierStokesMarkovDataset
        
    Returns:
        train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
    """
    # Create the full dataset
    full_dataset = NavierStokesMarkovDataset(filename, saved_folder, s=s, **kwargs)
    
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

# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# import os
# import glob
# import h5py
# import scipy.io
# import numpy as np
# import math as mt
# from einops import rearrange
# import scipy.fft

# # Some functions needed for loading the Navier-Stokes data

# def samples_fft(u):
#     return scipy.fft.fft2(u, norm='forward', workers=-1)

# def samples_ifft(u_hat):
#     return scipy.fft.ifft2(u_hat, norm='forward', workers=-1).real
    
# def downsample(u, N):
#     N_old = u.shape[-2]
#     freqs = scipy.fft.fftfreq(N_old, d=1/N_old)
#     sel = np.logical_and(freqs >= -N/2, freqs <= N/2-1)
#     u_hat = samples_fft(u)
#     u_hat_down = u_hat[:,:,sel,:][:,:,:,sel]
#     u_down = samples_ifft(u_hat_down)
#     return u_down    

# class NavierStokesMarkovDataset(Dataset):
#     def __init__(self, 
#                  filename, 
#                  saved_folder, 
#                  reduced_batch=1, 
#                  reduced_resolution=1, 
#                  reduced_resolution_t=1, 
#                  num_samples_max=-1,
#                  s=None,  # Target spatial resolution
#                  **kwargs):
        
#         assert reduced_resolution == 1, "reduced_resolution must be 1 when using parameter 's' for downsampling. Use 's' parameter instead of reduced_resolution for spatial downsampling."
        
#         root_path = os.path.join(os.path.abspath(saved_folder), filename)
#         print(f'Loading from file: {root_path}')
        
#         # Determine file type and load accordingly
#         file_extension = os.path.splitext(filename.lower())[1]
        
#         if file_extension in ['.hdf5', '.h5']:
#             self._load_hdf5_data(root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max)
#         elif file_extension == '.mat':
#             self._load_mat_data(root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max)
#         else:
#             raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .hdf5, .h5, .mat")
        
#         # Apply downsampling if s is specified and different from current resolution
#         current_spatial_size = self.data.shape[2]  # Assuming square grid
#         print('Current Spatial Size:', current_spatial_size)

#         if s is not None and s != current_spatial_size:
#             print(f"Downsampling from {current_spatial_size}x{current_spatial_size} to {s}x{s}")
            
#             # Reshape for downsampling: (batch, time, height, width, channels) 
#             # -> (batch*time*channels, 1, height, width)
#             batch_size, time_steps, height, width, channels = self.data.shape
#             data_reshaped = self.data.transpose(0, 1, 4, 2, 3)  # (batch, time, channels, height, width)
#             data_reshaped = data_reshaped.reshape(batch_size * time_steps * channels, 1, height, width)
            
#             # Downsample
#             data_downsampled = downsample(data_reshaped, s)
            
#             # Reshape back: (batch*time*channels, 1, s, s) -> (batch, time, s, s, channels)
#             data_downsampled = data_downsampled.reshape(batch_size, time_steps, channels, s, s)
#             data_downsampled = data_downsampled.transpose(0, 1, 3, 4, 2)  # (batch, time, height, width, channels)
            
#             self.data = data_downsampled
#             print(f"Data shape after downsampling: {self.data.shape}")
            
#             # Update grid for new resolution
#             if hasattr(self, 'grid') and self.grid is not None:
#                 current_height, current_width = s, s
#                 if file_extension == '.mat':
#                     # For MAT files, create uniform grid
#                     x_coords = np.linspace(0, 1, s)
#                     y_coords = np.linspace(0, 1, s)
#                 else:
#                     # For HDF5 files, try to maintain original grid structure
#                     x_coords = np.linspace(0, 1, s)  # Fallback to uniform if no original coords
#                     y_coords = np.linspace(0, 1, s)
                
#                 xx, yy = np.meshgrid(x_coords, y_coords)
#                 self.grid = np.stack([xx, yy], axis=-1)
#                 self.grid = torch.tensor(self.grid, dtype=torch.float)
        
#         # Extract the input sequence (all timesteps except last)
#         x = self.data[:, :-1]  # (batch, time-1, height, width, channels)
#         # Extract the output sequence (all timesteps except first)
#         y = self.data[:, 1:]   # (batch, time-1, height, width, channels)
        
#         # Flatten batch and time dimensions together
#         batch_size, time_steps = x.shape[0], x.shape[1]
#         height, width, channels = x.shape[2], x.shape[3], x.shape[4]
        
#         # Reshape to (batch*time, channels, height, width) format
#         self.x = torch.tensor(x, dtype=torch.float)
#         self.y = torch.tensor(y, dtype=torch.float)
        
#         self.x = rearrange(self.x, 'b t h w c -> (b t) c h w')
#         self.y = rearrange(self.y, 'b t h w c -> (b t) c h w')
        
#         assert len(self.x) == len(self.y), "Invalid input output pairs"
#         print(f"x shape: {self.x.shape}")
#         print(f"y shape: {self.y.shape}")
#         if hasattr(self, 'grid') and self.grid is not None:
#             print(f"grid shape: {self.grid.shape}")
    
#     def _load_hdf5_data(self, root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max):
#         """Load data from HDF5 file format"""
#         with h5py.File(root_path, 'r') as f:
#             keys = list(f.keys())
#             print(f"Available keys: {keys}")
            
#             # Load velocity data
#             velocity_data = np.array(f['velocity'], dtype=np.float32)
#             print(f"Velocity data shape: {velocity_data.shape}")
            
#             # Load particle data
#             particles_data = np.array(f['particles'], dtype=np.float32)
#             print(f"Particles data shape: {particles_data.shape}")
            
#             # Apply reductions
#             velocity_data = velocity_data[::reduced_batch, 
#                                          ::reduced_resolution_t, 
#                                          ::reduced_resolution, 
#                                          ::reduced_resolution, :]
            
#             particles_data = particles_data[::reduced_batch, 
#                                            ::reduced_resolution_t, 
#                                            ::reduced_resolution, 
#                                            ::reduced_resolution, :]
            
#             # Load time data if available
#             if 't' in keys:
#                 self.time = np.array(f['t'], dtype=np.float32)
#                 self.time = self.time[::reduced_batch, ::reduced_resolution_t]
            
#             # Load grid data if available
#             self.grid = None
#             if 'x-coordinate' in keys and 'y-coordinate' in keys:
#                 x_coords = np.array(f['x-coordinate'], dtype=np.float32)
#                 y_coords = np.array(f['y-coordinate'], dtype=np.float32)
#                 x_coords = x_coords[::reduced_resolution]
#                 y_coords = y_coords[::reduced_resolution]
#                 xx, yy = np.meshgrid(x_coords, y_coords)
#                 self.grid = np.stack([xx, yy], axis=-1)
#                 self.grid = torch.tensor(self.grid, dtype=torch.float)
#                 print(f"Grid shape: {self.grid.shape}")

#             # Apply sample limit
#             if num_samples_max > 0:
#                 num_samples_max = min(num_samples_max, velocity_data.shape[0])
#             else:
#                 num_samples_max = velocity_data.shape[0]

#             velocity_data = velocity_data[:num_samples_max]
#             particles_data = particles_data[:num_samples_max]
            
#             # Combine particles and velocity into a single data tensor
#             self.data = np.concatenate([particles_data, velocity_data], axis=-1)
#             print(f"Combined data shape: {self.data.shape}")
    
#     def _load_mat_data(self, root_path, reduced_batch, reduced_resolution, reduced_resolution_t, num_samples_max):
#         """Load data from MAT file format"""
#         # Load .mat file
#         mat_data = scipy.io.loadmat(root_path)
        
#         # Extract a, u, t from mat file
#         a = mat_data['a']  # Shape: (20, 512, 512)
#         u = mat_data['u']  # Shape: (20, 512, 512, 200)
#         t = mat_data['t']  # Shape: (1, 200)
        
#         print(f"a shape: {a.shape}")
#         print(f"u shape: {u.shape}")
#         print(f"t shape: {t.shape}")
        
#         # Expand a to have time dimension and concatenate with u
#         # a: (20, 512, 512) -> (20, 512, 512, 1)
#         a_expanded = np.expand_dims(a, axis=-1)
        
#         # Concatenate a and u along the last dimension
#         # Result: (20, 512, 512, 201) where first channel is 'a' and next 200 are 'u'
#         combined_data = np.concatenate([a_expanded, u], axis=-1)
#         print(f"Combined data shape: {combined_data.shape}")
        
#         # Convert to format: (batch, time, height, width, channels=1)
#         # Each of the 201 channels becomes a time step
#         batch_size, height, width, time_channels = combined_data.shape
        
#         # Reshape to (batch, time, height, width, 1)
#         self.data = combined_data.reshape(batch_size, height, width, time_channels, 1)
#         self.data = self.data.transpose(0, 3, 1, 2, 4)  # (batch, time, height, width, channels)
        
#         print(f"Reshaped data shape: {self.data.shape}")
        
#         # Apply reductions
#         self.data = self.data[::reduced_batch, 
#                              ::reduced_resolution_t, 
#                              ::reduced_resolution, 
#                              ::reduced_resolution, :]
        
#         # Store time data
#         self.time = t.flatten()  # Convert (1, 200) to (200,)
#         # Add time 0 for the 'a' channel
#         self.time = np.concatenate([[0], self.time])  # Now (201,)
#         self.time = self.time[::reduced_resolution_t]
        
#         # Create grid (assuming uniform grid from 0 to 1)
#         self.grid = None
#         current_height, current_width = self.data.shape[2], self.data.shape[3]
#         x_coords = np.linspace(0, 1, current_width)
#         y_coords = np.linspace(0, 1, current_height)
#         xx, yy = np.meshgrid(x_coords, y_coords)
#         self.grid = np.stack([xx, yy], axis=-1)
#         self.grid = torch.tensor(self.grid, dtype=torch.float)
#         print(f"Grid shape: {self.grid.shape}")

#         # Apply sample limit
#         if num_samples_max > 0:
#             num_samples_max = min(num_samples_max, self.data.shape[0])
#         else:
#             num_samples_max = self.data.shape[0]

#         self.data = self.data[:num_samples_max]
        
#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, idx):
#         if hasattr(self, 'grid') and self.grid is not None:
#             return self.x[idx], self.y[idx]
#         else:
#             return self.x[idx], self.y[idx]


# def cno_ns_markov_dataset(filename, saved_folder, data_normalizer=True, s=None, **kwargs):
#     """
#     Returns train, validation, and test datasets for Navier-Stokes with 0.8/0.1/0.1 ratio.
#     Uses min-max normalization to [0, 1] range.
#     Supports both HDF5 (.h5, .hdf5) and MAT (.mat) file formats.
    
#     Args:
#         filename: File name (supports .h5, .hdf5, .mat extensions)
#         saved_folder: Path to folder containing the file
#         data_normalizer: Whether to apply normalization
#         s: Target spatial resolution for downsampling (None to keep original)
#         **kwargs: Additional arguments to pass to NavierStokesMarkovDataset
        
#     Returns:
#         train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model
#     """
#     # Create the full dataset
#     full_dataset = NavierStokesMarkovDataset(filename, saved_folder, s=s, **kwargs)
    
#     # Calculate split sizes
#     dataset_size = len(full_dataset)
#     train_size = int(0.8 * dataset_size)
#     val_size = int(0.1 * dataset_size)
#     test_size = dataset_size - train_size - val_size
    
#     # Split dataset
#     train_dataset, val_dataset, test_dataset = random_split(
#         full_dataset, 
#         [train_size, val_size, test_size],
#         generator=torch.Generator().manual_seed(42)  # For reproducibility
#     )
    
#     min_data = None
#     max_data = None
#     min_model = None
#     max_model = None
    
#     if data_normalizer:
#         print('---------Computing min-max normalization statistics---------------')
#         temp_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
        
#         # Collect all training data for computing min-max statistics
#         x_train_all = []
#         y_train_all = []
#         for batch in temp_loader:
#             if len(batch) == 3:  # If grid is included
#                 x_batch, y_batch, _ = batch
#             else:
#                 x_batch, y_batch = batch
                
#             x_train_all.append(x_batch)
#             y_train_all.append(y_batch)
        
#         x_train_tensor = torch.cat(x_train_all, dim=0)
#         y_train_tensor = torch.cat(y_train_all, dim=0)
        
#         # Compute min-max statistics
#         min_data = float(x_train_tensor.min())
#         max_data = float(x_train_tensor.max())
#         min_model = float(y_train_tensor.min())
#         max_model = float(y_train_tensor.max())
        
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
#                 x, y = self.dataset[idx]
                
#                 # Apply min-max normalization to [0, 1]
#                 x_normalized = (x - self.min_data) / (self.max_data - self.min_data)
#                 y_normalized = (y - self.min_model) / (self.max_model - self.min_model)
                
#                 return x_normalized, y_normalized
        
#         # Apply normalization to each dataset
#         train_dataset = MinMaxNormalizedDataset(train_dataset, min_data, max_data, min_model, max_model)
#         val_dataset = MinMaxNormalizedDataset(val_dataset, min_data, max_data, min_model, max_model)
#         test_dataset = MinMaxNormalizedDataset(test_dataset, min_data, max_data, min_model, max_model)
    
#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Validation dataset size: {len(val_dataset)}")
#     print(f"Test dataset size: {len(test_dataset)}")
    
#     return train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model


# # # Example usage:
# # if __name__ == "__main__":
# #     # Example of how to use the modified function
# #     train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model = ns_markov_dataset(
# #         filename="your_file.h5",
# #         saved_folder="path/to/data",
# #         s=64,  # Downsample to 64x64
# #         data_normalizer=True
# #     )
    
# #     # You can now use these statistics at test time for denormalization:
# #     # original_value = normalized_value * (max_val - min_val) + min_val