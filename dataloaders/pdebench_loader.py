import torch
from dataloaders.sequential_dataset import SequentialDataSet
from torch.utils.data import DataLoader, Dataset
import os
import glob
import h5py
import numpy as np
import math as mt

def get_start_end(N, if_test, test_ratio, num_samples_max):
    if if_test: 
        # when testing, ignore num_samples_max
        start = int(N * (1-test_ratio))
        end = N
    elif num_samples_max > 0:
        if num_samples_max > int(N * (1-test_ratio)):
            raise ValueError(f"num_samples_max={num_samples_max} can't be larger than N * (1-test_ratio)={int(N * (1-test_ratio))}")
        start = 0
        end = num_samples_max
    else: 
        start = 0
        end = int(N * (1-test_ratio))
    return start, end

def read_h5_key(f : h5py.File, key : str, start : int, end : int):
    return f[key][start:end]

class FNODatasetSingle(SequentialDataSet):
    def __init__(self, 
                 filename,
                 init_step=0,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max = -1,
                 t_train = None,
                 t_test = None,
                 ):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """
        if t_train is None or t_test is None:
            raise ValueError("t_train and t_test must be specified")
        def read(f : h5py.File,  key : str, reduced_resolution : int = reduced_resolution):
            N = f[key].shape[0]
            start, end = get_start_end(N, if_test, test_ratio, num_samples_max)
            out = read_h5_key(f, key, start, end)
            # reduce t resolution
            idx_cfd = out.shape
            if len(idx_cfd)==3:  # 1D
                # (N, T, X) -> (N//reduced_batch, T//reduced_resolution_t, X//reduced_resolution)
                out = out[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
            elif len(idx_cfd)==4: #2D
                # (N, T, X, Y) -> (N//reduced_batch, T//reduced_resolution_t, X//reduced_resolution, Y//reduced_resolution)
                out = out[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
            elif len(idx_cfd)==5: #3D
                # (N, T, X, Y, Z) -> (N//reduced_batch, T//reduced_resolution_t, X//reduced_resolution, Y//reduced_resolution, Z//reduced_resolution)
                out = out[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution,::reduced_resolution]
            return out

        # Define path to files
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        if filename[-2:] != 'h5':
            print(f".HDF5 file extension is assumed hereafter")
        
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                if 'tensor' not in keys:
                    _data = np.array(read(f,'density'), dtype=np.float32)  # batch, time, x,...
                    idx_cfd = _data.shape
                    if len(idx_cfd)==3:  # 1D
                        self.data = np.zeros([idx_cfd[0],
                                              idx_cfd[2],
                                              idx_cfd[1],
                                              3],
                                            dtype=np.float32)
                        #density
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(read(f,'pressure'), dtype=np.float32)  # batch, time, x,...
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(read(f,'Vx'), dtype=np.float32)  # batch, time, x,...
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        self.data[...,2] = _data   # batch, x, t, ch

                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                        print(self.data.shape)
                    if len(idx_cfd)==4:  # 2D
                        self.data = np.zeros([idx_cfd[0],
                                              idx_cfd[2],
                                              idx_cfd[3],
                                              idx_cfd[1],
                                              4],
                                             dtype=np.float32)
                        # density
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(read(f,'pressure'), dtype=np.float32)  # batch, time, x,...
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(read(f,'Vx'), dtype=np.float32)  # batch, time, x,...
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(read(f,'Vy'), dtype=np.float32)  # batch, time, x,...
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 1))
                        self.data[...,3] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
                
                    if len(idx_cfd)==5:  # 3D
                        self.data = np.zeros([idx_cfd[0],
                                              idx_cfd[2],
                                              idx_cfd[3],
                                              idx_cfd[4],
                                              idx_cfd[1],
                                              5],
                                             dtype=np.float32)
                        # density
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,0] = _data   # batch, x, t, ch
                        # pressure
                        _data = np.array(read(f,'pressure'), dtype=np.float32)  # batch, time, x,...
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,1] = _data   # batch, x, t, ch
                        # Vx
                        _data = np.array(read(f,'Vx'), dtype=np.float32)  # batch, time, x,...
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,2] = _data   # batch, x, t, ch
                        # Vy
                        _data = np.array(read(f,'Vy'), dtype=np.float32)  # batch, time, x,...
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,3] = _data   # batch, x, t, ch
                        # Vz
                        _data = np.array(read(f,'Vz'), dtype=np.float32)  # batch, time, x,...
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data, (0, 2, 3, 4, 1))
                        self.data[...,4] = _data   # batch, x, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        z = np.array(f["z-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        z = torch.tensor(z, dtype=torch.float)
                        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
                        self.grid = torch.stack((X, Y, Z), axis=-1)[::reduced_resolution,\
                                                                    ::reduced_resolution,\
                                                                    ::reduced_resolution]
                                                                    
                else:  # scalar equations
                    ## data dim = [t, x1, ..., xd, v]
                    _data = np.array(read(f,'tensor', reduced_resolution=reduced_resolution,),  dtype=np.float32)  # batch, time, x,...
                    _data_hr = np.array(read(f,'tensor', reduced_resolution=1,), dtype=np.float32)  # batch, time, x,...
                    if len(_data.shape) == 3:  # 1D
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :], (0, 2, 1))
                        _data_hr = np.transpose(_data_hr[:, :, :], (0, 2, 1))
                        self.data = _data[:, :, :, None]  # batch, x, t, ch
                        self.data_hr = _data_hr[:, :, :, None]  # batch, x, t, ch

                        self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                        self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                    if len(_data.shape) == 4:  # 2D Darcy flow
                        # u: label
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        self.data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = np.concatenate([_data, self.data], axis=-1)
                        self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch

                        x = np.array(f["x-coordinate"], dtype=np.float32)
                        y = np.array(f["y-coordinate"], dtype=np.float32)
                        x = torch.tensor(x, dtype=torch.float)
                        y = torch.tensor(y, dtype=torch.float)
                        X, Y = torch.meshgrid(x, y, indexing='ij')
                        self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

        elif filename[-2:] == 'h5':  # SWE-2D (RDB)
            print(f".H5 file extension is assumed hereafter")
        
            with h5py.File(root_path, 'r') as f:
                keys = list(f.keys())
                keys.sort()
                data_arrays = [np.array(f[key]['data'], dtype=np.float32) for key in keys]
                _data = torch.from_numpy(np.stack(data_arrays, axis=0))   # [batch, nt, nx, ny, nc]
                _data = _data[:num_samples_max][::reduced_batch, ::reduced_resolution_t, ::reduced_resolution, ::reduced_resolution, ...]
                if len(_data.shape) == 4:  # diffusion-sorption
                    _data = torch.permute(_data, (0, 2, 1, 3))
                    _grid = np.array(f['0023']['grid']['x'], dtype=np.float32)
                    _grid = _grid[::reduced_resolution, ...]
                    _grid = torch.from_numpy(_grid).unsqueeze(-1)
                else:
                    _data = torch.permute(_data, (0, 2, 3, 1, 4))   # [batch, nx, ny, nt, nc]
                    gridx, gridy = np.array(f['0023']['grid']['x'], dtype=np.float32), np.array(f['0023']['grid']['y'], dtype=np.float32)
                    mgridX, mgridY = np.meshgrid(gridx, gridy, indexing='ij')
                    _grid = torch.stack((torch.from_numpy(mgridX), torch.from_numpy(mgridY)), axis=-1)
                    _grid = _grid[::reduced_resolution, ::reduced_resolution, ...]
                _tsteps_t = torch.from_numpy(np.array(f['0023']['grid']['t'], dtype=np.float32))
                tsteps_t = _tsteps_t[::reduced_resolution_t]
                self.data = _data
                self.grid = _grid
                self.tsteps_t = tsteps_t

        if not if_test and num_samples_max>0 and self.data.shape[0] != num_samples_max // reduced_batch:
            # there must have been some error
            raise ValueError(f"Something went wrong, num_samples_max={num_samples_max // reduced_batch} is not consistent with the data shape {self.data.shape}")

        # Time steps used as initial conditions
        # self.initial_step = inital_step

        self.data = self.data if torch.is_tensor(self.data) else torch.tensor(self.data)
        # self.data_hr = self.data_hr if torch.is_tensor(self.data_hr) else torch.tensor(self.data_hr)
        # data (B, S, T, D)
        n_time_steps = t_train if not if_test else t_test
        if n_time_steps > 0 or init_step>0:
            self.data = self.data[:,:, init_step:n_time_steps, :]
            # self.data_hr = self.data_hr[:,:, :n_time_steps, :]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        # return self.data[idx,...,:self.initial_step,:], self.data[idx], self.grid
        return self.data[idx], self.grid
    
    def input_shape(self):
        '''Returns a tuple input shape of the dataset (Sx, [Sy], T, V), where:
        Sx, [Sz], [Sz] = spatial dimension length
        T = number of timesteps
        V = number state variables
        :return: tuple
        '''
        return self.data.shape[1:]
