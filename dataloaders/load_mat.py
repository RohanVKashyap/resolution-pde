import os, sys
import torch
from dataloaders.sequential_dataset import SequentialDataSet
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import numpy as np
import scipy
from scipy import io
import h5py
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

"""Source code: https://github.com/RohanVKashyap/invariance-pde/blob/main/dataloaders/fno/burgers_loader.py"""

class BurgersFNOLoader(SequentialDataSet):
    def __init__(self, 
                 filename, 
                 saved_folder, 
                 reduced_resolution, 
                 num_samples_max=-1,
                 test_ratio=0.1,
                 if_test=False,
                 **kwargs,):
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        self.dataloader = MatReader(root_path)
        x_data = self.dataloader.read_field('a')[:, ::reduced_resolution]
        y_data = self.dataloader.read_field('u')[:, ::reduced_resolution]
        
        print(num_samples_max)
        if num_samples_max>0:
            num_samples_max  = min(num_samples_max, x_data.shape[0])
        else:
            num_samples_max = x_data.shape[0]
    
        self.x = rearrange(x_data, 'b m -> b m 1 1')
        self.y = rearrange(y_data, 'b m -> b m 1 1')

    def get_grid(self):
        size_x = self.x.shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(size_x, 1)
        return gridx

    def __getitem__(self, index):
        grid = self.get_grid()
        return self.x[index], self.y[index], grid

    def __len__(self):
        return len(self.x)
    
    def input_shape(self):
        '''Returns a tuple input shape of the dataset (Sx, [Sy], T, V), where:
        Sx, [Sz], [Sz] = spatial dimension length
        T = number of timesteps
        V = number state variables
        :return: tuple
        '''
        L = self.x.shape[-1]
        T, D = 1, 1
        return L, T, D

class FNOLoader2D(Dataset):
    def __init__(self, 
                 filename, 
                 saved_folder, 
                 reduced_resolution, 
                 reduced_resolution_t=1,
                 num_samples_max=-1,
                 test_ratio=0.1,
                 if_test=False,
                 t_train = -1,
                 t_test = -1,
                 chunk_train = False,
                 train_timesteps = None,
                 unfold = False,
                 scale = False, 
                 mean = 0.0,
                 std = 1.0,
                 **kwargs,):
        self.mean = mean
        self.std = std
        root_path = os.path.join(os.path.abspath(saved_folder), filename)
        
        if filename.endswith('.mat'):
            self.dataloader = MatReader(root_path)
            N = self.dataloader.read_field('a').shape[0]
            start, end = get_start_end(N, if_test, test_ratio, num_samples_max)
            print(f"Reading .mat file: " + root_path)
            x_data = torch.Tensor(self.dataloader.read_field('a')[start:end, ::reduced_resolution, ::reduced_resolution]).to(torch.float)
            y_data = torch.Tensor(self.dataloader.read_field('u')[start:end:, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution_t]).to(torch.float)
        else:
            with h5py.File(root_path, 'r') as f:
                N = f['a'].shape[0]
                start, end = get_start_end(N, if_test, test_ratio, num_samples_max)
                print(f"Reading .h5 file: " + root_path)
                x_data = torch.Tensor(f['a'][start:end, ::reduced_resolution, ::reduced_resolution]).to(torch.float)
                y_data = torch.Tensor(f['u'][start:end, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution_t]).to(torch.float)
        
        self.y = torch.cat((x_data.unsqueeze(-1), y_data), dim=-1)

        # y: (B, Sx, Sy, T)
        if t_train is None or t_test is None:
            raise ValueError("t_train and t_test must be specified")
        n_time_steps = t_train if not if_test else t_test
        if n_time_steps > 0:
            self.y = y_data[:, :, :, :n_time_steps]
        
        if scale: 
            self.y = (self.y -  self.mean) / self.std
        
        self.chunk_train = chunk_train
        if not if_test and self.chunk_train:
            
            assert train_timesteps is not None, "train_timesteps must be specified"
            # rearrange training data into chunks of n_time_steps
            # self.y_output = self.y[...,1:]
            # self.y = self.y[...,:-1]
            if unfold: 
                # make all possible combinations of n_time_steps using Unfold
                self.y = self.y.unfold(-1, train_timesteps, 1) 
                # self.y_output = self.y_output.unfold(-1, train_timesteps, 1)
                self.y = rearrange(self.y, "b sx sy t nt -> (b t) sx sy nt")
                # self.y_output = rearrange(self.y_output, "b sx sy t nt-> (b nt) sx sy t")
            else: 
                self.y = rearrange(self.y, "b sx sy (t1 t2)-> (b t1) sx sy t2", t2 = train_timesteps)
                # self.y_output = rearrange(self.y_output, "b sx sy (t1 t2) -> (b t1) sx sy t2", t2 = train_timesteps)

        # add 1 for state
        self.y = rearrange(self.y, 'b x y t -> b x y t 1')
        # self.y_output = rearrange(self.y_output, 'b x y t -> b x y t 1')
        ## stack x at the beginning of y

        self.grid = self.get_grid()
       

    def get_grid(self):
        Sx, Sy = self.y.shape[1:3]
        # make grid of size (Sx, Sy, 1, 1)
        gridx = np.linspace(0, 1, Sx)
        gridy = np.linspace(0, 1, Sy)
        gridx, gridy = np.meshgrid(gridx, gridy)
        grid = np.stack((gridx, gridy), axis=-1)
        grid = torch.tensor(grid, dtype=torch.float)
        # need to return (Sx, Sy, 1, 2)
        return grid

    def __getitem__(self, idx):
            return self.y[idx], self.grid

    def __len__(self):
        return len(self.y)
    
    def input_shape(self):
        '''Returns a tuple input shape of the dataset (Sx, [Sy], T, V), where:
        Sx, [Sz], [Sz] = spatial dimension length
        T = number of timesteps
        V = number state variables
        :return: tuple
        '''
        B, Sx, Sy, T, V = self.y.shape
        return (Sx, Sy, T, V)

    def unscale_data(self, u):
        '''Unscales the data
        '''
        return u * self.std + self.mean

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float