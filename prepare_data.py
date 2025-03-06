import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys

# import neuralop
# from neuralop.data.datasets import load_darcy_flow_small
# from neuralop.layers.embeddings import GridEmbedding2D

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()
        # x shape: (batch, H, W, channels)
        self.mean = torch.mean(x, 0)     # Shape: (H, W, channels)
        self.std = torch.std(x, 0)       # Shape: (H, W, channels)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, device='cuda:0'):
        std = self.std + self.eps
        mean = self.mean

        std = std.to(device)
        mean = mean.to(device)
        
        x = (x * std) + mean
        return x

def load_burger_data_from_mat(data_path1, data_path2=None, batch_size=16, split=[0.8, 0.1, 0.1]):
    """
    Load and preprocess Burger's equation dataset from .mat file(s)
    Returns train, validation, and test data loaders with 80/10/10 split
    
    Args:
        data_path1: Path to the first Burger's equation data file
        data_path2: Path to the second Burger's equation data file (optional)
        batch_size: Batch size for DataLoaders
        split: Train/validation/test split ratios
    """
    if data_path2:
        print(f"Loading Burger's equation data from {data_path1} and {data_path2}")
        # Load data from both .mat files
        data1 = scipy.io.loadmat(data_path1)
        data2 = scipy.io.loadmat(data_path2)
        
        # Check for keys in both files
        print(f"Keys in data1: {[k for k in data1.keys() if not k.startswith('_')]}")
        print(f"Keys in data2: {[k for k in data2.keys() if not k.startswith('_')]}")
        
        # Extract data from both files
        if 'a' in data1 and 'u' in data1 and 'a' in data2 and 'u' in data2:
            X1 = data1['a']  # Initial condition from file 1
            y1 = data1['u']  # Solution from file 1
            X2 = data2['a']  # Initial condition from file 2
            y2 = data2['u']  # Solution from file 2
            
            # Combine data from both files
            X = np.vstack([X1, X2])
            y = np.vstack([y1, y2])
        else:
            raise ValueError("Expected 'a' and 'u' keys not found in .mat files")
    else:
        print(f"Loading Burger's equation data from {data_path1}")
        # Load data from a single .mat file
        data = scipy.io.loadmat(data_path1)
        
        print(f"Keys in data: {[k for k in data.keys() if not k.startswith('_')]}")
        
        if 'a' in data and 'u' in data:
            X = data['a']  # Initial condition
            y = data['u']  # Solution at later time
        else:
            raise ValueError("Expected 'a' and 'u' keys not found in .mat file")
    
    print(f"Loaded data shapes: X={X.shape}, y={y.shape}")
    
    # Convert to PyTorch tensors and add channel dimension
    X = torch.from_numpy(X).float().unsqueeze(1)  # (2048, 1, 8192)
    y = torch.from_numpy(y).float().unsqueeze(1)  # (2048, 1, 8192)
    
    # Split into train, validation, and test sets
    total_samples = X.shape[0]
    train_size = int(total_samples * split[0])
    val_size = int(total_samples * split[1])
    
    # Training set
    X_train, y_train = X[:train_size], y[:train_size]
    
    # Validation set
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    
    # Test set
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Normalize data using UnitGaussianNormalizer based on training set only
    X_normalizer = UnitGaussianNormalizer(X_train)
    y_normalizer = UnitGaussianNormalizer(y_train)
    
    # Apply normalization to all sets
    X_train = X_normalizer.encode(X_train)
    X_val = X_normalizer.encode(X_val)
    X_test = X_normalizer.encode(X_test)
    
    y_train = y_normalizer.encode(y_train)
    y_val = y_normalizer.encode(y_val)
    y_test = y_normalizer.encode(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_normalizer, y_normalizer    

# def load_burger_data_from_mat(data_path, batch_size=16, split=[0.8, 0.1, 0.1]):
#     """
#     Load and preprocess Burger's equation dataset from .mat file
#     Returns train, validation, and test data loaders with 80/10/10 split
#     """
#     print(f"Loading data from {data_path}")
    
#     data = scipy.io.loadmat(data_path)
    
#     print(f"Keys in data: {[k for k in data.keys() if not k.startswith('_')]}")
    
#     if 'a' in data and 'u' in data:
#         X = data['a']  # Initial condition
#         y = data['u']  # Solution at later time
#     else:
#         raise ValueError("Expected 'a' and 'u' keys not found in .mat file")
    
#     print(f"Loaded data shapes: X={X.shape}, y={y.shape}")
    
#     X = torch.from_numpy(X).float().unsqueeze(1)  # (2048, 1, 8192)
#     y = torch.from_numpy(y).float().unsqueeze(1)  # (2048, 1, 8192)

#     # grid = torch.tensor(np.linspace(0, 2*np.pi, X.shape[2]), dtype=torch.float)
    
#     # Split into train, validation, and test sets
#     total_samples = X.shape[0]
#     train_size = int(total_samples * split[0])
#     val_size = int(total_samples * split[1])
    
#     # Training set
#     X_train, y_train = X[:train_size], y[:train_size]
    
#     # Validation set
#     X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    
#     # Test set
#     X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
#     print(f"Training set: {X_train.shape[0]} samples")
#     print(f"Validation set: {X_val.shape[0]} samples")
#     print(f"Test set: {X_test.shape[0]} samples")
    
#     # Normalize data using UnitGaussianNormalizer based on training set only
#     X_normalizer = UnitGaussianNormalizer(X_train)
#     y_normalizer = UnitGaussianNormalizer(y_train)
    
#     # Apply normalization to all sets
#     X_train = X_normalizer.encode(X_train)
#     X_val = X_normalizer.encode(X_val)
#     X_test = X_normalizer.encode(X_test)
    
#     y_train = y_normalizer.encode(y_train)
#     y_val = y_normalizer.encode(y_val)
#     y_test = y_normalizer.encode(y_test)
    
#     # Create DataLoaders
#     train_dataset = TensorDataset(X_train, y_train)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
#     val_dataset = TensorDataset(X_val, y_val)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
#     test_dataset = TensorDataset(X_test, y_test)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     return train_loader, val_loader, test_loader, X_normalizer, y_normalizer  

def load_darcy_data_from_mat(data_path1, data_path2, batch_size=16):
    """
    Load and preprocess Darcy Flow dataset from .mat files
    Returns train, validation, and test data loaders with 80/10/10 split
    """
    print(f"Loading data from {data_path1} and {data_path2}")
    # Load data from .mat files
    data1 = scipy.io.loadmat(data_path1)
    data2 = scipy.io.loadmat(data_path2)

    if 'coeff' in data1 and 'sol' in data1:
        X1 = data1['coeff']
        y1 = data1['sol']
        X2 = data2['coeff']
        y2 = data2['sol']

    elif 'Kcoeff' in data1 and 'sol' in data1:
        X1 = data1['Kcoeff']
        y1 = data1['sol']
        X2 = data2['Kcoeff']
        y2 = data2['sol']  
    else:
        # List keys for debugging
        print(f"Keys in data1: {[k for k in data1.keys() if not k.startswith('_')]}")
        print(f"Keys in data2: {[k for k in data2.keys() if not k.startswith('_')]}")
        raise ValueError("Expected 'coeff' and 'sol' keys not found in .mat files")

    # Combine data from both files
    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])

    print(f"Loaded data shapes: X={X.shape}, y={y.shape}")

    # Convert to PyTorch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    # Add channel dimension if not present
    if len(X.shape) == 3:  # (batch, height, width)
        X = X.unsqueeze(1)  # (batch, 1, height, width)
    if len(y.shape) == 3:
        y = y.unsqueeze(1)

    # Split into train, validation, and test sets - 80/10/10 split
    total_samples = X.shape[0]
    train_size = int(total_samples * 0.8)
    val_size = int(total_samples * 0.1)
    
    # Training set (80%)
    X_train, y_train = X[:train_size], y[:train_size]
    
    # Validation set (10%)
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    
    # Test set (10%)
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Normalize data using UnitGaussianNormalizer
    # Note: We normalize based on training set statistics only
    X_normalizer = UnitGaussianNormalizer(X_train)
    y_normalizer = UnitGaussianNormalizer(y_train)

    # Apply normalization to all sets
    X_train = X_normalizer.encode(X_train)
    X_val = X_normalizer.encode(X_val)
    X_test = X_normalizer.encode(X_test)

    y_train = y_normalizer.encode(y_train)
    y_val = y_normalizer.encode(y_val)
    y_test = y_normalizer.encode(y_test)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_normalizer, y_normalizer

def load_darcy_data(batch_size=16, ntrain=9000, ntest=1000):
    """
    Load and preprocess Darcy Flow dataset
    """
    # Load data from .npy files
    x = np.load('2D_DarcyFlow_beta0.01/nu.npy')
    y = np.load('2D_DarcyFlow_beta0.01/tensor.npy')
    
    # Convert to PyTorch tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    
    # Ensure dimensions are (batch, channels, height, width)
    if len(x.shape) == 3:  # (batch, height, width)
        x = x.unsqueeze(1)  # Add channel dimension -> (batch, 1, height, width)
    if len(y.shape) == 3:
        y = y.unsqueeze(1) 
    
    # Split into train and test sets
    x_train, y_train = x[:ntrain], y[:ntrain]
    x_test, y_test = x[ntrain:ntrain+ntest], y[ntrain:ntrain+ntest]
    
    # Normalize data
    x_normalizer = UnitGaussianNormalizer(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)
    
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_train = y_normalizer.encode(y_train)
    y_test = y_normalizer.encode(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, x_normalizer, y_normalizer