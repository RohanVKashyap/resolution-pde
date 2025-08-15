import matplotlib.pyplot as plt
import numpy as np
import operator
from functools import reduce
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.utils.rnn import pad_sequence
from collections.abc import Iterable

from models.custom_layer import UnitGaussianNormalizer
from dataloaders.burger_naive_markov import burger_markov_dataset
from dataloaders.ns_naive_markov import ns_markov_dataset
from dataloaders.burger_s4 import burger_window_dataset
from dataloaders.ns_s4 import ns_window_dataset
from dataloaders.darcy_loader import get_darcy_dataset

# class RelativeL2Loss(nn.Module):
#     def __init__(self, size_average=True, reduction=True):
#         """
#         Relative L2 Loss: ||x-y||₂/||y||₂
        
#         Args:
#             size_average (bool): If True, returns the mean of relative errors.
#                                 If False, returns the sum of relative errors.
#             reduction (bool): If False, returns individual errors without reduction.
#         """
#         super(RelativeL2Loss, self).__init__()
#         self.size_average = size_average
#         self.reduction = reduction
    
#     def forward(self, x, y):
#         """
#         Compute the relative L2 error between predictions and targets.
        
#         Args:
#             x (torch.Tensor): Prediction tensor
#             y (torch.Tensor): Target tensor
            
#         Returns:
#             torch.Tensor: Relative L2 error
#         """
#         num_examples = x.size()[0]
        
#         # Compute L2 norm of the difference (p=2)
#         diff_norms = torch.norm(
#             x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)
        
#         # Compute L2 norm of the target
#         y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)
        
#         # Add small epsilon to avoid division by zero
#         eps = 1e-8
#         rel_errors = diff_norms / (y_norms + eps)
        
#         if self.reduction:
#             if self.size_average:
#                 return torch.mean(rel_errors)
#             else:
#                 return torch.sum(rel_errors)
#         return rel_errors

def plot_predictions(batch_x, batch_y_denorm, batch_pred_denorm, pde_type=None, max_examples=4, save_path=None):
    """
    Plot predictions for 1D or 2D data.
    
    Args:
        batch_x: Input tensor (batch_size, channels, ...)
        batch_y_denorm: Denormalized ground truth (batch_size, channels, ...)
        batch_pred_denorm: Denormalized predictions (batch_size, channels, ...)
        pde_type: String identifier of the PDE type (optional)
        max_examples: Maximum number of examples to plot
        save_path: Path to save the figure (optional)
    
    Returns:
        fig: The created matplotlib figure
    """
    
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    batch_x = to_numpy(batch_x)
    batch_y_denorm = to_numpy(batch_y_denorm)
    batch_pred_denorm = to_numpy(batch_pred_denorm)
    
    num_examples = min(max_examples, batch_x.shape[0])
    
    # Check if data is 1D or 2D
    is_1d_data = len(batch_x[0, 0].shape) == 1
    
    if is_1d_data:
        # 1D data plotting (for Burger's equation)
        fig, axs = plt.subplots(num_examples, 1, figsize=(15, 4*num_examples))
        if num_examples == 1:
            axs = [axs] 
            
        for i in range(num_examples):
            x_axis = np.linspace(0, 2*np.pi, batch_x[i, 0].shape[0])
            axs[i].plot(x_axis, batch_x[i, 0], label='Input', linewidth=2)
            axs[i].plot(x_axis, batch_y_denorm[i, 0], label='Ground Truth', linewidth=2)
            axs[i].plot(x_axis, batch_pred_denorm[i, 0], label='Prediction', linewidth=2, linestyle='--')
            axs[i].set_title(f'Example {i+1}')
            axs[i].legend()
            axs[i].grid(True)
            
            if isinstance(batch_pred_denorm, torch.Tensor):
                error = torch.norm(batch_pred_denorm[i] - batch_y_denorm[i]) / torch.norm(batch_y_denorm[i])
                error_val = error.item()
            else:
                error = np.linalg.norm(batch_pred_denorm[i] - batch_y_denorm[i]) / np.linalg.norm(batch_y_denorm[i])
                error_val = error
                
            axs[i].set_xlabel(f'Rel. L2 Error: {error_val:.4f}')

    else:
        # 2D data plotting (for Darcy flow)
        fig, axs = plt.subplots(num_examples, 3, figsize=(15, 4*num_examples))
        if num_examples == 1:
            axs = [axs] 
            
        for i in range(num_examples):
            # Input
            im0 = axs[i, 0].imshow(batch_x[i, 0])
            axs[i, 0].set_title(f'Input {i+1}')
            fig.colorbar(im0, ax=axs[i, 0])
            
            # Ground Truth
            im1 = axs[i, 1].imshow(batch_y_denorm[i, 0])
            axs[i, 1].set_title(f'Ground Truth {i+1}')
            fig.colorbar(im1, ax=axs[i, 1])
            
            # Prediction
            im2 = axs[i, 2].imshow(batch_pred_denorm[i, 0])
            axs[i, 2].set_title(f'Prediction {i+1}')
            fig.colorbar(im2, ax=axs[i, 2])
            
            # Calculate relative L2 error
            if isinstance(batch_pred_denorm, torch.Tensor):
                error = torch.norm(batch_pred_denorm[i] - batch_y_denorm[i]) / torch.norm(batch_y_denorm[i])
                error_val = error.item()
            else:
                error = np.linalg.norm(batch_pred_denorm[i] - batch_y_denorm[i]) / np.linalg.norm(batch_y_denorm[i])
                error_val = error
                
            axs[i, 2].set_xlabel(f'Rel. L2 Error: {error_val:.4f}')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig   

def evaluate_super_resolution(model, test_loader, y_normalizer, t_resolutions, device='cuda'):
    """
    Evaluates model performance on test data at different spatial resolutions.
    
    Args:
        y_normalizer: Original normalizer for output data
        t_resolutions: List of downsampling factors to test
        
    Returns:
        Dictionary with resolution factors as keys and relative L2 losses as values
    """
    model.eval()
    loss_fn = RelativeL2Loss(size_average=True)
    results = {}
    
    all_x = []
    all_y = []
    
    with torch.no_grad():
        for x, y in test_loader:
            all_x.append(x)
            all_y.append(y)
    
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    print(f"Original test data shape: x={all_x.shape}, y={all_y.shape}")
    
    for res in t_resolutions:
        try:
            print(f"\nEvaluating at resolution factor: {res}")
            
            if res == 1:
                # For res=1, use original data
                downsampled_x = all_x
                downsampled_y = all_y
                
                # Create a DataLoader for the original data
                downsampled_dataset = TensorDataset(downsampled_x, downsampled_y)
                downsampled_loader = DataLoader(downsampled_dataset, batch_size=64, shuffle=False)
                
                # For res=1, use the original workflow without creating new normalizers
                total_loss = 0.0
                num_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in downsampled_loader:
                        # Move to device - no re-normalization needed
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        
                        # Forward pass
                        batch_pred = model(batch_x)
                        
                        # Denormalize with original normalizer
                        if y_normalizer is not None:
                            batch_pred = y_normalizer.decode(batch_pred, device=device)
                            batch_y = y_normalizer.decode(batch_y, device=device)
                        
                        # Calculate loss
                        loss = loss_fn(batch_pred, batch_y)
                        total_loss += loss.item()
                        num_batches += 1
                
                avg_loss = total_loss / num_batches
                
            else:
                # For 1D data
                if len(all_x.shape) == 3:  # [batch, channels, sequence_length]
                    # Downsample along spatial dimension
                    downsampled_x = all_x[:, :, ::res]
                    downsampled_y = all_y[:, :, ::res]
                # For 2D data
                elif len(all_x.shape) == 4:  # [batch, channels, height, width]
                    # Downsample along both spatial dimensions
                    downsampled_x = all_x[:, :, ::res, ::res]
                    downsampled_y = all_y[:, :, ::res, ::res]
                
                print(f"Downsampled shapes: x={downsampled_x.shape}, y={downsampled_y.shape}")
                
                # Create new normalizers specifically for this resolution
                # This ensures the statistics match the tensor shapes
                res_x_normalizer = UnitGaussianNormalizer(downsampled_x)
                res_y_normalizer = UnitGaussianNormalizer(downsampled_y)
                
                # Create a DataLoader for the downsampled data
                downsampled_dataset = TensorDataset(downsampled_x, downsampled_y)
                downsampled_loader = DataLoader(downsampled_dataset, batch_size=64, shuffle=False)
                
                # Evaluate model on downsampled data
                total_loss = 0.0
                num_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in downsampled_loader:
                        # Normalize inputs using resolution-specific normalizer
                        batch_x_normalized = res_x_normalizer.encode(batch_x)
                        
                        # Move to device
                        batch_x_normalized = batch_x_normalized.to(device)
                        batch_y = batch_y.to(device)
                        
                        # Forward pass
                        batch_pred_normalized = model(batch_x_normalized)
                        
                        # Denormalize with resolution-specific normalizer
                        batch_pred = res_y_normalizer.decode(batch_pred_normalized, device=device)
                        
                        # Calculate loss
                        loss = loss_fn(batch_pred, batch_y)
                        total_loss += loss.item()
                        num_batches += 1
                
                avg_loss = total_loss / num_batches
            
            results[res] = avg_loss
            print(f"Resolution factor {res} - Relative L2 Loss: {avg_loss:.6f}")

        except:
            print(f'Skipping Resolution factor {res} because not enough modes')
            continue    
    
    return results

def evaluate_higher_resolution(model, current_res, pde, data_path, reduced_batch, reduced_resolution_t, device='cuda'):
    """
    Evaluates model performance on test data at higher resolution than the current resolution.
    
    Args:
        model: Trained model to evaluate
        current_res: Current/base resolution (default: 64)
        pde: Type of PDE ('burger', 'darcy', or 'navier')
        device: Computation device ('cuda' or 'cpu')
        
    Returns:
        Dictionary with resolution factors as keys and relative L2 losses as values
    """
    model.eval()
    loss_fn = RelativeL2Loss(size_average=True)
    results = {}
    
    # Load appropriate dataset based on PDE type
    if 'burger' in pde:
        _, _, test_dataset, x_normalizer, y_normalizer = burger_markov_dataset(
                                            filename=data_path,
                                            saved_folder='.', 
                                            reduced_batch=reduced_batch, 
                                            reduced_resolution=1,
                                            reduced_resolution_t=reduced_resolution_t, 
                                            num_samples_max=-1)
    elif 'darcy' in pde:
        _, _, test_dataset, x_normalizer, y_normalizer = get_darcy_dataset(
                                    filename=data_path,
                                    saved_folder='.',
                                    reduced_resolution=reduced_batch,
                                    reduced_batch=1,
                                    num_samples_max=-1)
    elif 'navier' in pde:
        train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer = ns_markov_dataset(
                                                filename=data_path,
                                                saved_folder = '.', 
                                                reduced_batch=reduced_batch, 
                                                reduced_resolution=1, 
                                                reduced_resolution_t=reduced_resolution_t, 
                                                num_samples_max=-1)  
    
    # Create test loader for original data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Collect all test data
    all_x = []
    all_y = []
    
    with torch.no_grad():
        for x, y in test_loader:
            all_x.append(x)
            all_y.append(y)
    
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    print(f"Original test data shape: x={all_x.shape}, y={all_y.shape}")
    
    # Get original resolution
    original_res = all_x.shape[-1]  # Assuming this is 1024

    # Generate list of resolutions to test (upsampling from current_res)
    high_res = []
    test_res = current_res * 2  
    while test_res <= original_res:
        high_res.append(test_res)
        test_res = test_res * 2
    
    print(f"Testing resolutions: {high_res}")
    
    # Now evaluate at each resolution
    for target_res in high_res:
        try:
            print(f"\nEvaluating at resolution: {target_res}")
            
            # Calculate downsampling factor from original to target resolution
            target_factor = original_res // target_res
            
            # Downsample original data to target resolution
            if len(all_x.shape) == 3:  # 1D data
                downsampled_x = all_x[:, :, ::target_factor]
                downsampled_y = all_y[:, :, ::target_factor]
            elif len(all_x.shape) == 4:  # 2D data
                downsampled_x = all_x[:, :, ::target_factor, ::target_factor]
                downsampled_y = all_y[:, :, ::target_factor, ::target_factor]
            
            print(f"Downsampled shapes: x={downsampled_x.shape}, y={downsampled_y.shape}")
            
            # Create new normalizers specifically for this resolution
            res_x_normalizer = UnitGaussianNormalizer(downsampled_x)
            res_y_normalizer = UnitGaussianNormalizer(downsampled_y)
            
            # Create a DataLoader for the downsampled data
            downsampled_dataset = TensorDataset(downsampled_x, downsampled_y)
            downsampled_loader = DataLoader(downsampled_dataset, batch_size=64, shuffle=False)
            
            # Evaluate model on downsampled data
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in downsampled_loader:
                    # Normalize inputs using resolution-specific normalizer
                    batch_x_normalized = res_x_normalizer.encode(batch_x)
                    
                    # Move to device
                    batch_x_normalized = batch_x_normalized.to(device)
                    batch_y = batch_y.to(device)
                    
                    # Forward pass
                    batch_pred_normalized = model(batch_x_normalized)
                    
                    # Denormalize with resolution-specific normalizer
                    batch_pred = res_y_normalizer.decode(batch_pred_normalized, device=device)
                    
                    # Calculate loss
                    loss = loss_fn(batch_pred, batch_y)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches
            results[target_res] = avg_loss
            print(f"Resolution {target_res} - Relative L2 Loss: {avg_loss:.6f}")
            
        except Exception as e:
            print(f"Error evaluating resolution {target_res}: {e}")
            continue
    
    return results

def evaluate_s4_higher_resolution(model, current_res, pde, data_path, reduced_batch, window_size, reduced_resolution_t, device='cuda'):
    """
    Evaluates model performance on test data at higher resolution than the current resolution.
    
    Args:
        model: Trained model to evaluate
        current_res: Current/base resolution (default: 64)
        pde: Type of PDE ('burger', 'darcy', or 'navier')
        device: Computation device ('cuda' or 'cpu')
        
    Returns:
        Dictionary with resolution factors as keys and relative L2 losses as values
    """
    model.eval()
    loss_fn = RelativeL2Loss(size_average=True)
    results = {}
    
    print('--------Evaluating Higher Resolution------------')
    
    # Load appropriate dataset based on PDE type
    if 'burger' in pde:
        _, _, test_dataset, x_normalizer, y_normalizer = burger_window_dataset(
                                            filename=data_path,
                                            saved_folder = '.', 
                                            reduced_batch=reduced_batch, 
                                            reduced_resolution=1, 
                                            reduced_resolution_t=reduced_resolution_t,
                                            window_size=window_size, 
                                            num_samples_max=-1)

    
    elif 'navier' in pde:   
       train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer = ns_window_dataset(
                                                filename=data_path,
                                                saved_folder='.',
                                                reduced_batch=reduced_batch, 
                                                reduced_resolution=1,
                                                reduced_resolution_t=reduced_resolution_t,
                                                window_size=window_size, 
                                                flatten_window=True,           # Important for S4-FFNO 
                                                data_normalizer=False)    
    
    # Create test loader for original data  
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Collect all test data
    all_x = []
    all_y = []
    
    with torch.no_grad():
        for x, y in test_loader:
            all_x.append(x)
            all_y.append(y)
            break

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    print(f"Original test data shape: x={all_x.shape}, y={all_y.shape}")
    
    # Get original resolution
    original_res = all_x.shape[-1]  # Assuming this is 1024

    # Generate list of resolutions to test (upsampling from current_res)
    high_res = []
    test_res = current_res * 2  
    while test_res <= original_res:
        high_res.append(test_res)
        test_res = test_res * 2
    
    print(f"Testing resolutions: {high_res}")
    
    # Now evaluate at each resolution
    for target_res in high_res:
        try:
            print(f"\nEvaluating at resolution: {target_res}")
            
            # Calculate downsampling factor from original to target resolution
            target_factor = original_res // target_res
            
            # Downsample original data to target resolution
            if len(all_x.shape) == 3:  # 1D data
                downsampled_x = all_x[:, :, ::target_factor]
                downsampled_y = all_y[:, :, ::target_factor]
            elif len(all_x.shape) == 4:  # 2D data
                downsampled_x = all_x[:, :, ::target_factor, ::target_factor]
                downsampled_y = all_y[:, :, ::target_factor, ::target_factor]
            
            print(f"Downsampled shapes: x={downsampled_x.shape}, y={downsampled_y.shape}")
            
            # Create new normalizers specifically for this resolution
            res_x_normalizer = UnitGaussianNormalizer(downsampled_x)
            res_y_normalizer = UnitGaussianNormalizer(downsampled_y)
            
            # Create a DataLoader for the downsampled data
            downsampled_dataset = TensorDataset(downsampled_x, downsampled_y)
            downsampled_loader = DataLoader(downsampled_dataset, batch_size=64, shuffle=False)
            
            # Evaluate model on downsampled data
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in downsampled_loader:
                    # Normalize inputs using resolution-specific normalizer
                    batch_x_normalized = res_x_normalizer.encode(batch_x)
                    
                    # Move to device
                    batch_x_normalized = batch_x_normalized.to(device)
                    batch_y = batch_y.to(device)
                    
                    # Forward pass
                    batch_pred_normalized = model(batch_x_normalized)
                    
                    # Denormalize with resolution-specific normalizer
                    batch_pred = res_y_normalizer.decode(batch_pred_normalized, device=device)
                    
                    # Calculate loss
                    loss = loss_fn(batch_pred, batch_y)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches
            results[target_res] = avg_loss
            print(f"Resolution {target_res} - Relative L2 Loss: {avg_loss:.6f}")
            
        except Exception as e:
            print(f"Error evaluating resolution {target_res}: {e}")
            continue
    
    return results

def is_iterable(obj):
    return not isinstance(obj, str) and isinstance(obj, Iterable)   


################################################################
#  POSEIDON 
################################################################


def evaluate_pos_super_resolution(model, test_loader, y_normalizer, t_resolutions, time=1, device='cuda'):
    """
    Evaluates model performance on test data at different spatial resolutions.
    
    Args:
        y_normalizer: Original normalizer for output data
        t_resolutions: List of downsampling factors to test
        
    Returns:
        Dictionary with resolution factors as keys and relative L2 losses as values
    """
    model.eval()
    loss_fn = RelativeL2Loss(size_average=True)
    results = {}
    
    all_x = []
    all_y = []
    
    time_val = torch.tensor([time])
    
    with torch.no_grad():
        for x, y in test_loader:
            all_x.append(x)
            all_y.append(y)
    
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    print(f"Original test data shape: x={all_x.shape}, y={all_y.shape}")
    
    for res in t_resolutions:
        try:
            print(f"\nEvaluating at resolution factor: {res}")
            
            if res == 1:
                # For res=1, use original data
                downsampled_x = all_x
                downsampled_y = all_y
                
                # Create a DataLoader for the original data
                downsampled_dataset = TensorDataset(downsampled_x, downsampled_y)
                downsampled_loader = DataLoader(downsampled_dataset, batch_size=64, shuffle=False)
                
                # For res=1, use the original workflow without creating new normalizers
                total_loss = 0.0
                num_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in downsampled_loader:
                        # Move to device - no re-normalization needed
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        
                        # Forward pass
                        batch_pred = model(batch_x, time_val)['output']
                        
                        # Denormalize with original normalizer
                        if y_normalizer is not None:
                            batch_pred = y_normalizer.decode(batch_pred, device=device)
                            batch_y = y_normalizer.decode(batch_y, device=device)
                        
                        # Calculate loss
                        loss = loss_fn(batch_pred, batch_y)
                        total_loss += loss.item()
                        num_batches += 1
                
                avg_loss = total_loss / num_batches
                
            else:
                # For 1D data
                if len(all_x.shape) == 3:  # [batch, channels, sequence_length]
                    # Downsample along spatial dimension
                    downsampled_x = all_x[:, :, ::res]
                    downsampled_y = all_y[:, :, ::res]
                # For 2D data
                elif len(all_x.shape) == 4:  # [batch, channels, height, width]
                    # Downsample along both spatial dimensions
                    downsampled_x = all_x[:, :, ::res, ::res]
                    downsampled_y = all_y[:, :, ::res, ::res]
                
                print(f"Downsampled shapes: x={downsampled_x.shape}, y={downsampled_y.shape}")
                
                # Create new normalizers specifically for this resolution
                # This ensures the statistics match the tensor shapes
                res_x_normalizer = UnitGaussianNormalizer(downsampled_x)
                res_y_normalizer = UnitGaussianNormalizer(downsampled_y)
                
                # Create a DataLoader for the downsampled data
                downsampled_dataset = TensorDataset(downsampled_x, downsampled_y)
                downsampled_loader = DataLoader(downsampled_dataset, batch_size=64, shuffle=False)
                
                # Evaluate model on downsampled data
                total_loss = 0.0
                num_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in downsampled_loader:
                        # Normalize inputs using resolution-specific normalizer
                        batch_x_normalized = res_x_normalizer.encode(batch_x)
                        
                        # Move to device
                        batch_x_normalized = batch_x_normalized.to(device)
                        batch_y = batch_y.to(device)
                        
                        # Forward pass
                        batch_pred_normalized = model(batch_x_normalized, time_val)['output']
                        
                        # Denormalize with resolution-specific normalizer
                        batch_pred = res_y_normalizer.decode(batch_pred_normalized, device=device)
                        
                        # Calculate loss
                        loss = loss_fn(batch_pred, batch_y)
                        total_loss += loss.item()
                        num_batches += 1
                
                avg_loss = total_loss / num_batches
            
            results[res] = avg_loss
            print(f"Resolution factor {res} - Relative L2 Loss: {avg_loss:.6f}")

        except:
            print(f'Skipping Resolution factor {res} because not enough modes')
            continue    
    
    return results

def evaluate_pos_higher_resolution(model, current_res, pde, data_path, reduced_batch, reduced_resolution_t, time=1, device='cuda'):
    """
    Evaluates model performance on test data at higher resolution than the current resolution.
    
    Args:
        model: Trained model to evaluate
        current_res: Current/base resolution (default: 64)
        pde: Type of PDE ('burger', 'darcy', or 'navier')
        device: Computation device ('cuda' or 'cpu')
        
    Returns:
        Dictionary with resolution factors as keys and relative L2 losses as values
    """
    model.eval()
    loss_fn = RelativeL2Loss(size_average=True)
    results = {}
    
    # Load appropriate dataset based on PDE type
    if 'burger' in pde:
        _, _, test_dataset, x_normalizer, y_normalizer = burger_markov_dataset(
                                            filename=data_path,
                                            saved_folder='.', 
                                            reduced_batch=reduced_batch, 
                                            reduced_resolution=1,
                                            reduced_resolution_t=reduced_resolution_t, 
                                            num_samples_max=-1)
    elif 'darcy' in pde:
        _, _, test_dataset, x_normalizer, y_normalizer = get_darcy_dataset(
                                    filename=data_path,
                                    saved_folder='.',
                                    reduced_resolution=reduced_batch,
                                    reduced_batch=1,
                                    num_samples_max=-1)
    elif 'navier' in pde:
        train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer = ns_markov_dataset(
                                                filename=data_path,
                                                saved_folder = '.', 
                                                reduced_batch=reduced_batch, 
                                                reduced_resolution=1, 
                                                reduced_resolution_t=reduced_resolution_t, 
                                                num_samples_max=-1)  
    
    # Create test loader for original data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Collect all test data
    all_x = []
    all_y = []

    time_val = torch.tensor([time])
    
    with torch.no_grad():
        for x, y in test_loader:
            all_x.append(x)
            all_y.append(y)
    
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    print(f"Original test data shape: x={all_x.shape}, y={all_y.shape}")
    
    # Get original resolution
    original_res = all_x.shape[-1]  # Assuming this is 1024

    # Generate list of resolutions to test (upsampling from current_res)
    high_res = []
    test_res = current_res * 2  
    while test_res <= original_res:
        high_res.append(test_res)
        test_res = test_res * 2
    
    print(f"Testing resolutions: {high_res}")
    
    # Now evaluate at each resolution
    for target_res in high_res:
        try:
            print(f"\nEvaluating at resolution: {target_res}")
            
            # Calculate downsampling factor from original to target resolution
            target_factor = original_res // target_res
            
            # Downsample original data to target resolution
            if len(all_x.shape) == 3:  # 1D data
                downsampled_x = all_x[:, :, ::target_factor]
                downsampled_y = all_y[:, :, ::target_factor]
            elif len(all_x.shape) == 4:  # 2D data
                downsampled_x = all_x[:, :, ::target_factor, ::target_factor]
                downsampled_y = all_y[:, :, ::target_factor, ::target_factor]
            
            print(f"Downsampled shapes: x={downsampled_x.shape}, y={downsampled_y.shape}")
            
            # Create new normalizers specifically for this resolution
            res_x_normalizer = UnitGaussianNormalizer(downsampled_x)
            res_y_normalizer = UnitGaussianNormalizer(downsampled_y)
            
            # Create a DataLoader for the downsampled data
            downsampled_dataset = TensorDataset(downsampled_x, downsampled_y)
            downsampled_loader = DataLoader(downsampled_dataset, batch_size=64, shuffle=False)
            
            # Evaluate model on downsampled data
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in downsampled_loader:
                    # Normalize inputs using resolution-specific normalizer
                    batch_x_normalized = res_x_normalizer.encode(batch_x)
                    
                    # Move to device
                    batch_x_normalized = batch_x_normalized.to(device)
                    batch_y = batch_y.to(device)
                    
                    # Forward pass
                    batch_pred_normalized = model(batch_x_normalized, time_val)['output']
                    
                    # Denormalize with resolution-specific normalizer
                    batch_pred = res_y_normalizer.decode(batch_pred_normalized, device=device)
                    
                    # Calculate loss
                    loss = loss_fn(batch_pred, batch_y)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches
            results[target_res] = avg_loss
            print(f"Resolution {target_res} - Relative L2 Loss: {avg_loss:.6f}")
            
        except Exception as e:
            print(f"Error evaluating resolution {target_res}: {e}")
            continue
    
    return results
