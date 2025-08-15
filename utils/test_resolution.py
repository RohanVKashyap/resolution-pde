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
import os

from models.custom_layer import UnitGaussianNormalizer
from dataloaders.burger_naive_markov import burger_markov_dataset
from dataloaders.ns_naive_markov import navier_stokes_markov_dataset
from dataloaders.burger_s4 import burger_window_dataset
from dataloaders.ns_s4 import ns_window_dataset
from dataloaders.darcy_loader import get_darcy_dataset
from utils.utils import RelativeL2Loss

def test_higher_resolution(model, test_id, data_path, all_pde, model_type, current_res, reduced_batch, reduced_resolution_t, device='cuda'):
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
    for job_id, path, pde in zip(test_id, data_path, all_pde):
        checkpoint_path = os.path.join('./checkpoints', model_type, f'{pde}_{job_id}.pt')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        loss_fn = RelativeL2Loss(size_average=True)
        results = {}

        print('----------------------------------------------------------')
        print(f'Evaluating job_id:{job_id}')
        print('Loading data path:', path)
        print('Loading model path:', checkpoint_path)
        print('----------------------------------------------------------')
        
        # Load appropriate dataset based on PDE type
        if 'burger' in pde:
            _, _, test_dataset, x_normalizer, y_normalizer = burger_markov_dataset(
                                                filename=path,
                                                saved_folder='.', 
                                                reduced_batch=reduced_batch, 
                                                reduced_resolution=1,
                                                reduced_resolution_t=reduced_resolution_t, 
                                                num_samples_max=-1)
        elif 'darcy' in pde:
            _, _, test_dataset, x_normalizer, y_normalizer = get_darcy_dataset(
                                        filename=path,
                                        saved_folder='.',
                                        reduced_resolution=reduced_batch,
                                        reduced_batch=1,
                                        num_samples_max=-1)
        elif 'navier' in pde:
            _, _, test_dataset, x_normalizer, y_normalizer = navier_stokes_markov_dataset(
                                            filename=path,
                                            saved_folder='.', 
                                            reduced_batch=reduced_batch, 
                                            reduced_resolution=1, 
                                            reduced_resolution_t=reduced_resolution_t, 
                                            num_samples_max=-1)
        
        # Create test loader for original data
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
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
