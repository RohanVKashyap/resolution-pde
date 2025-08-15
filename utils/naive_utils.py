import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy
import os
from datetime import datetime
import operator
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.utils.rnn import pad_sequence
from collections.abc import Iterable

# from models.custom_layer import UnitGaussianNormalizer
from hydra.utils import instantiate
from utils.loss import RelativeL2Loss
from utils.plot_utils import plot_1d_pde_examples_multiple, plot_1d_pde_examples_compact, perform_frequency_analysis_1d_generic, save_numerical_results_generic

###################################
# NAIVE DOWNSAMPLING EVALUATION
###################################

def evaluate_1d_all_resolution(model, dataset_config, eval_dataset_target, current_res, test_resolutions, data_resolution, pde, 
                               saved_folder, reduced_batch, reduced_resolution_t, 
                               x_normalizer, y_normalizer, 
                               batch_size=8, time=1, 
                               model_type='ffno1d', device='cuda', plot_examples=True, 
                               save_dir=None, analyze_frequencies=True):
    """
    GENERIC evaluation function for ANY 1D PDE at multiple resolutions using naive downsampling.
    
    Args:
        model: Trained model to evaluate
        dataset_config: Dataset configuration
        current_res: Current/base resolution that the model was trained on
        test_resolutions: List of resolutions to test
        data_resolution: Maximum resolution available in the data
        saved_folder: Folder containing the data
        reduced_batch: Batch reduction factor
        reduced_resolution_t: Temporal resolution reduction factor
        x_normalizer: GaussianNormalizer for input data from training
        y_normalizer: GaussianNormalizer for output data from training
        pde: Type of PDE ('ks', 'burgers', 'advection', etc.) - for labeling only
        batch_size: Batch size for evaluation
        time: Time parameter for POS models
        model_type: Type of model ('ffno1d', 'pos', etc.)
        device: Computation device ('cuda' or 'cpu')
        plot_examples: Whether to plot example predictions vs targets
        save_dir: Directory to save images locally (optional)
        analyze_frequencies: Whether to perform frequency analysis (default: True)
        
    Returns:
        Dictionary with resolution factors as keys and relative L2 losses as values
    """
    model.eval()
    loss_fn = RelativeL2Loss(size_average=True, reduction=True)
    results = {}

    time_val = torch.tensor([time])      # For POS models
    
    print(f"Evaluating {pde.upper()} model at multiple resolutions")
    print(f"Using normalizer from training resolution {current_res}")
    
    print(f"Testing resolutions: {test_resolutions}")
    
    # Store examples for plotting - MODIFIED for 10 examples
    plot_data = {}
    
    # Store data for frequency analysis
    frequency_analysis_data = {}
    
    # Generate 10 random indices for consistent plotting across all resolutions
    num_plot_examples = 10
    # We'll determine these after loading the first dataset to ensure valid indices
    random_indices_plotting = None
    
    # Evaluate at each resolution
    for target_res in test_resolutions:
        try:
            # Calculate the reduced_resolution factor needed to get target_res from data_resolution
            if target_res > data_resolution:
                print(f"Warning: Target resolution {target_res} exceeds data resolution {data_resolution}. Skipping.")
                continue
                
            # Calculate how much to downsample: data_resolution -> target_res
            downsample_factor = data_resolution // target_res
            
            # Load dataset at target resolution using naive downsampling
            eval_config = dataset_config.copy()

            # OVERRIDE THE TARGET FOR EVALUATION (THIS IS ONLY FOR MULTI-RESOLUTION TO DEFAULT TO SINGLE RESOLUTION)
            if eval_dataset_target:
                eval_config['dataset_params']['_target_'] = eval_dataset_target

            eval_config['dataset_params']['reduced_batch'] = reduced_batch
            eval_config['dataset_params']['reduced_resolution_t'] = reduced_resolution_t
            eval_config['dataset_params']['reduced_resolution'] = downsample_factor  # Use naive downsampling
            eval_config['dataset_params']['data_normalizer'] = False   # No normalization - apply manually
            
            # Remove resize-related parameters if they exist
            if 's' in eval_config['dataset_params']:
                eval_config['dataset_params']['s'] = None
            
            # GENERIC dataset loading - works with any dataset function that returns the standard format
            _, _, original_test_dataset, _, _ = instantiate(eval_config.dataset_params)
            
            # Create test loader for original data
            test_loader = DataLoader(original_test_dataset, batch_size=batch_size, shuffle=False)

            # Initialize random indices on first resolution
            if random_indices_plotting is None:
                total_samples = len(original_test_dataset)
                random_indices_plotting = torch.randint(0, min(total_samples, batch_size * len(test_loader)), 
                                                       (num_plot_examples,)).tolist()
                print(f'Selected random indices for plotting: {random_indices_plotting}')

            # Evaluate model
            total_loss = 0.0
            num_batches = 0
            
            # For plotting - store examples from multiple batches
            examples_collected = 0
            plot_data[target_res] = {
                'predictions': [],
                'targets': [], 
                'inputs': []
            }
            
            # For frequency analysis - store sample data
            frequency_sample_stored = False
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                    # Handle both 2-element and 3-element batches (GENERIC)
                    if len(batch_data) == 3:
                        batch_x, batch_y, _ = batch_data  # Ignore spatial coordinates
                    else:
                        batch_x, batch_y = batch_data
                    
                    # Move to device
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    # Apply normalization using training statistics
                    if x_normalizer is not None:
                        batch_x_normalized = x_normalizer.encode(batch_x)
                    else:
                        batch_x_normalized = batch_x

                    # Store sample for frequency analysis
                    if analyze_frequencies and not frequency_sample_stored:
                        frequency_analysis_data[target_res] = {
                            'input_data': batch_x_normalized[0:1].cpu(),  # Take first sample
                            'target_resolution': current_res
                        }
                        frequency_sample_stored = True

                    # Forward pass (GENERIC - works with any model)
                    if model_type == 'pos':
                        batch_pred_normalized = model(batch_x_normalized, time_val)['output']
                    else:     
                        batch_pred_normalized = model(batch_x_normalized)

                    # Denormalize predictions using training statistics
                    if y_normalizer is not None:
                        batch_pred = y_normalizer.decode(batch_pred_normalized, device=device)
                    else:
                        batch_pred = batch_pred_normalized
                    
                    # Store examples for plotting - MODIFIED to collect 10 examples
                    if plot_examples and examples_collected < num_plot_examples:
                        # Calculate global indices for this batch
                        batch_start_idx = batch_idx * batch_size
                        batch_end_idx = batch_start_idx + batch_x.shape[0]
                        
                        # Check which of our target indices fall in this batch
                        for plot_idx, global_idx in enumerate(random_indices_plotting):
                            if (global_idx >= batch_start_idx and 
                                global_idx < batch_end_idx and 
                                examples_collected < num_plot_examples):
                                
                                # Convert global index to local batch index
                                local_idx = global_idx - batch_start_idx
                                
                                plot_data[target_res]['predictions'].append(
                                    batch_pred[local_idx].cpu().numpy()
                                )
                                plot_data[target_res]['targets'].append(
                                    batch_y[local_idx].cpu().numpy()
                                )
                                plot_data[target_res]['inputs'].append(
                                    batch_x[local_idx].cpu().numpy()  # Store unnormalized input
                                )
                                examples_collected += 1
                    
                    # Calculate loss on denormalized data (UNIVERSAL)
                    loss = loss_fn(batch_pred, batch_y) 
                    total_loss += loss.item()
                    num_batches += 1
            
            # Convert lists to numpy arrays for easier handling
            if plot_examples:
                plot_data[target_res]['predictions'] = np.array(plot_data[target_res]['predictions'])
                plot_data[target_res]['targets'] = np.array(plot_data[target_res]['targets'])
                plot_data[target_res]['inputs'] = np.array(plot_data[target_res]['inputs'])
                print(f"Collected {len(plot_data[target_res]['predictions'])} examples for resolution {target_res}")
            
            avg_loss = total_loss / num_batches
            results[target_res] = avg_loss
            print(f"Resolution {target_res} (downsample factor: {downsample_factor}) - Relative L2 Loss: {avg_loss:.6f}")

            # Clean up memory after each resolution 
            del original_test_dataset
            del test_loader
            if 'batch_x' in locals():
                del batch_x, batch_y, batch_x_normalized, batch_pred_normalized, batch_pred
            torch.cuda.empty_cache()  # Clear GPU cache
            print(f"Memory cleaned after resolution {target_res}")
            print('\n')
            
        except Exception as e:
            print(f"Error evaluating resolution {target_res}: {e}")
            import traceback
            traceback.print_exc()
            # Clean up memory even if there's an error
            if 'original_test_dataset' in locals():
                del original_test_dataset
            if 'test_loader' in locals():
                del test_loader
            torch.cuda.empty_cache()
            continue
    
    # Create subdirectories in save_dir
    if save_dir:
        prediction_plots_dir = os.path.join(save_dir, "prediction_plots")
        frequency_analysis_dir = os.path.join(save_dir, "frequency_analysis")
        os.makedirs(prediction_plots_dir, exist_ok=True)
        os.makedirs(frequency_analysis_dir, exist_ok=True)
    
    # Plot examples if requested - MODIFIED to handle 10 examples
    if plot_examples and plot_data:
        print("Creating prediction vs target plots...")
        plot_1d_pde_examples_multiple(plot_data, test_resolutions, pde=pde, save_dir=prediction_plots_dir, num_examples=num_plot_examples)
    
    # Perform frequency analysis if requested
    if analyze_frequencies and frequency_analysis_data and save_dir:
        print("Performing frequency analysis...")
        perform_frequency_analysis_1d_generic(frequency_analysis_data, current_res, pde, frequency_analysis_dir)
    
    # Save numerical results to a CSV file
    if save_dir:
        save_numerical_results_generic(results, current_res, pde, save_dir)
    
    # Print summary
    print("\n" + "="*50)
    print(f"{pde.upper()} EVALUATION SUMMARY")
    print("="*50)
    for res, loss in results.items():
        print(f"Resolution {res:3d}: {loss:.6f}")
    
    return results