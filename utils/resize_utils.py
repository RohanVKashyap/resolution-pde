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
from omegaconf import OmegaConf

from utils.res_utils import downsample, resize, downsample_1d, resize_1d
from utils.plot_utils import plot_1d_pde_examples_multiple, plot_1d_pde_examples_compact, perform_frequency_analysis_1d_generic, save_numerical_results_generic
from utils.plot_utils import perform_frequency_analysis, analyze_resize_frequencies, plot_frequency_retention_bar, plot_spatial_retention_bar
from utils.plot_utils import plot_energy_analysis, plot_frequency_summary, create_multi_resolution_frequency_comparison, save_numerical_results
from utils.plot_utils import plot_navier_stokes_examples, plot_single_channel, plot_multi_channel, create_error_plot, create_individual_plots, create_combined_channel_plot, plot_2d_pde_examples_multiple

from hydra.utils import instantiate
from utils.loss import RelativeL2Loss

def get_lower_resolutions(base_resolution, min_resolution=32):
    """
    Generate a list of lower resolutions derived by dividing the base resolution by powers of 2.

    Args:
        base_resolution (int): The original high resolution (e.g., 128, 256).
        min_resolution (int): Smallest resolution to consider (e.g., 32).

    Returns:
        List[int]: List of lower resolutions (e.g., [32, 64] from 128).
    """
    resolutions = []
    res = base_resolution // 2
    while res >= min_resolution:
        resolutions.insert(0, res)
        res = res // 2
    return resolutions + [base_resolution]    
    
################################################################
#  CNO 2D
################################################################    

def evaluate_cno_original_2d_all_resolution(model, dataset_config, eval_dataset_target, current_res, test_resolutions, data_resolution, pde, 
                                   saved_folder, reduced_batch, reduced_resolution_t, 
                                   normalization_type='minmax',
                                   min_data=None, max_data=None, min_model=None, max_model=None,
                                   x_normalizer=None, y_normalizer=None,
                                   resize_to_train=True,  # New parameter to control resizing behavior
                                   batch_size=8, time=1, 
                                   model_type='ffno2d', device='cuda', plot_examples=True, 
                                   save_dir=None):
    """
    GENERIC evaluation function for ANY 2D PDE at multiple resolutions.
    Supports both min-max and simple normalization, and can optionally use resizing.
    No frequency analysis for 2D version.
    
    Args:
        model: Trained model to evaluate
        dataset_config: Dataset configuration
        eval_dataset_target: Target dataset for evaluation
        current_res: Current/base resolution that the model was trained on
        test_resolutions: List of resolutions to test
        data_resolution: Maximum resolution available in the data
        pde: Type of PDE ('ns', 'darcy', 'shallow_water', etc.) - for labeling only
        saved_folder: Folder containing the data
        reduced_batch: Batch reduction factor
        reduced_resolution_t: Temporal resolution reduction factor
        normalization_type: Type of normalization - 'minmax' or 'simple'
        min_data, max_data: Input normalization statistics for min-max normalization
        min_model, max_model: Output normalization statistics for min-max normalization
        x_normalizer: Input normalizer for simple normalization
        y_normalizer: Output normalizer for simple normalization
        resize_to_train: If True, resize inputs to training resolution and outputs back to test resolution.
                        If False, pass inputs directly to model at test resolution (like evaluate_2d_all_resolution)
        batch_size: Batch size for evaluation
        time: Time parameter for POS models
        model_type: Type of model ('ffno2d', 'pos', etc.)
        device: Computation device ('cuda' or 'cpu')
        plot_examples: Whether to plot example predictions vs targets
        save_dir: Directory to save images locally (optional)
        
    Returns:
        Dictionary with resolution factors as keys and relative L2 losses as values
    """
    model.eval()
    loss_fn = RelativeL2Loss(size_average=True, reduction=True)
    results = {}

    time_val = torch.tensor([time])      # For POS models
    
    print(f"Evaluating 2D {pde.upper()} model at multiple resolutions")
    print(f"Using {normalization_type} normalization from training resolution {current_res}")
    print(f"Resize mode: {'ON - resize to training resolution' if resize_to_train else 'OFF - direct evaluation at test resolution'}")
    
    if normalization_type == 'minmax':
        print(f"  Input range: [{min_data:.6f}, {max_data:.6f}]")
        print(f"  Output range: [{min_model:.6f}, {max_model:.6f}]")
    else:
        print(f"  Using provided normalizers from training")
    
    print(f"Testing resolutions: {test_resolutions}")
    
    # Store examples for plotting - MODIFIED for 10 examples
    plot_data = {}
    
    # Generate 10 random indices for consistent plotting across all resolutions
    num_plot_examples = 10
    # We'll determine these after loading the first dataset to ensure valid indices
    random_indices_plotting = None
    
    # Define normalization/denormalization functions based on type
    if normalization_type == 'minmax':
        def normalize_input(data):
            return (data - min_data) / (max_data - min_data)
        
        def denormalize_output(data):
            return data * (max_model - min_model) + min_model
            
    elif normalization_type == 'simple':
        def normalize_input(data):
            if x_normalizer is not None:
                return x_normalizer.encode(data)
            else:
                return data
        
        def denormalize_output(data):
            if y_normalizer is not None:
                return y_normalizer.decode(data, device=device)
            else:
                return data
    else:
        raise ValueError(f"Invalid normalization_type: {normalization_type}. Must be 'minmax' or 'simple'")
    
    # Evaluate at each resolution
    for target_res in test_resolutions:
        try:
            # Load dataset at target resolution (GENERIC - works with any dataset config)
            eval_config = dataset_config.copy()

            # OVERRIDE THE TARGET FOR EVALUATION (THIS IS ONLY FOR MULTI-RESOLUTION TO DEFAULT TO SINGLE RESOLUTION)
            if eval_dataset_target:
                eval_config['dataset_params']['_target_'] = eval_dataset_target

                # Temporarily disable struct mode to add new keys
                OmegaConf.set_struct(eval_config['dataset_params'], False)
                
                # ADD: Override with evaluation-specific parameters (THIS IS ONLY FOR TRUE MULTI-RESOLUTION TRAINING)
                if 'eval_filename' in dataset_config['dataset_params']:
                    eval_config['dataset_params']['filename'] = dataset_config['dataset_params']['eval_filename']
                    
                if 'eval_saved_folder' in dataset_config['dataset_params']:
                    eval_config['dataset_params']['saved_folder'] = dataset_config['dataset_params']['eval_saved_folder']

                if 'reduced_resolution' not in dataset_config['dataset_params']:
                    eval_config['dataset_params']['reduced_resolution'] = 1

                if 's' not in dataset_config['dataset_params']:
                    eval_config['dataset_params']['s'] = target_res
                
                # Re-enable struct mode
                OmegaConf.set_struct(eval_config['dataset_params'], True)

            eval_config['dataset_params']['reduced_batch'] = reduced_batch
            eval_config['dataset_params']['reduced_resolution_t'] = reduced_resolution_t
            eval_config['dataset_params']['reduced_resolution'] = 1
            eval_config['dataset_params']['s'] = target_res            # Load at test resolution
            eval_config['dataset_params']['data_normalizer'] = False   # No normalization - apply manually
            
            # GENERIC dataset loading - works with any dataset function that returns the standard format
            if normalization_type == 'minmax':
                _, _, original_test_dataset, _, _, _, _ = instantiate(eval_config.dataset_params)
            else:  # simple
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
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                    # Handle both 2-element and 3-element batches (GENERIC)
                    if len(batch_data) == 3:
                        batch_x, batch_y, _ = batch_data  # Ignore spatial coordinates
                    else:
                        batch_x, batch_y = batch_data
                    
                    # Normalize inputs using TRAINING statistics
                    batch_x_normalized = normalize_input(batch_x)   # Test Resolution
                    
                    # Move to device
                    batch_x_normalized = batch_x_normalized.to(device)
                    batch_y = batch_y.to(device)

                    if resize_to_train:
                        # CNO RESIZE APPROACH: Resize to training resolution, forward pass, resize back
                        # Resize to training resolution using resize function
                        batch_x_resized = resize(batch_x_normalized, (current_res, current_res))  # Test -> Train Resolution
                        
                        # Forward pass at training resolution
                        if model_type == 'pos':
                            batch_pred_normalized = model(batch_x_resized, time_val)['output']
                        else:     
                            batch_pred_normalized = model(batch_x_resized)

                        # Resize predictions back to target test resolution
                        batch_pred_normalized_final = resize(batch_pred_normalized, (target_res, target_res))  # Train -> Test Resolution
                        
                    else:
                        # DIRECT APPROACH: Forward pass directly at test resolution (like evaluate_2d_all_resolution)
                        if model_type == 'pos':
                            batch_pred_normalized_final = model(batch_x_normalized, time_val)['output']
                        else:     
                            batch_pred_normalized_final = model(batch_x_normalized)
                    
                    # Denormalize predictions using TRAINING statistics
                    batch_pred = denormalize_output(batch_pred_normalized_final)
                    
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
                                    batch_x[local_idx].cpu().numpy()
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
            
            if resize_to_train:
                print(f"Resolution {target_res}x{target_res} (with resize to {current_res}x{current_res}) - Relative L2 Loss: {avg_loss:.6f}")
            else:
                print(f"Resolution {target_res}x{target_res} (direct evaluation) - Relative L2 Loss: {avg_loss:.6f}")

            # Clean up memory after each resolution 
            del original_test_dataset
            del test_loader
            if 'batch_x' in locals():
                del batch_x, batch_y, batch_x_normalized, batch_pred_normalized_final, batch_pred
                if resize_to_train and 'batch_x_resized' in locals():
                    del batch_x_resized, batch_pred_normalized
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
        os.makedirs(prediction_plots_dir, exist_ok=True)
    
    # Plot examples if requested - MODIFIED to handle 10 examples for 2D data
    if plot_examples and plot_data:
        print("Creating 2D prediction vs target plots...")
        plot_2d_pde_examples_multiple(plot_data, test_resolutions, pde=pde, save_dir=prediction_plots_dir, num_examples=num_plot_examples)
    
    # Save numerical results to a CSV file
    if save_dir:
        save_numerical_results_generic(results, current_res, pde, save_dir)
    
    # Print summary
    print("\n" + "="*50)
    print(f"2D {pde.upper()} EVALUATION SUMMARY")
    print("="*50)
    for res, loss in results.items():
        print(f"Resolution {res:3d}x{res:3d}: {loss:.6f}")
    
    return results

# def evaluate_cno_original_all_resolution(model, dataset_config, current_res, test_resolutions, data_resolution, pde, saved_folder, reduced_batch, reduced_resolution_t, 
#                                 reduced_resolution, min_data, max_data, min_model, max_model, batch_size=8, time=1, model_type='ffno2d', device='cuda', plot_examples=True, 
#                                 save_dir=None, analyze_frequencies=True):
#     """
#     Evaluates model performance on test data at multiple resolutions using resize function.
#     Now includes frequency analysis and saves everything to the figures directory.
    
#     Args:
#         model: Trained model to evaluate
#         current_res: Current/base resolution that the model was trained on
#         test_resolutions: List of resolutions to test
#         data_resolution: Maximum resolution available in the data
#         pde: Type of PDE ('burger', 'darcy', or 'navier')
#         data_path: Path to the data file
#         saved_folder: Folder containing the data
#         reduced_batch: Batch reduction factor
#         reduced_resolution_t: Temporal resolution reduction factor
#         reduced_resolution: 1
#         min_data, max_data: Input normalization statistics from training
#         min_model, max_model: Output normalization statistics from training
#         batch_size: Batch size for evaluation
#         time: Time parameter for POS models
#         model_type: Type of model ('ffno2d', 'pos', etc.)
#         device: Computation device ('cuda' or 'cpu')
#         plot_examples: Whether to plot example predictions vs targets
#         save_dir: Directory to save images locally (optional)
#         analyze_frequencies: Whether to perform frequency analysis (default: True)
        
#     Returns:
#         Dictionary with resolution factors as keys and relative L2 losses as values
#     """
#     model.eval()
#     loss_fn = RelativeL2Loss(size_average=True, reduction=True)
#     results = {}

#     time_val = torch.tensor([time])      # POS
    
#     # Load training dataset with current_res to get normalization statistics
#     print(f"Using provided normalization statistics from training resolution {current_res}:")
#     print(f"  Input range: [{min_data:.6f}, {max_data:.6f}]")
#     print(f"  Output range: [{min_model:.6f}, {max_model:.6f}]")
    
#     # Define resolutions to test
#     test_resolutions = [res for res in test_resolutions if res <= data_resolution]
    
#     print(f"Testing resolutions: {test_resolutions}")
    
#     # Store examples for plotting if Navier-Stokes
#     plot_data = {}
    
#     # Store data for frequency analysis
#     frequency_analysis_data = {}
    
#     # Evaluate at each resolution
#     for target_res in test_resolutions:
#         try:
#             if 'navier' in pde:
#                 # _, _, original_test_dataset, _, _, _, _ = cno_ns_markov_dataset(
#                 #     filename=data_path,
#                 #     saved_folder=saved_folder, 
#                 #     reduced_batch=reduced_batch, 
#                 #     reduced_resolution=1,    
#                 #     reduced_resolution_t=reduced_resolution_t, 
#                 #     num_samples_max=-1,
#                 #     s=target_res,          # Load at test resolution [32, 64, 128, 256, 512]
#                 #     data_normalizer=False,  # No normalization - we will apply it manually
#                 # )
#                 eval_config = dataset_config.copy()

#                 eval_config['dataset_params']['reduced_batch'] = reduced_batch
#                 eval_config['dataset_params']['reduced_resolution_t'] = reduced_resolution_t
#                 eval_config['dataset_params']['reduced_resolution'] = reduced_resolution
#                 eval_config['dataset_params']['s'] = target_res            # Load at test resolution [32, 64, 128, 256, 512]
#                 eval_config['dataset_params']['data_normalizer'] = False   # No normalization - we will apply it manually
                
#                 _, _, original_test_dataset, _, _, _, _ = instantiate(eval_config.dataset_params)
                
#             else:
#                 # Handle other PDEs similarly - you'll need to modify their dataset functions
#                 raise NotImplementedError(f"PDE type {pde} not implemented with new normalization")
            
#             # Apply normalization using training statistics
#             def normalize_data(data, min_val, max_val):
#                 return (data - min_val) / (max_val - min_val)
            
#             def denormalize_data(data, min_val, max_val):
#                 return data * (max_val - min_val) + min_val
            
#             # Create test loader for original data
#             resized_loader = DataLoader(original_test_dataset, batch_size=batch_size, shuffle=False)

#             # Evaluate model on resized data
#             total_loss = 0.0
#             num_batches = 0
            
#             # For plotting - store one random example
#             example_stored = False
            
#             # For frequency analysis - store sample data
#             frequency_sample_stored = False
            
#             with torch.no_grad():
#                 for batch_idx, (batch_x, batch_y) in enumerate(resized_loader):
#                     # Normalize inputs using TRAINING statistics
#                     batch_x_normalized = normalize_data(batch_x, min_data, max_data)   # Test Resolution
                    
#                     # Move to device
#                     batch_x_normalized = batch_x_normalized.to(device)
#                     batch_y = batch_y.to(device)

#                     # Store sample for frequency analysis (before resize)
#                     if analyze_frequencies and not frequency_sample_stored:
#                         frequency_analysis_data[target_res] = {
#                             'input_data': batch_x_normalized[0:1].cpu(),  # Take first sample
#                             'target_resolution': current_res
#                         }
#                         frequency_sample_stored = True

#                     batch_x_normalized = resize(batch_x_normalized, (current_res, current_res))  # Train Resolution
                    
#                     # Forward pass
#                     if model_type == 'pos':
#                         batch_pred_normalized = model(batch_x_normalized, time_val)['output']
#                     else:     
#                         batch_pred_normalized = model(batch_x_normalized)

#                     # Resize predictions back to target test resolution
#                     batch_pred_normalized_resized = resize(batch_pred_normalized, (target_res, target_res))
                    
#                     # Denormalize predictions using TRAINING statistics
#                     batch_pred = denormalize_data(batch_pred_normalized_resized, min_model, max_model)
                    
#                     # Store example for plotting (only for first batch and if Navier-Stokes)
#                     if plot_examples and 'navier' in pde and not example_stored:
#                         # Choose a random sample from the batch
#                         random_idx = torch.randint(0, batch_pred.shape[0], (1,)).item()
                        
#                         plot_data[target_res] = {
#                             'prediction': batch_pred[random_idx].cpu().numpy(),  # Shape: (1, H, W)
#                             'target': batch_y[random_idx].cpu().numpy(),        # Shape: (1, H, W)
#                             'input': batch_x[random_idx].cpu().numpy()          # Shape: (1, H, W)
#                         }
#                         example_stored = True
                    
#                     # Calculate loss on denormalized data
#                     loss = loss_fn(batch_pred, batch_y) 
#                     total_loss += loss.item()
#                     num_batches += 1
            
#             avg_loss = total_loss / num_batches
#             results[target_res] = avg_loss
#             print(f"Resolution {target_res} - Relative L2 Loss: {avg_loss:.6f}")

#             # Clean up memory after each resolution 
#             del original_test_dataset
#             del resized_loader
#             if 'batch_x' in locals():
#                 del batch_x, batch_y, batch_x_normalized, batch_pred_normalized, batch_pred_normalized_resized, batch_pred
#             torch.cuda.empty_cache()  # Clear GPU cache
#             print(f"Memory cleaned after resolution {target_res}")
            
#         except Exception as e:
#             print(f"Error evaluating resolution {target_res}: {e}")
#             import traceback
#             traceback.print_exc()
#             # Clean up memory even if there's an error
#             if 'original_test_dataset' in locals():
#                 del original_test_dataset
#             if 'resized_loader' in locals():
#                 del resized_loader
#             torch.cuda.empty_cache()
#             continue
    
#     # Create subdirectories in save_dir
#     if save_dir:
#         prediction_plots_dir = os.path.join(save_dir, "prediction_plots")
#         frequency_analysis_dir = os.path.join(save_dir, "frequency_analysis")
#         os.makedirs(prediction_plots_dir, exist_ok=True)
#         os.makedirs(frequency_analysis_dir, exist_ok=True)
    
#     # Plot examples if requested and Navier-Stokes
#     if plot_examples and 'navier' in pde and plot_data:
#         print("Creating prediction vs target plots...")
#         plot_navier_stokes_examples(plot_data, test_resolutions, save_dir=prediction_plots_dir)
    
#     # Perform frequency analysis if requested
#     if analyze_frequencies and frequency_analysis_data and save_dir:
#         print("Performing frequency analysis...")
#         perform_frequency_analysis(frequency_analysis_data, current_res, frequency_analysis_dir)
    
#     # Save numerical results to a CSV file
#     if save_dir:
#         save_numerical_results(results, current_res, save_dir)
    
#     # Print summary
#     print("\n" + "="*50)
#     print("EVALUATION SUMMARY")
#     print("="*50)
#     for res, loss in results.items():
#         print(f"Resolution {res:3d}: {loss:.6f}")
    
#     return results

############################################################################
# Resize 1D Function Evaluation   
#############################################################################

def evaluate_cno_original_1d_all_resolution(model, dataset_config, eval_dataset_target, current_res, test_resolutions, data_resolution, pde, 
                                   saved_folder, reduced_batch, reduced_resolution_t, 
                                   normalization_type='minmax',
                                   min_data=None, max_data=None, min_model=None, max_model=None,
                                   x_normalizer=None, y_normalizer=None,
                                   resize_to_train=True,  # New parameter to control resizing behavior
                                   batch_size=8, time=1, 
                                   model_type='ffno1d', device='cuda', plot_examples=True, 
                                   save_dir=None, analyze_frequencies=True):
    """
    GENERIC evaluation function for ANY 1D PDE at multiple resolutions.
    Supports both min-max and simple normalization, and can optionally use resizing.
    
    Args:
        model: Trained model to evaluate
        dataset_config: Dataset configuration
        eval_dataset_target: Target dataset for evaluation
        current_res: Current/base resolution that the model was trained on
        test_resolutions: List of resolutions to test
        data_resolution: Maximum resolution available in the data
        pde: Type of PDE ('ks', 'burgers', 'advection', etc.) - for labeling only
        saved_folder: Folder containing the data
        reduced_batch: Batch reduction factor
        reduced_resolution_t: Temporal resolution reduction factor
        normalization_type: Type of normalization - 'minmax' or 'simple'
        min_data, max_data: Input normalization statistics for min-max normalization
        min_model, max_model: Output normalization statistics for min-max normalization
        x_normalizer: Input normalizer for simple normalization
        y_normalizer: Output normalizer for simple normalization
        resize_to_train: If True, resize inputs to training resolution and outputs back to test resolution.
                        If False, pass inputs directly to model at test resolution (like evaluate_1d_all_resolution)
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
    print(f"Using {normalization_type} normalization from training resolution {current_res}")
    print(f"Resize mode: {'ON - resize to training resolution' if resize_to_train else 'OFF - direct evaluation at test resolution'}")
    
    if normalization_type == 'minmax':
        print(f"  Input range: [{min_data:.6f}, {max_data:.6f}]")
        print(f"  Output range: [{min_model:.6f}, {max_model:.6f}]")
    else:
        print(f"  Using provided normalizers from training")
    
    print(f"Testing resolutions: {test_resolutions}")
    
    # Store examples for plotting - MODIFIED for 10 examples
    plot_data = {}
    
    # Store data for frequency analysis
    frequency_analysis_data = {}
    
    # Generate 10 random indices for consistent plotting across all resolutions
    num_plot_examples = 10
    # We'll determine these after loading the first dataset to ensure valid indices
    random_indices_plotting = None
    
    # Define normalization/denormalization functions based on type
    if normalization_type == 'minmax':
        def normalize_input(data):
            return (data - min_data) / (max_data - min_data)
        
        def denormalize_output(data):
            return data * (max_model - min_model) + min_model
            
    elif normalization_type == 'simple':
        def normalize_input(data):
            if x_normalizer is not None:
                return x_normalizer.encode(data)
            else:
                return data
        
        def denormalize_output(data):
            if y_normalizer is not None:
                return y_normalizer.decode(data, device=device)
            else:
                return data
    else:
        raise ValueError(f"Invalid normalization_type: {normalization_type}. Must be 'minmax' or 'simple'")
    
    # Evaluate at each resolution
    for target_res in test_resolutions:
        try:
            # Load dataset at target resolution (GENERIC - works with any dataset config)
            eval_config = dataset_config.copy()

            # OVERRIDE THE TARGET FOR EVALUATION (THIS IS ONLY FOR MULTI-RESOLUTION TO DEFAULT TO SINGLE RESOLUTION)
            if eval_dataset_target:
                eval_config['dataset_params']['_target_'] = eval_dataset_target

                # Temporarily disable struct mode to add new keys
                OmegaConf.set_struct(eval_config['dataset_params'], False)
                
                # ADD: Override with evaluation-specific parameters (THIS IS ONLY FOR TRUE MULTI-RESOLUTION TRAINING)
                if 'eval_filename' in dataset_config['dataset_params']:
                    eval_config['dataset_params']['filename'] = dataset_config['dataset_params']['eval_filename']
                    
                if 'eval_saved_folder' in dataset_config['dataset_params']:
                    eval_config['dataset_params']['saved_folder'] = dataset_config['dataset_params']['eval_saved_folder']

                if 'reduced_resolution' not in dataset_config['dataset_params']:
                    eval_config['dataset_params']['reduced_resolution'] = 1

                if 's' not in dataset_config['dataset_params']:
                    eval_config['dataset_params']['s'] = target_res
                
                # Re-enable struct mode
                OmegaConf.set_struct(eval_config['dataset_params'], True)

            eval_config['dataset_params']['reduced_batch'] = reduced_batch
            eval_config['dataset_params']['reduced_resolution_t'] = reduced_resolution_t
            eval_config['dataset_params']['reduced_resolution'] = 1
            eval_config['dataset_params']['s'] = target_res            # Load at test resolution
            eval_config['dataset_params']['data_normalizer'] = False   # No normalization - apply manually
            
            # GENERIC dataset loading - works with any dataset function that returns the standard format
            if normalization_type == 'minmax':
                _, _, original_test_dataset, _, _, _, _ = instantiate(eval_config.dataset_params)
            else:  # simple
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
                    
                    # Normalize inputs using TRAINING statistics
                    batch_x_normalized = normalize_input(batch_x)   # Test Resolution
                    
                    # Move to device
                    batch_x_normalized = batch_x_normalized.to(device)
                    batch_y = batch_y.to(device)

                    # Store sample for frequency analysis (before any potential resize)
                    if analyze_frequencies and not frequency_sample_stored:
                        frequency_analysis_data[target_res] = {
                            'input_data': batch_x_normalized[0:1].cpu(),  # Take first sample
                            'target_resolution': current_res if resize_to_train else target_res
                        }
                        frequency_sample_stored = True

                    if resize_to_train:
                        # CNO RESIZE APPROACH: Resize to training resolution, forward pass, resize back
                        # Resize to training resolution using 1D resize
                        batch_x_resized = resize_1d(batch_x_normalized, current_res)  # Test -> Train Resolution
                        
                        # Forward pass at training resolution
                        if model_type == 'pos':
                            batch_pred_normalized = model(batch_x_resized, time_val)['output']
                        else:     
                            batch_pred_normalized = model(batch_x_resized)

                        # Resize predictions back to target test resolution
                        batch_pred_normalized_final = resize_1d(batch_pred_normalized, target_res)  # Train -> Test Resolution
                        
                    else:
                        # DIRECT APPROACH: Forward pass directly at test resolution (like evaluate_1d_all_resolution)
                        if model_type == 'pos':
                            batch_pred_normalized_final = model(batch_x_normalized, time_val)['output']
                        else:     
                            batch_pred_normalized_final = model(batch_x_normalized)
                    
                    # Denormalize predictions using TRAINING statistics
                    batch_pred = denormalize_output(batch_pred_normalized_final)
                    
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
                                    batch_x[local_idx].cpu().numpy()
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
            
            if resize_to_train:
                print(f"Resolution {target_res} (with resize to {current_res}) - Relative L2 Loss: {avg_loss:.6f}")
            else:
                print(f"Resolution {target_res} (direct evaluation) - Relative L2 Loss: {avg_loss:.6f}")

            # Clean up memory after each resolution 
            del original_test_dataset
            del test_loader
            if 'batch_x' in locals():
                del batch_x, batch_y, batch_x_normalized, batch_pred_normalized_final, batch_pred
                if resize_to_train and 'batch_x_resized' in locals():
                    del batch_x_resized, batch_pred_normalized
            torch.cuda.empty_cache()  # Clear GPU cache
            print(f"Memory cleaned after resolution {target_res}")
            
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


# def evaluate_s4_higher_resolution(model, current_res, pde, data_path, reduced_batch, window_size, reduced_resolution_t, device='cuda'):
#     """
#     Evaluates model performance on test data at higher resolution than the current resolution.
    
#     Args:
#         model: Trained model to evaluate
#         current_res: Current/base resolution (default: 64)
#         pde: Type of PDE ('burger', 'darcy', or 'navier')
#         device: Computation device ('cuda' or 'cpu')
        
#     Returns:
#         Dictionary with resolution factors as keys and relative L2 losses as values
#     """
#     model.eval()
#     loss_fn = RelativeL2Loss(size_average=True)
#     results = {}
    
#     print('--------Evaluating Higher Resolution------------')
    
#     # Load appropriate dataset based on PDE type
#     if 'burger' in pde:
#         _, _, test_dataset, x_normalizer, y_normalizer = burger_window_dataset(
#                                             filename=data_path,
#                                             saved_folder = '.', 
#                                             reduced_batch=reduced_batch, 
#                                             reduced_resolution=1, 
#                                             reduced_resolution_t=reduced_resolution_t,
#                                             window_size=window_size, 
#                                             num_samples_max=-1)

    
#     elif 'navier' in pde:   
#        train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer = ns_window_dataset(
#                                                 filename=data_path,
#                                                 saved_folder='.',
#                                                 reduced_batch=reduced_batch, 
#                                                 reduced_resolution=1,
#                                                 reduced_resolution_t=reduced_resolution_t,
#                                                 window_size=window_size, 
#                                                 flatten_window=True,           # Important for S4-FFNO 
#                                                 data_normalizer=False)    
    
#     # Create test loader for original data  
#     test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
#     # Collect all test data
#     all_x = []
#     all_y = []
    
#     with torch.no_grad():
#         for x, y in test_loader:
#             all_x.append(x)
#             all_y.append(y)
#             break

#     all_x = torch.cat(all_x, dim=0)
#     all_y = torch.cat(all_y, dim=0)
    
#     print(f"Original test data shape: x={all_x.shape}, y={all_y.shape}")
    
#     # Get original resolution
#     original_res = all_x.shape[-1]  # Assuming this is 1024

#     # Generate list of resolutions to test (upsampling from current_res)
#     high_res = []
#     test_res = current_res * 2  
#     while test_res <= original_res:
#         high_res.append(test_res)
#         test_res = test_res * 2
    
#     print(f"Testing resolutions: {high_res}")
    
#     # Now evaluate at each resolution
#     for target_res in high_res:
#         try:
#             print(f"\nEvaluating at resolution: {target_res}")
            
#             # Calculate downsampling factor from original to target resolution
#             target_factor = original_res // target_res
            
#             # Downsample original data to target resolution
#             if len(all_x.shape) == 3:  # 1D data
#                 downsampled_x = all_x[:, :, ::target_factor]
#                 downsampled_y = all_y[:, :, ::target_factor]
#             elif len(all_x.shape) == 4:  # 2D data
#                 downsampled_x = all_x[:, :, ::target_factor, ::target_factor]
#                 downsampled_y = all_y[:, :, ::target_factor, ::target_factor]
            
#             print(f"Downsampled shapes: x={downsampled_x.shape}, y={downsampled_y.shape}")
            
#             # Create new normalizers specifically for this resolution
#             res_x_normalizer = UnitGaussianNormalizer(downsampled_x)
#             res_y_normalizer = UnitGaussianNormalizer(downsampled_y)
            
#             # Create a DataLoader for the downsampled data
#             downsampled_dataset = TensorDataset(downsampled_x, downsampled_y)
#             downsampled_loader = DataLoader(downsampled_dataset, batch_size=64, shuffle=False)
            
#             # Evaluate model on downsampled data
#             total_loss = 0.0
#             num_batches = 0
            
#             with torch.no_grad():
#                 for batch_x, batch_y in downsampled_loader:
#                     # Normalize inputs using resolution-specific normalizer
#                     batch_x_normalized = res_x_normalizer.encode(batch_x)
                    
#                     # Move to device
#                     batch_x_normalized = batch_x_normalized.to(device)
#                     batch_y = batch_y.to(device)
                    
#                     # Forward pass
#                     batch_pred_normalized = model(batch_x_normalized)
                    
#                     # Denormalize with resolution-specific normalizer
#                     batch_pred = res_y_normalizer.decode(batch_pred_normalized, device=device)
                    
#                     # Calculate loss
#                     loss = loss_fn(batch_pred, batch_y)
#                     total_loss += loss.item()
#                     num_batches += 1
            
#             avg_loss = total_loss / num_batches
#             results[target_res] = avg_loss
#             print(f"Resolution {target_res} - Relative L2 Loss: {avg_loss:.6f}")
            
#         except Exception as e:
#             print(f"Error evaluating resolution {target_res}: {e}")
#             continue
    
#     return results

# def is_iterable(obj):
#     return not isinstance(obj, str) and isinstance(obj, Iterable)  