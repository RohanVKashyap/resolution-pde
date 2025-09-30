import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os

import pandas as pd

def evaluate_1d_rollout_all_resolution(model, dataset_config, eval_dataset_target, current_res, test_resolutions, 
                                     data_resolution, pde, saved_folder, reduced_batch, reduced_resolution_t, 
                                     x_normalizer, y_normalizer, 
                                     batch_size=8, rollout_steps=10, time=1, 
                                     model_type='ffno1d', device='cuda', plot_examples=True, 
                                     save_dir=None):
    """
    Args:
        model: Trained model to evaluate
        dataset_config: Dataset configuration from hydra config
        eval_dataset_target: Target dataset for evaluation (e.g., single-res for multi-res models)
        current_res: Current/base resolution that the model was trained on
        test_resolutions: List of resolutions to test
        data_resolution: Maximum resolution available in the data
        pde: Type of PDE ('ks', 'burgers', 'advection', etc.)
        saved_folder: Folder containing the data
        reduced_batch: Batch reduction factor
        reduced_resolution_t: Temporal resolution reduction factor
        x_normalizer: Normalizer for input data from training
        y_normalizer: Normalizer for output data from training
        batch_size: Batch size for evaluation
        rollout_steps: Number of autoregressive steps to predict
        time: Time parameter for POS models
        model_type: Type of model ('ffno1d', 'pos', etc.)
        device: Computation device
        plot_examples: Whether to plot rollout examples
        save_dir: Directory to save results
        
    Returns:
        Dictionary with resolution factors as keys and rollout losses as values
    """
    from utils.loss import RelativeL2Loss
    
    model.eval()
    loss_fn = RelativeL2Loss(size_average=True, reduction=True)
    results = {}
    
    time_val = torch.tensor([time]).to(device) if model_type == 'pos' else None
    
    print(f"Evaluating {pde.upper()} model rollout at multiple resolutions")
    print(f"Using normalizer from training resolution {current_res}")
    print(f"Rollout steps: {rollout_steps}")
    print(f"Testing resolutions: {test_resolutions}")
    
    # Store examples for plotting
    plot_data = {}
    
    # Generate random indices for consistent plotting across all resolutions
    num_plot_examples = 5  # Fewer examples for rollout plots
    random_indices_plotting = None
    
    # Evaluate at each resolution
    for target_res in test_resolutions:
        try:
            # Calculate the downsample factor needed to get target_res from data_resolution
            if target_res > data_resolution:
                print(f"Warning: Target resolution {target_res} exceeds data resolution {data_resolution}. Skipping.")
                continue
                
            downsample_factor = data_resolution // target_res
            
            print(f"\n--- Evaluating Rollout at Resolution {target_res} (downsample factor: {downsample_factor}) ---")
            
            # Load dataset at target resolution using the same config logic as teacher-forcing evaluation
            eval_config = dataset_config.copy()

            # OVERRIDE THE TARGET FOR EVALUATION (for multi-resolution to default to single resolution)
            if eval_dataset_target:
                eval_config['dataset_params']['_target_'] = eval_dataset_target

                # Temporarily disable struct mode to add new keys
                OmegaConf.set_struct(eval_config['dataset_params'], False)
                
                # Override with evaluation-specific parameters (for true multi-resolution training)
                if 'eval_filename' in dataset_config['dataset_params']:
                    eval_config['dataset_params']['filename'] = dataset_config['dataset_params']['eval_filename']
                if 'eval_saved_folder' in dataset_config['dataset_params']:
                    eval_config['dataset_params']['saved_folder'] = dataset_config['dataset_params']['eval_saved_folder']

                if 'reduced_resolution' not in dataset_config['dataset_params']:
                    eval_config['dataset_params']['reduced_resolution'] = downsample_factor  # Use naive downsampling    
                
                # Re-enable struct mode
                OmegaConf.set_struct(eval_config['dataset_params'], True)

            eval_config['dataset_params']['reduced_batch'] = reduced_batch
            eval_config['dataset_params']['reduced_resolution_t'] = reduced_resolution_t
            eval_config['dataset_params']['reduced_resolution'] = downsample_factor  # Use naive downsampling
            eval_config['dataset_params']['data_normalizer'] = False   # No normalization - apply manually
            
            # Remove resize-related parameters if they exist
            if 's' in eval_config['dataset_params']:
                eval_config['dataset_params']['s'] = None
            
            # Load dataset - this returns the rollout_test_dataset as the 4th element
            print(f"Loading dataset with config: {eval_config['dataset_params']['_target_']}")
            dataset_output = instantiate(eval_config.dataset_params)
            
            # Handle different dataset return formats
            if len(dataset_output) >= 4:
                # Both multi-res and single-res now return rollout_test_dataset as 4th element
                _, _, _, rollout_test_dataset = dataset_output[:4]
                print(f"Using rollout test dataset with {len(rollout_test_dataset)} trajectories")
            else:
                print(f"Warning: Dataset does not return rollout dataset. Skipping resolution {target_res}")
                continue
            
            # Create trajectory loader
            trajectory_loader = DataLoader(rollout_test_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize random indices on first resolution
            if random_indices_plotting is None:
                total_samples = len(rollout_test_dataset)
                available_samples = min(total_samples, batch_size * 3)  # First few batches
                random_indices_plotting = torch.randint(0, available_samples, (num_plot_examples,)).tolist()
                print(f'Selected random indices for plotting: {random_indices_plotting}')
            
            # Evaluate model with rollout
            total_loss = 0.0
            num_batches = 0
            
            # For plotting - store rollout examples
            examples_collected = 0
            plot_data[target_res] = {
                'rollout_predictions': [],
                'targets': [],
                'initial_conditions': []
            }
            
            with torch.no_grad():
                for batch_idx, batch_trajectories in enumerate(trajectory_loader):
                    # batch_trajectories shape: (batch_size, time_steps, spatial_points)
                    batch_trajectories = batch_trajectories.to(device)
                    
                    # Check if we have enough timesteps for rollout
                    available_timesteps = batch_trajectories.shape[1]
                    actual_rollout_steps = min(rollout_steps, available_timesteps - 1)
                    
                    if actual_rollout_steps <= 0:
                        print(f"  Not enough timesteps ({available_timesteps}) for rollout")
                        continue
                    
                    # Extract initial condition (t=0)
                    initial_condition = batch_trajectories[:, 0, :]  # (batch, spatial)
                    
                    # Apply input normalization (same as teacher-forcing evaluation)
                    if x_normalizer is not None:
                        # Add channel dimension for normalizer: (batch, 1, spatial)
                        initial_condition_norm = x_normalizer.encode(initial_condition.unsqueeze(1)).squeeze(1)
                    else:
                        initial_condition_norm = initial_condition
                    
                    # Perform rollout
                    try:
                        rollout_predictions = perform_rollout_1d(
                            model=model,
                            initial_condition=initial_condition_norm,
                            rollout_steps=actual_rollout_steps,
                            model_type=model_type,
                            time_val=time_val,
                            device=device
                        )
                        
                        # Denormalize predictions (same as teacher-forcing evaluation)
                        if y_normalizer is not None:
                            # rollout_predictions shape: (batch, rollout_steps, spatial)
                            # Reshape for normalizer: (batch*rollout_steps, 1, spatial)
                            batch_size_curr, rollout_len, spatial_dim = rollout_predictions.shape
                            rollout_flat = rollout_predictions.reshape(-1, 1, spatial_dim)
                            rollout_denorm_flat = y_normalizer.decode(rollout_flat, device=device)
                            rollout_denorm = rollout_denorm_flat.reshape(batch_size_curr, rollout_len, spatial_dim)
                        else:
                            rollout_denorm = rollout_predictions
                        
                        # Get ground truth for comparison
                        ground_truth = batch_trajectories[:, 1:actual_rollout_steps+1, :]  # (batch, rollout_steps, spatial)
                        
                        # Calculate loss
                        batch_loss = loss_fn(rollout_denorm, ground_truth)
                        total_loss += batch_loss.item()
                        num_batches += 1
                        
                        # Store examples for plotting
                        if plot_examples and examples_collected < num_plot_examples:
                            batch_start_idx = batch_idx * batch_size
                            batch_end_idx = batch_start_idx + batch_trajectories.shape[0]
                            
                            for global_idx in random_indices_plotting:
                                if (global_idx >= batch_start_idx and 
                                    global_idx < batch_end_idx and 
                                    examples_collected < num_plot_examples):
                                    
                                    local_idx = global_idx - batch_start_idx
                                    
                                    plot_data[target_res]['rollout_predictions'].append(
                                        rollout_denorm[local_idx].cpu().numpy()
                                    )
                                    plot_data[target_res]['targets'].append(
                                        ground_truth[local_idx].cpu().numpy()
                                    )
                                    plot_data[target_res]['initial_conditions'].append(
                                        initial_condition[local_idx].cpu().numpy()
                                    )
                                    examples_collected += 1
                        
                    except Exception as e:
                        print(f"  Error during rollout for batch {batch_idx}: {e}")
                        continue
            
            # Convert lists to numpy arrays for easier handling
            if plot_examples and examples_collected > 0:
                plot_data[target_res]['rollout_predictions'] = np.array(plot_data[target_res]['rollout_predictions'])
                plot_data[target_res]['targets'] = np.array(plot_data[target_res]['targets'])
                plot_data[target_res]['initial_conditions'] = np.array(plot_data[target_res]['initial_conditions'])
                print(f"Collected {len(plot_data[target_res]['rollout_predictions'])} rollout examples for resolution {target_res}")
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                results[target_res] = avg_loss
                print(f"Resolution {target_res} (downsample factor: {downsample_factor}) - Rollout Loss ({actual_rollout_steps} steps): {avg_loss:.6f}")
            else:
                print(f"No valid batches processed for resolution {target_res}")
            
            # Clean up memory after each resolution (same as teacher-forcing evaluation)
            del rollout_test_dataset
            del trajectory_loader
            if 'batch_trajectories' in locals():
                del batch_trajectories, initial_condition, initial_condition_norm
            if 'rollout_predictions' in locals():
                del rollout_predictions, rollout_denorm, ground_truth
            torch.cuda.empty_cache()  # Clear GPU cache
            print(f"Memory cleaned after resolution {target_res}")
            
        except Exception as e:
            print(f"Error evaluating rollout at resolution {target_res}: {e}")
            import traceback
            traceback.print_exc()
            # Clean up memory even if there's an error
            torch.cuda.empty_cache()
            continue
    
    # Create subdirectories in save_dir
    if save_dir:
        rollout_plots_dir = os.path.join(save_dir, "rollout_plots")
        os.makedirs(rollout_plots_dir, exist_ok=True)
    
    # Plot rollout examples if requested
    if plot_examples and plot_data and save_dir:
        print("Creating rollout prediction plots...")
        plot_1d_rollout_examples_multiple(plot_data, test_resolutions, pde=pde, 
                                         save_dir=rollout_plots_dir, num_examples=num_plot_examples,
                                         rollout_steps=rollout_steps)
    
    # Save numerical results (same as teacher-forcing evaluation)
    if save_dir:
        save_rollout_results_csv(results, current_res, pde, save_dir, rollout_steps)
    
    # Print summary
    print("\n" + "="*60)
    print(f"{pde.upper()} ROLLOUT EVALUATION SUMMARY ({rollout_steps} steps)")
    print("="*60)
    for res, loss in results.items():
        print(f"Resolution {res:3d}: {loss:.6f}")
    
    return results

def perform_rollout_1d(model, initial_condition, rollout_steps, model_type='ffno1d', 
                           time_val=None, device='cuda', x_normalizer=None, y_normalizer=None):
    predictions = []
    current_state = initial_condition  # (batch, spatial) - already normalized
    
    for step in range(rollout_steps):
        model_input = current_state.unsqueeze(1)  # (batch, 1, spatial)
        
        if model_type == 'pos' and time_val is not None:
            next_state_norm = model(model_input, time_val)['output']
        else:
            next_state_norm = model(model_input)
        
        if next_state_norm.dim() == 3 and next_state_norm.shape[1] == 1:
            next_state_norm = next_state_norm.squeeze(1)
        
        predictions.append(next_state_norm.unsqueeze(1))
        
        # KEY FIX: Denormalize then renormalize for next step
        if y_normalizer is not None and x_normalizer is not None:
            next_state_denorm = y_normalizer.decode(next_state_norm.unsqueeze(1), device=device).squeeze(1)
            current_state = x_normalizer.encode(next_state_denorm.unsqueeze(1)).squeeze(1)
        else:
            current_state = next_state_norm
    
    return torch.cat(predictions, dim=1)

# def perform_rollout_1d(model, initial_condition, rollout_steps, model_type='ffno1d', 
#                       time_val=None, device='cuda'):
#     """
#     Perform autoregressive rollout for 1D equations.
    
#     Args:
#         model: Trained model
#         initial_condition: Shape (batch, spatial) - normalized
#         rollout_steps: Number of steps to predict
#         model_type: Type of model
#         time_val: Time parameter for POS models
#         device: Computing device
    
#     Returns:
#         predictions: Shape (batch, rollout_steps, spatial) - normalized (will be denormalized later)
#     """
#     predictions = []
#     current_state = initial_condition  # (batch, spatial)
    
#     for step in range(rollout_steps):
#         # Prepare input for model
#         if current_state.dim() == 2:  # (batch, spatial)
#             model_input = current_state.unsqueeze(1)  # Add channel: (batch, 1, spatial)
#         else:
#             model_input = current_state
        
#         # Model prediction
#         if model_type == 'pos' and time_val is not None:
#             next_state_norm = model(model_input, time_val)['output']
#         else:
#             next_state_norm = model(model_input)
        
#         # Handle model output shape
#         if next_state_norm.dim() == 3 and next_state_norm.shape[1] == 1:
#             next_state_norm = next_state_norm.squeeze(1)  # Remove channel: (batch, spatial)
        
#         predictions.append(next_state_norm.unsqueeze(1))  # Add time dimension: (batch, 1, spatial)
        
#         # Use prediction as input for next step (keep normalized)
#         current_state = next_state_norm
    
#     # Concatenate all predictions: (batch, rollout_steps, spatial)
#     return torch.cat(predictions, dim=1)

def plot_1d_rollout_examples_multiple(plot_data, test_resolutions, pde, save_dir, 
                                     num_examples=5, rollout_steps=10):
    """
    Plot rollout predictions vs targets for multiple resolutions.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(len(test_resolutions), num_examples, 
                           figsize=(4*num_examples, 3*len(test_resolutions)))
    if len(test_resolutions) == 1:
        axes = axes.reshape(1, -1)
    if num_examples == 1:
        axes = axes.reshape(-1, 1)
    
    for res_idx, resolution in enumerate(test_resolutions):
        if resolution not in plot_data:
            continue
            
        data = plot_data[resolution]
        
        for ex_idx in range(min(num_examples, len(data['rollout_predictions']))):
            ax = axes[res_idx, ex_idx]
            
            # Plot rollout prediction and target
            rollout_pred = data['rollout_predictions'][ex_idx]  # Shape: (rollout_steps, spatial)
            target = data['targets'][ex_idx]  # Shape: (rollout_steps, spatial)
            initial_cond = data['initial_conditions'][ex_idx]  # Shape: (spatial,)
            
            # Create spatial grid
            spatial_dim = rollout_pred.shape[-1]
            x_grid = np.linspace(0, 1, spatial_dim)
            
            # Plot initial condition
            ax.plot(x_grid, initial_cond, 'g-', linewidth=2, label='Initial (t=0)', alpha=0.8)
            
            # Plot final states
            ax.plot(x_grid, rollout_pred[-1], 'r--', linewidth=2, 
                   label=f'Predicted (t={rollout_steps})', alpha=0.8)
            ax.plot(x_grid, target[-1], 'b-', linewidth=2, 
                   label=f'Ground Truth (t={rollout_steps})', alpha=0.8)
            
            ax.set_title(f'Res {resolution}, Ex {ex_idx+1}')
            ax.grid(True, alpha=0.3)
            if ex_idx == 0:
                ax.set_ylabel(f'Resolution {resolution}')
            if res_idx == len(test_resolutions) - 1:
                ax.set_xlabel('Spatial Domain')
            if res_idx == 0 and ex_idx == 0:
                ax.legend()
    
    plt.suptitle(f'{pde.upper()} Rollout Predictions ({rollout_steps} steps)')
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f'{pde}_rollout_examples.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Rollout plots saved to: {save_path}")
    
    plt.show()

def save_rollout_results_csv(results, current_res, pde, save_dir, rollout_steps):
    """
    Save rollout evaluation results to CSV file.
    """
    import pandas as pd
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {'Resolution': res, 'Rollout_Loss': loss, 'Rollout_Steps': rollout_steps}
        for res, loss in results.items()
    ])
    
    # Save to CSV
    csv_path = os.path.join(save_dir, f'{pde}_rollout_results_trained_res_{current_res}_steps_{rollout_steps}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Rollout results saved to: {csv_path}")
    
    return csv_path