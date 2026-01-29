import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os

from hydra.utils import instantiate
from utils.loss import RelativeL2Loss
from utils.frequency_error import decompose_error_by_frequency_1d
from scipy.ndimage import uniform_filter1d

def evaluate_multiresolution_training_analysis(model_checkpoints, model, dataset_config, eval_dataset_target,
                                              test_resolution, data_resolution, pde,
                                              saved_folder, reduced_batch, reduced_resolution_t,
                                              batch_size=8, time=1, model_type='ffno1d', 
                                              x_normalizer=None, y_normalizer=None,
                                              checkpoint_resolutions=None,
                                              device='cuda', save_dir=None):
    """
    Evaluate MULTIPLE models (trained with different alphas) on the SAME test dataset.
    Args:
        model_checkpoints: Dict like {'alpha_0.00': model1, 'alpha_0.05': model2, ...}
        test_resolution: Single resolution to test on (e.g., 256 for full resolution)
        ... other args same as before ...
    """
    
    print(f"Multi-Resolution Training Analysis")
    print(f"Testing all models on resolution: {test_resolution}")
    print(f"Number of models (alphas): {len(model_checkpoints)}")
    
    # Load test dataset ONCE (same for all models)
    eval_config = dataset_config.copy()
    
    if eval_dataset_target:
        eval_config['dataset_params']['_target_'] = eval_dataset_target
        OmegaConf.set_struct(eval_config['dataset_params'], False)
        
        if 'eval_filename' in dataset_config['dataset_params']:
            eval_config['dataset_params']['filename'] = dataset_config['dataset_params']['eval_filename']
        if 'eval_saved_folder' in dataset_config['dataset_params']:
            eval_config['dataset_params']['saved_folder'] = dataset_config['dataset_params']['eval_saved_folder']

        if 'reduced_resolution' not in dataset_config['dataset_params']:
            eval_config['dataset_params']['reduced_resolution'] = 1
        
        OmegaConf.set_struct(eval_config['dataset_params'], True)

    eval_config['dataset_params']['reduced_batch'] = reduced_batch
    eval_config['dataset_params']['reduced_resolution_t'] = reduced_resolution_t
    eval_config['dataset_params']['reduced_resolution'] = 1
    eval_config['dataset_params']['data_normalizer'] = False
    
    if 's' in eval_config['dataset_params']:
        eval_config['dataset_params']['s'] = None
    
    _, _, original_test_dataset, _, _, _ = instantiate(eval_config.dataset_params)
    test_loader = DataLoader(original_test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Test dataset size: {len(original_test_dataset)}")
    
    # Store results for each alpha
    frequency_data = {}
    loss_results = {}
    
    # Evaluate each model
    for alpha_name, checkpoint_path in model_checkpoints.items():
        print(f"\nEvaluating model: {alpha_name}")
        
        model.eval()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded Model Checkpoint: checkpoint_path")
        
        # Accumulate predictions and targets
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        loss_fn = RelativeL2Loss(size_average=True, reduction=True)
        time_val = torch.tensor([time])
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                # Handle both 2-element and 3-element batches
                if len(batch_data) == 3:
                    batch_x, batch_y, _ = batch_data
                else:
                    batch_x, batch_y = batch_data
                
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Apply normalization
                if x_normalizer is not None:
                    batch_x_normalized = x_normalizer.encode(batch_x)
                else:
                    batch_x_normalized = batch_x

                # Forward pass
                if model_type == 'pos':
                    batch_pred_normalized = model(batch_x_normalized, time_val)['output']
                else:     
                    batch_pred_normalized = model(batch_x_normalized)

                # Denormalize predictions
                if y_normalizer is not None:
                    batch_pred = y_normalizer.decode(batch_pred_normalized, device=device)
                else:
                    batch_pred = batch_pred_normalized
                
                # NEW: Accumulate for frequency analysis
                all_predictions.append(batch_pred.cpu())
                all_targets.append(batch_y.cpu())
                
                # Calculate loss
                loss = loss_fn(batch_pred, batch_y) 
                total_loss += loss.item()
                num_batches += 1
        
        # Store accumulated data
        frequency_data[alpha_name] = {
            'predictions': torch.cat(all_predictions, dim=0),
            'targets': torch.cat(all_targets, dim=0)
        }
        
        avg_loss = total_loss / num_batches
        loss_results[alpha_name] = avg_loss
        print(f"{alpha_name}: Resolution {test_resolution} Loss = {avg_loss:.6f}, Samples = {len(frequency_data[alpha_name]['predictions'])}")

        # Cleanup
        del all_predictions, all_targets
        torch.cuda.empty_cache()
    
    # Create frequency analysis plots
    if save_dir:
        frequency_dir = os.path.join(save_dir, "frequency_analysis")
        os.makedirs(frequency_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("CREATING FREQUENCY ANALYSIS PLOTS")
        print("="*60)
        
        create_multiresolution_training_plots(
            frequency_data, 
            alpha_names=list(model_checkpoints.keys()),
            pde=pde,
            test_resolution=test_resolution,
            save_dir=frequency_dir,
            checkpoint_resolutions=checkpoint_resolutions
        )
    
    return loss_results, frequency_data

def create_multiresolution_training_plots(frequency_data, alpha_names, pde, test_resolution, save_dir, checkpoint_resolutions):
    """
    Error per frequency for different alphas - ICML version with 2 panels only.
    Enhanced publication-ready styling with subplot labels in titles.
    """
    
    # Decompose error for each alpha
    decomposition_results = {}
    
    for alpha_name in alpha_names:
        y_pred = frequency_data[alpha_name]['predictions']
        y_true = frequency_data[alpha_name]['targets']
        
        print(f"Decomposing {alpha_name}...")
        
        error_per_mode, solution_per_mode, frequencies = decompose_error_by_frequency_1d(
            y_pred, y_true
        )
        
        decomposition_results[alpha_name] = {
            'error': error_per_mode,
            'solution': solution_per_mode,
            'frequencies': frequencies
        }
    
    # Professional color palette for publications (colorblind-friendly)
    colors_palette = [
        '#2E3192',  # Deep blue (α=0.0)
        '#00B4D8',  # Cyan (α=0.05)
        '#B565D8',  # Purple (α=0.4)
        '#4A4A4A',  # Gray (α=1.0)
        '#06A77D',  # Teal
        '#F77F00',  # Orange
        '#6A4C93'   # Dark purple
    ]
    colors = colors_palette[:len(alpha_names)]
    
    # Distinct markers
    markers_list = ['o', 's', '^', 'D', 'v', 'P', '*']
    markers = markers_list[0]
    
    # Create 1x2 subplot with white background - PUBLICATION READY
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    
    # Store line objects for shared legend
    legend_lines = []
    legend_labels = []
    
    # ============ PDE-DEPENDENT MARKEVERY ============
    # Get typical frequency data length
    first_alpha = alpha_names[0]
    freq_length = len(decomposition_results[first_alpha]['frequencies'])
    
    # Adjust markevery based on PDE and resolution
    if pde.lower() == 'burger':
        # Burgers has more points (1024), needs much sparser markers
        markevery_base = max(1, freq_length // 25)  # ~20 markers (was //50 → ~50 markers)
    else:  # 'ks' or 'ns'
        # KS/NS have fewer points (256-512), can use denser markers
        markevery_base = max(1, freq_length // 70)  # ~7-10 markers
    
    print(f'PDE: {pde}, Frequency data length: {freq_length}, markevery: {markevery_base}')
    # =================================================
    
    # Plot 1: Error per mode for different alphas
    for idx, alpha_name in enumerate(alpha_names):
        data = decomposition_results[alpha_name]

        # UPDATE
        if pde == 'burger':
            window_size = 31  # Adjust this (odd number)
            data['error'] = uniform_filter1d(data['error'], size=window_size, mode='nearest')

        line, = ax1.semilogy(data['frequencies'], data['error'],
                    label=f'α = {alpha_name}', 
                    color=colors[idx], 
                    linewidth=2.5,
                    marker=markers[0],  # marker=markers[idx] (UPDATE)
                    markevery=markevery_base,  # Use PDE-dependent markevery
                    markersize=7,
                    markerfacecolor=colors[idx],
                    markeredgecolor='white',
                    markeredgewidth=1.5,
                    alpha=0.95,
                    zorder=3)
        legend_lines.append(line)
        legend_labels.append(f'α = {alpha_name}')
    
    ax1.set_xlabel('Frequency', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Error per Mode', fontsize=15, fontweight='bold')
    ax1.set_title('(a) Error Decomposition by Fourier Mode', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=1.2, color='gray', zorder=0)
    ax1.grid(True, which='minor', alpha=0.12, linestyle='-', linewidth=0.7, color='gray', zorder=0)
    
    # Plot 2: Solution spectral decay (excluding frequency=0)
    data = decomposition_results[first_alpha]
    
    # Remove frequency=0 point for better visualization
    freq_mask = data['frequencies'] > 0
    frequencies_filtered = data['frequencies'][freq_mask]
    solution_filtered = data['solution'][freq_mask]
    
    # ============ PDE-DEPENDENT MARKEVERY FOR PANEL 2 ============
    if pde.lower() == 'burger':
        markevery_panel2 = max(1, len(frequencies_filtered) // 25)  # ~20 markers
    else:
        markevery_panel2 = max(1, len(frequencies_filtered) // 70)  # ~7-10 markers
    # ==============================================================
    
    ax2.semilogy(frequencies_filtered, solution_filtered, 
                color='#2E3192',  # Consistent with first alpha color
                linewidth=2.5,
                marker='o',
                markersize=7,
                markevery=markevery_panel2,  # Use PDE-dependent markevery
                markerfacecolor='#2E3192',
                markeredgecolor='white',
                markeredgewidth=1.5,
                alpha=0.95,
                zorder=3)
    ax2.set_xlabel('Frequency', fontsize=15, fontweight='bold')
    ax2.set_ylabel('Amplitude', fontsize=15, fontweight='bold')
    ax2.set_title(f'(b) Solution Spectral Decay', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=1.2, color='gray', zorder=0)
    ax2.grid(True, which='minor', alpha=0.12, linestyle='-', linewidth=0.7, color='gray', zorder=0)
    
    # Apply enhanced styling to both subplots
    for ax in [ax1, ax2]:
        # Tick parameters - publication quality
        ax.tick_params(axis='both', which='major', labelsize=13, width=1.8, length=8,
                      direction='out', colors='black')
        ax.tick_params(axis='both', which='minor', width=1.2, length=5,
                      direction='out', colors='black')
        
        # Professional spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
            spine.set_color('black')
        
        # Ensure proper spacing
        ax.margins(x=0.02)
    
    # Main title with training/test info
    res_str = ', '.join([f'$\\mathcal{{G}}_{{{res}}}$' for res in checkpoint_resolutions])
    plt.suptitle(f'{pde.upper()} Error Decomposition: Train: {res_str}, Test: $\\mathcal{{G}}_{{{test_resolution}}}$',
                 fontsize=15, fontweight='bold', y=1.0)
    
    # Create single legend at the bottom center - ICML style
    legend = fig.legend(legend_lines, legend_labels, 
                       loc='lower center', 
                       bbox_to_anchor=(0.5, -0.08),
                       ncol=len(alpha_names),
                       fontsize=12,
                       frameon=True,
                       framealpha=1.0,
                       edgecolor='black',
                       fancybox=False,
                       shadow=False,
                       columnspacing=1.5,
                       handletextpad=0.6,
                       borderpad=0.7)
    legend.get_frame().set_linewidth(1.5)
    for text in legend.get_texts():
        text.set_fontweight('medium')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.88)
    
    # Save with publication-quality DPI
    save_path_png = os.path.join(save_dir, f'{pde}_frequency_analysis_icml.png')
    save_path_pdf = os.path.join(save_dir, f'{pde}_frequency_analysis_icml.pdf')
    
    plt.savefig(save_path_png, dpi=400, bbox_inches='tight', format='png', facecolor='white')
    plt.savefig(save_path_pdf, bbox_inches='tight', format='pdf', facecolor='white')
    
    print(f"✓ Saved PNG: {save_path_png}")
    print(f"✓ Saved PDF: {save_path_pdf}")
    plt.close()

# THIS IS CORRECT. THE CURRENT VERSION GIVES NEATER BURGER PLOTS.
# def create_multiresolution_training_plots(frequency_data, alpha_names, pde, test_resolution, save_dir, checkpoint_resolutions):
#     """
#     Error per frequency for different alphas - ICML version with 2 panels only.
#     Enhanced publication-ready styling with subplot labels in titles.
#     """
    
#     # Decompose error for each alpha
#     decomposition_results = {}
    
#     for alpha_name in alpha_names:
#         y_pred = frequency_data[alpha_name]['predictions']
#         y_true = frequency_data[alpha_name]['targets']
        
#         print(f"Decomposing {alpha_name}...")
        
#         error_per_mode, solution_per_mode, frequencies = decompose_error_by_frequency_1d(
#             y_pred, y_true
#         )
        
#         decomposition_results[alpha_name] = {
#             'error': error_per_mode,
#             'solution': solution_per_mode,
#             'frequencies': frequencies
#         }
    
#     # Professional color palette for publications (colorblind-friendly)
#     # Using distinct, high-contrast colors suitable for print
#     colors_palette = [
#         '#2E3192',  # Deep blue (α=0.0)
#         '#00B4D8',  # Cyan (α=0.05)
#         '#B565D8',  # Red (α=0.4)
#         '#4A4A4A',  # Purple (α=1.0)
#         '#06A77D',  # Teal
#         '#F77F00',  # Orange
#         '#6A4C93'   # Dark purple
#     ]
#     colors = colors_palette[:len(alpha_names)]
    
#     # Distinct markers (UPDATE)
#     markers_list = ['o', 's', '^', 'D', 'v', 'P', '*']
#     markers = markers_list[0]
    
#     # Create 1x2 subplot with white background - PUBLICATION READY
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     fig.patch.set_facecolor('white')
    
#     # Store line objects for shared legend
#     legend_lines = []
#     legend_labels = []
    
#     # Plot 1: Error per mode for different alphas
#     for idx, alpha_name in enumerate(alpha_names):
#         data = decomposition_results[alpha_name]

#         markevery = max(1, len(data['frequencies'])//70)  # Adjust for clarity
#         print('Frequency data length:', len(data['frequencies']))

#         line, = ax1.semilogy(data['frequencies'], data['error'],
#                     label=f'α = {alpha_name}', 
#                     color=colors[idx], 
#                     linewidth=2.5,
#                     marker=markers[0],  # marker=markers[idx] (UPDATE)
#                     markevery=markevery,
#                     markersize=7,
#                     markerfacecolor=colors[idx],
#                     markeredgecolor='white',
#                     markeredgewidth=1.5,
#                     alpha=0.95,
#                     zorder=3)
#         legend_lines.append(line)
#         legend_labels.append(f'α = {alpha_name}')
    
#     ax1.set_xlabel('Frequency', fontsize=15, fontweight='bold')
#     ax1.set_ylabel('Error per Mode', fontsize=15, fontweight='bold')
#     ax1.set_title('(a) Error Decomposition by Fourier Mode', fontsize=14, fontweight='bold', pad=10)
#     ax1.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=1.2, color='gray', zorder=0)
#     ax1.grid(True, which='minor', alpha=0.12, linestyle='-', linewidth=0.7, color='gray', zorder=0)
    
#     # Plot 2: Solution spectral decay (excluding frequency=0)
#     first_alpha = alpha_names[0]
#     data = decomposition_results[first_alpha]
    
#     # Remove frequency=0 point for better visualization
#     freq_mask = data['frequencies'] > 0
#     frequencies_filtered = data['frequencies'][freq_mask]
#     solution_filtered = data['solution'][freq_mask]
    
#     markevery = max(1, len(frequencies_filtered)//70)
    
#     ax2.semilogy(frequencies_filtered, solution_filtered, 
#                 color='#2E3192',  # Consistent with first alpha color
#                 linewidth=2.5,
#                 marker='o',
#                 markersize=7,
#                 markevery=markevery,
#                 markerfacecolor='#2E3192',
#                 markeredgecolor='white',
#                 markeredgewidth=1.5,
#                 alpha=0.95,
#                 zorder=3)
#     ax2.set_xlabel('Frequency', fontsize=15, fontweight='bold')
#     ax2.set_ylabel('Amplitude', fontsize=15, fontweight='bold')
#     ax2.set_title(f'(b) Solution Spectral Decay', fontsize=14, fontweight='bold', pad=10)
#     ax2.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=1.2, color='gray', zorder=0)
#     ax2.grid(True, which='minor', alpha=0.12, linestyle='-', linewidth=0.7, color='gray', zorder=0)
    
#     # Apply enhanced styling to both subplots
#     for ax in [ax1, ax2]:
#         # Tick parameters - publication quality
#         ax.tick_params(axis='both', which='major', labelsize=13, width=1.8, length=8,
#                       direction='out', colors='black')
#         ax.tick_params(axis='both', which='minor', width=1.2, length=5,
#                       direction='out', colors='black')
        
#         # Professional spines
#         for spine in ax.spines.values():
#             spine.set_linewidth(1.8)
#             spine.set_color('black')
        
#         # Ensure proper spacing
#         ax.margins(x=0.02)
    
#     # Main title with training/test info - adjusted position to avoid overlap
#     res_str = ', '.join([f'$\\mathcal{{G}}_{{{res}}}$' for res in checkpoint_resolutions])
#     plt.suptitle(f'{pde.upper()} Error Decomposition: Train: {res_str}, Test: $\\mathcal{{G}}_{{{test_resolution}}}$',
#                  fontsize=15, fontweight='bold', y=1.0)
    
#     # Create single legend at the bottom center - ICML style
#     legend = fig.legend(legend_lines, legend_labels, 
#                        loc='lower center', 
#                        bbox_to_anchor=(0.5, -0.08),
#                        ncol=len(alpha_names),
#                        fontsize=12,
#                        frameon=True,
#                        framealpha=1.0,
#                        edgecolor='black',
#                        fancybox=False,
#                        shadow=False,
#                        columnspacing=1.5,
#                        handletextpad=0.6,
#                        borderpad=0.7)
#     legend.get_frame().set_linewidth(1.5)
#     for text in legend.get_texts():
#         text.set_fontweight('medium')
    
#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.18, top=0.88)  # Make room for legend and avoid title overlap
    
#     # Save with publication-quality DPI
#     save_path_png = os.path.join(save_dir, f'{pde}_frequency_analysis_icml.png')
#     save_path_pdf = os.path.join(save_dir, f'{pde}_frequency_analysis_icml.pdf')
    
#     plt.savefig(save_path_png, dpi=400, bbox_inches='tight', format='png', facecolor='white')
#     plt.savefig(save_path_pdf, bbox_inches='tight', format='pdf', facecolor='white')
    
#     print(f"✓ Saved PNG: {save_path_png}")
#     print(f"✓ Saved PDF: {save_path_pdf}")
#     plt.close()