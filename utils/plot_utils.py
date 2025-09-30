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

from models.custom_layer import UnitGaussianNormalizer
from hydra.utils import instantiate

############################
# PLOT 1D
############################


def plot_1d_pde_examples_multiple(plot_data, test_resolutions, pde="PDE", save_dir=None, num_examples=10):
    """
    Plot multiple prediction vs target examples for 1D PDE across different resolutions.
    
    Args:
        plot_data: Dictionary with resolution as key and dict containing arrays of predictions, targets, inputs
        test_resolutions: List of resolutions tested
        pde: Name of PDE for labeling
        save_dir: Directory to save plots (optional)
        num_examples: Number of examples to plot
    """
    import matplotlib.pyplot as plt
    
    # Create a large figure with subplots
    # Each row shows one example across all resolutions
    fig, axes = plt.subplots(num_examples, len(test_resolutions), 
                            figsize=(4*len(test_resolutions), 3*num_examples))
    
    # Handle case where we only have one resolution or one example
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    elif len(test_resolutions) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'{pde.upper()} Predictions vs Targets - {num_examples} Examples', fontsize=16, y=0.98)
    
    # Plot each example
    for example_idx in range(num_examples):
        for res_idx, resolution in enumerate(test_resolutions):
            ax = axes[example_idx, res_idx]
            
            # Check if we have data for this example and resolution
            if (resolution in plot_data and 
                example_idx < len(plot_data[resolution]['predictions'])):
                
                prediction = plot_data[resolution]['predictions'][example_idx]
                target = plot_data[resolution]['targets'][example_idx]
                
                # Handle different dimensionalities
                if prediction.ndim > 1:
                    prediction = prediction.squeeze()
                if target.ndim > 1:
                    target = target.squeeze()
                
                # Create spatial coordinates
                x_coords = np.linspace(0, 1, len(prediction))
                
                # Plot prediction and target
                ax.plot(x_coords, target, 'b-', label='Target', linewidth=2, alpha=0.8)
                ax.plot(x_coords, prediction, 'r--', label='Prediction', linewidth=2, alpha=0.8)
                
                # Calculate error for this example
                error = np.mean((prediction - target)**2)
                
                # Set title
                if example_idx == 0:
                    ax.set_title(f'Resolution {resolution}\nMSE: {error:.4f}', fontsize=10)
                else:
                    ax.set_title(f'MSE: {error:.4f}', fontsize=10)
                
                # Add legend only for first example
                if example_idx == 0 and res_idx == 0:
                    ax.legend(fontsize=8)
                
                # Set labels
                if res_idx == 0:
                    ax.set_ylabel(f'Example {example_idx+1}\nu(x)', fontsize=9)
                if example_idx == num_examples - 1:
                    ax.set_xlabel('x', fontsize=9)
                
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
                
            else:
                # No data available for this example/resolution
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Resolution {resolution}', fontsize=10)
                if res_idx == 0:
                    ax.set_ylabel(f'Example {example_idx+1}', fontsize=9)
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        filename = f'{pde.lower()}_predictions_vs_targets_{num_examples}_examples.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved prediction plot: {filepath}")
    
    plt.show()


def plot_1d_pde_examples_compact(plot_data, test_resolutions, pde="PDE", save_dir=None, num_examples=10):
    """
    Alternative compact plotting - shows all examples for each resolution in separate figures.
    """
    import matplotlib.pyplot as plt
    
    for resolution in test_resolutions:
        if resolution not in plot_data:
            continue
            
        # Calculate number of rows and columns for subplots
        rows = int(np.ceil(np.sqrt(num_examples)))
        cols = int(np.ceil(num_examples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        fig.suptitle(f'{pde.upper()} Resolution {resolution} - {num_examples} Examples', fontsize=14)
        
        # Flatten axes for easier indexing
        if num_examples > 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = [axes]
        
        predictions = plot_data[resolution]['predictions']
        targets = plot_data[resolution]['targets']
        
        for i in range(num_examples):
            ax = axes[i]
            
            if i < len(predictions):
                prediction = predictions[i].squeeze()
                target = targets[i].squeeze()
                
                x_coords = np.linspace(0, 1, len(prediction))
                
                ax.plot(x_coords, target, 'b-', label='Target', linewidth=2, alpha=0.8)
                ax.plot(x_coords, prediction, 'r--', label='Prediction', linewidth=2, alpha=0.8)
                
                error = np.mean((prediction - target)**2)
                ax.set_title(f'Example {i+1} (MSE: {error:.4f})', fontsize=10)
                
                if i == 0:
                    ax.legend(fontsize=8)
                    
            else:
                ax.set_visible(False)
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        
        # Hide extra subplots
        for i in range(num_examples, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_dir:
            filename = f'{pde.lower()}_resolution_{resolution}_examples.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        plt.show()
    


def perform_frequency_analysis_1d_generic(frequency_data, current_res, pde, save_dir):
    """
    GENERIC 1D frequency analysis for any PDE
    
    Args:
        frequency_data: Dictionary with resolution data
        current_res: Training resolution
        pde: Type of PDE for labeling
        save_dir: Directory to save analysis plots
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, len(frequency_data), figsize=(5*len(frequency_data), 8))
    if len(frequency_data) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (res, data) in enumerate(frequency_data.items()):
        input_data = data['input_data'][0, 0].numpy()  # Shape: (spatial,)
        
        # Compute 1D FFT
        fft_data = np.fft.fft(input_data)
        frequencies = np.fft.fftfreq(len(input_data))
        
        # Plot spatial data
        x = np.linspace(0, 1, len(input_data))
        axes[0, i].plot(x, input_data)
        axes[0, i].set_title(f'Spatial Data - Resolution {res}')
        axes[0, i].set_xlabel('Spatial Coordinate')
        axes[0, i].set_ylabel('Field Value')
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot frequency spectrum (magnitude)
        axes[1, i].plot(frequencies[:len(frequencies)//2], np.abs(fft_data[:len(fft_data)//2]))
        axes[1, i].set_title(f'Frequency Spectrum - Resolution {res}')
        axes[1, i].set_xlabel('Frequency')
        axes[1, i].set_ylabel('Magnitude')
        axes[1, i].set_yscale('log')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'{pde.upper()} 1D Frequency Analysis', fontsize=14, fontweight='bold')
    
    if save_dir:
        filename = f'{pde.lower()}_1d_frequency_analysis_{timestamp}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Saved 1D {pde.upper()} frequency analysis to {save_dir}")
    
    plt.show()


def save_numerical_results_generic(results, current_res, pde, save_dir):
    """Save numerical results to CSV and text files - GENERIC version"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    csv_path = os.path.join(save_dir, f'{pde.lower()}_1d_evaluation_results_{timestamp}.csv')
    with open(csv_path, 'w') as f:
        f.write("Resolution,Relative_L2_Loss,Training_Resolution,PDE_Type\n")
        for res, loss in results.items():
            f.write(f"{res},{loss:.6f},{current_res},{pde.upper()}\n")
    
    # Save detailed summary
    summary_path = os.path.join(save_dir, f'{pde.lower()}_1d_evaluation_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{pde.upper()} 1D EVALUATION SUMMARY\n")
        f.write(f"{'='*50}\n")
        f.write(f"Training Resolution: {current_res}\n")
        f.write(f"PDE Type: {pde.upper()}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"RESULTS:\n")
        for res, loss in sorted(results.items()):
            operation = "UPSAMPLING" if res > current_res else "DOWNSAMPLING" if res < current_res else "SAME"
            f.write(f"  {res}: {loss:.6f} ({operation})\n")
        
        if results:
            best_res = min(results, key=results.get)
            worst_res = max(results, key=results.get)
            f.write(f"\nBest performance: {best_res} (Loss: {results[best_res]:.6f})\n")
            f.write(f"Worst performance: {worst_res} (Loss: {results[worst_res]:.6f})\n")
    
    print(f"Saved numerical results: {csv_path}")
    print(f"Saved summary: {summary_path}")


########################
# PLOT 2D
########################
    
def perform_frequency_analysis(frequency_data, current_res, save_dir):
    """
    Perform frequency analysis for all resolution pairs and save plots
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Analyzing frequency distributions...")
    print(f"Training resolution: {current_res}")
    print(f"Test resolutions: {list(frequency_data.keys())}")
    
    # Create individual frequency analysis for each resolution pair
    for test_res, data in frequency_data.items():
        print(f"\nAnalyzing {test_res}→{current_res} frequency distribution...")
        
        input_data = data['input_data']  # Shape: (1, 1, test_res, test_res)
        
        # Perform frequency analysis
        fig = analyze_resize_frequencies(input_data, test_res, current_res)
        
        # Save individual analysis
        individual_path = os.path.join(save_dir, f'frequency_analysis_{test_res}_to_{current_res}_{timestamp}.png')
        fig.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved frequency analysis: {individual_path}")
    
    # Create comparison plot for all resolutions
    if len(frequency_data) > 1:
        print("Creating multi-resolution frequency comparison...")
        comparison_fig = create_multi_resolution_frequency_comparison(frequency_data, current_res)
        comparison_path = os.path.join(save_dir, f'frequency_comparison_all_resolutions_{timestamp}.png')
        comparison_fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close(comparison_fig)
        print(f"Saved frequency comparison: {comparison_path}")


def analyze_resize_frequencies(input_data, input_res, output_res):
    """
    Analyze frequency distribution for a specific resize operation
    """
    x = input_data  # Shape: (1, 1, input_res, input_res)
    out_size = (output_res, output_res)
    
    # Step 1: Forward FFT
    f = torch.fft.rfft2(x, norm='backward')
    
    # Step 2: Create target frequency tensor
    f_z = torch.zeros((*x.shape[:-2], out_size[0], out_size[1]//2 + 1), 
                      dtype=f.dtype, device=f.device)
    
    # Step 3: Calculate bounds with correct bounds checking
    top_freqs1 = min((f.shape[-2] + 1) // 2, (out_size[0] + 1) // 2)
    top_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    bot_freqs1 = min(f.shape[-2] // 2, out_size[0] // 2)
    bot_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    
    # Step 4: Copy frequencies
    f_z[..., :top_freqs1, :top_freqs2] = f[..., :top_freqs1, :top_freqs2]
    if bot_freqs1 > 0:
        f_z[..., -bot_freqs1:, :bot_freqs2] = f[..., -bot_freqs1:, :bot_freqs2]
    
    # Calculate amplitudes for visualization
    f_amplitude = torch.abs(f[0, 0])  
    f_z_amplitude = torch.abs(f_z[0, 0])  
    
    # Determine operation type
    if output_res > input_res:
        operation_type = "UPSAMPLING"
    elif output_res < input_res:
        operation_type = "DOWNSAMPLING"
    else:
        operation_type = "NO CHANGE"
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original frequency spectrum
    im1 = axes[0, 0].imshow(f_amplitude.cpu().numpy(), cmap='viridis', aspect='auto')    # UPDATE
    axes[0, 0].set_title(f'Original Frequency Spectrum\n{f.shape[-2]}×{f.shape[-1]} ({input_res}→FFT)')
    axes[0, 0].set_xlabel(f'Frequency bins (0-{f.shape[-1]-1})')
    axes[0, 0].set_ylabel(f'Spatial frequency (0-{f.shape[-2]-1})')
    plt.colorbar(im1, ax=axes[0, 0], label='Amplitude')
    
    # Add rectangles showing copied regions
    if top_freqs1 > 0 and top_freqs2 > 0:
        rect1 = patches.Rectangle((0, 0), top_freqs2, top_freqs1, linewidth=3, 
                                 edgecolor='red', facecolor='none', label='Copied (top)')
        axes[0, 0].add_patch(rect1)
    
    if bot_freqs1 > 0 and bot_freqs2 > 0:
        bottom_start = f.shape[-2] - bot_freqs1
        rect2 = patches.Rectangle((0, bottom_start), bot_freqs2, bot_freqs1, 
                                 linewidth=3, edgecolor='orange', facecolor='none', 
                                 label='Copied (bottom)')
        axes[0, 0].add_patch(rect2)
    
    axes[0, 0].legend()
    
    # 2. Target frequency spectrum
    im2 = axes[0, 1].imshow(f_z_amplitude.cpu().numpy(), cmap='viridis', aspect='auto')    # UPDATE
    axes[0, 1].set_title(f'Target Frequency Spectrum\n{f_z.shape[-2]}×{f_z.shape[-1]} ({output_res}→FFT)')
    axes[0, 1].set_xlabel(f'Frequency bins (0-{f_z.shape[-1]-1})')
    axes[0, 1].set_ylabel(f'Spatial frequency (0-{f_z.shape[-2]-1})')
    plt.colorbar(im2, ax=axes[0, 1], label='Amplitude')
    
    # 3. Frequency bin retention
    plot_frequency_retention_bar(axes[0, 2], f.shape[-1], f_z.shape[-1], top_freqs2)
    
    # 4. Spatial frequency retention
    plot_spatial_retention_bar(axes[1, 0], f.shape[-2], f_z.shape[-2], top_freqs1, bot_freqs1)
    
    # 5. Energy analysis
    plot_energy_analysis(axes[1, 1], f_amplitude, f_z_amplitude)
    
    # 6. Summary statistics
    plot_frequency_summary(axes[1, 2], input_res, output_res, operation_type, 
                          f_amplitude, f_z_amplitude, top_freqs1, top_freqs2, bot_freqs1, bot_freqs2)
    
    plt.tight_layout()
    plt.suptitle(f'Frequency Analysis: {input_res}×{input_res} → {output_res}×{output_res} ({operation_type})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def plot_frequency_retention_bar(ax, original_width, target_width, retained_bins):
    """Plot frequency bin retention bar chart"""
    bins = np.arange(original_width)
    retention_status = np.zeros(original_width)
    retention_status[:retained_bins] = 1
    
    colors = ['green' if status == 1 else 'red' for status in retention_status]
    
    ax.bar(bins, np.ones(original_width), color=colors, alpha=0.7)
    ax.set_title('Frequency Bin Retention')
    ax.set_xlabel('Original frequency bins')
    ax.set_ylabel('Retention status')
    
    retention_pct = (retained_bins / original_width) * 100
    ax.text(0.02, 0.98, f'Retained: {retained_bins}/{original_width} ({retention_pct:.1f}%)',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_spatial_retention_bar(ax, original_height, target_height, top_freqs, bot_freqs):
    """Plot spatial frequency retention bar chart"""
    spatial_bins = np.arange(original_height)
    retention_status = np.zeros(original_height)
    
    if top_freqs > 0:
        retention_status[:top_freqs] = 1
    if bot_freqs > 0:
        retention_status[-bot_freqs:] = 1
    
    colors = ['green' if status == 1 else 'red' for status in retention_status]
    
    ax.bar(spatial_bins, np.ones(original_height), color=colors, alpha=0.7, width=1)
    ax.set_title('Spatial Frequency Retention')
    ax.set_xlabel('Spatial frequency index')
    ax.set_ylabel('Retention status')
    
    retained_total = top_freqs + bot_freqs
    retained_pct = (retained_total / original_height) * 100
    
    ax.text(0.02, 0.98, f'Retained: {retained_total}/{original_height} ({retained_pct:.1f}%)',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_energy_analysis(ax, f_amp, f_z_amp):
    """Plot energy analysis"""
    original_energy = torch.sum(f_amp**2).item()
    retained_energy = torch.sum(f_z_amp**2).item()
    
    energy_ratio = (retained_energy / original_energy * 100) if original_energy > 0 else 0
    
    categories = ['Original', 'Retained', 'Lost']
    energies = [original_energy, retained_energy, original_energy - retained_energy]
    colors = ['blue', 'green', 'red']
    
    bars = ax.bar(categories, energies, color=colors, alpha=0.7)
    ax.set_title('Energy Analysis')
    ax.set_ylabel('Energy')
    ax.set_yscale('log')
    
    # Add percentage annotations
    for i, (bar, energy) in enumerate(zip(bars, energies)):
        height = bar.get_height()
        if i == 0:
            pct_text = "100%"
        elif i == 1:
            pct_text = f"{energy_ratio:.1f}%"
        else:
            pct_text = f"{100-energy_ratio:.1f}%"
        
        ax.text(bar.get_x() + bar.get_width()/2., height,
                pct_text, ha='center', va='bottom')


def plot_frequency_summary(ax, input_res, output_res, operation_type, f_amp, f_z_amp, 
                          top_freqs1, top_freqs2, bot_freqs1, bot_freqs2):
    """Plot summary statistics"""
    ax.axis('off')
    
    total_original = f_amp.numel()
    total_target = f_z_amp.numel()
    
    retained_coeffs = top_freqs1 * top_freqs2 + bot_freqs1 * bot_freqs2
    retention_ratio = retained_coeffs / total_original * 100
    
    retained_energy = torch.sum(f_z_amp**2).item()
    total_energy = torch.sum(f_amp**2).item()
    energy_ratio = (retained_energy / total_energy * 100) if total_energy > 0 else 0
    
    stats_text = f"""
    FREQUENCY ANALYSIS SUMMARY
    {'='*40}
    
    Operation: {operation_type}
    {input_res}×{input_res} → {output_res}×{output_res}
    
    COEFFICIENTS:
    Original: {total_original:,}
    Target: {total_target:,}
    Retained: {retained_coeffs:,} ({retention_ratio:.1f}%)
    
    ENERGY:
    Preserved: {energy_ratio:.1f}%
    Lost: {100-energy_ratio:.1f}%
    
    FREQUENCY MAPPING:
    Low freq: {top_freqs1} spatial bins
    High freq: {bot_freqs1} spatial bins
    Freq bands: {top_freqs2} bins
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))


def create_multi_resolution_frequency_comparison(frequency_data, current_res):
    """Create comparison plot showing all resolution pairs"""
    
    resolutions = sorted(frequency_data.keys())
    fig, axes = plt.subplots(len(resolutions), 2, figsize=(12, 4*len(resolutions)))
    
    if len(resolutions) == 1:
        axes = axes.reshape(1, -1)
    
    for i, test_res in enumerate(resolutions):
        input_data = frequency_data[test_res]['input_data']
        
        # Analyze this resolution pair
        f = torch.fft.rfft2(input_data, norm='backward')
        f_z = torch.zeros((1, 1, current_res, current_res//2 + 1), dtype=f.dtype)
        
        # Copy frequencies with bounds checking
        top_freqs1 = min((f.shape[-2] + 1) // 2, (current_res + 1) // 2)
        top_freqs2 = min(f.shape[-1], current_res // 2 + 1)
        bot_freqs1 = min(f.shape[-2] // 2, current_res // 2)
        bot_freqs2 = min(f.shape[-1], current_res // 2 + 1)
        
        f_z[..., :top_freqs1, :top_freqs2] = f[..., :top_freqs1, :top_freqs2]
        if bot_freqs1 > 0:
            f_z[..., -bot_freqs1:, :bot_freqs2] = f[..., -bot_freqs1:, :bot_freqs2]
        
        # Plot original and target spectra
        f_amp = torch.abs(f[0, 0])
        f_z_amp = torch.abs(f_z[0, 0])
        
        im1 = axes[i, 0].imshow(f_amp.cpu().numpy(), cmap='viridis', aspect='auto')   # UPDATE
        axes[i, 0].set_title(f'Input: {test_res}×{test_res} → {f.shape[-2]}×{f.shape[-1]}')
        axes[i, 0].set_ylabel(f'{test_res}→{current_res}')
        
        im2 = axes[i, 1].imshow(f_z_amp.cpu().numpy(), cmap='viridis', aspect='auto')    # UPDATE
        axes[i, 1].set_title(f'Output: {current_res}×{current_res} → {f_z.shape[-2]}×{f_z.shape[-1]}')
        
        # Add retention statistics
        retention_ratio = (f_z_amp.numel() / f_amp.numel()) * 100
        energy_ratio = (torch.sum(f_z_amp**2) / torch.sum(f_amp**2) * 100).item()
        
        axes[i, 1].text(0.02, 0.98, f'Retention: {retention_ratio:.1f}%\nEnergy: {energy_ratio:.1f}%',
                        transform=axes[i, 1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle(f'Multi-Resolution Frequency Analysis (Target: {current_res}×{current_res})', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    return fig


def save_numerical_results(results, current_res, save_dir):
    """Save numerical results to CSV and text files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    csv_path = os.path.join(save_dir, f'evaluation_results_{timestamp}.csv')
    with open(csv_path, 'w') as f:
        f.write("Resolution,Relative_L2_Loss,Training_Resolution\n")
        for res, loss in results.items():
            f.write(f"{res},{loss:.6f},{current_res}\n")
    
    # Save detailed summary
    summary_path = os.path.join(save_dir, f'evaluation_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"EVALUATION SUMMARY\n")
        f.write(f"{'='*50}\n")
        f.write(f"Training Resolution: {current_res}×{current_res}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"RESULTS:\n")
        for res, loss in sorted(results.items()):
            operation = "UPSAMPLING" if res > current_res else "DOWNSAMPLING" if res < current_res else "SAME"
            f.write(f"  {res}×{res}: {loss:.6f} ({operation})\n")
        
        if results:
            best_res = min(results, key=results.get)
            worst_res = max(results, key=results.get)
            f.write(f"\nBest performance: {best_res}×{best_res} (Loss: {results[best_res]:.6f})\n")
            f.write(f"Worst performance: {worst_res}×{worst_res} (Loss: {results[worst_res]:.6f})\n")
    
    print(f"Saved numerical results: {csv_path}")
    print(f"Saved summary: {summary_path}")



def plot_navier_stokes_examples(plot_data, test_resolutions, save_dir=None):
    """
    Plot prediction vs target examples for Navier-Stokes at different resolutions.
    Save images locally. Handles both single channel and multi-channel data.
    
    Args:
        plot_data: Dictionary with resolution as key and prediction/target data as values
        test_resolutions: List of resolutions that were tested
        save_dir: Directory to save images locally (optional)
    """
    num_resolutions = len(plot_data)
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine number of channels from first resolution data
    first_resolution = list(plot_data.keys())[0]
    sample_data = plot_data[first_resolution]['input']
    num_channels = sample_data.shape[0]  # Shape: (C, H, W)
    
    print(f"Detected {num_channels} channel(s) in the data")
    
    # Handle different channel cases
    if num_channels == 1:
        # Single channel - plot as before
        plot_single_channel(plot_data, num_resolutions, save_dir, timestamp)
    elif num_channels == 3:
        # Multi-channel - plot each channel separately and combined
        plot_multi_channel(plot_data, num_resolutions, save_dir, timestamp)
    else:
        # For other channel counts, plot first channel only with warning
        print(f"Warning: {num_channels} channels detected. Plotting only the first channel.")
        plot_single_channel(plot_data, num_resolutions, save_dir, timestamp, channel_idx=0)


def plot_single_channel(plot_data, num_resolutions, save_dir, timestamp, channel_idx=0):
    """Plot single channel data"""
    # Create subplots: 3 rows (input, prediction, target) x num_resolutions columns
    fig, axes = plt.subplots(3, num_resolutions, figsize=(4*num_resolutions, 12))
    
    # Handle case where there's only one resolution
    if num_resolutions == 1:
        axes = axes.reshape(3, 1)
    
    for col_idx, resolution in enumerate(sorted(plot_data.keys())):
        data = plot_data[resolution]
        
        # Extract data (take specified channel)
        input_data = data['input'][channel_idx]      # Shape: (H, W)
        pred_data = data['prediction'][channel_idx]   # Shape: (H, W)
        target_data = data['target'][channel_idx]     # Shape: (H, W)
        
        # Plot input
        im1 = axes[0, col_idx].imshow(input_data, cmap='RdBu_r', aspect='equal')
        axes[0, col_idx].set_title(f'Input\nRes: {resolution}x{resolution}')
        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])
        plt.colorbar(im1, ax=axes[0, col_idx], shrink=0.8)
        
        # Plot prediction
        im2 = axes[1, col_idx].imshow(pred_data, cmap='RdBu_r', aspect='equal')
        axes[1, col_idx].set_title(f'Prediction\nRes: {resolution}x{resolution}')
        axes[1, col_idx].set_xticks([])
        axes[1, col_idx].set_yticks([])
        plt.colorbar(im2, ax=axes[1, col_idx], shrink=0.8)
        
        # Plot target
        im3 = axes[2, col_idx].imshow(target_data, cmap='RdBu_r', aspect='equal')
        axes[2, col_idx].set_title(f'Target\nRes: {resolution}x{resolution}')
        axes[2, col_idx].set_xticks([])
        axes[2, col_idx].set_yticks([])
        plt.colorbar(im3, ax=axes[2, col_idx], shrink=0.8)
        
        # Add column label
        if col_idx == 0:
            channel_label = f" (Ch {channel_idx})" if channel_idx > 0 else ""
            axes[0, col_idx].set_ylabel(f'Input{channel_label}', fontsize=14, fontweight='bold')
            axes[1, col_idx].set_ylabel(f'Prediction{channel_label}', fontsize=14, fontweight='bold')
            axes[2, col_idx].set_ylabel(f'Target{channel_label}', fontsize=14, fontweight='bold')
    
    channel_suffix = f"_ch{channel_idx}" if channel_idx > 0 else ""
    plt.suptitle(f'Navier-Stokes: Model Predictions vs Targets at Different Resolutions{channel_suffix}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save main comparison plot
    if save_dir:
        main_plot_path = os.path.join(save_dir, f'navier_stokes_comparison{channel_suffix}_{timestamp}.png')
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved main comparison plot to: {main_plot_path}")
    
    plt.show()
    plt.close()
    
    # Create error plot
    create_error_plot(plot_data, num_resolutions, save_dir, timestamp, channel_idx)
    
    # Create individual resolution plots
    if save_dir:
        create_individual_plots(plot_data, save_dir, timestamp, channel_idx)


def plot_multi_channel(plot_data, num_resolutions, save_dir, timestamp):
    """Plot multi-channel (3-channel) data"""
    channel_names = ['Channel 0', 'Channel 1', 'Channel 2']
    channel_cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r']  # You can customize these
    
    # Plot each channel separately
    for ch_idx in range(3):
        print(f"Plotting {channel_names[ch_idx]}...")
        plot_single_channel(plot_data, num_resolutions, save_dir, timestamp, channel_idx=ch_idx)
    
    # Create a combined RGB-like visualization (if appropriate for your data)
    if save_dir:
        create_combined_channel_plot(plot_data, num_resolutions, save_dir, timestamp)


def create_error_plot(plot_data, num_resolutions, save_dir, timestamp, channel_idx=0):
    """Create error analysis plot"""
    fig2, axes2 = plt.subplots(1, num_resolutions, figsize=(4*num_resolutions, 4))
    
    if num_resolutions == 1:
        axes2 = [axes2]
    
    for col_idx, resolution in enumerate(sorted(plot_data.keys())):
        data = plot_data[resolution]
        pred_data = data['prediction'][channel_idx]
        target_data = data['target'][channel_idx]
        error_data = np.abs(pred_data - target_data)
        
        # Plot absolute error
        im = axes2[col_idx].imshow(error_data, cmap='Reds', aspect='equal')
        axes2[col_idx].set_title(f'Absolute Error\nRes: {resolution}x{resolution}')
        axes2[col_idx].set_xticks([])
        axes2[col_idx].set_yticks([])
        plt.colorbar(im, ax=axes2[col_idx], shrink=0.8)
        
        # Add error statistics
        mean_error = np.mean(error_data)
        max_error = np.max(error_data)
        axes2[col_idx].text(0.02, 0.98, f'Mean: {mean_error:.4f}\nMax: {max_error:.4f}', 
                           transform=axes2[col_idx].transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    channel_suffix = f"_ch{channel_idx}" if channel_idx > 0 else ""
    plt.suptitle(f'Absolute Error: |Prediction - Target|{channel_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save error plot
    if save_dir:
        error_plot_path = os.path.join(save_dir, f'navier_stokes_error{channel_suffix}_{timestamp}.png')
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved error plot to: {error_plot_path}")
    
    plt.show()
    plt.close()


def create_individual_plots(plot_data, save_dir, timestamp, channel_idx=0):
    """Create individual resolution plots for detailed analysis"""
    for resolution in sorted(plot_data.keys()):
        data = plot_data[resolution]
        
        # Create individual plot for this resolution
        fig_ind, axes_ind = plt.subplots(2, 2, figsize=(10, 10))
        
        input_data = data['input'][channel_idx]
        pred_data = data['prediction'][channel_idx]
        target_data = data['target'][channel_idx]
        error_data = np.abs(pred_data - target_data)
        
        # Input
        im1 = axes_ind[0, 0].imshow(input_data, cmap='RdBu_r', aspect='equal')
        axes_ind[0, 0].set_title('Input')
        axes_ind[0, 0].set_xticks([])
        axes_ind[0, 0].set_yticks([])
        plt.colorbar(im1, ax=axes_ind[0, 0])
        
        # Prediction
        im2 = axes_ind[0, 1].imshow(pred_data, cmap='RdBu_r', aspect='equal')
        axes_ind[0, 1].set_title('Prediction')
        axes_ind[0, 1].set_xticks([])
        axes_ind[0, 1].set_yticks([])
        plt.colorbar(im2, ax=axes_ind[0, 1])
        
        # Target
        im3 = axes_ind[1, 0].imshow(target_data, cmap='RdBu_r', aspect='equal')
        axes_ind[1, 0].set_title('Target')
        axes_ind[1, 0].set_xticks([])
        axes_ind[1, 0].set_yticks([])
        plt.colorbar(im3, ax=axes_ind[1, 0])
        
        # Error
        im4 = axes_ind[1, 1].imshow(error_data, cmap='Reds', aspect='equal')
        axes_ind[1, 1].set_title('Absolute Error')
        axes_ind[1, 1].set_xticks([])
        axes_ind[1, 1].set_yticks([])
        plt.colorbar(im4, ax=axes_ind[1, 1])
        
        # Add statistics
        mean_error = np.mean(error_data)
        max_error = np.max(error_data)
        std_error = np.std(error_data)
        axes_ind[1, 1].text(0.02, 0.98, f'Mean: {mean_error:.4f}\nMax: {max_error:.4f}\nStd: {std_error:.4f}', 
                           transform=axes_ind[1, 1].transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        channel_suffix = f"_ch{channel_idx}" if channel_idx > 0 else ""
        plt.suptitle(f'Navier-Stokes Analysis - Resolution {resolution}x{resolution}{channel_suffix}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save individual plot
        ind_plot_path = os.path.join(save_dir, f'navier_stokes_res_{resolution}{channel_suffix}_{timestamp}.png')
        plt.savefig(ind_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved individual plot for resolution {resolution} to: {ind_plot_path}")
        
        plt.close()


def create_combined_channel_plot(plot_data, num_resolutions, save_dir, timestamp):
    """Create a combined visualization showing all channels together"""
    # For 3-channel Navier-Stokes data, it's better to show channels side by side
    # rather than trying to combine them as RGB, since each channel has physical meaning
    
    fig, axes = plt.subplots(3, 3*num_resolutions, figsize=(5*num_resolutions, 12))
    
    if num_resolutions == 1:
        axes = axes.reshape(3, 3)
    
    channel_names = ['Channel 0', 'Channel 1', 'Channel 2']
    row_names = ['Input', 'Prediction', 'Target']
    
    for res_idx, resolution in enumerate(sorted(plot_data.keys())):
        data = plot_data[resolution]
        
        for ch_idx in range(3):
            col_idx = res_idx * 3 + ch_idx
            
            # Input row
            im1 = axes[0, col_idx].imshow(data['input'][ch_idx], cmap='RdBu_r', aspect='equal')
            axes[0, col_idx].set_title(f'{channel_names[ch_idx]}\nRes: {resolution}x{resolution}')
            axes[0, col_idx].set_xticks([])
            axes[0, col_idx].set_yticks([])
            plt.colorbar(im1, ax=axes[0, col_idx], shrink=0.6)
            
            # Prediction row
            im2 = axes[1, col_idx].imshow(data['prediction'][ch_idx], cmap='RdBu_r', aspect='equal')
            axes[1, col_idx].set_xticks([])
            axes[1, col_idx].set_yticks([])
            plt.colorbar(im2, ax=axes[1, col_idx], shrink=0.6)
            
            # Target row
            im3 = axes[2, col_idx].imshow(data['target'][ch_idx], cmap='RdBu_r', aspect='equal')
            axes[2, col_idx].set_xticks([])
            axes[2, col_idx].set_yticks([])
            plt.colorbar(im3, ax=axes[2, col_idx], shrink=0.6)
            
            # Add row labels only for first column
            if col_idx == 0:
                axes[0, col_idx].set_ylabel('Input', fontsize=14, fontweight='bold')
                axes[1, col_idx].set_ylabel('Prediction', fontsize=14, fontweight='bold')
                axes[2, col_idx].set_ylabel('Target', fontsize=14, fontweight='bold')
    
    plt.suptitle('Navier-Stokes: All Channels Side-by-Side Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = os.path.join(save_dir, f'navier_stokes_all_channels_{timestamp}.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved all channels comparison plot to: {combined_plot_path}")
    
    plt.close()   


def plot_2d_pde_examples_multiple(plot_data, test_resolutions, pde="PDE", save_dir=None, num_examples=10):
    """
    Plot multiple prediction vs target examples for 2D PDE across different resolutions.
    
    Args:
        plot_data: Dictionary with resolution as key and dict containing arrays of predictions, targets, inputs
        test_resolutions: List of resolutions tested
        pde: Name of PDE for labeling
        save_dir: Directory to save plots (optional)
        num_examples: Number of examples to plot
    """
    import matplotlib.pyplot as plt
    
    # For 2D data, we'll create separate plots for each example to avoid overcrowding
    # Create a summary plot showing one example across all resolutions
    fig, axes = plt.subplots(3, len(test_resolutions), 
                            figsize=(4*len(test_resolutions), 12))
    
    # Handle case where we only have one resolution
    if len(test_resolutions) == 1:
        axes = axes.reshape(3, 1)
    
    fig.suptitle(f'2D {pde.upper()} Predictions vs Targets - Example 1', fontsize=16, y=0.98)
    
    # Plot the first example across all resolutions
    example_idx = 0
    for res_idx, resolution in enumerate(test_resolutions):
        if (resolution in plot_data and 
            example_idx < len(plot_data[resolution]['predictions'])):
            
            # Get data for first example - assume single channel for visualization
            input_data = plot_data[resolution]['inputs'][example_idx]
            prediction = plot_data[resolution]['predictions'][example_idx]
            target = plot_data[resolution]['targets'][example_idx]
            
            # Handle multi-channel data by taking first channel
            if input_data.ndim == 3:  # (C, H, W)
                input_data = input_data[0]
                prediction = prediction[0]
                target = target[0]
            elif input_data.ndim == 4:  # (B, C, H, W)
                input_data = input_data[0, 0]
                prediction = prediction[0, 0]
                target = target[0, 0]
            
            # Plot input
            im1 = axes[0, res_idx].imshow(input_data, cmap='RdBu_r', aspect='equal')
            axes[0, res_idx].set_title(f'Input\nRes: {resolution}x{resolution}')
            axes[0, res_idx].set_xticks([])
            axes[0, res_idx].set_yticks([])
            plt.colorbar(im1, ax=axes[0, res_idx], shrink=0.8)
            
            # Plot prediction
            im2 = axes[1, res_idx].imshow(prediction, cmap='RdBu_r', aspect='equal')
            axes[1, res_idx].set_title(f'Prediction\nRes: {resolution}x{resolution}')
            axes[1, res_idx].set_xticks([])
            axes[1, res_idx].set_yticks([])
            plt.colorbar(im2, ax=axes[1, res_idx], shrink=0.8)
            
            # Plot target
            im3 = axes[2, res_idx].imshow(target, cmap='RdBu_r', aspect='equal')
            axes[2, res_idx].set_title(f'Target\nRes: {resolution}x{resolution}')
            axes[2, res_idx].set_xticks([])
            axes[2, res_idx].set_yticks([])
            plt.colorbar(im3, ax=axes[2, res_idx], shrink=0.8)
            
            # Calculate and display error
            error = np.mean((prediction - target)**2)
            axes[2, res_idx].text(0.02, 0.98, f'MSE: {error:.4f}', 
                                 transform=axes[2, res_idx].transAxes, 
                                 verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add row labels
            if res_idx == 0:
                axes[0, res_idx].set_ylabel('Input', fontsize=14, fontweight='bold')
                axes[1, res_idx].set_ylabel('Prediction', fontsize=14, fontweight='bold')
                axes[2, res_idx].set_ylabel('Target', fontsize=14, fontweight='bold')
        else:
            # No data available
            for row in range(3):
                axes[row, res_idx].text(0.5, 0.5, 'No data', ha='center', va='center', 
                                      transform=axes[row, res_idx].transAxes)
                axes[row, res_idx].set_title(f'Resolution {resolution}x{resolution}')
    
    plt.tight_layout()
    
    # Save main comparison plot
    if save_dir:
        filename = f'{pde.lower()}_2d_main_comparison.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved main 2D comparison plot: {filepath}")
    
    plt.show()
    
    # Create individual example plots for detailed analysis
    if save_dir:
        create_individual_2d_example_plots(plot_data, test_resolutions, pde, save_dir, num_examples)


def create_individual_2d_example_plots(plot_data, test_resolutions, pde, save_dir, num_examples):
    """Create individual plots for each example at each resolution"""
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for resolution in test_resolutions:
        if resolution not in plot_data:
            continue
            
        # Create plots for up to num_examples
        actual_examples = min(num_examples, len(plot_data[resolution]['predictions']))
        
        for ex_idx in range(actual_examples):
            # Get data for this example
            input_data = plot_data[resolution]['inputs'][ex_idx]
            prediction = plot_data[resolution]['predictions'][ex_idx]
            target = plot_data[resolution]['targets'][ex_idx]
            
            # Handle multi-channel data by taking first channel
            if input_data.ndim == 3:  # (C, H, W)
                input_data = input_data[0]
                prediction = prediction[0]
                target = target[0]
            elif input_data.ndim == 4:  # (B, C, H, W)
                input_data = input_data[0, 0]
                prediction = prediction[0, 0]
                target = target[0, 0]
            
            # Create 2x2 subplot for this example
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            # Input
            im1 = axes[0, 0].imshow(input_data, cmap='RdBu_r', aspect='equal')
            axes[0, 0].set_title('Input')
            axes[0, 0].set_xticks([])
            axes[0, 0].set_yticks([])
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Prediction
            im2 = axes[0, 1].imshow(prediction, cmap='RdBu_r', aspect='equal')
            axes[0, 1].set_title('Prediction')
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Target
            im3 = axes[1, 0].imshow(target, cmap='RdBu_r', aspect='equal')
            axes[1, 0].set_title('Target')
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Error
            error_data = np.abs(prediction - target)
            im4 = axes[1, 1].imshow(error_data, cmap='Reds', aspect='equal')
            axes[1, 1].set_title('Absolute Error')
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
            plt.colorbar(im4, ax=axes[1, 1])
            
            # Add error statistics
            mean_error = np.mean(error_data)
            max_error = np.max(error_data)
            std_error = np.std(error_data)
            axes[1, 1].text(0.02, 0.98, f'Mean: {mean_error:.4f}\nMax: {max_error:.4f}\nStd: {std_error:.4f}', 
                           transform=axes[1, 1].transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle(f'2D {pde.upper()} - Resolution {resolution}x{resolution} - Example {ex_idx+1}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save individual plot
            filename = f'{pde.lower()}_2d_res_{resolution}_example_{ex_idx+1}_{timestamp}.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved individual 2D plot: {filepath}")
            
            plt.close()
    
    print(f"Created individual plots for {actual_examples} examples at each resolution")


