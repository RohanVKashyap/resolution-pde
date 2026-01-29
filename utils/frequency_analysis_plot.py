import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.frequency_error import decompose_error_by_frequency_1d

def perform_frequency_analysis_1d(frequency_data, test_resolutions, current_res, pde, save_dir):
    """
    Perform frequency decomposition analysis for 1D PDE predictions.
    
    Args:
        frequency_data: Dict with resolution keys, each containing 'predictions' and 'targets' tensors
        test_resolutions: List of resolutions tested
        current_res: Resolution the model was trained on
        pde: PDE name (for labeling)
        save_dir: Directory to save plots
    """
    
    # Decompose error for each resolution
    decomposition_results = {}
    
    for res in test_resolutions:
        if res not in frequency_data:
            continue
            
        y_pred = frequency_data[res]['predictions']  # (N, C, H)
        y_true = frequency_data[res]['targets']      # (N, C, H)
        
        print(f"Decomposing error for resolution {res}...")
        print(f"  Data shape: {y_pred.shape}")
        
        # Decompose error by frequency
        error_per_mode, solution_per_mode, frequencies = decompose_error_by_frequency_1d(
            y_pred, y_true, num_modes=None
        )
        
        decomposition_results[res] = {
            'error': error_per_mode,
            'solution': solution_per_mode,
            'frequencies': frequencies
        }
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_resolutions)))
    
    # Plot 1: Error per mode for different resolutions
    for idx, res in enumerate(test_resolutions):
        if res not in decomposition_results:
            continue
        data = decomposition_results[res]
        axes[0, 0].semilogy(data['frequencies'], data['error'], 
                           label=f'Res {res}', color=colors[idx], linewidth=2, marker='o', markersize=3)
    axes[0, 0].set_xlabel('Frequency (cycles per sample)', fontsize=11)
    axes[0, 0].set_ylabel('L2 Error per Mode (log scale)', fontsize=11)
    axes[0, 0].set_title('Error Decomposition by Fourier Mode', fontsize=12)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Solution spectral decay (should be same for all)
    first_res = test_resolutions[0]
    if first_res in decomposition_results:
        data = decomposition_results[first_res]
        axes[0, 1].semilogy(data['frequencies'], data['solution'], 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Frequency (cycles per sample)', fontsize=11)
        axes[0, 1].set_ylabel('Solution Magnitude (log scale)', fontsize=11)
        axes[0, 1].set_title(f'{pde.upper()} Solution Spectral Decay', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Normalized error for different resolutions
    for idx, res in enumerate(test_resolutions):
        if res not in decomposition_results:
            continue
        data = decomposition_results[res]
        normalized_error = data['error'] / (data['solution'] + 1e-10)
        axes[1, 0].semilogy(data['frequencies'], normalized_error,
                           label=f'Res {res}', color=colors[idx], linewidth=2, marker='o', markersize=3)
    axes[1, 0].set_xlabel('Frequency (cycles per sample)', fontsize=11)
    axes[1, 0].set_ylabel('Normalized Error (log scale)', fontsize=11)
    axes[1, 0].set_title('Normalized Error: Error/Solution Magnitude', fontsize=12)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error ratio comparison
    # Compare error at different frequencies relative to baseline
    # if len(test_resolutions) > 1:
    #     baseline_res = test_resolutions[0]  # Usually lowest resolution
    #     baseline_error = decomposition_results[baseline_res]['error']
        
    #     for idx, res in enumerate(test_resolutions[1:], 1):
    #         if res not in decomposition_results:
    #             continue
    #         data = decomposition_results[res]
    #         error_ratio = data['error'] / (baseline_error + 1e-10)
    #         axes[1, 1].plot(data['frequencies'], error_ratio,
    #                       label=f'Res {res} / Res {baseline_res}', 
    #                       color=colors[idx], linewidth=2, marker='o', markersize=3)
    #     axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Baseline')
    #     axes[1, 1].set_xlabel('Frequency (cycles per sample)', fontsize=11)
    #     axes[1, 1].set_ylabel('Error Ratio', fontsize=11)
    #     axes[1, 1].set_title('Error Improvement vs Baseline', fontsize=12)
    #     axes[1, 1].legend(fontsize=9)
    #     axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{pde.upper()}: Frequency Analysis (Trained on {current_res})', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{pde}_frequency_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved frequency analysis plot: {save_path}")
    plt.close()
    
    # Print statistics
    print("\nFrequency Analysis Summary:")
    for res in test_resolutions:
        if res not in decomposition_results:
            continue
        data = decomposition_results[res]
        n_modes = len(data['frequencies'])
        n_low = n_modes // 10
        n_high = n_modes // 10
        
        print(f"\nResolution {res}:")
        print(f"  Low freq error (avg first 10%): {data['error'][:n_low].mean():.6f}")
        print(f"  High freq error (avg last 10%): {data['error'][-n_high:].mean():.6f}")
        print(f"  Error ratio (high/low): {data['error'][-n_high:].mean() / (data['error'][:n_low].mean() + 1e-10):.2f}x")