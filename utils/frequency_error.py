import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter

########################################################
"""Error Decomposition by Frequency Modes"""
########################################################

# def decompose_error_by_frequency_1d(y_hat, y, num_modes=None):
#     """
#     Decompose error and solution magnitude across Fourier modes.
#     This shows WHERE (which frequencies) the model is making errors.
#     """
#     device = y.device
#     B, C, H = y.shape
    
#     # Compute FFT
#     y_hat_fft = torch.fft.rfft(y_hat, dim=-1)  # (B, C, n_freq)
#     y_fft = torch.fft.rfft(y, dim=-1)          # (B, C, n_freq)
    
#     n_freq = y_fft.shape[-1]
#     if num_modes is None:
#         num_modes = n_freq
#     else:
#         num_modes = min(num_modes, n_freq)
    
#     # Get frequencies
#     frequencies = torch.fft.rfftfreq(H, device=device).cpu().numpy()
    
#     # Compute error and magnitude directly in Fourier space
#     error_per_mode = torch.abs(y_hat_fft - y_fft).pow(2).sum(dim=(0,1)).sqrt().cpu().numpy()
#     solution_magnitude_per_mode = torch.abs(y_fft).pow(2).sum(dim=(0,1)).sqrt().cpu().numpy()
    
#     return error_per_mode[:num_modes], solution_magnitude_per_mode[:num_modes], frequencies[:num_modes]

def decompose_error_by_frequency_1d(y_hat, y, num_modes=None):
    """
    Decompose error across different Fourier modes.
    Shows error contribution per frequency mode AND solution magnitude per mode.
    
    Args:
        y_hat: predictions (B, channels, H)
        y: targets (B, channels, H)
        num_modes: number of modes to analyze (if None, use all)
    
    Returns:
        error_per_mode: L2 error in each Fourier mode
        solution_magnitude_per_mode: magnitude of solution in each mode
        frequencies: frequency values for each mode
    """
    device = y.device
    B, C, H = y.shape
    
    # Compute FFT
    y_hat_fft = torch.fft.rfft(y_hat, dim=-1)  # (B, C, n_freq)
    y_fft = torch.fft.rfft(y, dim=-1)          # (B, C, n_freq)
    
    n_freq = y_fft.shape[-1]
    if num_modes is None:
        num_modes = n_freq
    else:
        num_modes = min(num_modes, n_freq)
    
    # Get frequencies
    frequencies = torch.fft.rfftfreq(H, device=device).cpu().numpy()
    
    # Initialize arrays
    error_per_mode = np.zeros(num_modes)
    solution_magnitude_per_mode = np.zeros(num_modes)
    
    for mode_idx in range(num_modes):
        # Extract single mode
        y_hat_mode = torch.zeros_like(y_hat_fft)
        y_mode = torch.zeros_like(y_fft)
        
        y_hat_mode[..., mode_idx] = y_hat_fft[..., mode_idx]
        y_mode[..., mode_idx] = y_fft[..., mode_idx]
        
        # Inverse FFT to get spatial contribution of this mode
        y_hat_spatial = torch.fft.irfft(y_hat_mode, n=H, dim=-1)
        y_spatial = torch.fft.irfft(y_mode, n=H, dim=-1)
        
        # Compute L2 error in this mode
        error_per_mode[mode_idx] = torch.norm(y_hat_spatial - y_spatial).item()
        
        # Compute magnitude of solution in this mode
        solution_magnitude_per_mode[mode_idx] = torch.norm(y_spatial).item()
    
    return error_per_mode, solution_magnitude_per_mode, frequencies[:num_modes]

def decompose_error_by_frequency_2d(y_hat, y, num_radial_bins=64):
    """
    Decompose error across different radial frequency bands in 2D.
    Returns radially-averaged error and solution magnitude.
    
    Args:
        y_hat: predictions (B, channels, H, W)
        y: targets (B, channels, H, W)
        num_radial_bins: number of radial frequency bins
    
    Returns:
        error_per_bin: L2 error in each radial frequency bin
        solution_magnitude_per_bin: magnitude of solution in each bin
        radial_freqs: center frequency of each bin
    """
    device = y.device
    B, C, H, W = y.shape
    
    # Compute 2D FFT
    y_hat_fft = torch.fft.rfft2(y_hat, dim=(-2, -1))
    y_fft = torch.fft.rfft2(y, dim=(-2, -1))
    
    # Compute radial frequency
    freq_y = torch.fft.fftfreq(H, device=device)
    freq_x = torch.fft.rfftfreq(W, device=device)
    fy = freq_y.view(-1, 1)
    fx = freq_x.view(1, -1)
    radial_freq = torch.sqrt(fy**2 + fx**2)
    
    # Define radial bins
    # cycles per sample (Nyquist frequency), 1 sample per grid point, Nyquist frequency: Half the sampling rate = 0.5
    max_freq = 0.5    
    freq_bins = np.linspace(0, max_freq, num_radial_bins + 1)
    
    error_per_bin = np.zeros(num_radial_bins)
    solution_magnitude_per_bin = np.zeros(num_radial_bins)
    radial_freqs = np.zeros(num_radial_bins)
    
    for i in range(num_radial_bins):
        freq_start = freq_bins[i]
        freq_end = freq_bins[i + 1]
        
        # Create mask for this frequency band
        mask = (radial_freq >= freq_start) & (radial_freq < freq_end)
        
        # Skip if bin is empty
        if mask.sum() == 0:
            error_per_bin[i] = 0.0
            solution_magnitude_per_bin[i] = 0.0
            radial_freqs[i] = (freq_start + freq_end) / 2
            continue
        
        # Expand mask to match FFT dimensions: (1, 1, H, W//2+1)
        mask_tensor = mask.unsqueeze(0).unsqueeze(0)
        
        # Extract frequency band
        y_hat_band_fft = y_hat_fft * mask_tensor
        y_band_fft = y_fft * mask_tensor
        
        # Inverse FFT to get spatial contribution
        y_hat_spatial = torch.fft.irfft2(y_hat_band_fft, s=(H, W), dim=(-2, -1))
        y_spatial = torch.fft.irfft2(y_band_fft, s=(H, W), dim=(-2, -1))
        
        # Compute L2 error and solution magnitude
        error_per_bin[i] = torch.norm(y_hat_spatial - y_spatial).item()
        solution_magnitude_per_bin[i] = torch.norm(y_spatial).item()
        
        radial_freqs[i] = (freq_start + freq_end) / 2
    
    return error_per_bin, solution_magnitude_per_bin, radial_freqs