import torch

def lowpass_filter_1d(data, cutoff_ratio=0.25):
    """
    Apply low-pass filter by zeroing out high-frequency Fourier modes.
    
    Args:
        data: (batch, time, channels, spatial_dim) or (batch, time, spatial_dim) - can be on GPU or CPU
        cutoff_ratio: fraction of frequencies to keep (0.25 = keep lowest 25%)
                     0.5 = Nyquist/2, 0.25 = Nyquist/4, etc.
    
    Returns:
        filtered_data: same shape and device as input, high frequencies removed
    """
    device = data.device
    original_shape = data.shape
    original_ndim = data.ndim
    
    # Handle case where channels dimension doesn't exist
    if original_ndim == 3:  # (batch, time, spatial_dim)
        data = data.unsqueeze(2)  # Add channels dim: (batch, time, 1, spatial_dim)
    
    # Take FFT along spatial dimension
    data_fft = torch.fft.rfft(data, dim=-1)  # (batch, time, channels, n_freq)
    
    # Determine cutoff frequency
    n_freqs = data_fft.size(-1)
    cutoff_idx = int(n_freqs * cutoff_ratio)
    
    # Zero out high frequencies
    data_fft[..., cutoff_idx:] = 0
    
    # Inverse FFT back to spatial domain
    filtered_data = torch.fft.irfft(data_fft, n=data.shape[-1], dim=-1)
    
    # Remove channels dimension if it was added
    if original_ndim == 3:
        filtered_data = filtered_data.squeeze(2)
    
    return filtered_data.to(device)

def lowpass_filter_2d(data, cutoff_ratio=0.25):
    """
    Apply low-pass filter by zeroing out high-frequency Fourier modes in 2D.
    Args:
        data: (batch, time, channels, spatial_dim, spatial_dim) or (batch, time, spatial_dim, spatial_dim)
              Can be on GPU or CPU. Assumes height == width.
        cutoff_ratio: fraction of spatial resolution to keep
                      e.g., 32/256 = 0.125 means keep frequencies up to resolution 32
    Returns:
        filtered_data: same shape and device as input, high frequencies removed
    """
    device = data.device
    original_shape = data.shape
    original_ndim = data.ndim
    
    # Handle case where channels dimension doesn't exist
    if original_ndim == 4:  # (batch, time, spatial_dim, spatial_dim)
        data = data.unsqueeze(2)  # Add channels dim: (batch, time, 1, spatial_dim, spatial_dim)
    
    # Take 2D FFT along spatial dimensions (last two dimensions)
    data_fft = torch.fft.rfft2(data, dim=(-2, -1))  # (batch, time, channels, spatial_dim, spatial_dim//2+1)
    
    # Determine cutoff frequency
    spatial_dim = data.shape[-1]
    n_freq = data_fft.shape[-1]
    
    # Create frequency grids
    freq_y = torch.fft.fftfreq(spatial_dim, device=device)  # (spatial_dim,)
    freq_x = torch.fft.rfftfreq(spatial_dim, device=device)  # (spatial_dim//2+1,)
    
    # Cutoff frequency (same as 1D logic)
    # When you sample data at some rate, the Nyquist frequency is the highest frequency you can represent, which is half the sampling rate.
    # torch.fft.fftfreq() and torch.fft.rfftfreq() return frequencies in "cycles per sample"
    # These frequencies range from 0 to 0.5 (for positive frequencies)
    # 0.5 = Nyquist = maximum representable frequency
    # This means: If cutoff_ratio = 0.125 (want 32 out of 256),  cutoff_freq = 0.0625 (in cycles per sample), Frequencies range from [-0.5, 0.5]
    # We keep |freq| <= 0.0625. his keeps 12.5% of the frequency range on each side. Total: 25% of frequency bins = 0.125 * 2
    cutoff_freq = cutoff_ratio * 0.5
    
    # Create mask: keep frequencies below cutoff in BOTH dimensions (rectangular)
    # This matches the 1D behavior more closely
    mask_y = (torch.abs(freq_y) <= cutoff_freq).to(device)  # (spatial_dim,)
    mask_x = (torch.abs(freq_x) <= cutoff_freq).to(device)  # (spatial_dim//2+1,)
    
    # Create 2D mask as outer product
    mask = mask_y.view(-1, 1) * mask_x.view(1, -1)  # (spatial_dim, spatial_dim//2+1)
    
    # Broadcast mask to match data_fft shape: (1, 1, 1, spatial_dim, n_freq)
    mask = mask.view(1, 1, 1, spatial_dim, n_freq)
    data_fft = data_fft * mask
    
    # Inverse FFT back to spatial domain
    filtered_data = torch.fft.irfft2(data_fft, s=(spatial_dim, spatial_dim), dim=(-2, -1))
    
    # Remove channels dimension if it was added
    if original_ndim == 4:
        filtered_data = filtered_data.squeeze(2)
    
    return filtered_data.to(device)

# cutoff_ratio = 64/256 = 0.25 (25% of spatial resolution)
# cutoff_freq = 0.25 * 0.5 = 0.125 (12.5% of Nyquist frequency)
# This keeps the lowest 64 cycles and removes everything above

# 32/256 = 0.125 → keeps exactly 32 cycles ✓
# 64/256 = 0.25 → keeps exactly 64 cycles ✓
# 128/256 = 0.5 → keeps exactly 128 cycles ✓