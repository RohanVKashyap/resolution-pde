import torch
import os
import scipy.io
import numpy as np
import math as mt
from einops import rearrange
import scipy.fft

# Some functions needed for loading the Navier-Stokes data
################################################################
#  2D Data
################################################################

def samples_fft(u):
    return scipy.fft.fft2(u, norm='forward', workers=-1)

def samples_ifft(u_hat):
    return scipy.fft.ifft2(u_hat, norm='forward', workers=-1).real
    
def downsample(u, N):
    N_old = u.shape[-2]
    freqs = scipy.fft.fftfreq(N_old, d=1/N_old)
    sel = np.logical_and(freqs >= -N/2, freqs <= N/2-1)
    u_hat = samples_fft(u)
    u_hat_down = u_hat[:,:,sel,:][:,:,:,sel]
    u_down = samples_ifft(u_hat_down)
    return u_down    

def resize(x, out_size, permute=False):
    if permute:
        x = x.permute(0, 3, 1, 2)
    f = torch.fft.rfft2(x, norm='backward')
    f_z = torch.zeros((*x.shape[:-2], out_size[0], out_size[1]//2 + 1), dtype=f.dtype, device=f.device)
    # 2k+1 -> (2k+1 + 1) // 2 = k+1 and (2k+1)//2 = k
    top_freqs1 = min((f.shape[-2] + 1) // 2, (out_size[0] + 1) // 2)
    top_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)

    # 2k -> (2k + 1) // 2 = k and (2k)//2 = k
    bot_freqs1 = min(f.shape[-2] // 2, out_size[0] // 2)
    bot_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)

    f_z[..., :top_freqs1, :top_freqs2] = f[..., :top_freqs1, :top_freqs2]
    f_z[..., -bot_freqs1:, :bot_freqs2] = f[..., -bot_freqs1:, :bot_freqs2]
    # x_z = torch.fft.ifft2(f_z, s=out_size).real
    x_z = torch.fft.irfft2(f_z, s=out_size).real
    x_z = x_z * (out_size[0] / x.shape[-2]) * (out_size[1] / x.shape[-1])
    # f_z[..., -f.shape[-2]//2:, :f.shape[-1]] = f[..., :f.shape[-2]//2+1, :]
    if permute:
        x_z = x_z.permute(0, 2, 3, 1)
    return x_z


################################################################
#  1D Data
################################################################

def samples_fft_1d(u):
    return scipy.fft.fft(u, norm='forward', workers=-1, axis=-1)

def samples_ifft_1d(u_hat):
    return scipy.fft.ifft(u_hat, norm='forward', workers=-1, axis=-1).real

def downsample_1d(u, N):
    """
    Downsample 1D data using FFT
    
    Args:
        u: Input array of shape (..., N_old) where N_old is the spatial dimension
        N: Desired output size for the last dimension
    
    Returns:
        Downsampled array of shape (..., N)
    """
    N_old = u.shape[-1]
    
    # Create frequency grid for 1D
    freqs = scipy.fft.fftfreq(N_old, d=1/N_old)
    
    # Select frequencies within the desired range
    sel = np.logical_and(freqs >= -N/2, freqs <= N/2-1)
    
    # Take 1D FFT
    u_hat = samples_fft_1d(u)
    
    # Select frequencies (only need to index the last dimension)
    u_hat_down = u_hat[..., sel]
    
    # Inverse FFT to get downsampled signal
    u_down = samples_ifft_1d(u_hat_down)
    
    return u_down

def resize_1d(x, out_size):
    """
    Resize 1D data using FFT interpolation
    
    Args:
        x: Input tensor of shape (..., N) where N is the spatial dimension
        out_size: Integer, desired output size for the last dimension
    
    Returns:
        Resized tensor of shape (..., out_size)
    """
    # Get the input size of the last dimension
    in_size = x.shape[-1]
    
    # Take 1D real FFT along the last dimension
    f = torch.fft.rfft(x, norm='backward')
    
    # Create output frequency tensor
    f_z = torch.zeros((*x.shape[:-1], out_size//2 + 1), dtype=f.dtype, device=f.device)
    
    # Determine how many frequency components to copy
    max_freqs = min(f.shape[-1], out_size//2 + 1)
    
    # Copy the frequency components
    f_z[..., :max_freqs] = f[..., :max_freqs]
    
    # Inverse FFT to get resized signal
    x_z = torch.fft.irfft(f_z, n=out_size).real
    
    # Scale using the size
    x_z = x_z * (out_size / in_size)
    
    return x_z