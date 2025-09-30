import argparse
import os
import sys
import math
import numpy as np
import torch
import h5py
import random
import matplotlib.pyplot as plt
from datetime import datetime

import time
from contextlib import contextmanager

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('/home/rvk')  # Add the path where utils is located

from typing import Tuple, List, Optional
from scipy.integrate import solve_ivp
from copy import copy
from equations.PDEs import PDE, KdV, KS, Heat
from utils.res_utils import downsample_1d, resize_1d


def check_files(pde: PDE, modes: dict, viscosity: float, nx: int, L: float, lmax: int, end_time: float, nt_effective: int, nt: int) -> None:
    """
    Check if data files exist and replace them if wanted.
    Args:
        pde (PDE): pde at hand [KS, KdV, Heat]
        modes (dict): mode ([train, valid, test]), replace, num_samples, training suffix
        viscosity (float): viscosity value for directory naming
        nx (int): spatial resolution for directory naming
        L (float): domain length for directory naming
        lmax (int): maximum frequency for directory naming
        end_time (float): end time for directory naming
        nt_effective (int): effective time steps for directory naming
        nt (int): total time steps for directory naming
    Returns:
            None
    """
    for mode, replace, num_samples, suffix in modes:
        save_name = f"/data/user_data/rvk/ks/res_{nx}/visc_{viscosity}_L{L}_lmax{lmax}_et{end_time}_nte{nt_effective}_nt{nt}/" + "_".join([str(pde), mode])
        if mode == "train":
            save_name = save_name + "_" + str(num_samples)
        if suffix:
            save_name = save_name + "_" + suffix
        if (replace == True):
            if os.path.exists(f'{save_name}.h5'):
                os.remove(f'{save_name}.h5')
                print(f'File {save_name}.h5 is deleted.')
            else:
                print(f'No file {save_name}.h5 exists yet.')
        else:
            print(f'File {save_name}.h5 is kept.')

def check_directory(viscosity: float, nx: int, L: float, lmax: int, end_time: float, nt_effective: int, nt: int) -> None:
    """
    Check if data and log directories exist, and create otherwise.
    Args:
        viscosity (float): viscosity value for directory naming
        nx (int): spatial resolution for directory naming
        L (float): domain length for directory naming
        lmax (int): maximum frequency for directory naming
        end_time (float): end time for directory naming
        nt_effective (int): effective time steps for directory naming
        nt (int): total time steps for directory naming
    """
    data_dir = f'/data/user_data/rvk/ks/res_{nx}/visc_{viscosity}_L{L}_lmax{lmax}_et{end_time}_nte{nt_effective}_nt{nt}'
    
    if os.path.exists(data_dir):
        print(f'Data directory {data_dir} exists and will be written to.')
    else:
        os.makedirs(data_dir, exist_ok=True)
        print(f'Data directory {data_dir} created.')

    log_dir = f'{data_dir}/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f'Log directory {log_dir} created.')


def initial_conditions(A: np.ndarray, phi: np.ndarray, l: np.ndarray, L: float):
    """
    Return initial conditions based on initial parameters.
    Args:
        A (np.ndarray): amplitude of different sine waves
        phi (np.ndarray): phase shift of different sine waves
        l (np.ndarray): frequency of different sine waves
        L (float): length of the spatial domain
    Returns:
        None
    """
    def fnc(x):
        u = np.sum(A * np.sin(2 * np.pi * l * x / L + phi), -1)
        return u
    return fnc

def params(pde: PDE, batch_size: int, device: torch.cuda.device="cpu") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get initial parameters for KdV, KS, and Burgers' equation.
    Args:
        pde (PDE): pde at hand [KS, KdV, Heat]
        batch_size (int): batch size
        device: device (cpu/gpu)
    Returns:
        np.ndarray: amplitude of different sin waves
        np.ndarray: phase shift
        np.ndarray: space dependent frequency
    """
    A = np.random.rand(1, pde.N) - 0.5
    phi = 2.0 * np.pi * np.random.rand(1, pde.N)
    l = np.random.randint(pde.lmin, pde.lmax, (1, pde.N))
    return A, phi, l

def inv_cole_hopf(psi0: np.ndarray, scale: float = 10.) -> np.ndarray:
    """
    Inverse Cole-Hopf transformation to obtain Heat equation out of initial conditions of Burgers' equation.
    Args:
        psi0 (np.ndarray): Burgers' equation (at arbitrary timestep) which gets transformed into Heat equation
        scale (float): scaling factor for transformation
    Returns:
        np.ndarray: transformed Heat equation
    """
    psi0 = psi0 - np.amin(psi0)
    psi0 = scale * 2 * ((psi0 / np.amax(psi0)) - 0.5)
    psi0 = np.exp(psi0)
    return psi0

def generate_trajectories(pde: PDE,
                          mode: str,
                          num_samples: int,
                          suffix: str,
                          batch_size: int,
                          viscosity: float,
                          nx: int,
                          L: float,
                          lmax: int,
                          end_time: float,
                          nt_effective: int,
                          nt: int,
                          device: torch.cuda.device="cpu") -> None:
    """
    Generate data trajectories for KdV, KS equation on periodic spatial domains.
    Args:
        pde (PDE): pde at hand [KS, KdV, Heat]
        mode (str): [train, valid, test]
        num_samples (int): how many trajectories do we create
        suffix (str): naming suffix for special trajectories
        batch_size (int): batch size
        viscosity (float): viscosity value for directory naming
        nx (int): spatial resolution for directory naming
        L (float): domain length for directory naming
        lmax (int): maximum frequency for directory naming
        end_time (float): end time for directory naming
        nt_effective (int): effective time steps for directory naming
        nt (int): total time steps for directory naming
        device: device (cpu/gpu)
    Returns:
        None
    """

    # parallel data generation is not yet implemented
    assert(batch_size == 1)
    num_batches = num_samples // batch_size

    pde_string = str(pde)
    print(f'Equation: {pde_string}')
    print(f'Mode: {mode}')
    print(f'Number of samples: {num_samples}')

    sys.stdout.flush()

    save_name = f"/data/user_data/rvk/ks/res_{nx}/visc_{viscosity}_L{L}_lmax{lmax}_et{end_time}_nte{nt_effective}_nt{nt}/" + "_".join([pde_string, mode])
    if mode == "train":
        save_name = save_name + "_" + str(num_samples)
    if suffix:
        save_name = save_name + '_' + suffix
    h5f = h5py.File("".join([save_name, '.h5']), 'a')
    dataset = h5f.create_group(mode)

    tcoord = {}
    xcoord = {}
    dx = {}
    dt = {}
    h5f_u = {}

    # Tolerance of the solver
    tol = 1e-9
    nt = pde.grid_size[0]
    nx = pde.grid_size[1]
    # The field u, the coordinations (xcoord, tcoord) and dx, dt are saved
    # Only nt_effective time steps of each trajectories are saved
    h5f_u = dataset.create_dataset(f'pde_{pde.nt_effective}-{nx}', (num_samples, pde.nt_effective, nx), dtype=float)
    xcoord = dataset.create_dataset(f'x', (num_samples, nx), dtype=float)
    dx = dataset.create_dataset(f'dx', (num_samples,), dtype=float)
    tcoord = dataset.create_dataset(f't', (num_samples, pde.nt_effective), dtype=float)
    dt = dataset.create_dataset(f'dt', (num_samples,), dtype=float)

    for idx in range(num_batches):

        T = pde.tmax
        L_actual = pde.L  # Use the L from PDE object

        t = np.linspace(pde.tmin, T, nt)
        x = np.linspace(0, (1 - 1.0 / nx) * L_actual, nx)

        # Parameters for initial conditions
        A, omega, l = params(pde, batch_size, device=device)

        # Initial condition of the equation at end
        u0 = initial_conditions(A, omega, l, L_actual)(x[:, None])

        # We use the initial condition of Burgers' equation and inverse Cole-Hopf transform it into the Heat equation
        if pde_string == 'Heat':
            u0 = inv_cole_hopf(u0)

        # We use pseudospectral reconstruction as spatial solver
        spatial_method = pde.pseudospectral_reconstruction

        # Solving for the full trajectories
        # For integration in time, we use an implicit Runge-Kutta method of Radau IIA family, order 5
        solved_trajectory = solve_ivp(fun=spatial_method,
                                      t_span=[t[0], t[-1]],
                                      y0=u0,
                                      method='Radau',
                                      t_eval=t,
                                      args=(L_actual, ),
                                      atol=tol,
                                      rtol=tol)

        # Saving the trajectories, if successfully solved
        if solved_trajectory.success:
            sol = solved_trajectory.y.T[-pde.nt_effective:]
            h5f_u[idx:idx+1, :, :] = sol
            xcoord[idx:idx+1, :] = x
            dx[idx:idx+1] = L_actual/nx
            tcoord[idx:idx + 1, :] = t[-pde.nt_effective:]
            dt[idx:idx+1] = T/(nt-1)

        else:
            print("Solution was not successful.")

        print("Solved indices: {:d} : {:d}".format(idx * batch_size, (idx + 1) * batch_size - 1))
        print("Solved batches: {:d} of {:d}".format(idx + 1, num_batches))
        sys.stdout.flush()
        sys.stderr.flush()

    print()
    print("Data saved")
    print()
    print()
    h5f.close()


def distance_by_timestep(data: torch.Tensor, loss=torch.nn.MSELoss()) -> List[float]:
    """
    Calculate distances between consecutive timesteps.
    Args:
        data: Tensor of shape (B, L, T, D) - (Batch, Length, Time, Dimensions)
        loss: Loss function to use
    Returns:
        List of distance values between consecutive timesteps
    """
    B, Sx, T, D = data.shape
    distances = []
    
    for t in range(T - 1):
        # Reshape to (B, L*D) and calculate loss between consecutive timesteps
        curr_step = data[..., t, :].reshape(B, -1)
        next_step = data[..., t + 1, :].reshape(B, -1)
        distance = loss(curr_step, next_step).item()
        distances.append(distance)
    
    return distances

def plot_distances(distances: List[float], title: str, loss_id: str, save_path: Optional[str] = None) -> str:
    """
    Create a neat and clean plot of temporal distances.
    Args:
        distances: List of distance values
        title: Plot title
        loss_id: Label for the loss metric
        save_path: Optional path to save the plot
    Returns:
        Path where the plot was saved
    """
    # Set up the plot with better aesthetics
    plt.style.use('default')  # Reset to clean style
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    
    # Create x-axis values
    timesteps = range(len(distances))
    
    # Plot with enhanced styling
    ax.plot(timesteps, distances,
            color='#1f77b4',
            linewidth=2.5,
            alpha=0.8,
            marker='o',
            markersize=4,
            markerfacecolor='#1f77b4',
            markeredgecolor='white',
            markeredgewidth=0.5,
            label=loss_id)
    
    # Enhanced title and labels
    ax.set_title(f'{title} - Temporal Evolution Rate',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time Step Transition', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{loss_id} Between Consecutive Steps', fontsize=12, fontweight='bold')
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_facecolor('#fafafa')
    
    # Better tick formatting
    ax.tick_params(labelsize=10, colors='#333333')
    
    # Enhanced legend
    ax.legend(loc='upper right', fontsize=11,
              framealpha=0.9, fancybox=True, shadow=True)
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1.2)
    
    # Add statistics annotation
    mean_val = np.mean(distances)
    std_val = np.std(distances)
    min_val = np.min(distances)
    max_val = np.max(distances)
    stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nRange: [{min_val:.4f}, {max_val:.4f}]'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Distance plot saved to: {save_path}")
    
    plt.close()  # Close to free memory
    return save_path

def create_visualization_plots(viscosity: float, 
                             nx: int, 
                             L: float, 
                             lmax: int, 
                             nt_effective: int,
                             end_time: float,
                             nt: int,
                             job_id: str = None, 
                             num_plots: int = 10,
                             resolutions: List[int] = [32, 64, 128, 256, 512],
                             dataset_type: str = 'valid') -> Tuple[str, List[str]]:
    """
    Enhanced visualization function with multiple resolutions and distance analysis.
    Args:
        viscosity (float): viscosity value for finding the file
        nx (int): spatial resolution for finding the file
        L (float): domain length for finding the file
        lmax (int): maximum frequency for finding the file
        nt_effective (int): effective time steps
        end_time (float): end time for directory naming
        nt (int): total time steps for directory naming
        job_id (str): job ID for creating unique save directory (optional)
        num_plots (int): number of random samples to plot
        resolutions (List[int]): list of target resolutions for plotting
        dataset_type (str): type of dataset ('train', 'valid', 'test')
    Returns:
        Tuple of (save_directory, list_of_saved_plots)
    """
    
    # Auto-generate job_id if not provided
    if job_id is None:
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id:
            job_id = f"job_{slurm_job_id}_visc_{viscosity}_L{L}_lmax{lmax}_et{end_time}_nte{nt_effective}_nt{nt}"
        else:
            job_id = f"visc_{viscosity}_L{L}_lmax{lmax}_et{end_time}_nte{nt_effective}_nt{nt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Auto-generated job_id: {job_id}")
    
    # Define file path with updated directory structure
    filename = f'/data/user_data/rvk/ks/res_{nx}/visc_{viscosity}_L{L}_lmax{lmax}_et{end_time}_nte{nt_effective}_nt{nt}/KS_{dataset_type}.h5'
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"{dataset_type.capitalize()} file {filename} not found. Skipping visualization.")
        return None, []
    
    # Create save directory
    save_path = f'./figures/{job_id}'
    os.makedirs(save_path, exist_ok=True)
    print(f"Created figure directory: {save_path}")
    
    saved_plots = []
    
    try:
        # Load data
        with h5py.File(filename, 'r') as f:
            u = f[dataset_type][f'pde_{nt_effective}-{nx}'][:]
            print(f"Loaded {dataset_type} data with shape: {u.shape}")
        
        # Validate data shape
        if len(u.shape) != 3:
            raise ValueError(f"Expected 3D data (samples, time, space), got shape {u.shape}")
        
        total_samples, total_time_steps, spatial_size = u.shape
        
        # Randomly select sample indices
        if total_samples < num_plots:
            print(f"Warning: Only {total_samples} samples available, using all of them.")
            num_plots = total_samples
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        sample_indices = sorted(random.sample(range(total_samples), num_plots))
        print(f"Selected random sample indices: {sample_indices}")
        
        # Select time indices (evenly distributed, including more points)
        num_time_points = min(14, total_time_steps)  # Use up to 14 time points
        if total_time_steps < num_time_points:
            time_indices = list(range(total_time_steps))
        else:
            time_indices = np.linspace(0, total_time_steps - 1, num_time_points, dtype=int).tolist()
        print(f"Selected time indices: {time_indices}")
        
        # Calculate and plot distances (only once)
        print("Calculating temporal distances...")
        # Reshape data for distance calculation: (B, L, T, D)
        u_reshaped = torch.from_numpy(u[:, :, :, None].transpose(0, 2, 1, 3))
        distances = distance_by_timestep(u_reshaped)
        
        # Create distance plot
        distance_plot_path = os.path.join(save_path, f'temporal_distances_{dataset_type}_visc_{viscosity}_L{L}_lmax{lmax}_et{end_time}_nte{nt_effective}_nt{nt}.png')
        plot_distances(distances, f'KS ({dataset_type.upper()})', 'MSE', distance_plot_path)
        saved_plots.append(distance_plot_path)
        
        # Generate plots for each resolution
        for res in resolutions:
            print(f"\nGenerating plots for resolution {res}...")
            
            # Set up the main evolution plot
            plt.style.use('default')
            fig, axes = plt.subplots(len(sample_indices), len(time_indices),
                                    figsize=(max(24, len(time_indices) * 2), max(10, len(sample_indices) * 3)), 
                                    facecolor='white')
            
            # Handle single sample or single time case
            if len(sample_indices) == 1 and len(time_indices) == 1:
                axes = np.array([[axes]])
            elif len(sample_indices) == 1:
                axes = axes.reshape(1, -1)
            elif len(time_indices) == 1:
                axes = axes.reshape(-1, 1)
            
            # Enhanced color scheme
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # Create spatial coordinates for x-axis
            x_coords = np.linspace(0, L, res)
            
            for row, sample_idx in enumerate(sample_indices):
                sample_color = colors[row % len(colors)]
                
                for col, t in enumerate(time_indices):
                    ax = axes[row, col]
                    
                    # Get and resize data
                    u_sample = u[sample_idx, t, :]
                    if res > nx:
                        # Convert to tensor for resize_1d
                        u_sample_tensor = torch.from_numpy(u_sample) if isinstance(u_sample, np.ndarray) else u_sample
                        u_sample_resized = resize_1d(u_sample_tensor, res).numpy()
                    elif res < nx:
                        u_sample_resized = downsample_1d(u_sample, res)
                    else:
                        u_sample_resized = u_sample
                    
                    # Plot with enhanced styling
                    ax.plot(x_coords, u_sample_resized, color=sample_color,
                           linewidth=2.5, alpha=0.9)
                    
                    # Enhanced styling
                    ax.set_xlabel('Spatial coordinate', fontsize=10, fontweight='bold')
                    ax.set_ylabel('u(x,t)', fontsize=10, fontweight='bold')
                    
                    # Enhanced title with more information
                    u_min, u_max = u_sample_resized.min(), u_sample_resized.max()
                    ax.set_title(f'Sample {sample_idx}, t = {t}\nRange: [{u_min:.2f}, {u_max:.2f}]',
                                fontsize=10, fontweight='bold', pad=10)
                    
                    # Enhanced grid and styling
                    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
                    ax.set_facecolor('#fafafa')
                    
                    # Keep consistent y-axis limits for each sample
                    if res != nx:
                        if res > nx:
                            # Convert to tensor for resize_1d
                            u_sample_tensor = torch.from_numpy(u[sample_idx]) if isinstance(u[sample_idx], np.ndarray) else u[sample_idx]
                            sample_data = resize_1d(u_sample_tensor, res).numpy()
                        else:
                            sample_data = np.stack([downsample_1d(u[sample_idx, i], res) 
                                                  for i in range(u.shape[1])])
                        sample_min = sample_data.min()
                        sample_max = sample_data.max()
                    else:
                        sample_min = u[sample_idx].min()
                        sample_max = u[sample_idx].max()
                    
                    margin = (sample_max - sample_min) * 0.05
                    ax.set_ylim(sample_min - margin, sample_max + margin)
                    
                    # Enhanced tick styling
                    ax.tick_params(labelsize=9, colors='#333333')
                    
                    # Add subtle border
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#cccccc')
                        spine.set_linewidth(1.2)
            
            # Enhanced main title
            main_title = (f'Kuramoto-Sivashinsky Evolution - Random Samples ({dataset_type.upper()})\n'
                         f'Original Resolution: {nx} → Resized to: {res} | '
                         f'Samples: {sample_indices} | Time Steps: {len(time_indices)} | '
                         f'Data Shape: {u.shape}')
            
            plt.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)
            
            # Enhanced layout
            plt.tight_layout(rect=[0, 0, 1, 0.94])
            fig.patch.set_facecolor('white')
            
            # Save evolution plot with new naming
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            evolution_filename = (f'ks_evolution_samples_{"-".join(map(str, sample_indices))}_'
                                 f'res_{res}_visc_{viscosity}_L{L}_lmax{lmax}_et{end_time}_nte{nt_effective}_nt{nt}_'
                                 f'{dataset_type}_{timestamp}.png')
            evolution_filepath = os.path.join(save_path, evolution_filename)
            
            plt.savefig(evolution_filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.2)
            saved_plots.append(evolution_filepath)
            plt.close()
        
        # Print summary
        print(f"\nVisualization Summary:")
        print(f"  - Dataset: {dataset_type}")
        print(f"  - Random samples plotted: {sample_indices}")
        print(f"  - Time points per sample: {len(time_indices)}")
        print(f"  - Original resolution: {nx}")
        print(f"  - Plot resolutions: {resolutions}")
        print(f"  - Distance analysis: Calculated {len(distances)} temporal transitions")
        print(f"  - Total plots saved: {len(saved_plots)}")
        print(f"  - Output directory: {save_path}")
        
        for i, plot_path in enumerate(saved_plots, 1):
            print(f"    {i}. {os.path.basename(plot_path)}")
        
        return save_path, saved_plots
        
    except Exception as e:
        print(f"Error in plotting: {e}")
        return save_path, saved_plots

def generate_data(experiment: str,
                  starting_time : float,
                  end_time: float,
                  L: float,
                  nx: int,
                  nt: int,
                  nt_effective: int,
                  num_samples_train: int,
                  num_samples_valid: int,
                  num_samples_test: int,
                  batch_size: int=1,
                  device: torch.cuda.device="cpu",
                  viscosity: float = 1.0,
                  lmax: int = 3
                  ) -> None:
    """
    Generate data for KdV, KS equation on periodic spatial domains.
    Args:
        experiment (str): pde at hand [KS, KdV, Heat]
        starting_time (float): starting time of PDE solving
        end_time (float): end time of PDE solving
        L (float): length of the spatial domain
        nx (int): spatial resolution
        nt (int): temporal resolution
        num_samples_train (int): number of trajectories created for training
        num_samples_valid (int): number of trajectories created for validation
        num_samples_test (int): number of trajectories created for testing
        batch_size (int): batch size
        device: device (cpu/gpu)
        viscosity (float): viscosity parameter
        lmax (int): maximum frequency for initial conditions
    Returns:
        None
    """
    print(f'Generating data')
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    if args.log:
        logfile = f'/data/user_data/rvk/ks/res_{nx}/visc_{viscosity}_L{L}_lmax{lmax}_et{end_time}_nte{nt_effective}_nt{nt}/log/{experiment}_time{timestring}.csv'
        print(f'Writing to log file {logfile}')
        sys.stdout = open(logfile, 'w')

    # Create instances of PDE
    if experiment == 'KdV':
        pde = KdV(tmin=starting_time,
                  tmax=end_time,
                  grid_size=(nt, nx),
                  nt_effective=nt_effective,
                  L=L,
                  device=device)
    elif experiment == 'KS':
        pde = KS(tmin=starting_time,
                 tmax=end_time,
                 grid_size=(nt, nx),
                 nt_effective=nt_effective,
                 L=L,
                 device=device, 
                 viscosity=viscosity,
                 lmax=lmax)

    elif experiment == 'Burgers':
        # Heat equation is generated; afterwards trajectories are transformed via Cole-Hopf transformation.
        # L is not set for Burgers equation, since it is very sensitive. Default value is 2*math.pi.
        pde = Heat(tmin=starting_time,
                 tmax=end_time,
                 grid_size=(nt, nx),
                 nt_effective=nt_effective,
                 device=device)

    else:
        raise Exception("Wrong experiment")

    # Check if train, valid and test files already exist and replace if wanted
    files = {("train", True, num_samples_train, args.suffix),
             ("valid", num_samples_valid > 0, num_samples_valid, args.suffix),
             ("test", num_samples_test > 0, num_samples_test, args.suffix)}
    check_files(pde, files, viscosity, nx, L, lmax, end_time, nt_effective, nt)

    # Obtain trajectories for different modes
    for mode, _, num_samples, suffix in files:
        if num_samples > 0:
            generate_trajectories(pde=pde,
                                  mode=mode,
                                  num_samples=num_samples,
                                  suffix=suffix,
                                  batch_size=batch_size,
                                  viscosity=viscosity,
                                  nx=nx,
                                  L=L,
                                  lmax=lmax,
                                  end_time=end_time,
                                  nt_effective=nt_effective,
                                  nt=nt,
                                  device=device)

def main(args: argparse) -> None:
    """
    Main method for data generation.
    """
    # Print all arguments for debugging
    print("=" * 60)
    print("ARGUMENTS:")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"{arg:20}: {value}")
    print("=" * 60)
    
    check_directory(args.viscosity, args.nx, args.L, args.lmax, args.end_time, args.nt_effective, args.nt)

    if args.device != "cpu":
        raise NotImplementedError

    if args.batch_size != 1:
        raise NotImplementedError
    
    start_time = time.time()
    generate_data(experiment=args.experiment,
                  starting_time=0.0,
                  end_time=args.end_time,
                  L=args.L,
                  nt=args.nt,
                  nt_effective=args.nt_effective,
                  nx=args.nx,
                  num_samples_train=args.train_samples,
                  num_samples_valid=args.valid_samples,
                  num_samples_test=args.test_samples,
                  batch_size=args.batch_size,
                  device=args.device,
                  viscosity=args.viscosity,
                  lmax=args.lmax)
    elapsed = time.time() - start_time
    print(f"Operation took {elapsed:.2f} seconds")
    
    # Generate enhanced visualization plots after data generation
    if args.create_plots:
        # Plot validation data if available
        if args.valid_samples > 0:
            print("\nGenerating validation plots...")
            save_path, plots = create_visualization_plots(
                viscosity=args.viscosity,
                nx=args.nx,
                L=args.L,
                lmax=args.lmax,
                nt_effective=args.nt_effective,
                end_time=args.end_time,
                nt=args.nt,
                job_id=None,  # Will auto-generate
                num_plots=args.num_plots,
                resolutions=getattr(args, 'plot_resolutions', [32, 64, 128, 256, 512]),
                dataset_type='valid'
            )
        
        # Plot test data if available
        if args.test_samples > 0:
            print("\nGenerating test plots...")
            save_path, plots = create_visualization_plots(
                viscosity=args.viscosity,
                nx=args.nx,
                L=args.L,
                lmax=args.lmax,
                nt_effective=args.nt_effective,
                end_time=args.end_time,
                nt=args.nt,
                job_id=None,  # Will auto-generate
                num_plots=args.num_plots,
                resolutions=getattr(args, 'plot_resolutions', [32, 64, 128, 256, 512]),
                dataset_type='test'
            )


def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating PDE data')
    parser.add_argument('--experiment', type=str, default='KdV',
                        help='Experiment for which data should create for: [KdV, KS, Burgers]')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Used device')
    parser.add_argument('--end_time', type=float, default=100.,
                        help='How long do we want to simulate')
    parser.add_argument('--nt', type=int, default=250,
                        help='Time steps used for solving')
    parser.add_argument('--nt_effective', type=int, default=140,
                        help='Solved timesteps used for training')
    parser.add_argument('--nx', type=int, default=256,
                        help='Spatial resolution')
    parser.add_argument('--L', type=float, default=128.,
                        help='Length for which we want to solve the PDE')
    parser.add_argument('--train_samples', type=int, default=2 ** 5,
                        help='Samples in the training dataset')
    parser.add_argument('--valid_samples', type=int, default=2 ** 5,
                        help='Samples in the validation dataset')
    parser.add_argument('--test_samples', type=int, default=2 ** 5,
                        help='Samples in the test dataset')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size used for creating training, val, and test dataset. So far the code only works for batch_size==1')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for additional datasets')
    parser.add_argument('--log', type=str2bool, default=False,
                        help='pipe the output to log file')
    parser.add_argument('--viscosity', type=float, default=1.0,
                        help='Viscosity of the KS equation (only used in the case of KS)')
    parser.add_argument('--lmax', type=int, default=3,
                        help='Maximum frequency of the initial conditions')
    parser.add_argument('--create_plots', type=str2bool, default=True,
                        help='Whether to create visualization plots after data generation')
    parser.add_argument('--num_plots', type=int, default=10,
                        help='Number of random samples to plot from validation dataset')
    parser.add_argument('--plot_resolutions', type=int, nargs='+', default=[32, 64, 128, 256, 512],
                        help='List of target resolutions for plotting (e.g., --plot_resolutions 32 64 128 256 512)')

    args = parser.parse_args()
    main(args)