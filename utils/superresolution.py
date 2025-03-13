import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
import os
from dataloaders.load_data import UnitGaussianNormalizer, load_burger_data_from_mat
from legacy.model import FNO1d

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_burger_test_data(data_path, batch_size=16):
    """Load only test portion of Burger's equation data"""
    data = scipy.io.loadmat(data_path)
    
    if 'a' in data and 'u' in data:
        X = data['a']  # Initial condition
        y = data['u']  # Solution
    else:
        raise ValueError("Expected 'a' and 'u' keys not found in .mat file")
    
    # Convert to PyTorch tensors and add channel dimension
    X = torch.from_numpy(X).float().unsqueeze(1)
    y = torch.from_numpy(y).float().unsqueeze(1)
    
    # Use last 10% as test set
    total_samples = X.shape[0]
    test_start = int(total_samples * 0.9)
    X_test, y_test = X[test_start:], y[test_start:]
    
    # Create simple normalizer based on test data (or load pre-trained normalizers)
    X_normalizer = UnitGaussianNormalizer(X_test)
    y_normalizer = UnitGaussianNormalizer(y_test)
    
    # Normalize test data
    X_test_normalized = X_normalizer.encode(X_test)
    y_test_normalized = y_normalizer.encode(y_test)
    
    # Create DataLoader
    test_dataset = TensorDataset(X_test_normalized, y_test_normalized)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader, X_test, y_test, X_normalizer, y_normalizer

def downsample_and_evaluate(model, test_data, test_target, x_normalizer, y_normalizer, resolution_scales=[1, 2, 4, 8, 16]):
    """Test model on different resolution scales"""
    results = {}
    
    for scale in resolution_scales:
        original_resolution = test_data.shape[2]
        print(f"\nTesting at resolution scale 1/{scale} ({original_resolution // scale} points)")
        
        # Downsample the test data and target
        if scale > 1:
            downsampled_data = test_data[:, :, ::scale]
            downsampled_target = test_target[:, :, ::scale]
        else:
            downsampled_data = test_data
            downsampled_target = test_target
        
        # Create new normalizers specifically for this resolution
        # This ensures the statistics match the tensor shapes
        downsampled_x_normalizer = UnitGaussianNormalizer(downsampled_data)
        downsampled_y_normalizer = UnitGaussianNormalizer(downsampled_target)
        
        # Normalize using the new normalizers
        downsampled_data_normalized = downsampled_x_normalizer.encode(downsampled_data)
        
        # Move to device
        downsampled_data_normalized = downsampled_data_normalized.to(device)
        downsampled_target = downsampled_target.to(device)
        
        # Run model inference
        model.eval()
        with torch.no_grad():
            # Get prediction
            downsampled_pred_normalized = model(downsampled_data_normalized)
            
            # Denormalize using the resolution-specific normalizer
            downsampled_pred = downsampled_y_normalizer.decode(downsampled_pred_normalized)
            
            # Calculate L2 error
            l2_error = F.mse_loss(downsampled_pred, downsampled_target).item()
            
            # Store results
            results[scale] = {
                'downsampled_data': downsampled_data.cpu(),
                'downsampled_target': downsampled_target.cpu(),
                'downsampled_pred': downsampled_pred.cpu(),
                'l2_error': l2_error,
            }
            
            print(f"Scale 1/{scale}: L2 Error = {l2_error:.6f}")
    
    return results

def plot_super_resolution_results(test_data, test_target, results, num_samples=3, save_path=None):
    """Plot input, ground truth, and predictions at different resolutions"""
    resolution_scales = list(results.keys())
    
    for i in range(num_samples):
        plt.figure(figsize=(12, 8))
        
        for j, scale in enumerate(resolution_scales):
            plt.subplot(len(resolution_scales), 1, j + 1)
            
            # Get downsampled data for this scale
            downsampled_data = results[scale]['downsampled_data'][i, 0].numpy()
            downsampled_target = results[scale]['downsampled_target'][i, 0].numpy()
            downsampled_pred = results[scale]['downsampled_pred'][i, 0].numpy()
            
            # Plot all three at the downsampled resolution
            plt.plot(downsampled_data, 'r-', label='Initial condition')
            plt.plot(downsampled_target, 'g-', label='Ground Truth')
            plt.plot(downsampled_pred, 'b-', label='FNO prediction')
            
            plt.title(f"Resolution Scale 1/{scale}: {downsampled_data.shape[0]} points, L2 Error = {results[scale]['l2_error']:.6f}")
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(f"{save_path}_sample_{i+1}.png")
        
        plt.show()
    
    # Create summary plot of errors vs resolution
    plt.figure(figsize=(10, 6))
    scales = list(results.keys())
    errors = [results[scale]['l2_error'] for scale in scales]
    
    plt.plot(scales, errors, 'o-')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Downsampling Factor')
    plt.ylabel('L2 Error')
    plt.title('Model Performance at Different Resolutions')
    plt.grid(True)
    
    if save_path:
        plt.savefig(f"{save_path}_error_summary.png")
    
    plt.show()

def main():
    # Path to your trained model
    model_path = '/home/rvk/checkpoints/burger_fno_1d/burger_fno_1d_4336857.pt'
    
    # Path to test data
    data_path = 'data/burger_data_mat/burgers_data_R10.mat'  # Update with your actual path
    
    # Resolution scales to test (e.g., 1, 2, 4, 8, 16)
    resolution_scales = [1, 2, 4, 8, 16]
    
    # Load test data
    test_loader, X_test, y_test, X_normalizer, y_normalizer = load_burger_test_data(data_path)
    
    # Load your trained model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration from checkpoint if available
    # Otherwise, use default values
    modes = 16
    width = 64
    if 'config' in checkpoint:
        modes = checkpoint.get('config', {}).get('modes', modes)
        width = checkpoint.get('config', {}).get('width', width)
    
    print(f"Creating model with modes={modes}, width={width}")
    
    # Initialize model with correct parameters
    model = FNO1d(
        in_channels=1,
        out_channels=1,
        modes=modes,
        width=width
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model at different resolutions
    results = downsample_and_evaluate(model, X_test, y_test, X_normalizer, y_normalizer, resolution_scales)
    
    # Plot and save results
    save_dir = 'resolution_test_results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'burger_resolution_test')
    
    plot_super_resolution_results(X_test, y_test, results, num_samples=3, save_path=save_path)
    
    # Print summary
    print("\nSummary of L2 Errors at Different Resolutions:")
    for scale in resolution_scales:
        print(f"Resolution Scale 1/{scale}: {results[scale]['l2_error']:.6f}")

if __name__ == "__main__":
    main()