import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
import os
from prepare_data import UnitGaussianNormalizer, load_burger_data_from_mat
from model import FNO1d

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
    
    original_resolution = test_data.shape[2]
    
    for scale in resolution_scales:
        print(f"\nTesting at resolution scale 1/{scale} ({original_resolution // scale} points)")
        
        # Downsample the test data
        if scale > 1:
            downsampled_data = test_data[:, :, ::scale]
        else:
            downsampled_data = test_data
            
        # Normalize the downsampled data
        downsampled_data_normalized = x_normalizer.encode(downsampled_data)
        
        # Move to device
        downsampled_data_normalized = downsampled_data_normalized.to(device)
        
        # Run model inference
        model.eval()
        with torch.no_grad():
            downsampled_pred_normalized = model(downsampled_data_normalized)
            
            # Denormalize the prediction
            downsampled_pred = y_normalizer.decode(downsampled_pred_normalized.cpu())
            
            # Upsample prediction to original resolution using linear interpolation
            if scale > 1:
                # Shape: [batch, channels, seq_len]
                batch_size, channels, downsampled_length = downsampled_pred.shape
                upsampled_pred = F.interpolate(
                    downsampled_pred, 
                    size=original_resolution, 
                    mode='linear', 
                    align_corners=True
                )
            else:
                upsampled_pred = downsampled_pred
                
            # Calculate L2 error compared to original high-res ground truth
            l2_error = torch.mean((upsampled_pred - test_target)**2).item()
            rel_l2_error = torch.sqrt(torch.sum((upsampled_pred - test_target)**2) / torch.sum(test_target**2)).item()
            
            results[scale] = {
                'downsampled_data': downsampled_data,
                'upsampled_pred': upsampled_pred,
                'l2_error': l2_error,
                'rel_l2_error': rel_l2_error
            }
            
            print(f"Scale 1/{scale}: L2 Error = {l2_error:.6f}, Relative L2 Error = {rel_l2_error:.6f}")
    
    return results

def plot_super_resolution_results(test_data, test_target, results, num_samples=3, save_path=None):
    """Plot original input, ground truth, and predictions at different resolutions"""
    resolution_scales = list(results.keys())
    
    for i in range(num_samples):
        plt.figure(figsize=(15, 10))
        
        # Plot input
        plt.subplot(len(resolution_scales) + 2, 1, 1)
        plt.plot(test_data[i, 0].cpu().numpy())
        plt.title(f"Sample {i+1}: Input Initial Condition")
        plt.grid(True)
        
        # Plot ground truth
        plt.subplot(len(resolution_scales) + 2, 1, 2)
        plt.plot(test_target[i, 0].cpu().numpy(), 'k-', label='Ground Truth')
        plt.title("Ground Truth (High Resolution)")
        plt.grid(True)
        plt.legend()
        
        # Plot predictions at different resolutions
        for j, scale in enumerate(resolution_scales):
            plt.subplot(len(resolution_scales) + 2, 1, j + 3)
            
            # Plot ground truth
            plt.plot(test_target[i, 0].cpu().numpy(), 'k-', label='Ground Truth')
            
            # Plot downsampled points (what the model sees)
            downsampled = results[scale]['downsampled_data'][i, 0].cpu().numpy()
            downsampled_indices = np.arange(0, test_target.shape[2], scale)
            plt.plot(downsampled_indices, downsampled, 'rx', markersize=4, label=f'Downsampled Points (1/{scale})')
            
            # Plot upsampled prediction
            upsampled = results[scale]['upsampled_pred'][i, 0].cpu().numpy()
            plt.plot(upsampled, 'b-', label=f'Prediction (1/{scale}→Full)')
            
            plt.title(f"Resolution Scale 1/{scale}: L2 Error = {results[scale]['l2_error']:.6f}")
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
    errors = [results[scale]['rel_l2_error'] for scale in scales]
    
    plt.plot(scales, errors, 'o-')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Downsampling Factor')
    plt.ylabel('Relative L2 Error')
    plt.title('Super-resolution Performance')
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
    
    model = FNO1d(
        in_channels=1,
        out_channels=1,
        modes=16,  
        width=64 
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model at different resolutions
    results = downsample_and_evaluate(model, X_test, y_test, X_normalizer, y_normalizer, resolution_scales)
    
    # Plot and save results
    save_dir = 'super_resolution_results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'burger_super_res')
    
    plot_super_resolution_results(X_test, y_test, results, num_samples=3, save_path=save_path)
    
    # Print summary
    print("\nSummary of Relative L2 Errors:")
    for scale in resolution_scales:
        print(f"Resolution Scale 1/{scale}: {results[scale]['rel_l2_error']:.6f}")

if __name__ == "__main__":
    main()
