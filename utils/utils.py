import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_predictions(batch_x, batch_y_denorm, batch_pred_denorm, pde_type=None, max_examples=4, save_path=None):
    """
    Plot predictions for 1D or 2D data.
    
    Args:
        batch_x: Input tensor (batch_size, channels, ...)
        batch_y_denorm: Denormalized ground truth (batch_size, channels, ...)
        batch_pred_denorm: Denormalized predictions (batch_size, channels, ...)
        pde_type: String identifier of the PDE type (optional)
        max_examples: Maximum number of examples to plot
        save_path: Path to save the figure (optional)
    
    Returns:
        fig: The created matplotlib figure
    """
    
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    batch_x = to_numpy(batch_x)
    batch_y_denorm = to_numpy(batch_y_denorm)
    batch_pred_denorm = to_numpy(batch_pred_denorm)
    
    num_examples = min(max_examples, batch_x.shape[0])
    
    # Check if data is 1D or 2D
    is_1d_data = len(batch_x[0, 0].shape) == 1
    
    if is_1d_data:
        # 1D data plotting (for Burger's equation)
        fig, axs = plt.subplots(num_examples, 1, figsize=(15, 4*num_examples))
        if num_examples == 1:
            axs = [axs] 
            
        for i in range(num_examples):
            x_axis = np.linspace(0, 2*np.pi, batch_x[i, 0].shape[0])
            axs[i].plot(x_axis, batch_x[i, 0], label='Input', linewidth=2)
            axs[i].plot(x_axis, batch_y_denorm[i, 0], label='Ground Truth', linewidth=2)
            axs[i].plot(x_axis, batch_pred_denorm[i, 0], label='Prediction', linewidth=2, linestyle='--')
            axs[i].set_title(f'Example {i+1}')
            axs[i].legend()
            axs[i].grid(True)
            
            if isinstance(batch_pred_denorm, torch.Tensor):
                error = torch.norm(batch_pred_denorm[i] - batch_y_denorm[i]) / torch.norm(batch_y_denorm[i])
                error_val = error.item()
            else:
                error = np.linalg.norm(batch_pred_denorm[i] - batch_y_denorm[i]) / np.linalg.norm(batch_y_denorm[i])
                error_val = error
                
            axs[i].set_xlabel(f'Rel. L2 Error: {error_val:.4f}')

    else:
        # 2D data plotting (for Darcy flow)
        fig, axs = plt.subplots(num_examples, 3, figsize=(15, 4*num_examples))
        if num_examples == 1:
            axs = [axs] 
            
        for i in range(num_examples):
            # Input
            im0 = axs[i, 0].imshow(batch_x[i, 0])
            axs[i, 0].set_title(f'Input {i+1}')
            fig.colorbar(im0, ax=axs[i, 0])
            
            # Ground Truth
            im1 = axs[i, 1].imshow(batch_y_denorm[i, 0])
            axs[i, 1].set_title(f'Ground Truth {i+1}')
            fig.colorbar(im1, ax=axs[i, 1])
            
            # Prediction
            im2 = axs[i, 2].imshow(batch_pred_denorm[i, 0])
            axs[i, 2].set_title(f'Prediction {i+1}')
            fig.colorbar(im2, ax=axs[i, 2])
            
            # Calculate relative L2 error
            if isinstance(batch_pred_denorm, torch.Tensor):
                error = torch.norm(batch_pred_denorm[i] - batch_y_denorm[i]) / torch.norm(batch_y_denorm[i])
                error_val = error.item()
            else:
                error = np.linalg.norm(batch_pred_denorm[i] - batch_y_denorm[i]) / np.linalg.norm(batch_y_denorm[i])
                error_val = error
                
            axs[i, 2].set_xlabel(f'Rel. L2 Error: {error_val:.4f}')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig       