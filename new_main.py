import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import scipy
import wandb  
import argparse

from prepare_data import UnitGaussianNormalizer, load_darcy_data, load_darcy_data_from_mat, load_burger_data_from_mat
from training import train, evaluate
from model import FNO1d, FNO2d, FFNO1D, FFNO2d
from utils import plot_predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    parser = argparse.ArgumentParser()
    # Basic training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=500, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use_normalizer', type=int, default=-1, help='Output Normalization during training')

    # Model selection
    parser.add_argument('--model_type', type=str, default='fno', choices=['fno', 'ffno'], help='Model type (fno or ffno)')
    
    # Common model parameters
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier Modes')
    parser.add_argument('--width', type=int, default=32, help='Width/hidden dimension of the layers')
    parser.add_argument('--n_blocks', type=int, default=4, help='Number of operator blocks in FNO or layers in FFNO')
    
    # FFNO-specific parameters
    parser.add_argument('--factor', type=int, default=4, help='Feed-forward expansion factor (FFNO only)')
    parser.add_argument('--ff_weight_norm', type=int, default=1, help='Use weight normalization (FFNO only)')
    parser.add_argument('--n_ff_layers', type=int, default=2, help='Number of feed-forward layers (FFNO only)')
    parser.add_argument('--layer_norm', type=int, default=1, help='Use layer normalization (FFNO only)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (FFNO only)')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'low-pass', 'no-fourier'], 
                       help='Fourier mode for FFNO (full, low-pass, or no-fourier)')

    # Project and data parameters
    parser.add_argument('--project_name', type=str, default='darcy_fno2d', help='Project')
    parser.add_argument('--pde', type=str, default=None, help='PDE')
    parser.add_argument('--data_path1', type=str, default=None, help='Data Path 1')
    parser.add_argument('--data_path2', type=str, default=None, help='Data Path 2')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint path')
    
    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    modes = args.modes
    width = args.width
    n_blocks = args.n_blocks
    model_type = args.model_type

    # Load appropriate dataset based on PDE type
    if 'burger' in args.pde:
        train_loader, val_loader, test_loader, x_normalizer, y_normalizer = load_burger_data_from_mat(
            args.data_path1, args.data_path2, batch_size=batch_size)  
    elif 'darcy' in args.pde:
        train_loader, val_loader, test_loader, x_normalizer, y_normalizer = load_darcy_data_from_mat(
            args.data_path1, args.data_path2, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported PDE type: {args.pde}")
    
    # Get input shapes from a batch
    x_sample, y_sample = next(iter(train_loader))
    in_channels = x_sample.shape[1]
    out_channels = y_sample.shape[1]

    print(f"Input shape: {x_sample.shape}")
    print(f"Output shape: {y_sample.shape}")

    # Initialize model based on PDE type and model_type
    if 'burger' in args.pde:
        if model_type == 'fno':
            model = FNO1d(
                in_channels=in_channels,
                out_channels=out_channels,
                modes=args.modes,
                width=width,
                n_blocks=n_blocks).to(device)
        else:  # FFNO
            model = FFNO1D(
                in_channels=in_channels + 1,  # +1 for grid coordinates
                out_channels=out_channels,
                hidden_dim=width,
                n_layers=n_blocks,
                n_modes=modes,
                factor=args.factor,
                ff_weight_norm=bool(args.ff_weight_norm),
                n_ff_layers=args.n_ff_layers,
                layer_norm=bool(args.layer_norm),
                dropout=args.dropout,
                mode=args.mode).to(device)
    elif 'darcy' in args.pde:
        if model_type == 'fno':
            model = FNO2d(
                in_channels=in_channels,
                out_channels=out_channels,
                modes1=modes,
                modes2=modes,
                width=width,
                n_blocks=n_blocks).to(device)
        else:  # FFNO
            model = FFNO2d(
                in_channels=in_channels + 2,  # +2 for grid coordinates
                out_channels=out_channels,
                hidden_dim=width,
                n_layers=n_blocks,
                n_modes=modes,
                factor=args.factor,
                ff_weight_norm=bool(args.ff_weight_norm),
                n_ff_layers=args.n_ff_layers,
                layer_norm=bool(args.layer_norm),
                dropout=args.dropout,
                mode=args.mode).to(device)
    
    # Initialize grid attribute for FFNO models
    if model_type == 'ffno':
        if hasattr(model, 'grid'):
            model.grid = None
    
    # Initialize optimizer and scheduler
    iterations = epochs * len(train_loader)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    # Initialize WandB
    wandb.init(
        project=args.project_name,
        config={
            'model_type': model_type,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'modes': modes,
            'width': width,
            'pde': args.pde,
            'factor': args.factor if model_type == 'ffno' else None,
            'ff_weight_norm': args.ff_weight_norm if model_type == 'ffno' else None,
            'n_ff_layers': args.n_ff_layers if model_type == 'ffno' else None,
            'layer_norm': args.layer_norm if model_type == 'ffno' else None,
            'dropout': args.dropout if model_type == 'ffno' else None,
            'mode': args.mode if model_type == 'ffno' else None,
        })
    
    # Train model
    loss_history, val_loss_history = train(
        model, train_loader, val_loader, optimizer, scheduler, y_normalizer, args.use_normalizer, epochs=epochs)

    # Create save directory and evaluate model
    save_dir = os.path.join(args.checkpoint_dir, args.project_name)
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    avg_l2_loss = evaluate(model, test_loader, y_normalizer, args.pde, job_id)

    # Save model checkpoint
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'{args.project_name}_{model_type}_{job_id}.pt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'l2_loss': avg_l2_loss,
    }, checkpoint_path)
    
    print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()