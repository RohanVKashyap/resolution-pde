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
from model import FNO1d, FNO2d
from utils import plot_predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=500, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier Modes')
    parser.add_argument('--width', type=int, default=32, help='Width of the FNO layers')
    parser.add_argument('--n_blocks', type=int, default=4, help='FNO Blocks')
    parser.add_argument('--use_normalizer', type=int, default=-1, help='Output Normalization during training')

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

    if 'burger' in args.pde:
        train_loader, val_loader, test_loader, x_normalizer, y_normalizer = load_burger_data_from_mat(
            args.data_path1, args.data_path2, batch_size = batch_size)  

    elif 'darcy' in args.pde:
        train_loader, val_loader, test_loader, x_normalizer, y_normalizer = load_darcy_data_from_mat(
            args.data_path1, args.data_path2, batch_size)
    
    x_sample, y_sample = next(iter(train_loader))
    in_channels = x_sample.shape[1]
    out_channels = y_sample.shape[1]

    print(f"Input shape: {x_sample.shape}")
    print(f"Output shape: {y_sample.shape}")

    if 'burger' in args.pde:
        # Initialize model
        model = FNO1d(
            in_channels = in_channels,
            out_channels = out_channels,
            modes = args.modes,
            width = width,
            n_blocks = n_blocks).to(device)

    elif 'darcy' in args.pde:   
        # Initialize model
        model = FNO2d(
            in_channels = in_channels,
            out_channels = out_channels,
            modes1 = modes,
            modes2 = modes,
            width = width,
            n_blocks = n_blocks,
            ).to(device) 

    
    # Initialize optimizer
    iterations = epochs * len(train_loader)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    wandb.init(
        project=args.project_name,
        config={
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'modes': modes,
            'width': width,
            'pde': args.pde})
    
    # Train model
    loss_history, val_loss_history = train(
        model, train_loader, val_loader, optimizer, scheduler, y_normalizer, args.use_normalizer, epochs = epochs)

    save_dir = os.path.join(args.checkpoint_dir, args.project_name)
    job_id = os.environ.get('SLURM_JOB_ID', 'local') 

    avg_l2_loss = evaluate(model, test_loader, y_normalizer, args.pde, job_id)

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'{args.project_name}_{job_id}.pt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
    }, checkpoint_path)
    
    print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()