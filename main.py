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
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from dataloaders.load_data import load_darcy_data, load_darcy_data_from_mat, load_burger_data_from_mat
from train.training import train, evaluate
from models.custom_layer import UnitGaussianNormalizer
from models.fno import FNO1d, FNO2d
from models.ffno import FFNO1D, FFNO2d
from models.unet import UNet1d, UNet2d
from utils.utils import plot_predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(args: DictConfig):
    logging.info(OmegaConf.to_yaml(args))

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    modes = args.model.modes
    width = args.model.width
    n_blocks = args.model.n_blocks
    model_type = args.model_type
    res_scale = args.res_scale
    use_normalizer = args.use_normalizer

    # Load appropriate dataset based on PDE type
    if 'burger' in args.dataset.pde:
        train_loader, val_loader, test_loader, x_normalizer, y_normalizer = load_burger_data_from_mat(
            args.dataset.data_path1, args.dataset.data_path2, res_scale = res_scale, batch_size=batch_size)  
    elif 'darcy' in args.dataset.pde:
        train_loader, val_loader, test_loader, x_normalizer, y_normalizer = load_darcy_data_from_mat(
            args.dataset.data_path1, args.dataset.data_path2, res_scale = res_scale, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported PDE type: {args.pde}")
    
    # Get input shapes from a batch
    x_sample, y_sample = next(iter(train_loader))
    in_channels = x_sample.shape[1]
    out_channels = y_sample.shape[1]

    print(f"Input shape: {x_sample.shape}")
    print(f"Output shape: {y_sample.shape}")


    # Initialize model based on PDE type and model_type
    if 'burger' in args.dataset.pde:
        if model_type == 'fno':
            model = FNO1d(
                in_channels=in_channels,
                out_channels=out_channels,
                modes=args.modes,
                width=width,
                n_blocks=n_blocks).to(device)
            
        elif model_type == 'ffno':  # FFNO
            model = FFNO1D(
                in_channels=in_channels + 1,  # +1 for grid coordinates
                out_channels=out_channels,
                hidden_dim=width,
                n_layers=n_blocks,
                n_modes=modes,
                factor=args.factor,
                ff_weight_norm=args.ff_weight_norm,
                n_ff_layers=args.n_ff_layers,
                layer_norm=args.layer_norm,
                grid=None,
                dropout=args.dropout,
                mode=args.mode).to(device)
            
        elif model_type == 'unet':
            model = UNet1d(
               in_channels=in_channels,
               out_channels=out_channels).to(device)
               
    elif 'darcy' in args.dataset.pde:
        if model_type == 'fno':
            model = FNO2d(
                in_channels=in_channels,
                out_channels=out_channels,
                modes1=modes,
                modes2=modes,
                width=width,
                n_blocks=n_blocks).to(device)
            
        elif model_type == 'ffno':
            model = FFNO2d(
                in_channels=in_channels + 2,  # +2 for grid coordinates
                out_channels=out_channels,
                hidden_dim=width,
                n_layers=n_blocks,
                n_modes=modes,
                factor=args.factor,
                ff_weight_norm=args.ff_weight_norm,
                n_ff_layers=args.n_ff_layers,
                layer_norm=args.layer_norm,
                grid=None,
                dropout=args.dropout,
                mode=args.mode).to(device)

        elif model_type == 'unet':
            model = UNet2d(
               in_channels=in_channels,
               out_channels=out_channels).to(device)    
    
    # Initialize optimizer and scheduler
    iterations = epochs * len(train_loader)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    # Initialize WandB
    wandb.init(
        project=args.dataset.project_name,
        config={
            'model_type': model_type,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'modes': modes,
            'width': width,
            'pde': args.dataset.pde,
            'factor': args.model.factor if model_type == 'ffno' else None,
            'ff_weight_norm': args.model.ff_weight_norm if model_type == 'ffno' else None,
            'n_ff_layers': args.model.n_ff_layers if model_type == 'ffno' else None,
            'layer_norm': args.model.layer_norm if model_type == 'ffno' else None,
            'dropout': args.model.dropout if model_type == 'ffno' else None,
            'mode': args.model.mode if model_type == 'ffno' else None,
        })
    
    # Train model
    loss_history, val_loss_history = train(
        model, train_loader, val_loader, optimizer, scheduler, y_normalizer, use_normalizer, epochs=epochs)

    # Create save directory and evaluate model
    save_dir = os.path.join(args.checkpoint_dir, args.dataset.project_name)
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    avg_l2_loss = evaluate(model, test_loader, y_normalizer, args.dataset.pde, job_id)

    # Save model checkpoint
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'{args.dataset.project_name}_{model_type}_{job_id}.pt')

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
