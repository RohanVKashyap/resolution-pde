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
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import h5py

from dataloaders.burger_s4 import burger_window_dataset
from dataloaders.ns_s4 import ns_window_dataset
from train.training import train, evaluate
from models.custom_layer import UnitGaussianNormalizer
from utils.utils import plot_predictions, RelativeL2Loss, evaluate_super_resolution, evaluate_s4_higher_resolution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# path: lr_${learning_rate}_modes_${model.modes}_ep_${epochs}_${now:%Y-%m-%d_%H-%M-%S}
# git clone https://RohanVKashyap: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@github.com/RohanVKashyap/resolution-pde.git
# python3 new_main.py model=ffno_1d/ffno_1d dataset=burger/burger_0.01

@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(args: DictConfig):
    logging.info(OmegaConf.to_yaml(args))

    print(f"Project name: {args.project_name}")
    print(f"Model name: {args.model._target_}")
    print(f"PDE Dataset: {args.dataset.pde}")

    batch_size = args.training.batch_size
    learning_rate = args.training.learning_rate
    epochs = args.training.epochs
    use_normalizer = args.training.use_normalizer

    # Load appropriate dataset based on PDE type
    if 'burger' in args.dataset.pde:
        train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer = burger_window_dataset(
                                                filename=args.dataset.data_path1,
                                                saved_folder = '.', 
                                                reduced_batch=args.dataset.reduced_batch, 
                                                reduced_resolution=args.dataset.reduced_resolution, 
                                                reduced_resolution_t=args.dataset.reduced_resolution_t,
                                                window_size=args.dataset.window_size, 
                                                num_samples_max=-1)

    elif 'navier' in args.dataset.pde:   
        train_dataset, val_dataset, test_dataset, x_normalizer, y_normalizer = ns_window_dataset(
                                                filename=args.dataset.data_path1,
                                                saved_folder='.',
                                                reduced_batch=args.dataset.reduced_batch, 
                                                reduced_resolution=args.dataset.reduced_resolution, 
                                                reduced_resolution_t=args.dataset.reduced_resolution_t,
                                                window_size=args.dataset.window_size, 
                                                flatten_window=True,           # Important for S4-FFNO 
                                                data_normalizer=True)
                                                    
    else:
        raise ValueError(f"Unsupported PDE type: {args.pde}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input shapes from a batch
    x_sample, y_sample = next(iter(train_loader))
    print(type(x_sample), torch.mean(x_sample).item(), torch.mean(y_sample).item(), torch.std(x_sample).item(), torch.std(y_sample).item())

    print(f"Sample Input shape: {x_sample.shape}")
    print(f"Sample Output shape: {y_sample.shape}")

    # Initialize model based on PDE type and model_type
    model = instantiate(
            args.model,
            _recursive_=False).to(device)  
    
    # model.train() 
    # total_l2_loss = 0.0
    # num_batches = 0
    # loss_fn = RelativeL2Loss(size_average=True)

    # print('-----Hi------', model(x_sample.to(device)).shape)

    # with torch.no_grad():
    #     for batch_x, batch_y in test_loader:
    #         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #         batch_pred = model(batch_x)
    #         print(batch_x.shape, batch_y.shape, batch_pred.shape)
    #         loss = loss_fn(batch_pred, batch_y)
    #         print('----loss------,', loss)
    #         break

    # avg_l2_loss = evaluate(model, test_loader, y_normalizer, args.dataset.pde)
    # print('average loss:', avg_l2_loss)

    # out = model(x_sample.to(device))
    # print('Sample Output:', out.shape)
    # loss_fn = RelativeL2Loss(size_average=True)
    # print('Loss:', loss_fn(y_sample.to(device), torch.rand_like(y_sample).to(device)))
    
    # Initialize optimizer and scheduler
    iterations = epochs * len(train_loader)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Initialize WandB
    model_type = args.model._target_.split(".")[-1].lower()
    job_id = os.environ.get('SLURM_JOB_ID', 'local')

    wandb.init(
        project=model_type,
        config={
            'model_type': model_type,
            'project_name': args.project_name,
            'pde': args.dataset.pde,
            'job_id': job_id,
            'reduced_resolution': args.dataset.reduced_resolution,
            'reduced_resolution_t': args.dataset.reduced_resolution_t,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'modes': args.model.n_modes if 'ffno' in model_type else None,
            'd_model': args.model.d_model if hasattr(args.model, 'd_model') else None,
            'width': args.model.width if hasattr(args.model, 'width') else None,
            'factor': args.model.factor if 'ffno' in model_type else None,
            'ff_weight_norm': args.model.ff_weight_norm  if 'ffno' in model_type else None,
            'n_ff_layers': args.model.n_ff_layers  if 'ffno' in model_type else None,
            'layer_norm': args.model.layer_norm  if 'ffno' in model_type else None,
            'dropout': args.model.dropout  if 'ffno' in model_type else None,
            'mode': args.model.mode  if 'ffno' in model_type else None,
            'previous_history': args.dataset.window_size  if 's4' in model_type else None,
        })
    
    # Train model
    loss_history, val_loss_history = train(
        model, train_loader, val_loader, optimizer, scheduler, y_normalizer, use_normalizer, epochs=epochs)
    
    # Test model
    avg_l2_loss = evaluate(model, test_loader, y_normalizer, args.dataset.pde, job_id)

    # Save model checkpoint
    save_dir = os.path.join(args.checkpoint_dir, model_type)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'{args.dataset.pde}_{job_id}.pt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'l2_loss': avg_l2_loss,
    }, checkpoint_path)
    
    print(f"Model saved to {checkpoint_path}")

    # Superresolution Results
    t_resolutions = [1, 2, 4, 8, 16, 32]
    resolution_results = evaluate_super_resolution(model, test_loader, y_normalizer,
                                                   t_resolutions, device=device)
    
    current_res = x_sample.shape[-1]
    high_resolution_results = evaluate_s4_higher_resolution(model, current_res, args.dataset.pde, 
                    args.dataset.data_path1, args.dataset.reduced_batch, args.dataset.window_size, args.dataset.reduced_resolution_t)
    
    wandb.log({"super_resolution": wandb.Table(
    columns=["Resolution Factor", "Relative L2 Loss"],
    data=[[res, loss] for res, loss in resolution_results.items()])})

    wandb.log({"higher_resolution": wandb.Table(
    columns=["Resolution Factor", "Relative L2 Loss"],
    data=[[res, loss] for res, loss in high_resolution_results.items()])})
    
    print("\nSummary of Super-Resolution Evaluation:")
    for res, loss in resolution_results.items():
        print(f"Resolution factor {res}: Relative L2 Loss = {loss:.6f}")
    
    print("\nSummary of Higher-Resolution Evaluation:")
    for res, loss in high_resolution_results.items():
        print(f"Resolution {res}: Relative L2 Loss = {loss:.6f}")      

if __name__ == "__main__":
    main()