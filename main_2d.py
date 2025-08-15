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

from dataloaders.ns_resize_markov import cno_ns_markov_dataset
from train.training import train, cno_evaluate
from models.custom_layer import UnitGaussianNormalizer
from utils.utils import plot_predictions, RelativeL2Loss 
from utils.resize_utils import get_lower_resolutions, evaluate_cno_original_all_resolution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# path: lr_${learning_rate}_modes_${model.modes}_ep_${epochs}_${now:%Y-%m-%d_%H-%M-%S}
# git clone https://RohanVKashyap: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@github.com/RohanVKashyap/resolution-pde.git
# python3 new_main.py model=ffno_1d/ffno_1d dataset=burger/burger_0.01

# This was earlier cno_original_mai.py (CORRECT)

@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(args: DictConfig):
    logging.info(OmegaConf.to_yaml(args))

    print(f"Project name: {args.project_name}")
    print(f"Model name: {args.model._target_}")
    # print(f"Dataset name: {args.dataset['dataset']['_target_']}")
    print(f"PDE Dataset: {args.dataset['pde']}")

    batch_size = args.training.batch_size
    learning_rate = args.training.learning_rate
    epochs = args.training.epochs
    use_normalizer = args.training.use_normalizer 

    data_resolution = args.dataset['original_res']                            # Original Data Resolution (512)
    train_resolution = args.dataset['dataset_params']['s']                    # Training Resolution (128)

    model_type = args.model._target_.split(".")[-1].lower()   

    # Load appropriate dataset based on PDE type
    if 'navier' in args.dataset['pde']:
        print('---------------------')
        # train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model = cno_ns_markov_dataset(
        #                                                     filename=args.dataset.data_path1,
        #                                                     saved_folder = '.', 
        #                                                     reduced_batch=args.dataset.reduced_batch, 
        #                                                     reduced_resolution=1,                       # CNO uses 's' always set this to 1
        #                                                     reduced_resolution_t=args.dataset.reduced_resolution_t, 
        #                                                     s=args.dataset.s, 
        #                                                     data_normalizer=True,
        #                                                     num_samples_max=-1)

        # Instantiate the dataset function
        train_dataset, val_dataset, test_dataset, min_data, max_data, min_model, max_model = instantiate(
            args.dataset.dataset_params,
            use_strain_orientation=True)     
                                                    
    else:
        raise ValueError(f"Unsupported PDE type: {args.pde}")
    
    print(val_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input shapes from a batch
    x_sample, y_sample = next(iter(train_loader))
    print(type(x_sample), torch.mean(x_sample).item(), torch.mean(y_sample).item(), torch.std(x_sample).item(), torch.std(y_sample).item())

    # print(f"Sample Input shape: {x_sample.shape}")
    # print(f"Sample Output shape: {y_sample.shape}")
    
    # Initialize model based on PDE type and model_type: CNO Model
    if model_type == 'cno':
        model = instantiate(
                args.model,
                _recursive_=False,
                in_size=train_resolution).to(device)
        
    elif model_type == 'pos':  
        from scOT.model import ScOT, ScOTConfig
        model_config = ScOTConfig(**args.model)  
        model = ScOT.from_pretrained('camlab-ethz/Poseidon-B', config=model_config, 
                                    ignore_mismatched_sizes=True).to(device) 

    else:
        model = instantiate(
                args.model,
                _recursive_=False).to(device) 
        print(model)
    
    # Count and print total parameters in millions
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total Model Parameters: {total_params / 1e6:.2f}M')
    print(f'Trainable Parameters: {trainable_params / 1e6:.2f}M')
    
    # print(model)
    # out = model(x_sample.to(device))
    # print('Sample Output:', out.shape)
    # loss_fn = RelativeL2Loss(size_average=True)
    # print('Loss:', loss_fn(y_sample.to(device), torch.rand_like(y_sample).to(device)))
    
    # # Initialize optimizer and scheduler: Vanilla CNO Training
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # Initialize optimizer and scheduler: Original CNO Training
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # Initialize WandB

    job_id = os.environ.get('SLURM_JOB_ID', 'local')

    wandb.init(
        project=model_type,
        config={
            'model_type': model_type,
            'project_name': args.project_name,
            'pde': args.dataset['pde'],
            'job_id': job_id,
            'reduced_resolution': args.dataset['dataset_params']['reduced_resolution'],
            'reduced_resolution_t': args.dataset['dataset_params']['reduced_resolution_t'],
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
        })
    
    # Train model  
    loss_history, val_loss_history = train(
                        model, train_loader, val_loader, 
                        optimizer, scheduler, y_normalizer=None, use_normalizer=use_normalizer, 
                        time=1, model_type=model_type, epochs=epochs)
    
    # Test model
    avg_l2_loss = cno_evaluate(model=model, 
                               test_loader=test_loader, 
                               min_data=min_data, 
                               max_data=max_data, 
                               min_model=min_model, 
                               max_model=max_model, 
                               pde=args.dataset['pde'], 
                               time=1, 
                               model_type=model_type,
                               job_id=job_id)

    # Save model checkpoint
    save_dir = os.path.join(args.checkpoint_dir, model_type)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'{args.dataset['pde']}_{job_id}.pt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'l2_loss': avg_l2_loss,
    }, checkpoint_path)
    
    print(f"Model saved to {checkpoint_path}")

    # Create directory for evaluation plots with job_id subdirectory
    figures_dir = os.path.join("figures", job_id)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Created/verified figures directory: {figures_dir}")
    
    # Superresolution Results
    test_resolutions = get_lower_resolutions(args.dataset['max_test_resolution'])       # Original Data Resolution (512)   
    # resolution_results = evaluate_cno_original_all_resolution(
    #                 model=model,
    #                 dataset_config=args.dataset,
    #                 current_res=args.dataset['dataset_params']['s'],
    #                 test_resolutions=test_resolutions,
    #                 data_resolution=data_resolution,
    #                 pde=args.dataset['pde'],
    #                 data_path=args.dataset['dataset_params']['filename'],
    #                 saved_folder='.',
    #                 reduced_batch=args.dataset['dataset_params']['reduced_resolution'],
    #                 reduced_resolution_t=args.dataset['dataset_params']['reduced_resolution_t'],  
    #                 min_data=min_data,
    #                 max_data=max_data,
    #                 min_model=min_model,
    #                 max_model=max_model,
    #                 batch_size=batch_size,
    #                 time=1,
    #                 model_type=model_type,
    #                 device='cuda',
    #                 plot_examples=True,
    #                 save_dir=figures_dir, 
    #                 analyze_frequencies=False,     # Plot Frequency Histogram                  
    #             )  
    
    resolution_results = evaluate_cno_original_all_resolution(
                    model=model,
                    dataset_config=args.dataset,
                    current_res=args.dataset['dataset_params']['s'],
                    test_resolutions=test_resolutions,
                    data_resolution=data_resolution,
                    pde=args.dataset['pde'],
                    saved_folder='.',
                    reduced_batch=args.dataset['dataset_params']['reduced_batch'],
                    reduced_resolution_t=args.dataset['dataset_params']['reduced_resolution_t'],  
                    reduced_resolution=1, # Use resize function for upsampling/downsampling 
                    min_data=min_data,
                    max_data=max_data,
                    min_model=min_model,
                    max_model=max_model,
                    batch_size=batch_size,
                    time=1,
                    model_type=model_type,
                    device='cuda',
                    plot_examples=True,
                    save_dir=figures_dir, 
                    analyze_frequencies=False,     # Plot Frequency Histogram                  
                )  
    
    wandb.log({"super_resolution": wandb.Table(
    columns=["Resolution Factor", "Relative L2 Loss"],
    data=[[res, loss] for res, loss in resolution_results.items()])})

    print("\nSummary of Super-Resolution Evaluation:")
    for res, loss in resolution_results.items():
        print(f"Resolution factor {res}: Relative L2 Loss = {loss:.6f}")  

    print(f"\nEvaluation plots saved to: {figures_dir}")
    print(f"Wandb run: {wandb.run.url}")
    
    # Finish wandb run
    wandb.finish()    

if __name__ == "__main__":
    main()