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

from train.mres_training import create_grouped_dataloaders 
from utils.resize_utils import get_lower_resolutions

from utils.loss import RelativeL2Loss
from utils.multiresolution_analysis import evaluate_multiresolution_training_analysis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# git clone https://RohanVKashyap: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@github.com/RohanVKashyap/resolution-pde.git

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
    
    if 's' in args.dataset['dataset_params']:
        train_resolution = args.dataset['dataset_params']['s']
        print('Loaded Data using downsample_1d / resize_1d function')
    elif 'reduced_resolution' in args.dataset['dataset_params']:
        print('Loaded Data using naive downsampling')
        train_resolution = data_resolution // args.dataset['dataset_params']['reduced_resolution']
    else:
        print('This is True Multi-Resolution Training. Train Resolution is the highest data resolution in the multi-resolution data') 
        train_resolution = args.dataset['original_res']   

    normalizer_type = args.dataset['dataset_params']['normalization_type']
    evaluation_type = args.dataset['evaluation_type']

    model_type = args.model._target_.split(".")[-1].lower()  
    print('Model Type:', model_type) 

    # Load Data
    if any(substring in args.dataset['pde'] for substring in ['ks', 'burger', 'ns']):      
        print('---------------------')
        # Instantiate the dataset function: KS dataset has 2 additional files (default): KS_valid.h5, ks_test.h5
        data_ = instantiate(args.dataset.dataset_params, use_strain_orientation=True)
        # train_dataset, val_dataset, test_dataset = data_[:3]
        train_dataset, val_dataset, test_dataset, _ = data_[:4]          
                                                    
    else:
        raise ValueError(f"Unsupported PDE type: {args.pde}")
    

    if normalizer_type == 'simple':
        # x_normalizer, y_normalizer = data_[3:]
        x_normalizer, y_normalizer = data_[4:]
        min_data = max_data = min_model = max_model = None
    else:
        # min_data, max_data, min_model, max_model = data_[3:]
        min_data, max_data, min_model, max_model = data_[4:]
        x_normalizer = y_normalizer = None

    print(val_dataset)    
    
    if not args.dataset['train_mres']:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader, val_loader, test_loader = create_grouped_dataloaders(
                                  train_dataset, val_dataset, test_dataset, batch_size=batch_size)    
    
    # Get input shapes from a batch
    x_sample, y_sample = next(iter(train_loader))
    print(type(x_sample), torch.mean(x_sample).item(), torch.mean(y_sample).item(), torch.std(x_sample).item(), torch.std(y_sample).item())
    
    if model_type == 'cno1d':
        model = instantiate(
                args.model,
                _recursive_=False,
                size=args.dataset['cno_train_size']).to(device)
        
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

    # Initialize WandB
    job_id = os.environ.get('SLURM_JOB_ID', 'local')

    # Create directory for evaluation plots with job_id subdirectory
    figures_dir = os.path.join("figures", job_id)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Created/verified figures directory: {figures_dir}")
    
    # Superresolution Results
    # test_resolutions = get_lower_resolutions(data_resolution)       # Original Data Resolution (512) 
    test_resolutions = get_lower_resolutions(args.dataset['max_test_resolution']) 
    
    # This is only for multi-resolution (single-resolution this is None)
    eval_dataset_target=args.dataset['dataset_params'].get('eval_dataset_target', None)

    # Save Directory
    figures_dir = os.path.join("figures_paper", job_id)
    os.makedirs(figures_dir, exist_ok=True)
    
    model_checkpoints=args.dataset['model_checkpoints']
    loss_results, frequency_data = evaluate_multiresolution_training_analysis(
        model_checkpoints=model_checkpoints,
        model=model,
        dataset_config=args.dataset,
        eval_dataset_target=eval_dataset_target,
        test_resolution=data_resolution,  # IMPORTANT (RESOLUTION TO PLOT THE FREQUENCY ANALYSIS)
        data_resolution=data_resolution,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        pde=args.dataset['pde'],
        saved_folder=args.dataset['dataset_params']['saved_folder'],
        reduced_batch=args.dataset['dataset_params']['reduced_batch'],
        reduced_resolution_t=args.dataset['dataset_params']['reduced_resolution_t'],
        batch_size=args.training.batch_size,
        time=1,
        checkpoint_resolutions=args.dataset['checkpoint_resolutions'],  # The trained resolutions at which the checkpoints are loaded
        model_type=model_type,
        device=device,
        save_dir=figures_dir
    ) 

if __name__ == "__main__":
    main()