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

from train.training import train, evaluate
from models.custom_layer import UnitGaussianNormalizer
from utils.loss import RelativeL2Loss
from train.mres_training import create_grouped_dataloaders 
from utils.resize_utils import get_lower_resolutions, evaluate_cno_original_1d_all_resolution
from utils.naive_utils import evaluate_1d_all_resolution
from utils.autoregressive_step import evaluate_1d_rollout_all_resolution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(args: DictConfig):
    logging.info(OmegaConf.to_yaml(args))

    print(f"Project name: {args.project_name}")
    print(f"Model name: {args.model._target_}")
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

    print(f"Data resolution: {data_resolution}")
    print(f"Train resolution: {train_resolution}")
    print(f"Model type: {model_type}")

    # Load Data for normalizer information
    print("Loading dataset to get normalizer...")
    if any(substring in args.dataset['pde'] for substring in ['ks', 'burger', 'ns']):      
        print('---------------------')
        # Instantiate the dataset function: KS dataset has 2 additional files (default): KS_valid.h5, ks_test.h5
        data_ = instantiate(args.dataset.dataset_params, use_strain_orientation=True)
        # train_dataset, val_dataset, test_dataset = data_[:3]
        train_dataset, val_dataset, test_dataset, _ = data_[:4]          
                                                    
    else:
        raise ValueError(f"Unsupported PDE type: {args.dataset['pde']}")

    # Extract normalizers based on type
    if normalizer_type == 'simple':
        x_normalizer, y_normalizer = data_[4:]
        min_data = max_data = min_model = max_model = None
    else:
        min_data, max_data, min_model, max_model = data_[4:]
        x_normalizer = y_normalizer = None

    print("Dataset loaded successfully")
    print(f"Normalizer type: {normalizer_type}")

    # Initialize model (same architecture as training)
    if model_type == 'cno1d':
        model = instantiate(
                args.model,
                _recursive_=False,
                size=32).to(device)    # in_size: UPDATE
        
    elif model_type == 'pos':  
        from scOT.model import ScOT, ScOTConfig
        model_config = ScOTConfig(**args.model)  
        model = ScOT.from_pretrained('camlab-ethz/Poseidon-B', config=model_config, 
                                    ignore_mismatched_sizes=True).to(device) 
    else:
        model = instantiate(
                args.model,
                _recursive_=False).to(device) 

    print(f"Model architecture: {model_type}")
    
    # Count and print total parameters in millions
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Model Parameters: {total_params / 1e6:.2f}M')
    print(f'Trainable Parameters: {trainable_params / 1e6:.2f}M')

    # Load checkpoint
    checkpoint_path = args.dataset.checkpoint_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Checkpoint loaded successfully")
    print(f"Previous training L2 loss: {checkpoint.get('l2_loss', 'N/A')}")
    
    # Set model to evaluation mode
    model.eval()

    # Create directory for evaluation plots
    job_id = os.environ.get('SLURM_JOB_ID', 'test_rollout')
    figures_dir = os.path.join("figures", f"{job_id}_rollout_eval")
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Created/verified figures directory: {figures_dir}")

    # Get test resolutions
    test_resolutions = get_lower_resolutions(args.dataset['max_test_resolution'])
    print(f"Test resolutions: {test_resolutions}")
    
    # This is only for multi-resolution (single-resolution this is None)
    eval_dataset_target = args.dataset['dataset_params'].get('eval_dataset_target', None)
    
    print("\n" + "="*60)
    print("STARTING ROLLOUT EVALUATION")
    print("="*60)
    
    resolution_results = evaluate_1d_all_resolution(
        model=model,
        dataset_config=args.dataset,
        eval_dataset_target=eval_dataset_target if args.dataset['train_mres'] else None,  # This is only for multi-resolution
        current_res=train_resolution,   # data_resolution / reduced_resolution
        test_resolutions=test_resolutions,
        data_resolution=data_resolution,   # UPDATE: LINE 1088 in cno_utils.py (IMPORTANT)
        pde=args.dataset['pde'],
        saved_folder=args.dataset['dataset_params']['saved_folder'],
        reduced_batch=args.dataset['dataset_params']['reduced_batch'],
        reduced_resolution_t=args.dataset['dataset_params']['reduced_resolution_t'], 
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer, 
        batch_size=batch_size,
        time=1,
        model_type=model_type,
        device='cuda',
        plot_examples=True,
        save_dir=figures_dir, 
        analyze_frequencies=False,     # Plot Frequency Histogram                  
    )  
    
    # Run rollout evaluation
    rollout_results = evaluate_1d_rollout_all_resolution(
        model=model,
        dataset_config=args.dataset,
        eval_dataset_target=eval_dataset_target if args.dataset['train_mres'] else None,
        current_res=train_resolution,
        test_resolutions=test_resolutions,
        data_resolution=data_resolution,
        pde=args.dataset['pde'],
        saved_folder=args.dataset['dataset_params']['saved_folder'],
        reduced_batch=args.dataset['dataset_params']['reduced_batch'],
        reduced_resolution_t=args.dataset['dataset_params']['reduced_resolution_t'],
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        batch_size=batch_size,
        rollout_steps=args.dataset.rollout_steps,   # IMPORTANT
        time=1,
        model_type=model_type,
        device='cuda',
        plot_examples=True,
        save_dir=figures_dir
    )   

    print("\n" + "="*60)
    print("ROLLOUT EVALUATION COMPLETED")
    print("="*60)
    print(f"Results saved to: {figures_dir}")
    
    print("\nSummary of Rollout Evaluation:")
    for res, loss in rollout_results.items():
        print(f"Resolution {res}: Rollout Loss ({args.dataset.rollout_steps} steps) = {loss:.6f}")

    # Optionally initialize wandb for logging results
    if args.get('log_to_wandb', False):
        wandb.init(
            project=f"{model_type}_rollout_eval",
            config={
                'model_type': model_type,
                'pde': args.dataset['pde'],
                'checkpoint_path': checkpoint_path,
                'rollout_steps': args.dataset.rollout_steps,
                'test_resolutions': test_resolutions,
            }
        )
        
        wandb.log({"rollout_evaluation": wandb.Table(
            columns=["Resolution", "Rollout Loss", "Rollout Steps"],
            data=[[res, loss, args.dataset.rollout_steps] for res, loss in rollout_results.items()])
        })
        
        print(f"Results logged to wandb: {wandb.run.url}")
        wandb.finish()

if __name__ == "__main__":
    main()