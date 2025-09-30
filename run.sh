#!/bin/bash
#SBATCH --job-name=burger
#SBATCH --partition=general
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=output_logs/%x-%j.out
#SBATCH --error=output_logs/%x-%j.err
#SBATCH --nodes=1

# Create output_logs directory if it doesn't exist
mkdir -p output_logs

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

echo "GPU information:"
nvidia-smi
echo "CUDA devices:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

pwd

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/rvk/pde

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate /data/user_data/rvk/fno16

# Load any necessary modules
# module load python/3.8
# module load cuda/11.2 
# burger_fno_1d
# darcy_fno2d
# burger_superresolution
# gpu:2
# output_logs/%x-%j.err
# output_logs/%x-%j.out
# gres=gpu:A100_80GB:2

# Activate virtual environment
# source pde/bin/activate
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate /data/user_data/rvk/pde

# 1. Create environment under custom path
# conda create --prefix /data/user_data/rvk/ns_env python=3.9 pytorch=1.13.1 -c pytorch
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate /data/user_data/rvk/ns_env

# Create environment
# conda create -n pdebench_env
# conda activate pdebench_env

# # Install core packages
# conda install numpy scipy matplotlib -y

# # Install PyTorch with CUDA 12.x support
# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# # Install JAX with CUDA 12 support  
# pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # Install Hydra and other required packages
# pip install hydra-core omegaconf einops h5py

# # Verify installation
# python -c "
# import torch
# import jax
# print('PyTorch CUDA available:', torch.cuda.is_available())
# print('PyTorch CUDA version:', torch.version.cuda)
# print('JAX devices:', jax.devices())
# print('Number of GPUs (JAX):', len(jax.devices()))
# "



# pip3 install hydra-core omegaconf einops h5py

# Make sure you're in the right environment
# conda activate pdebench_env

# # Install JAX with CUDA 12 support for your RTX A6000 GPUs
# pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # Install other required packages
# pip install hydra-core omegaconf einops h5py numpy matplotlib

# # Install PyTorch if needed
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# # Test all imports
# python -c "
# import jax
# import jax.numpy as jnp
# import hydra
# from omegaconf import DictConfig
# import numpy as np
# print('JAX version:', jax.__version__)
# print('JAX devices:', jax.devices())
# print('Number of GPUs:', len(jax.devices()))
# print('All packages working!')
# "


# conda activate /data/user_data/rvk/wellenv

# pip3 install the_well

# Run the Python script with arguments
# srun python3 superresolution.py
# srun python3 new_main.py model=ffno_1d/ffno_1d dataset=burger/burger_0.001
# srun python3 s4_main.py model=s4_1d/s4_1d dataset=burger/burger_0.001
# run python3 new_main.py model=unet/unet_1d dataset=burger/burger_0.001
# srun python3 s4_main.py model=s4_2d/s4_2d dataset=navier_stokes/ns_512_0
# srun python3 s1.py model=s4_2d/s4_2d dataset=navier_stokes/ns_512_0_t2
# srun python3 new_main.py model=unet/unet_2d dataset=navier_stokes/ns_512_0

# srun python3 new_main.py model=ffno_2d/ns_ffno_2d dataset=navier_stokes/ns_512_0   
# srun python3 new_main.py model=ffno_1d/ffno_1d dataset=burger/burger_0.001
# srun python3 pos_main.py model=pos/pos dataset=navier_stokes/ns_512_0
# srun python3 pos_main.py model=pos/pos dataset=navier_stokes/ns_512_0_t2
# srun python3 cno_main.py model=cno_2d/cno_2d dataset=navier_stokes/ns_512_0
# srun python3 cno_original_main.py model=cno_2d/cno_2d_original dataset=navier_stokes/ns_512_0
# srun python3 cno_original_main.py model=ffno_2d/ns_ffno_2d dataset=navier_stokes/ns_512_0
# srun python3 cno_original_main.py model=pos/pos dataset=navier_stokes/ns_512_0
# srun python3 cno_original_mat.py model=ffno_2d/ns_ffno_2d dataset=navier_stokes/ns_512_0_t2

# srun python3 LPSDA/generate/generate_data.py --experiment=KS \
#                                              --train_samples=2048 \
#                                              --valid_samples=256 \
#                                              --test_samples=256 \
#                                              --L=64 \
#                                              --nt=51 \
#                                              --nx=256 \
#                                              --nt_effective=51 \
#                                              --viscosity=0.1 \
#                                              --end_time=5.0 \
#                                              --lmax=8

# srun python3 LPSDA/generate/generate_data.py --experiment=KS \
#                                              --train_samples=2048 \
#                                              --valid_samples=256 \
#                                              --test_samples=256 \
#                                              --L=64 \
#                                              --nt=51 \
#                                              --nx=400 \
#                                              --nt_effective=51 \
#                                              --viscosity=0.075 \
#                                              --end_time=5.0 \
#                                              --lmax=8

# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_true_mres4

# srun python3 -u data_generation/ns_2d.py
# srun python3 main_2d.py model=ffno_2d/ffno_2d dataset=ns/ns_naive_true_mres

# srun python3 LPSDA/generate/generate_data.py --experiment=KS \
#                                              --train_samples=2048 \
#                                              --valid_samples=256 \
#                                              --test_samples=256 \
#                                              --L=64 \
#                                              --nt=51 \
#                                              --nx=32 \
#                                              --nt_effective=51 \
#                                              --viscosity=0.001 \
#                                              --end_time=1.0 \
#                                              --lmax=4

# srun python3 LPSDA/generate/generate_data.py --experiment=KS \
#                                              --train_samples=2048 \
#                                              --valid_samples=256 \
#                                              --test_samples=256 \
#                                              --L=4 \
#                                              --nt=51 \
#                                              --nx=32 \
#                                              --nt_effective=51 \
#                                              --viscosity=0.1 \
#                                              --end_time=5.0 \
#                                              --lmax=8

# srun python3 LPSDA/generate/generate_data.py --experiment=KS \
#                                              --train_samples=2048 \
#                                              --valid_samples=256 \
#                                              --test_samples=256 \
#                                              --L=64 \
#                                              --nt=500 \
#                                              --nx=65 \
#                                              --nt_effective=140 \
#                                              --viscosity=1.0 \
#                                              --end_time=100.0 \
#                                              --lmax=3 \

# srun python3 LPSDA/generate/generate_data.py --experiment=KS \
#                                              --nx=512 \
#                                              --L=64 \
#                                              --viscosity=0.03 \
#                                              --lmax=8 \
#                                              --end_time=5.0 \
#                                              --nt=51 \
#                                              --nt_effective=51 \
#                                              --train_samples=2048 \
#                                              --valid_samples=256 \
#                                              --test_samples=256


# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks

# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_512
# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_mres1_16

# srun python3 naive_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_mres_8
# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_1024
# srun python3 naive_main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_naive

# srun python3 pino-closure-models/ks/solver/KS_solver.py
# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_pino_1024

# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_mres1
# srun python3 plot_burgers.py

# Run with relative config path
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate pdebench_env
# cd /home/rvk/PDEBench/pdebench/data_gen/data_gen_NLE/BurgersEq
# # srun python3 burgers_multi_solution_Hydra.py \
# #     --config-path="config" \
# #     +multi=1e-3.yaml
# srun python3 burgers_multi_solution_Hydra.py \
#     --config-path="config" \
#     +multi=config_1e-4.yaml

# cd /home/rvk/PDEBench/pdebench/data_gen/data_gen_NLE
# srun python3 Data_Merge.py \
#     args.type=burgers \
#     args.dim=1 \
#     args.savedir="/data/user_data/rvk/pdebench_gen/burgers_1024_0.0001/"

# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres
# srun python3 main_1d.py model=unet/unet_1d dataset=burger/burger_naive_true_mres2
# srun python3 main_1d.py model=unet/unet_1d dataset=burger/burger_naive_true_mres3
# srun python3 main_1d.py model=unet/unet_1d dataset=ks/ks_naive_true_mres
# srun python3 main_1d.py model=unet/unet_1d dataset=ks/ks_naive_true_mres1

# # Add these lines to your existing script instead of the srun commands:

# Run both commands in parallel
srun python3 main_1d.py model=unet/unet_1d dataset=burger/burger_naive_true_mres

# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_true_mres
# srun python3 correct_main_2d.py model=ffno_2d/ffno_2d dataset=ns/ns_naive_true_mres4
# srun python3 autoregressive_eval.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_true_mres
# srun python3 main_2d.py model=ffno_2d/ffno_2d dataset=ns/ns_naive_true_mres
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_resize_true_mres

# srun python3 cno_original_main_1d.py model=cno_1d/cno_1d dataset=ks/ks
# srun python3 ks_plots.py
# the-well-download --base-path /data/user_data/rvk/well/ --dataset active_matter --split train

echo "Job completed at $(date)"