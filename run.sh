#!/bin/bash
#SBATCH --job-name=burger
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/rvk/pde

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
#                                              --nx=512 \
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

srun python3 naive_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_mres_8
# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_1024
# srun python3 naive_main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_naive

# srun python3 pino-closure-models/ks/solver/KS_solver.py
# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_pino_1024


# srun python3 cno_original_main_1d.py model=cno_1d/cno_1d dataset=ks/ks
# srun python3 ks_plots.py
# the-well-download --base-path /data/user_data/rvk/well/ --dataset active_matter --split train

echo "Job completed at $(date)"