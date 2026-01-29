#!/bin/bash
#SBATCH --job-name=burger
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=output_new_logs/%x-%j.out
#SBATCH --error=output_new_logs/%x-%j.err
#SBATCH --nodes=1

# export PYTHONPATH=/home/rvk:$PYTHONPATH

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

# Load any necessary modules
# module load python/3.8
# module load cuda/11.2 
# burger_fno_1d
# darcy_fno2d
# burger_superresolution
# gpu:2
# output_logs/%x-%j.err
# output_logs/%x-%j.out
#Original: #SBATCH --partition=debug

# Activate virtual environment
# source pde/bin/activate
# conda activate /data/user_data/rvk/ns_env
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate /data/user_data/rvk/ns_env

# Run the Python script with arguments
# srun python3 superresolution.py
# srun python3 new_main.py model=ffno_1d/ffno_1d dataset=burger/burger_0.01
# srun python3 new_main.py model=unet/unet_1d dataset=burger/burger_0.01
# srun python3 s4_main.py model=s4_1d/s4_1d dataset=burger/burger_0.01
# srun python3 new_main.py model=ffno_2d/ns_ffno_2d dataset=navier_stokes/ns_512_0_t2
# srun python3 s4_main.py model=s4_2d/s4_2d dataset=navier_stokes/ns_512_0_t2
# srun python3 s4_main.py model=s4_2d/s4_2d dataset=navier_stokes/ns_512_0_t2
# srun python3 s1.py model=s4_2d/s4_2d dataset=navier_stokes/ns_512_0
# srun python3 new_main.py model=unet/unet_2d dataset=navier_stokes/ns_512_0_t2

# srun python3 new_main.py model=ffno_2d/ns_ffno_2d dataset=navier_stokes/ns_512_0_t2  
# srun python3 new_main.py model=ffno_1d/ffno_1d dataset=burger/burger_mres 
# srun python3 pos_main.py model=pos/pos dataset=navier_stokes/ns_512_0_t2
# srun python3 pos_main.py model=pos/pos dataset=navier_stokes/ns_512_0_t4
# srun python3 cno_original_main.py model=cno_2d/cno_2d_original dataset=navier_stokes/ns_512_0_t2
# srun python3 cno_original_main.py model=ffno_2d/ns_ffno_2d dataset=navier_stokes/ns_512_0_t2
# srun python3 cno_original_main.py model=pos/pos dataset=navier_stokes/ns_512_0_t2
# srun python3 cno_original_mat.py model=ffno_2d/ns_ffno_2d dataset=navier_stokes/ns_512_0_t2
# srun python3 cno_original_main.py model=cno_2d/cno_2d_original dataset=navier_stokes/ns_512_0_t2
# srun python3 cno_original_main.py model=ffno_2d/ns_ffno_2d dataset=navier_stokes/ns_512_0_t2
# srun python3 cno_original_main.py model=cno_2d/cno_2d_original dataset=navier_stokes/ns_active_t2

# srun python3 cno_original_main.py model=ffno_2d/ns_ffno_2d dataset=navier_stokes/ns_active_t2
# srun python3 cno_original_main_1d.py model=cno_1d/cno_1d dataset=ks/ks_2
# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_2

# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_256
# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_mres2_16

# srun python3 naive_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_8
# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_mres_16
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_mres
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_resize
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_resize_true_mres1
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres1

# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres4
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_true_mres1
# srun python3 correct_main_2d.py model=ffno_2d/ffno_2d dataset=ns/ns_naive_true_mres5
# srun python3 main_1d.py model=unet/unet_1d dataset=ks/ks_naive_true_mres2
# srun python3 main_1d.py model=unet/unet_1d dataset=ks/ks_naive_true_mres3

# srun python3 frequency_evaluation.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_true_mres1
# srun python3 frequency_evaluation.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres1
srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_true_mres1
# srun python3 main_1d.py model=cno_1d/cno_1d dataset=ks/ks_naive_true_mres1
# srun python3 frequency_evaluation.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres1
# srun python3 main_1d.py model=ffno_1d/ffno_1d_large dataset=ks/ks_naive_true_mres1
# srun python3 -u data_generation/ns1_2d.py
# srun python3 main_1d.py model=unet/unet_1d dataset=burger/burger_naive_true_mres1
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_true_mres1
# srun python3 main_1d.py model=cno_1d/cno_1d dataset=ks/ks_naive_true_mres1
# srun python3 main_1d.py model=unet/unet_1d dataset=ks/ks_naive_true_mres1
# srun python3 main_2d.py model=ffno_2d/ffno_2d dataset=ns/ns_naive_true_mres1
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres1
# srun python3 main_1d.py model=unet/unet_1d dataset=burger/burger_naive_true_mres1
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_true_mres1
# srun python3 main_1d.py model=cno_1d/cno_1d dataset=burger/burger_naive_true_mres1
# srun python3 main_2d.py model=ffno_2d/ffno_2d dataset=ns/ns_naive_true_mres1
# srun python3 main_2d.py model=cno_2d/cno_2d_original dataset=ns/ns_naive_true_mres
# srun python3 main_2d.py model=cno_2d/cno_2d_original dataset=ns/ns_naive_true_mres1
# srun python3 main_2d.py model=cno_2d/cno_2d_original dataset=ns/ns_naive_true_mres3
# srun python3 main_1d.py model=unet/unet_1d dataset=ks/ks_naive_true_mres4
# srun python3 main_2d.py model=ffno_2d/ffno_2d dataset=ns/ns_naive_true_mres1
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres1

# python3 main_1d.py model=unet/unet_1d dataset=burger/burger_naive_true_mres2 &
# python3 main_1d.py model=unet/unet_1d dataset=burger/burger_naive_true_mres3 &

# wait


# srun python3 main_2d.py model=ffno_2d/ffno_2d dataset=ns/ns_naive_true_mres1
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_true_mres
# srun python3 main_2d.py model=ffno_2d/ffno_2d dataset=ns/ns_naive_true_mres1
# srun python3 -u data_generation/ns1_2d.py

# conda activate pdebench_env
# cd /home/rvk/PDEBench/pdebench/data_gen/data_gen_NLE/BurgersEq
# # srun python3 burgers_multi_solution_Hydra.py \
# #     --config-path="config" \
# #     +multi=1e-3.yaml
# srun python3 burgers_multi_solution_Hydra.py \
#     --config-path="config" \
#     +multi=config1_1e-4.yaml

# cd /home/rvk/PDEBench/pdebench/data_gen/data_gen_NLE
# srun python3 Data_Merge.py \
#     args.type=burgers \
#     args.dim=1 \
#     args.savedir="/data/user_data/rvk/pdebench_gen/burgers_512_0.0001/"

# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres1
# srun python3 main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres1

# srun python3 cno_original_main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_pino_512


# srun python3 LPSDA/generate/generate_data.py --experiment=KS \
#                                              --train_samples=2048 \
#                                              --valid_samples=256 \
#                                              --test_samples=256 \
#                                              --L=64 \
#                                              --nt=51 \
#                                              --nx=64 \
#                                              --nt_effective=51 \
#                                              --viscosity=0.001 \
#                                              --end_time=5.0 \
#                                              --lmax=8

# srun python3 LPSDA/generate/generate_data.py --experiment=KS \
#                                              --train_samples=2048 \
#                                              --valid_samples=256 \
#                                              --test_samples=256 \
#                                              --L=64 \
#                                              --nt=51 \
#                                              --nx=512 \
#                                              --nt_effective=51 \
#                                              --viscosity=0.05 \
#                                              --end_time=5.0 \
#                                              --lmax=8

# srun python3 LPSDA/generate/generate_data.py --experiment=KS \
#                                              --train_samples=2048 \
#                                              --valid_samples=256 \
#                                              --test_samples=256 \
#                                              --L=64 \
#                                              --nt=51 \
#                                              --nx=64 \
#                                              --nt_effective=51 \
#                                              --viscosity=0.001 \
#                                              --end_time=1.0 \
#                                              --lmax=4  

# srun python3 LPSDA/generate/generate_data.py --experiment=KS \
#                                              --nx=32 \
#                                              --L=4.0 \
#                                              --viscosity=0.1 \
#                                              --lmax=15 \
#                                              --end_time=8.0 \
#                                              --nt=200 \
#                                              --nt_effective=100 \
#                                              --train_samples=2048 \
#                                              --valid_samples=256 \
#                                              --test_samples=256

echo "Job completed at $(date)"