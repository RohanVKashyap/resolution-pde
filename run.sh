#!/bin/bash
#SBATCH --job-name=darcy_fno2d
#SBATCH --partition=debug
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:2
#SBATCH --output=output_logs/%x-%j.out
#SBATCH --error=output_logs/%x-%j.err
#SBATCH --nodes=1

# Create output_logs directory if it doesn't exist
mkdir -p output_logs

echo "Job started on $(hostname) at $(date)"

# Load any necessary modules
# module load python/3.8
# module load cuda/11.2 
# burger_fno_1d
# darcy_fno2d

# Activate virtual environment
# source pde/bin/activate

# Run the Python script with arguments
srun python3 new_main.py \
    --batch_size 32 \
    --epochs 200 \
    --learning_rate 2.5e-3 \
    --factor 4 \
    --ff_weight_norm 1 \
    --n_ff_layers 2 \
    --layer_norm 1 \
    --mode full \
    --modes 61 \
    --width 64 \
    --n_blocks 24 \
    --project_name darcy_flow_fno2d \
    --pde darcy_flow \
    --use_normalizer -1 \
    --model_type ffno \
    --data_path1 data/darcy_data_mat/piececonst_r241_N1024_smooth1.mat \
    --data_path2 data/darcy_data_mat/piececonst_r241_N1024_smooth2.mat \
    --checkpoint_dir checkpoints


# srun python3 main.py \
#     --batch_size 64 \
#     --epochs 200 \
#     --learning_rate 1e-3 \
#     --modes 12 \
#     --width 32 \
#     --n_blocks 4 \
#     --project_name darcy_flow_fno2d \
#     --pde darcy_flow \
#     --data_path1 data/darcy_data_mat/piececonst_r421_N1024_smooth1.mat \
#     --data_path2 data/darcy_data_mat/piececonst_r421_N1024_smooth2.mat \
#     --checkpoint_dir checkpoints  

# srun python3 main.py \
#     --batch_size 64 \
#     --epochs 200 \
#     --learning_rate 1e-3 \
#     --modes 16 \
#     --width 64 \
#     --n_blocks 4 \
#     --project_name burger_fno_1d \
#     --pde burger \
#     --data_path1 data/burger_data_mat/burgers_data_R10.mat \
#     --checkpoint_dir checkpoints      

echo "Job completed at $(date)"