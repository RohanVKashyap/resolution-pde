#!/bin/bash
#SBATCH --job-name=hydra
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
echo "Job ID: $SLURM_JOB_ID"

# Load any necessary modules
# module load python/3.8
# module load cuda/11.2 
# burger_fno_1d
# darcy_fno2d
# burger_superresolution
# gpu:2

# Activate virtual environment
# source pde/bin/activate

# Run the Python script with arguments
# srun python3 superresolution.py
srun python3 main.py     

echo "Job completed at $(date)"
