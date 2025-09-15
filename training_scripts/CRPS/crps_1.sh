#!/bin/bash
#SBATCH -J CRPS_600e
#SBATCH -t 03-00:00:00
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#
# --gres=gpu:8
# --ntasks-per-node=8
# --cpus-per-task=16
# --dependency=afterany:JOB_ID

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate BZ31
wandb online

cd /proj/berzelius-2022-164/users/x_erila/neural-lam
git switch main

# Standard arguments
DIFFUSION_MODEL="--diffusion_model graph_fm --graph hierarchical-3"
RUN_NAME="--wandb_run_name CRPS_600e"

python3 neural_lam/train_model.py --model CRPS $RUN_NAME  --n_workers 16 --pred_residual --border_condition --epochs 600 --batch_size 6