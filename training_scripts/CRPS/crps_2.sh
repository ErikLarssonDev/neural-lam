#!/bin/bash
#SBATCH -J CRPS_AR4_1000e
#SBATCH -t 03-00:00:00
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate BZ31
wandb online

cd /proj/berzelius-2022-164/users/x_erila/neural-lam
git switch main

# Standard arguments
RUN_NAME="--wandb_run_name CRPS_AR4_1000e"
CHECKPOINT="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_600e_res-CRPS-6x128-07_25_09-6319/last.ckpt"

# NOTE: Need lower batch size for CRPS as we are training with 2 predictions per sample
python3 neural_lam/train_model.py --model CRPS $RUN_NAME  --n_workers 16 --pred_residual --border_condition --epochs 1000 --batch_size 1 --lr 0.0001 --ar_steps 4 --load $CHECKPOINT