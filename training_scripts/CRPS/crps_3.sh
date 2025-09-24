#!/bin/bash
#SBATCH -J CRPS_AR6_1200e
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
RUN_NAME="--wandb_run_name CRPS_AR6_1200e"
CHECKPOINT="CRPS_res_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_1000e_res-CRPS-6x128-07_26_16-6370/last.ckpt""

# NOTE: Need lower batch size for CRPS as we are training with 2 predictions per sample
python3 neural_lam/train_model.py --model CRPS $RUN_NAME  --n_workers 16 --pred_residual --border_condition --epochs 1200 --batch_size 1 --lr 0.00001 --ar_steps 6 --load $CHECKPOINT