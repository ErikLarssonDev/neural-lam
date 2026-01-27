#!/bin/bash
#SBATCH -J EDM_xsmall
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:2
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name EDM_xsmall"
    
# Activate environment
source ~/.bashrc
# source /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/envs/neural-lam/bin/activate
mamba activate clim

cd /mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam
# cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

# Switch to the correct branch
git switch clim-downscaling

# Activate wandb
wandb online

EDM_small_25e=""

# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --batch_size 2 --epochs 25 --lr 0.00001 --val_interval 10 --hidden_dim 32

EDM_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_xsmall-diffusion-6x32-01_24_13-8505/last.ckpt"
python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --batch_size 10 --epochs 25 --lr 0.00001 --val_interval 10 --hidden_dim 32 --eval val --load $EDM_xsmall_25e
    
