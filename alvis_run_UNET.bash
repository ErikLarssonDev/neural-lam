#!/bin/bash
#SBATCH -J UNET_13_var
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name UNET_13_var"
    
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

UNET_small_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_small-unet-6x64-01_24_11-5862/last.ckpt"
UNET_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_xsmall-unet-6x32-01_24_10-0012/last.ckpt"
UNET_xxsmall="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_xxsmall-unet-6x32-01_26_09-6705/last.ckpt"

DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/clim_config_2.yaml"

python3 neural_lam/train_model.py --model unet $RUN_NAME --data_config $DATA_CONFIG --n_workers 16 --batch_size 2 --epochs 25 --lr 0.00001 --val_interval 5 

# python3 neural_lam/train_model.py --model unet $RUN_NAME --data_config $DATA_CONFIG --n_workers 16 --batch_size 10 --epochs 25 --lr 0.00001 --val_interval 10 --eval val --load $UNET_xxsmall

# Big model 50 epochs -> 29 hours.
# Big model 25 epochs -> 15 hours. 20 hours on 3 GPUs.

# 125 hours in parallell.
# 6 runs --> 20 hours per run.
# 3 runs --> 40 hours per run.
