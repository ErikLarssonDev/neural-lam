#!/bin/bash
#SBATCH -J CORDEX
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:2
#SBATCH -t 10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name UNET_hist_oro"

# Activate environment
source ~/.bashrc
# source /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/envs/neural-lam/bin/activate
mamba activate clim

# Switch to the correct directory
# cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam
cd /mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam

# Switch to the correct branch
git switch clim-downscaling-inference

# Activate wandb
wandb online

# Data config
# DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/ALPS_hist_future_global_std_config.yaml"
# DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/ALPS_hist_future_local_std_config.yaml"

# NOTE: Global std seems to perform better.
DATA_CONFIG="neural_lam/ALPS_hist_global_std_config.yaml"
# DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/ALPS_hist_local_std_config.yaml"

# Saved models
SI_64_hist_oro="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_ALPS_hist_global_std-SI-6x64-02_11_09-0235/last.ckpt"
SI_128_hist_oro="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_ALPS_hist_global_std-SI-6x128-01_17_09-6495/last.ckpt"
SI_256_hist_oro="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_ALPS_hist_global_std-SI-6x256-02_11_09-1539/last.ckpt"

UNET_hist_oro="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_ALPS_hist_global_std-unet-6x128-02_11_09-4618/last.ckpt"
UNET_future_oro=""

# EDM
# Training
# python3 neural_lam/train_model.py --model diffusion --data_config $DATA_CONFIG --diffusion_model edm $RUN_NAME  --n_workers 16 --batch_size 24 --epochs 50 --lr 0.00001 --val_interval 5

# Testing
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --load $EDM_50e_stand --eval val

# SI
# Training
# python3 neural_lam/train_model.py --model SI --data_config $DATA_CONFIG --diffusion_model song_unet $RUN_NAME  --n_workers 16 --batch_size 12 --epochs 50 --lr 0.00001 --sampler euler --sampler_steps 100 --val_interval 5 --hidden_dim 256

# Testing
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME --data_config $DATA_CONFIG --n_workers 16 --batch_size 24 --epochs 50 --lr 0.00001 --load $SI_256_hist_oro --hidden_dim 256 --eval val --sampler euler --sampler_steps 100

# UNET
# Train
# python3 neural_lam/train_model.py --model unet --data_config $DATA_CONFIG $RUN_NAME --n_workers 16 --batch_size 24 --epochs 50 --lr 0.00001

# Eval
python3 neural_lam/train_model.py --model unet $RUN_NAME --data_config $DATA_CONFIG --n_workers 16 --batch_size 24 --epochs 50 --lr 0.00001 --load $UNET_hist_oro --eval val

# CorrDiff
# python3 neural_lam/train_model.py --model CorrDiff --data_config $DATA_CONFIG --n_workers 16 --batch_size 12 --epochs 50 --lr 0.00001 --mean_model_ckpt_path $UNET_hist_oro --pred_residual --val_interval 5 

# Training
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --val_interval 10

# Eval
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --eval test --subset_ds --load $CorrDiff_Static_50e --save_output