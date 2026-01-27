#!/bin/bash
#SBATCH -J SI_xxsmall
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=A100fat:2
#SBATCH -t 10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name SI_xxsmall"
    
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

SI_50e_stand="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_Static_50e-SI-6x128-12_13_01-4575/last.ckpt"
SI_small_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_small-SI-6x64-01_24_20-2262/last.ckpt"
SI_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_xsmall-SI-6x32-01_24_10-5547/last.ckpt"
SI_xxsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_xxsmall-SI-6x32-01_26_03-1571/last.ckpt"

# Training
# python3 neural_lam/train_model.py \
#     --model SI \
#     --diffusion_model song_unet \
#     $RUN_NAME  \
#     --n_workers 16 \
#     --batch_size 4 \
#     --epochs 25 \
#     --lr 0.00001 \
#     --sampler euler \
#     --sampler_steps 50 \
#     --val_interval 10 \
#     --hidden_dim 32 \
#     --channel_mult "1,1,1,1"

# Testing SI with static input data
python3 neural_lam/train_model.py \
    --model SI \
    --diffusion_model song_unet \
    $RUN_NAME  \
    --n_workers 16 \
    --batch_size 20 \
    --epochs 50 \
    --lr 0.00001 \
    --load $SI_xxsmall_25e \
    --eval val \
    --sampler euler \
    --sampler_steps 40 \
    --ensemble_size 1 \
    --hidden_dim 32 \
    --channel_mult "1,1,1,1" \
