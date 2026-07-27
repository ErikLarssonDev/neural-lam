#!/bin/bash
#SBATCH -J SI_13_var_150e
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=A100fat:4
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

# Projects:
# NAISS2025-22-1196
# naiss2025-1-11


# GPUs:
# --gpus-per-node=A100fat:3

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name SI_13_var_100e"
    
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

DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/clim_config_2.yaml"

SI_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_13_var-SI-6x128-03_23_14-4660/last.ckpt"
SI_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_13_var-SI-6x128-03_24_10-0270/last.ckpt"
SI_75e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_13_var-SI-6x128-03_25_12-7982/last.ckpt"
SI_100e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_13_var-SI-6x128-04_17_11-4866/last.ckpt"
# Training
python3 neural_lam/train_model.py \
    --model SI \
    --diffusion_model song_unet \
    $RUN_NAME  \
    --data_config $DATA_CONFIG \
    --n_workers 16 \
    --batch_size 4 \
    --epochs 150 \
    --lr 0.00001 \
    --sampler euler \
    --sampler_steps 50 \
    --val_interval 5 \
    --load $SI_100e \
    --restore_opt \

# Testing SI with static input data
# python3 neural_lam/train_model.py \
#     --model SI \
#     --diffusion_model song_unet \
#     $RUN_NAME  \
#     --data_config $DATA_CONFIG \
#     --n_workers 16 \
#     --batch_size 4 \
#     --epochs 75 \
#     --lr 0.00001 \
#     --sampler euler \
#     --sampler_steps 100 \
#     --val_interval 5 \
#     --load $SI_100e \
#     --restore_opt \
#     --eval val \
