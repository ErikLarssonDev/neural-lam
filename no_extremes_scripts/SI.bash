#!/bin/bash
#SBATCH -J SI_no_extremes
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name SI_no_extremes"
# DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/no_extremes_nordic.yaml"
DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/no_extremes.yaml"

    
# Activate environment
source ~/.bashrc
mamba activate clim

cd /mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam

# Switch to the correct branch
git switch clim-downscaling

# Activate wandb
wandb online

# Training
python3 neural_lam/train_model.py \
    --model SI \
    --diffusion_model song_unet \
    $RUN_NAME  \
    --data_config $DATA_CONFIG \
    --n_workers 16 \
    --batch_size 2 \
    --epochs 25 \
    --lr 0.00001 \
    --sampler euler \
    --sampler_steps 50 \
    --val_interval 10 \

# Testing SI with static input data
# python3 neural_lam/train_model.py \
#     --model SI \
#     --diffusion_model song_unet \
#     $RUN_NAME  \
#     --n_workers 16 \
#     --batch_size 20 \
#     --epochs 50 \
#     --lr 0.00001 \
#     --load $SI_xxsmall_25e \
#     --eval val \
#     --sampler euler \
#     --sampler_steps 100 \

