#!/bin/bash
#SBATCH -J CorrDiff_Static_50e
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 3-00:00:00
# -A NAISS2024-22-955 -p alvis
# NAISS2024/6-323 
# NAISS 2025/22-1196


export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name CorrDiff_Static_50e"

# Activate environment
source ~/.bashrc
# source /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/envs/neural-lam/bin/activate
mamba activate clim

# Switch to the correct directory
# cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam
cd /mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam

# Switch to the correct branch
git switch clim-downscaling

# Activate wandb
wandb online

# Saved models
SI_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_50e-SI-6x128-07_10_12-7283/last.ckpt"

# Saved models with static input data
SI_50e_stand="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_Static_50e-SI-6x128-12_13_01-4575/last.ckpt"
EDM_50e_stand="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_Static_50e-diffusion-6x128-12_13_21-4237/last.ckpt"
UNET_50e_stand="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_Static_50e-unet-6x128-12_12_16-6743/last.ckpt"
CorrDiff_Static_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrDiff_Static_50e-CorrDiff-6x128-12_15_19-5090/last.ckpt"

# EDM
# Training
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001

# Training trial
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 16 --batch_size 4 --epochs 1

# Testing
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --load $EDM_50e_stand --eval test --subset_ds

# Testing trial
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 2 --eval test --n_example_pred 2 --batch_size 2 --subset_ds --ensemble_size 5 --sampler edm --sampler_steps 20 --load "/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_t2m-diffusion-6x128-07_05_07-8839/last.ckpt"

# SI
# Training
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 # --load $SI_25e

# Training trial
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 16 --batch_size 2 --epochs 2 --subset_ds

# Testing
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --load $SI_50e_stand --eval test --subset_ds --sampler euler --sampler_steps 50
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME --n_workers 16 --eval test --n_example_pred 1 --batch_size 8 --ensemble_size 5 (5-25) --sampler_steps 100 (10-100) --sampler euler_2 (euler/euler_2) --load $SI_25e 

# Testing trial
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 2 --eval test --n_example_pred 2 --batch_size 2 --ensemble_size 5 --sampler_steps 100 --subset_ds --load $SI_50e --sampler euler_2

# UNET
# Train
# python3 neural_lam/train_model.py --model unet $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001

# Eval
# python3 neural_lam/train_model.py --model unet $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --load $UNET_50e_stand --eval test --subset_ds

# CorrDiff
# python3 neural_lam/train_model.py --model CorrDiff --n_workers 4 --batch_size 2 --epochs 50 --lr 0.00001 --subset_ds

# Training
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 

# Eval
python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --eval test --subset_ds --load $CorrDiff_Static_50e

