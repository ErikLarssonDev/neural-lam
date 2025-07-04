#!/bin/bash
#SBATCH -J SI_25e
#SBATCH -A NAISS2024-22-955 -p alvis
#SBATCH -N 1 --gpus-per-node=A100fat:4
#SBATCH -t 1-00:00:00

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name SI_25e"

# Activate environment
source ~/.bashrc
mamba activate clim

# Switch to the correct directory
cd /mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam

# Switch to the correct branch
git switch clim-downscaling

# Activate wandb
wandb online

# Saved models
# EDM_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_25e-diffusion-6x128-05_23_12-8415/last.ckpt"
EDM_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_25e-diffusion-6x128-05_27_08-4988/last.ckpt"
EDM_25e_t2m="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_25e_t2m-diffusion-6x128-06_10_08-4629/last.ckpt"

#####
EDM_25e_stand="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_25e-diffusion-6x128-06_21_13-1220/last.ckpt"
EDM_25e_stand_lr="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_25e-diffusion-6x128-06_21_13-2566/last.ckpt"
EDM_50e_stand_lr="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_50e_lr-diffusion-6x128-07_01_10-5721/last.ckpt"
EMD_60e_stand_lr="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_60e-diffusion-6x128-07_03_08-2151/last.ckpt"



# EDM
# Training
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --batch_size 4 --epochs 60 --lr 0.000001 --load $EDM_25e_stand_lr

# Training trial
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 16 --batch_size 4 --epochs 1

# Testing

# Testing trial
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 2 --eval test --n_example_pred 2 --batch_size 2 --subset_ds --ensemble_size 5 --sampler edm --sampler_steps 20 --load "/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_60e-diffusion-6x128-07_03_08-2151/last.ckpt"

# SI
# Training
python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --batch_size 4 --epochs 25 --lr 0.00001

# Training trial
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 16 --batch_size 2 --epochs 2 --subset_ds

# Testing
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME --n_workers 2 --eval val --n_example_pred 1 --batch_size 8 --ensemble_size 5 --sampler_steps 100

# Testing trial
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 2 --eval test --n_example_pred 1 --batch_size 2 --ensemble_size 5 --sampler_steps 100 --subset_ds