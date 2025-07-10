#!/bin/bash
#SBATCH -J SI_50e
#SBATCH -A NAISS2024-22-955 -p alvis
#SBATCH -N 1 --gpus-per-node=A100fat:4
#SBATCH -t 1-00:00:00

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name SI_50e"

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
EDM_25e_t2m="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_t2m-diffusion-6x128-07_05_07-8839/last.ckpt" # Diffusion model trained for 25 epochs on temperature, NOTE: This model does not work with the current codebase, it is from an older version of the code

#####
EDM_25e_stand_lr="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_25e-diffusion-6x128-06_21_13-2566/last.ckpt" # Diffusion model trained for 25 epochs with standardized data and lower learning rate
EDM_50e_stand_lr="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_50e_lr-diffusion-6x128-07_01_10-5721/last.ckpt"
EMD_60e_stand_lr="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_60e-diffusion-6x128-07_03_08-2151/last.ckpt"

####
SI_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_25e-SI-6x128-07_07_07-4062/last.ckpt" # SI model trained for 25 epochs

# EDM
# Training
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --batch_size 4 --epochs 25 --lr 0.00001 # --load $EDM_25e_stand_lr

# Training trial
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 16 --batch_size 4 --epochs 1

# Testing

# Testing trial
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 2 --eval test --n_example_pred 2 --batch_size 2 --subset_ds --ensemble_size 5 --sampler edm --sampler_steps 20 --load "/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_t2m-diffusion-6x128-07_05_07-8839/last.ckpt"

# SI
# Training
python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --batch_size 4 --epochs 50 --lr 0.00001 --load $SI_25e

# Training trial
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 16 --batch_size 2 --epochs 2 --subset_ds

# Testing
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME --n_workers 16 --eval test --n_example_pred 1 --batch_size 8 --ensemble_size 5 (5-25) --sampler_steps 100 (10-100) --sampler euler_2 (euler/euler_2) --load $SI_25e 

# Testing trial
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 2 --eval test --n_example_pred 2 --batch_size 2 --ensemble_size 5 --sampler_steps 100 --subset_ds --load "/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_25e-SI-6x128-07_07_07-4062/last.ckpt" --sampler euler_2