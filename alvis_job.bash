#!/bin/bash
#SBATCH -J A40CorrDiff_res
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:4
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

# -A NAISS2024-22-955 -p alvis
# NAISS2024/6-323 
# NAISS 2025/22-1196


export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name CorrDiff_res"
    
# Activate environment
source ~/.bashrc
# source /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/envs/neural-lam/bin/activate
mamba activate clim

cd /mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam
# Switch to the correct directory
# cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

# Switch to the correct branch
git switch clim-downscaling

# Activate wandb
wandb online

# Saved models
SI_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_50e-SI-6x128-07_10_12-7283/last.ckpt"

# Saved models with static input data
SI_50e_stand="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_Static_50e-SI-6x128-12_13_01-4575/last.ckpt"
SI_linear_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_Linear-SI-6x128-01_19_11-9386/last.ckpt"
EDM_50e_stand="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_Static_50e-diffusion-6x128-12_13_21-4237/last.ckpt"
UNET_50e_stand="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_Static_50e-unet-6x128-12_12_16-6743/last.ckpt"
CorrDiff_Static_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrDiff_Static_50e-CorrDiff-6x128-12_15_19-5090/last.ckpt"
CorrSI_mean_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrSI_mean-CorrDiff-6x128-01_21_10-4310/last.ckpt"
CorrSI_mean_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrSI_mean-CorrDiff-6x128-01_22_17-1490/last.ckpt"

# EDM
# Training
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --batch_size 2 --epochs 25 --lr 0.00001 --val_interval 10 # EDM_res

# Training trial
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 16 --batch_size 4 --epochs 1

# Testing
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --batch_size 10 --epochs 50 --lr 0.00001 --load $EDM_50e_stand --eval val # --save_output --subset_ds

# Testing trial
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 2 --eval test --n_example_pred 2 --batch_size 2 --subset_ds --ensemble_size 5 --sampler edm --sampler_steps 20 --load "/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_t2m-diffusion-6x128-07_05_07-8839/last.ckpt"

# SI
# Training
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --sampler euler --sampler_steps 50 --val_interval 10 --keep_cond # SI_keep_cond

# Training trial
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 1 --batch_size 2 --epochs 2 --subset_ds

# Testing
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --batch_size 10 --epochs 50 --lr 0.00001 --load $SI_50e_stand --eval val --sampler euler --sampler_steps 100 --ensemble_size 5 --diffusion_fn g_sigma --sigma_coef_sampling 1.5 # --save_output
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --batch_size 10 --epochs 50 --lr 0.00001 --load $SI_linear_50e --eval val --sampler euler --sampler_steps 50 --ensemble_size 5 --beta_fn linear # SI_linear_eval
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --batch_size 10 --epochs 50 --lr 0.00001 --load $SI_50e_stand --eval val --sampler euler --sampler_steps 500 --subset_ds--diffusion_fn g_sigma_pow --sigma_coef_sampling 1.0 # --save_output  
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME --n_workers 16 --eval test --n_example_pred 1 --batch_size 8 --ensemble_size 5 (5-25) --sampler_steps 100 (10-100) --sampler euler_2 (euler/euler_2) --load $SI_25e 

# Testing trial
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 2 --eval val --n_example_pred 1 --batch_size 2 --ensemble_size 5 --subset_ds --load /mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_Static_50e-SI-6x128-12_13_01-4575/last.ckpt --sampler euler --wandb_run_name Tuning --sampler_steps 500 --diffusion_fn g_sigma --sigma_coef_sampling 2.0 

# UNET
# Train
# python3 neural_lam/train_model.py --model unet $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001

# Eval
# python3 neural_lam/train_model.py --model unet $RUN_NAME --n_workers 16 --batch_size 10 --epochs 50 --lr 0.00001 --load /mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_Static_50e-unet-6x128-12_12_16-6743/last.ckpt --eval val --subset_ds # --save_output --subset_ds

# CorrDiff
# python3 neural_lam/train_model.py --model CorrDiff --n_workers 4 --batch_size 2 --epochs 50 --lr 0.00001 --subset_ds --pred_residual 

# Training
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --val_interval 10 # CorrDiff
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --val_interval 10 --pred_residual # CorrDiff_res
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --val_interval 10 --sampler_steps 50 --residual_model SI --diffusion_model song_unet # CorrSI
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --val_interval 5 --sampler_steps 50 --residual_model SI_mean --diffusion_model song_unet --load $CorrSI_mean_25e --restore_opt # CorrSI_mean

# Eval
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 2 --epochs 50 --lr 0.00001 --eval val --load $CorrDiff_Static_50e  --subset_ds # --save_output
# python3 neural_lam/train_model.py --model CorrDiff --n_workers 16 --batch_size 2 --epochs 25 --lr 0.00001 --val_interval 10 --sampler_steps 5 --residual_model SI_mean --diffusion_model song_unet --sampler euler --eval val --subset_ds
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 2 --batch_size 2 --epochs 25 --lr 0.00001 --val_interval 10 --sampler_steps 50 --residual_model SI_mean --diffusion_model song_unet --eval val --load $CorrSI_mean_25e --subset_ds --diffusion_fn g_sigma --sigma_coef_sampling 2.0 # CorrSI_mean


# Tips and tricks on alvis:

# Monitoring jobs:
# job_stats.py # Opens a page in the terminal that updates every few seconds with info about your jobs
# jobinfo # Gives a summary of available and used resources


# CorrSI_mean_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrSI_mean-CorrDiff-6x128-01_22_17-1490/last.ckpt"
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 10 --epochs 25 --lr 0.00001 --val_interval 10 --sampler_steps 50 --residual_model SI_mean --diffusion_model song_unet --eval val --load $CorrSI_mean_50e