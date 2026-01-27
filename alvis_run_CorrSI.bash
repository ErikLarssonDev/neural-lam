#!/bin/bash
#SBATCH -J CorrSI_LQ_xsmall
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:3
#SBATCH -t 04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name CorrSI_LQ_xsmall"
    
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

# CorrDiff_Static_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrDiff_Static_50e-CorrDiff-6x128-12_15_19-5090/last.ckpt"

# # Testing CorrDiff with static input data
# python3 neural_lam/train_model.py \
#     --model CorrDiff \
#     $RUN_NAME \
#     --n_workers 16 \
#     --batch_size 10 \
#     --epochs 50 \
#     --lr 0.00001 \
#     --eval test \
#     --ensemble_size 20 \
#     --load $CorrDiff_Static_50e # --save_output

UNET_64="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_small-unet-6x64-01_24_11-5862/last.ckpt"
UNET_32="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_xsmall-unet-6x32-01_24_10-0012/last.ckpt"

# Can do either SI or SI_mean as residual model
# SI from mean to HQ

CorrSI_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrSI_xsmall-CorrDiff-6x32-01_25_09-0002/last.ckpt"
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 2 --epochs 25 --lr 0.00001 --val_interval 10 --hidden_dim 32 --mean_model_ckpt_path $UNET_32 --residual_model SI_mean --diffusion_model song_unet --sampler_steps 50
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 10 --epochs 25 --lr 0.00001 --val_interval 10 --hidden_dim 32 --mean_model_ckpt_path $UNET_32 --residual_model SI_mean --diffusion_model song_unet --sampler_steps 50 --load $CorrSI_xsmall_25e --eval val 


# SI From LQ to HQ
CorrSI_LQ_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrSI_xsmall-CorrDiff-6x32-01_25_14-2624/last.ckpt"
# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 2 --epochs 25 --lr 0.00001 --val_interval 10 --hidden_dim 32 --mean_model_ckpt_path $UNET_32 --residual_model SI --diffusion_model song_unet --sampler_steps 50
python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 10 --epochs 25 --lr 0.00001 --val_interval 10 --hidden_dim 32 --mean_model_ckpt_path $UNET_32 --residual_model SI --diffusion_model song_unet --sampler_steps 50 --load $CorrSI_LQ_xsmall_25e --eval val

