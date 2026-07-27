#!/bin/bash
#SBATCH -J CorrD_13_var
#SBATCH -A NAISS2025-1-11 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name CorrD_13_var"
    
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
UNET_xxsmall="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_xxsmall-unet-6x32-01_26_09-6705/last.ckpt"

CorrrDiff_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrD_xsmall-CorrDiff-6x32-01_25_09-6120/last.ckpt"

# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 10 --epochs 25 --lr 0.00001 --val_interval 10 --pred_residual --hidden_dim 32 --mean_model_ckpt_path $UNET_32 # --channel_mult "1,1,1,1"

# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 10 --epochs 25 --lr 0.00001 --val_interval 10 --pred_residual --hidden_dim 32 --mean_model_ckpt_path $UNET_32 --load $CorrrDiff_xsmall_25e --eval val --ensemble_size 5

# 13 Variables
DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/clim_config_2.yaml"
UNET_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_13_var-unet-6x128-03_24_02-5844/last.ckpt"
CorrDiff_13_19e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrD_13_var-CorrDiff-6x128-04_15_19-1970/last.ckpt"
CorrDiff_13_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrD_13_var-CorrDiff-6x128-04_16_19-3137/last.ckpt"

python3 neural_lam/train_model.py \
    --model CorrDiff $RUN_NAME \
    --data_config $DATA_CONFIG \
    --n_workers 16 \
    --batch_size 2 \
    --epochs 150 \
    --lr 0.00001 \
    --val_interval 5 \
    --pred_residual \
    --mean_model_ckpt_path $UNET_25e \
    --load $CorrDiff_13_50e \
    --restore_opt \
    --eval val \
    --sampler_steps 50 \