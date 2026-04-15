#!/bin/bash
#SBATCH -J CorrrDiff_xsmall_25e_100s_test_r2_metrics_only
#SBATCH -A naiss2025-1-11 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name CorrrDiff_xsmall_25e_100s_test_r2_metrics_only"
    
# Activate environment
source ~/.bashrc
# source /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/envs/neural-lam/bin/activate
mamba activate clim

cd /mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam
# cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

# Switch to the correct branch
git switch clim-downscaling-inference

# Activate wandb
wandb online

DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/clim_config_r2.yaml"


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

# python3 neural_lam/train_model.py \
#     --model CorrDiff $RUN_NAME \
#     --n_workers 16 \
#     --batch_size 2 \
#     --epochs 25 \
#     --lr 0.00001 \
#     --val_interval 10 \
#     --pred_residual \
#     --hidden_dim 32 \
#     --mean_model_ckpt_path $UNET_32 \
#     --load $CorrrDiff_xsmall_25e \
#     --restore_opt \


# 50 steps 3 GPUs
# 5 ens -> 2.25 h
# 20 ens -> 9 h
# 50 ens -> 22,5 h
# 100 ens -> 45 h

python3 neural_lam/train_model.py \
    --model CorrDiff $RUN_NAME \
    --n_workers 16 \
    --batch_size 10 \
    --epochs 25 \
    --lr 0.00001 \
    --val_interval 10 \
    --pred_residual \
    --mean_model_ckpt_path $UNET_32 \
    --load $CorrrDiff_xsmall_25e \
    --eval test \
    --sampler_steps 50 \
    --ensemble_size 20 \
    --data_config $DATA_CONFIG \
    # --save_output \


