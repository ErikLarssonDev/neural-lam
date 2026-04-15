#!/bin/bash
#SBATCH -J SI_xsmall_25e_100s_test_r2_metrics
#SBATCH -A naiss2025-1-11 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

# Projects:
# NAISS2025-22-1196
# naiss2025-1-11

# squeue --me --format="%.18i %.30j %.8u %.10T %.10M %.10l %.6D %R"

# GPUs:
# --gpus-per-node=A100fat:3

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name SI_xsmall_25e_100s_test_r2_metrics"
    
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

SI_50e_stand="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_Static_50e-SI-6x128-12_13_01-4575/last.ckpt"
SI_small_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_small-SI-6x64-01_24_20-2262/last.ckpt"
SI_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_xsmall-SI-6x32-01_24_10-5547/last.ckpt"
SI_xxsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_xxsmall-SI-6x32-01_26_03-1571/last.ckpt"

DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/clim_config_r2.yaml"

SI_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_13_var-SI-6x128-03_23_14-4660/last.ckpt"
SI_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_13_var-SI-6x128-03_24_10-0270/last.ckpt"
SI_75e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_13_var-SI-6x128-03_25_12-7982/last.ckpt"

# Training
# python3 neural_lam/train_model.py \
#     --model SI \
#     --diffusion_model song_unet \
#     $RUN_NAME  \
#     --n_workers 16 \
#     --batch_size 2 \
#     --epochs 50 \
#     --lr 0.00001 \
#     --sampler euler \
#     --sampler_steps 50 \
#     --val_interval 5 \
#     --load $SI_xsmall_25e \
#     --restore_opt \
    # --data_config $DATA_CONFIG \

# 100 steps 3 GPUs
# 5 ens -> 2.25 h
# 20 ens -> 9 h
# 50 ens -> 22,5 h
# 100 ens -> 45 h

# Testing SI with static input data
python3 neural_lam/train_model.py \
    --model SI \
    --diffusion_model song_unet \
    $RUN_NAME  \
    --n_workers 16 \
    --batch_size 10 \
    --epochs 50 \
    --lr 0.00001 \
    --load $SI_xsmall_25e \
    --eval test \
    --sampler euler \
    --sampler_steps 100 \
    --ensemble_size 20 \
    --data_config $DATA_CONFIG \
    # --save_output \
    # --diffusion_fn g_opt \
#     --correction_steps 1 \
#     --snr 0.1 \
#     --corr_tmin 0.1 \
    # --subset_ds \
