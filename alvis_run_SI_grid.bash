#!/bin/bash
#SBATCH -J SI_grid
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:1
#SBATCH -t 05:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

# Projects:
# NAISS2025-22-1196
# naiss2025-1-11


# GPUs:
# --gpus-per-node=A100fat:3

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name SI_corrector"
    
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

# Training
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
#     --sampler_steps 50 \
#     --val_interval 5 \
#     --load $SI_50e \
#     --restore_opt \

# Testing SI with static input data
SAMPLER_STEPS_LIST=(21 40 100)
CORRECTION_STEPS_LIST=(1 2)
SNR_LIST=(0.01 0.05 0.1 0.3 0.4 0.5)
CORR_TMIN_LIST=(0.3 0.5 0.7 0.9)


# =========================
# LOOP
# =========================

for SAMPLER_STEPS in "${SAMPLER_STEPS_LIST[@]}"; do
for CORRECTION_STEPS in "${CORRECTION_STEPS_LIST[@]}"; do
for SNR in "${SNR_LIST[@]}"; do
for CORR_TMIN in "${CORR_TMIN_LIST[@]}"; do

RUN_NAME="--wandb_run_name SI_corr_ss${SAMPLER_STEPS}_cs${CORRECTION_STEPS}_snr${SNR}_ct${CORR_TMIN}"

echo "======================================"
echo "Running:"
echo "sampler_steps=$SAMPLER_STEPS"
echo "correction_steps=$CORRECTION_STEPS"
echo "snr=$SNR"
echo "corr_tmin=$CORR_TMIN"
echo "======================================"

python3 neural_lam/train_model.py \
    --model SI \
    --diffusion_model song_unet \
    $RUN_NAME \
    --n_workers 16 \
    --batch_size 1 \
    --epochs 50 \
    --lr 0.00001 \
    --load $SI_xsmall_25e \
    --eval val \
    --sampler euler \
    --sampler_steps $SAMPLER_STEPS \
    --ensemble_size 5 \
    --correction_steps $CORRECTION_STEPS \
    --snr $SNR \
    --corr_tmin $CORR_TMIN \
    --subset_ds

done
done
done
done
