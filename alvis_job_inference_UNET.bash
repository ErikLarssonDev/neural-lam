#!/bin/bash
#SBATCH -J neural-lam
#SBATCH -A NAISS2025-1-11 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 01:00:00
#SBATCH --output=val_UNet_small_r2.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name UNET_small"

# Switch to the correct directory
cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

output_path="output/270126/val_UNet_small_r2"
batch_size=5
n_workers=16
model="unet"
data_config="neural_lam/clim_config_r2.yaml"

mkdir -p $output_path
cp alvis_job_inference_UNET.bash $output_path
cp $data_config $output_path

# Saved models
# model_checkpoint="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_Static_50e-SI-6x128-12_13_01-4575/last.ckpt"
#model_checkpoint="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrDiff_Static_50e-CorrDiff-6x128-12_15_19-5090/last.ckpt"
#model_checkpoint="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrSI_mean-CorrDiff-6x128-01_22_17-1490/last.ckpt"
UNET_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_xsmall-unet-6x32-01_24_10-0012/last.ckpt"


# Define the local repository path
REPO_PATH="/mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam"

# Run the container
apptainer exec \
  --bind ${REPO_PATH}:/opt/neural-lam \
  --pwd /opt/neural-lam \
  ~/neural-lam.sif \
  python3 neural_lam/train_model.py \
    --model $model \
    --data_config $data_config \
    --output_path $output_path \
    --n_workers $n_workers \
    --eval val \
    --batch_size $batch_size \
    --load $UNET_xsmall_25e \
    --hidden_dim 32 \
    --save_output