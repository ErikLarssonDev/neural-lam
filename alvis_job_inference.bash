#!/bin/bash
#SBATCH -J neural-lam
#SBATCH -A NAISS2025-1-11 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:1
#SBATCH -t 1-00:00:00
#SBATCH --output=val_SI.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name val_SI"

# Switch to the correct directory
cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

output_path="output/230126/val_SI"
ensemble_size=5
sampler_steps=50
sampler="euler"
batch_size=8
n_workers=8

mkdir -p $output_path
cp alvis_job_inference.bash $output_path

# Saved models
model_checkpoint="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_Static_50e-SI-6x128-12_13_01-4575/last.ckpt"

# Define the local repository path
REPO_PATH="/mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam"

# Run the container
apptainer exec \
  --bind ${REPO_PATH}:/opt/neural-lam \
  --pwd /opt/neural-lam \
  ~/neural-lam.sif python3 neural_lam/train_model.py \
  --model SI \
  --data_config neural_lam/clim_config.yaml \
  --diffusion_model song_unet \
  --output_path $output_path \
  --n_workers $n_workers \
  --eval test \
  --batch_size $batch_size \
  --ensemble_size $ensemble_size \
  --sampler_steps $sampler_steps \
  --load $model_checkpoint \
  --sampler $sampler \
  --save_output