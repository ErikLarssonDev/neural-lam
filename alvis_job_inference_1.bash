#!/bin/bash
#SBATCH -J SI
#SBATCH -A NAISS2025-1-11 -p alvis # NAISS2025-1-11 or NAISS2024-22-955
#SBATCH -N 1 --gpus-per-node=A40:4
#SBATCH -t 7-00:00:00
#SBATCH --output=r2_1985_1990.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name SI_50e"

# Switch to the correct directory
cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

output_path="output/120825/exp9_long/1985-1990"
data_config="output/120825/exp9_long/clim_config_inference_1985-1990.yaml"
ensemble_size=100
sampler_steps=50
sampler="euler"
batch_size=8

mkdir -p $output_path
cp alvis_job_inference.bash $output_path

# Saved models
SI_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_50e-SI-6x128-07_10_12-7283/last.ckpt"

# Define the local repository path
REPO_PATH="/mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam"

# Run the container
apptainer exec \
  --bind ${REPO_PATH}:/opt/neural-lam \
  --pwd /opt/neural-lam \
  ~/neural-lam.sif python3 neural_lam/train_model.py \
  --model SI \
  --data_config $data_config \
  --diffusion_model song_unet \
  --output_path $output_path \
  --n_workers 16 \
  --eval test \
  --batch_size $batch_size \
  --ensemble_size $ensemble_size \
  --sampler_steps $sampler_steps \
  --load $SI_50e \
  --sampler $sampler \
  --save_output