#!/bin/bash
#SBATCH -J SI
#SBATCH -A NAISS2024-22-955 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:4
#SBATCH -t 1-00:00:00
#SBATCH --output=exp8.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name SI_50e"

# Switch to the correct directory
cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

output_path="output/250725/exp8"
ensemble_size=25
sampler_steps=100
sampler="euler"

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
  --data_config neural_lam/clim_config_inference.yaml \
  --diffusion_model song_unet \
  --output_path $output_path \
  --n_workers 16 \
  --eval test \
  --n_example_pred 99999 \
  --batch_size 1 \
  --ensemble_size $ensemble_size \
  --sampler_steps $sampler_steps \
  --load $SI_50e \
  --sampler $sampler \
  --save_output