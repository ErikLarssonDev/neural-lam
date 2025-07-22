#!/bin/bash
#SBATCH -J SI
#SBATCH -A NAISS2024-22-955 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 1-00:00:00

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name SI_50e"

# Switch to the correct directory
cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

# Activate wandb
apptainer exec ~/neural-lam.sif wandb off

# Saved models
SI_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_50e-SI-6x128-07_10_12-7283/last.ckpt"

# Inference
apptainer exec ~/neural-lam.sif python3 neural_lam/train_model.py \
  --model SI \
  --data_config neural_lam/clim_config_inference.yaml \
  --diffusion_model song_unet \
  --output_path output/220725/steps_10 \
  --n_workers 16 \
  --eval test \
  --n_example_pred 99999 \
  --batch_size 1 \
  --ensemble_size 4 \
  --sampler_steps 10 \
  --load $SI_50e \
  --sampler euler_2 \
  --save_output