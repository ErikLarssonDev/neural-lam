#!/bin/bash
#SBATCH -J neural-lam
#SBATCH -A NAISS2025-1-11 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 14:00:00
#SBATCH --output=SI_200.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name test_SI_small"

# Switch to the correct directory
cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

# If we run on multiple GPUs, check so that we don't overwrite output from other runs.
output_path="output/280126/SI_200"
ensemble_size=5 # 5 - val, 20 - test, should probably run 5 per GPU.
sampler_steps=200
sampler="euler"
batch_size=5
n_workers=16
diffusion_model="song_unet"
model="SI"
data_config="neural_lam/clim_config.yaml"

mkdir -p $output_path
cp alvis_job_inference_SI.bash $output_path
cp $data_config $output_path

# Saved models
SI_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_xsmall-SI-6x32-01_24_10-5547/last.ckpt"

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
    --diffusion_model $diffusion_model \
    --output_path $output_path \
    --n_workers $n_workers \
    --eval val \
    --batch_size $batch_size \
    --ensemble_size $ensemble_size \
    --sampler_steps $sampler_steps \
    --load $SI_xsmall_25e \
    --sampler $sampler \
    --hidden_dim 32 \
    --save_output