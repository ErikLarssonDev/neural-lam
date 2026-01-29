#!/bin/bash
#SBATCH -J neural-lam
#SBATCH -A NAISS2025-1-11 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 24:00:00
#SBATCH --output=EDM_100.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name test_EDM_small"

# Switch to the correct directory
cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

# If we run on multiple GPUs, check so that we don't overwrite output from other runs.
output_path="output/280126/EDM_100"
ensemble_size=5 # 5 - val, 20 - test, should probably run 5 per GPU.
sampler_steps=100
sampler="edm"
batch_size=5
n_workers=16
diffusion_model="edm"
model="diffusion" 
data_config="neural_lam/clim_config.yaml"

mkdir -p $output_path
cp alvis_job_inference_EDM.bash $output_path
cp $data_config $output_path

# Saved models
# model_checkpoint="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_Static_50e-SI-6x128-12_13_01-4575/last.ckpt"
#model_checkpoint="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrDiff_Static_50e-CorrDiff-6x128-12_15_19-5090/last.ckpt"
#model_checkpoint="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrSI_mean-CorrDiff-6x128-01_22_17-1490/last.ckpt"
EDM_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_xsmall-diffusion-6x32-01_24_13-8505/last.ckpt"


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
    --load $EDM_xsmall_25e \
    --sampler $sampler \
    --hidden_dim 32 \
    --save_output