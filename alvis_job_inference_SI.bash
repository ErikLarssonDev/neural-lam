#!/bin/bash
#SBATCH -J neural-lam
#SBATCH -A NAISS2025-1-11 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 2-00:00:00
#SBATCH --output=test_SI_cont.out

#export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name test_SI_small"

# Switch to the correct directory
cd /mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

# If we run on multiple GPUs, check so that we don't overwrite output from other runs.
output_path="output/230126/test_SI_small"
ensemble_size=20 # 5 - val, 20 - test, should probably run 5 per GPU.
sampler_steps=40
sampler="euler"
batch_size=8
n_workers=8
diffusion_model="song_unet"
model="SI"
data_config="neural_lam/clim_config.yaml"

mkdir -p $output_path
cp alvis_job_inference_SI.bash $output_path
cp $data_config $output_path

# Saved models
# model_checkpoint="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_Static_50e-SI-6x128-12_13_01-4575/last.ckpt"
#model_checkpoint="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrDiff_Static_50e-CorrDiff-6x128-12_15_19-5090/last.ckpt"
#model_checkpoint="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrSI_mean-CorrDiff-6x128-01_22_17-1490/last.ckpt"
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
    --eval test \
    --batch_size $batch_size \
    --ensemble_size $ensemble_size \
    --sampler_steps $sampler_steps \
    --load $SI_xsmall_25e \
    --sampler $sampler \
    --hidden_dim 32 \
    --save_output