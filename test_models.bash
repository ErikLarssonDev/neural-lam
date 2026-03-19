#!/bin/bash
#SBATCH -J TEST_no_extremes
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name TEST"
# DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/no_extremes_nordic.yaml"
DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/no_extremes.yaml"
# DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/clim_config.yaml"

    
# Activate environment
source ~/.bashrc
mamba activate clim

cd /mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam

# Switch to the correct branch
git switch clim-downscaling

# Activate wandb
wandb online

echo -e "Testing models \n"

# Training
echo -e "Training SI \n"
echo -e "\n"
echo -e "\n"
echo -e "\n"

python3 neural_lam/train_model.py \
    --model SI \
    --diffusion_model song_unet \
    $RUN_NAME  \
    --data_config $DATA_CONFIG \
    --n_workers 16 \
    --batch_size 1 \
    --epochs 2 \
    --lr 0.00001 \
    --sampler euler \
    --sampler_steps 50 \
    --val_interval 1 \
    --subset_ds

# Testing SI with static input data
echo -e "Testing SI \n"
echo -e "\n"
echo -e "\n"
echo -e "\n"
python3 neural_lam/train_model.py \
    --model SI \
    --diffusion_model song_unet \
    $RUN_NAME  \
    --data_config $DATA_CONFIG \
    --n_workers 16 \
    --batch_size 1 \
    --epochs 2 \
    --lr 0.00001 \
    --eval val \
    --sampler euler \
    --sampler_steps 50 \
    --subset_ds

# UNET
echo -e "Training UNET \n"
echo -e "\n"
echo -e "\n"
echo -e "\n"
python3 neural_lam/train_model.py --model unet $RUN_NAME --data_config $DATA_CONFIG --n_workers 16 --batch_size 1 --epochs 2 --lr 0.00001 --val_interval 1 --subset_ds

echo -e "Testing UNET \n"
echo -e "\n"
echo -e "\n"
echo -e "\n"
python3 neural_lam/train_model.py --model unet $RUN_NAME --data_config $DATA_CONFIG --n_workers 16 --batch_size 1 --epochs 2 --lr 0.00001 --val_interval 1 --subset_ds --eval val

# EDM
echo -e "Training EDM \n"
echo -e "\n"
echo -e "\n"
echo -e "\n"
python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME --data_config $DATA_CONFIG --n_workers 16 --batch_size 1 --epochs 2 --lr 0.00001 --val_interval 1 --subset_ds
echo -e "Testing EDM \n"
echo -e "\n"
echo -e "\n"
echo -e "\n"
python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME --data_config $DATA_CONFIG --n_workers 16 --batch_size 1 --epochs 2 --lr 0.00001 --val_interval 1 --subset_ds --eval val

# CorrDiff
echo -e "Training CorrDiff \n"
echo -e "\n"
echo -e "\n"
echo -e "\n"
python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --data_config $DATA_CONFIG --n_workers 16 --batch_size 1 --epochs 2 --lr 0.00001 --val_interval 1 --pred_residual --subset_ds # --mean_model_ckpt_path $UNET 
echo -e "Testing CorrDiff \n"
echo -e "\n"
echo -e "\n"
echo -e "\n"
python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --data_config $DATA_CONFIG --n_workers 16 --batch_size 1 --epochs 2 --lr 0.00001 --val_interval 1 --pred_residual --subset_ds --eval val




