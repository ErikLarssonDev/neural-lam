#!/bin/bash
#SBATCH -J CorrD_xxsmall
#SBATCH -A NAISS2025-22-1196 -p alvis
#SBATCH -N 1 --gpus-per-node=A100fat:1
#SBATCH -t 20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out

export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name CorrD_xxsmall"

# DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/no_extremes_nordic.yaml"
DATA_CONFIG="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/neural_lam/no_extremes.yaml"
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

UNET=""

python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --data_config $DATA_CONFIG --n_workers 16 --batch_size 10 --epochs 25 --lr 0.00001 --val_interval 10 --pred_residual --mean_model_ckpt_path $UNET 

# python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 16 --batch_size 10 --epochs 25 --lr 0.00001 --val_interval 10 --pred_residual --mean_model_ckpt_path $UNET --load $CorrrDiff_xsmall_25e --eval val --ensemble_size 5

