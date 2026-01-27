#!/bin/bash

RUN_NAME="--wandb_run_name SAMPLES_UNET"

UNET_small_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_small-unet-6x64-01_24_11-5862/last.ckpt"
UNET_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_xsmall-unet-6x32-01_24_10-0012/last.ckpt"
UNET_xxsmall="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_xxsmall-unet-6x32-01_26_09-6705/last.ckpt"

python3 neural_lam/train_model.py --model unet $RUN_NAME --n_workers 2 --batch_size 1 --epochs 25 --lr 0.00001 --hidden_dim 32 --val_interval 10 --eval val --load $UNET_xsmall_25e --subset_ds --save_output

SI_50e_stand="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_Static_50e-SI-6x128-12_13_01-4575/last.ckpt"
SI_small_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_small-SI-6x64-01_24_20-2262/last.ckpt"
SI_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_xsmall-SI-6x32-01_24_10-5547/last.ckpt"
SI_xxsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_xxsmall-SI-6x32-01_26_03-1571/last.ckpt"

RUN_NAME="--wandb_run_name SAMPLES_SI"

# Testing SI with static input data
python3 neural_lam/train_model.py \
    --model SI \
    --diffusion_model song_unet \
    $RUN_NAME  \
    --n_workers 2 \
    --batch_size 1 \
    --epochs 50 \
    --lr 0.00001 \
    --load $SI_xsmall_25e \
    --eval val \
    --sampler euler \
    --sampler_steps 40 \
    --ensemble_size 5 \
    --hidden_dim 32 \
    --subset_ds \
    --save_output \

RUN_NAME="--wandb_run_name SAMPLES_CorrDiff"

CorrrDiff_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/CorrD_xsmall-CorrDiff-6x32-01_25_09-6120/last.ckpt"

python3 neural_lam/train_model.py --model CorrDiff $RUN_NAME --n_workers 2 --batch_size 1 --epochs 25 --lr 0.00001 --val_interval 10 --pred_residual --hidden_dim 32 --mean_model_ckpt_path $UNET_xsmall_25e --subset_ds --save_output --load $CorrrDiff_xsmall_25e --eval val --ensemble_size 5

EDM_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/EDM_xsmall-diffusion-6x32-01_24_13-8505/last.ckpt"

RUN_NAME="--wandb_run_name SAMPLES_EDM"

python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 2 --batch_size 1 --epochs 25 --lr 0.00001 --val_interval 10 --hidden_dim 32 --eval val --load $EDM_xsmall_25e --subset_ds --save_output