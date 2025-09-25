#!/bin/bash
#SBATCH -J EDM_Spectra
#SBATCH -t 03-00:00:00
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate BZ31
wandb online

cd /proj/berzelius-2022-164/users/x_erila/neural-lam
git switch main

RUN_NAME="--wandb_run_name CRPS_Samples"
CRPS_res_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_res_1200e-CRPS-6x128-07_27_10-5686/last.ckpt"
CRPS_AR_4_res_1600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_AR_4_res_1600e-CRPS-6x128-08_22_17-9414/last.ckpt"

python3 neural_lam/train_model.py \
        --model CRPS $RUN_NAME\
        --n_workers 16\
        --pred_residual\
        --border_condition\
        --batch_size 4\
        --load $CRPS_AR_4_res_1600e\
        --eval test\
        --n_example_pred 1\
        --ensemble_size 1\
        --save_output \

# RUN_NAME="--wandb_run_name EDM_Spectra"
# EDM_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/paper_checkpoints/Diffusion-LAM/last.ckpt" # Best model

# python3 neural_lam/train_model.py\
#     --model diffusion\
#     --diffusion_model edm\
#      $RUN_NAME\
#      --n_workers 16\
#      --pred_residual\
#      --border_condition\
#      --eval test\
#      --n_example_pred 1 \
#      --batch_size 8 \
#      --ensemble_size 1 \
#      --load $EDM_1200e \
#      --sampler heun \
