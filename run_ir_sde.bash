#!/bin/bash
#SBATCH -J IR_SDE_neural_lam_200e
#SBATCH -t 2-00:00:00
#SBATCH --gpus=1
#SBATCH -C "fat"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate BZ31
wandb online

cd /proj/berzelius-2022-164/users/x_erila/neural-lam
git switch downscaling

# Standard arguments
MODEL="--model ir_sde"
# DIFFUSION_MODEL="--diffusion_model graph_fm --graph hierarchical-3"
RUN_NAME="--wandb_run_name IR_SDE_neural_lam_200e"

# Paths to saved models

# Execute Python script with arguments
# Train
python3 neural_lam/train_model.py --model ir_sde --diffusion_model song_unet $RUN_NAME  --n_workers 16 --batch_size 12 --epochs 200  --sampler_steps 100 # --load $IR_SDE_600e

# Test
# python3 neural_lam/train_model.py --model ir_sde --diffusion_model song_unet $RUN_NAME --n_workers 2 --eval val --n_example_pred 1 --batch_size 8 --ensemble_size 5 --sampler_steps 100 --load $IR_SDE_600e  

# Trial train
# python3 neural_lam/train_model.py --model ir_sde --diffusion_model song_unet --n_workers 16 --batch_size 12 --sampler_steps 100 --epochs 5 --subset_ds

# Trial test
# python3 neural_lam/train_model.py --model ir_sde --diffusion_model song_unet --n_workers 2 --eval test --n_example_pred 1 --batch_size 4 --subset_ds --ensemble_size 3 --sampler_steps 100 --load "/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/IR_SDE_neural_lam_200e-ir_sde-6x128-03_14_16-5847/last.ckpt"
