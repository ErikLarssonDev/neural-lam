#!/bin/bash
#SBATCH -J SI_1200e
#SBATCH -t 0-05:00:00
#SBATCH --gpus=8
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
MODEL="--model SI"
# DIFFUSION_MODEL="--diffusion_model graph_fm --graph hierarchical-3"
RUN_NAME="--wandb_run_name SI_1200e"

# Paths to saved models
SI_200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_neural_lam_200e-SI-6x128-04_14_16-7715/last.ckpt"
SI_600e_bug="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_neural_lam_600e-SI-6x128-04_15_08-8344/last.ckpt"
SI_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_neural_lam_600e-SI-6x128-04_15_19-6421/last.ckpt"
SI_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_1000e-SI-6x128-04_16_17-6061/last.ckpt"

# Execute Python script with arguments
# Train
python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --batch_size 12 --epochs 1200 --lr 0.00001 --load $SI_1000e

# Test
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME --n_workers 2 --eval val --n_example_pred 1 --batch_size 8 --ensemble_size 5 --sampler_steps 100 --load $IR_SDE_600e   

# Trial train
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 16 --batch_size 12 --epochs 200

# Trial test
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 2 --eval test --n_example_pred 1 --batch_size 4 --subset_ds --ensemble_size 1 --sampler_steps 100 --save_steps --subset_ds --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_1000e-SI-6x128-04_16_17-6061/last.ckpt --sigma_coef 0.5
