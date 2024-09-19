#!/bin/bash
#SBATCH -J eval_diff_graph_fm_hier
#SBATCH -t 3-00:00:00
#SBATCH --gpus=1 -C "thin"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate BZ31
wandb online

cd /proj/berzelius-2022-164/users/x_erila/neural-lam
git switch prob_model_lam

# Path to your Python script
PYTHON_SCRIPT_PATH="neural_lam.train_model"

MODEL="diffusion" # N_O, WNO2d, diffusion
DIFFUSION_MODEL="--diffusion_model graph_fm --graph hierarchical"
RUN_NAME="--wandb_run_name diff_graph_fm_hier"
PATH_TO_MODEL="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/diff_graph_fm_hier-diffusion-4x64-09_16_13-7088/min_val_loss.ckpt"

# Execute Python script with arguments
python3 train_model.py "--model" $MODEL $DIFFUSION_MODEL "--n_workers" 16 $RUN_NAME "--pred_residual" --batch_size 10 --val_interval 10 --eval val --load $PATH_TO_MODEL
# Fix Wandb run name
# python3 train_model.py --model diffusion --diffusion_model graph_fm --graph hierarchical --pred_residual --batch_size 10 --eval val --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graphcast_diff_200e-diffusion-4x64-09_06_13-0341/last.ckpt --dataset meps_example

# Sanity check 1: 10 min
# Sanity check 2: 
# Batch time training: 0.5 s / batch => 6 min / epoch
# Batch time validation: train * 19 * 39 = 19 * 39 * 0.5 s = 6 min => 169 * 6 min = 17 h / epoch, estimated 27 h for 1 epochs and 10 min per batch.