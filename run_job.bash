#!/bin/bash
#SBATCH -J FNO_res_200e_large
#SBATCH -t 3-00:00:00
#SBATCH --gpus=1 -C "thin"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate BZ31
wandb online

cd /proj/berzelius-2022-164/users/x_erila/neural-lam

# Path to your Python script
PYTHON_SCRIPT_PATH="train_model.py"

MODEL="N_O" # N_O, WNO2d, diffusion
NEURAL_OPERATOR="--neural_operator FNO" # FNO, SFNO, UNO
RUN_NAME="FNO_res_200e_large"
PATH_TO_MODEL="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/FNO2d-4x64-08_27_09-9685/last.ckpt"

# Execute Python script with arguments
python3 $PYTHON_SCRIPT_PATH "--model" $MODEL "--n_workers" 16 "--wandb_run_name" $RUN_NAME "--pred_residual"