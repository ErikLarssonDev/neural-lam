#!/bin/bash
#SBATCH -J weather_forecasting_WNO2d
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

MODEL="WNO2d" # FNO2d, WNO2d, diffusion

# Execute Python script with JSON configuration
python3 $PYTHON_SCRIPT_PATH "--model" $MODEL