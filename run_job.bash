#!/bin/bash
#SBATCH -J edm_state
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
DIFFUSION_MODEL="--diffusion_model edm --graph hierarchical"
RUN_NAME="--wandb_run_name edm_state"
PATH_TO_MODEL="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/diff_graph_fm_hier-diffusion-4x64-09_16_13-7088/last.ckpt"
DIFFUSION_MODEL_400e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/diff_graph_fm_hier_400e-diffusion-4x64-09_23_09-3756/last.ckpt"
DIFFUSION_GRAPH_FM_CON_NORM_200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/diff_con_layer_norm_res-diffusion-4x64-10_03_10-5623/last.ckpt"
LOSS_DIFF_GRAPH_CON_NORM_RES="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/loss_res_graph_fm_con_norm_no_border-diffusion-4x64-10_15_11-6256/last.ckpt"
LOSS_DIFF_EDM_CON_NORM_RES="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/loss_diff_edm_no_border-diffusion-4x64-10_15_10-5635/last.ckpt"
NORM_LOSS_DIFF_GRAPH_CON_NORM_RES="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/norm_res-diffusion-4x64-10_29_14-5875/last.ckpt"

# Execute Python script with arguments
python3 train_model.py "--model" $MODEL $DIFFUSION_MODEL "--n_workers" 16 $RUN_NAME --batch_size 4 --val_interval 10 # --pred_residual # --load $NORM_LOSS_DIFF_GRAPH_CON_NORM_RES --eval test --subset_ds 1 --sampler heun

# python3 train_model.py --model diffusion --diffusion_model graph_fm --graph hierarchical --pred_residual --batch_size 10 --subset_ds 1 --eval test  --sampler heun --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/loss_res_graph_fm_con_norm_no_border-diffusion-4x64-10_15_11-6256/last.ckpt 
# python3 train_model.py --model diffusion --diffusion_model edm --pred_residual --batch_size 4 --wandb_run_name std_1_loss_diff_edm_no_border --val_interval 10 --n_workers 16 # --eval val --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/loss_diff_edm_no_border-diffusion-4x64-10_15_10-5635/last.ckpt
# python3 train_model.py --model swin_u2 --pred_residual --wandb_project neural_lam_wavelet --val_interval 10 --batch_size 10 --n_workers 4
# Sanity check 1: 10 min
# Sanity check 2: 
# Batch time training: 0.5 s / batch => 6 min / epoch
# Batch time validation: train * 19 * 39 = 19 * 39 * 0.5 s = 6 min => 169 * 6 min = 17 h / epoch, estimated 27 h for 1 epochs and 10 min per batch.

# python3 create_mesh.py --graph hierarchical --hierarchical 1

# python3 train_model.py --model graph_fm --graph hierarchical --batch_size 10 --val_interval 10 --wandb_run_name graph_fm