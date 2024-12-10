#!/bin/bash
#SBATCH -J BORDER_GRAPH_FM_400e
#SBATCH -t 3-00:00:00
#SBATCH --gpus=1 -C "thin"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate BZ31
wandb online

cd /proj/berzelius-2022-164/users/x_erila/neural-lam
git switch prob-model-boundary

# Path to your Python script
PYTHON_SCRIPT_PATH="neural_lam.train_model"

MODEL="diffusion" # N_O, WNO2d, diffusion
DIFFUSION_MODEL="--diffusion_model graph_fm --graph hierarchical"
RUN_NAME="--wandb_run_name BORDER_GRAPH_FM_400e"

# Paths to saved models
PATH_TO_MODEL="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/diff_graph_fm_hier-diffusion-4x64-09_16_13-7088/last.ckpt"
DIFFUSION_MODEL_400e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/diff_graph_fm_hier_400e-diffusion-4x64-09_23_09-3756/last.ckpt"
DIFFUSION_GRAPH_FM_CON_NORM_200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/diff_con_layer_norm_res-diffusion-4x64-10_03_10-5623/last.ckpt"
LOSS_DIFF_GRAPH_CON_NORM_RES="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/loss_res_graph_fm_con_norm_no_border-diffusion-4x64-10_15_11-6256/last.ckpt"
LOSS_DIFF_EDM_CON_NORM_RES="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/loss_diff_edm_no_border-diffusion-4x64-10_15_10-5635/last.ckpt"
NORM_LOSS_DIFF_GRAPH_CON_NORM_RES="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/norm_res-diffusion-4x64-10_29_14-5875/last.ckpt"
NORM_LOSS_DIFF_EDM="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/edm_res-diffusion-6x128-10_31_09-3448/last.ckpt"
NORM_LOSS_STATE_EDM="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/edm_state-diffusion-6x128-10_31_09-9003/last.ckpt"
NORM_LOSS_RES_GRAPH_FM="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm-diffusion-6x128-10_31_09-8082/last.ckpt"
NORM_LOSS_STATE_GRAPH_FM="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_state-diffusion-6x128-10_31_09-1230/last.ckpt"
NORM_LOSS_RES_GRAPH_FM_SIGMA_002="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_sigmamin_0.002-diffusion-6x128-11_07_11-2917/last.ckpt"
NORM_LOSS_RES_GRAPH_FM_SIGMA_0002="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_sigmamin_0.0002-diffusion-6x128-11_07_11-8652/last.ckpt"
NORM_LOSS_RES_GRAPH_FM_400e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_sigmamin_0.0002-diffusion-6x128-11_08_10-2170/last.ckpt"
BORDER_GRAPH_FM="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_border_condition-diffusion-6x128-12_02_14-8805/last.ckpt"
BORDER_GRAPH_FM_300e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_border_condition_300e-diffusion-6x128-12_06_07-7658/last.ckpt"
COSINE_BORDER_GRAPH_FM_400e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/cosine_graph_fm_border_condition_400e-diffusion-6x128-12_09_07-5373/last.ckpt"
BORDER_GRAPH_FM_400e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_border_condition_400e-diffusion-6x128-12_09_07-5907/last.ckpt"

# Execute Python script with arguments
python3 neural_lam/train_model.py "--model" $MODEL $DIFFUSION_MODEL "--n_workers" 16 $RUN_NAME --batch_size 4 --val_interval 10 --pred_residual --border_condition --load $BORDER_GRAPH_FM_400e --eval test # --lr 0.00001 --epochs 400 --load $BORDER_GRAPH_FM_300e --lr_scheduler cosine #   # --eval test --sampler heun # --plot_diffusion_steps

# python3 neural_lam/train_model.py --model diffusion --diffusion_model graph_fm --graph hierarchical --pred_residual --batch_size 10 --border_condition --subset_ds --lr_scheduler cosine --epochs 300 --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_border_condition-diffusion-6x128-12_02_14-8805/last.ckpt
# python3 train_model.py --model diffusion --diffusion_model edm --pred_residual --batch_size 4 --wandb_run_name std_1_loss_diff_edm_no_border --val_interval 10 --n_workers 16 # --eval val --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/loss_diff_edm_no_border-diffusion-4x64-10_15_10-5635/last.ckpt
# python3 neural_lam/train_model.py --model graph_efm --graph hierarchical --pred_residual --batch_size 10 --border_condition --subset_ds --eval test 
# Sanity check 1: 10 min
# Sanity check 2: 
# Batch time training: 0.5 s / batch => 6 min / epoch
# Batch time validation: train * 19 * 39 = 19 * 39 * 0.5 s = 6 min => 169 * 6 min = 17 h / epoch, estimated 27 h for 1 epochs and 10 min per batch.

############################################################################################################
# Bash script to create a graph
############################################################################################################

# python3 create_mesh.py --graph hierarchical --hierarchical 1
# python3 neural_lam/build_rectangular_graph.py --output_dir graphs/hierarchical --archetype hierarchical
