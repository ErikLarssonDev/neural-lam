#!/bin/bash
#SBATCH -J SI_1000e_weighted
#SBATCH -t 1-00:00:00
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate BZ31
wandb online

cd /proj/berzelius-2022-164/users/x_erila/neural-lam
git switch prob-model-boundary

# Standard arguments
MODEL="--model diffusion"
DIFFUSION_MODEL="--diffusion_model graph_fm --graph hierarchical-3"
RUN_NAME="--wandb_run_name SI_1000e_weighted"

# Paths to saved models
BORDER_GRAPH_FM_400e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_border_condition_400e-diffusion-6x128-12_09_07-5907/last.ckpt"
BORDER_GRAPH_FM_500e_sigma_0002="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/BORDER_GRAPH_FM_500e_sigma_0002-diffusion-6x128-12_12_16-6574/last.ckpt"
level_3_200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/diff_128x2-diffusion-2x128-01_08_16-3450/last.ckpt"
level_3_400e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/3_level-diffusion-2x128-01_09_13-1229/last.ckpt" # Maybe 0298 will be better
level_3_64_400e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/3_level-diffusion-2x64-01_09_13-0555/last.ckpt"
EDM_400e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/diffusion-6x128-01_09_16-5035/last.ckpt"

level_3_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_128_600e-diffusion-2x128-01_10_09-8022/last.ckpt"
level_3_64_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_128_600e-diffusion-2x64-01_10_09-3224/last.ckpt"
EDM_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/diffusion-6x128-01_10_09-8192/last.ckpt"
EDM_800e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/diffusion-6x128-01_10_18-1813/last.ckpt"
EDM_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/edm_1000e-diffusion-6x128-01_13_11-7178/last.ckpt"
EDM_EQUAL_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/edm_equal-diffusion-6x128-01_13_10-4867/last.ckpt"
EDM_LARGE_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_large-diffusion-6x128-01_12_17-5739/last.ckpt"
EDM_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_1200e-diffusion-6x128-01_14_08-9197/last.ckpt" # Best model
EDM_1600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_1600e-diffusion-6x128-01_17_00-4121/last.ckpt"

EDM_1800e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_1800e-diffusion-6x128-01_17_21-0168/last.ckpt"
EDM_RES_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_RESIDUAL_1000e-diffusion-6x128-01_17_14-1630/last.ckpt"
EDM_NOISE_1800e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/NOISE_EDM_1800e-diffusion-6x128-01_17_21-7249/last.ckpt"

EDM_1400e_NOISE="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_1400e_NOISE-diffusion-6x128-01_18_11-3482/last.ckpt"
EDM_1400e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_1400e-diffusion-6x128-01_18_11-5436/last.ckpt"
EDM_2000e_NOISE="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_NOISE_2000e-diffusion-6x128-01_18_09-9304/last.ckpt"

EDM_RES_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_RES_1200e-diffusion-6x128-01_19_08-7897/last.ckpt"

EDM_RES_1400e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_RES_1400e-diffusion-6x128-01_19_16-9474/last.ckpt"
EDM_2000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_2000e-diffusion-6x128-01_19_16-8718/last.ckpt"

# The loss is not weighted based on the variable
SI_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_600e-SI-6x128-05_06_17-8464/last.ckpt"
SI_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_600e-SI-6x128-05_07_08-7772/last.ckpt" # Wierd name, forgot to change it
SI_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_1200e-SI-6x128-05_07_18-0323/last.ckpt"

# With weighted loss
SI_600e_weighted="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_600e_weighted-SI-6x128-05_20_18-5352/last.ckpt"

# Execute Python script with arguments
# Train
# python3 neural_lam/train_model.py $MODEL $DIFFUSION_MODEL $RUN_NAME --n_workers 16 --pred_residual --border_condition --vertical_propnets 1 --batch_size 12  --processor_layers 2 --hidden_dim 128 --epochs 800 --load $level_3_600e --lr 0.0001
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --pred_residual --border_condition --epochs 2000 --batch_size 12 --load $EDM_1800e --lr 0.0000001 # --encoder_type residual # --noise_aug_prob 0.5
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --border_condition --epochs 600 --batch_size 12
python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --border_condition --epochs 1000 --batch_size 12 --output_std --load $SI_600e_weighted --lr 0.0001 

# Test
# python3 neural_lam/train_model.py $MODEL $DIFFUSION_MODEL $RUN_NAME --n_workers 2 --pred_residual --border_condition --vertical_propnets 1 --batch_size 18  --processor_layers 2 --hidden_dim 128 --eval val --n_example_pred 0 --ensemble_size 5 --load $level_3_600e
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME --n_workers 2 --pred_residual --border_condition --eval test --n_example_pred 1 --batch_size 8 --ensemble_size 5 --load $EDM_1200e --sampler edm --sampler_steps 10 # --encoder_type residual
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME --n_workers 2 --border_condition --eval val --n_example_pred 1 --batch_size 8 --ensemble_size 25 --load $SI_1200e --sampler_steps 100


# Trial train
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --graph hierarchical-3 --n_workers 16 --pred_residual --border_condition --vertical_propnets 1 --batch_size 12  --processor_layers 2 --hidden_dim 128
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 16 --pred_residual --border_condition --resample_filter [1,3,3,1] --channel_mult [2, 2, 2, 2] --encoder_type standard --attn_resolutions [134, 68, 34, 18]
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 16 --border_condition --subset_ds --output_std --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_600e-SI-6x128-05_07_08-7772/last.ckpt

# Trial test
# Batch size 18 for GraphFM
# Batch size (56 max but 32 more stable) for EDM
# python3 neural_lam/train_model.py --model diffusion --diffusion_model graph_fm --graph hierarchical-3 --n_workers 2 --pred_residual --border_condition --vertical_propnets 1 --eval test --n_example_pred 0 --batch_size 18 --processor_layers 2 --hidden_dim 128
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 2 --pred_residual --border_condition --eval test --n_example_pred 1 --batch_size 1 --subset_ds --ensemble_size 2 --sampler ddpm --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_1200e-diffusion-6x128-01_14_08-9197/last.ckpt
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 2 --border_condition --eval test --n_example_pred 1 --batch_size 1 --subset_ds --ensemble_size 5 --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_1200e-SI-6x128-05_07_18-0323/last.ckpt --sampler_steps 100


# Eval
# "thin" batch size 8
# "fat" batch size 16
# N batch size 128
# python3 neural_lam/train_model.py "--model" $MODEL $DIFFUSION_MODEL --n_workers 16 $RUN_NAME --batch_size 18 --val_interval 10 --pred_residual --border_condition --load $BORDER_GRAPH_FM_400e --eval val --n_example_pred 0 # --sigma_min 0.0002 # --sampler heun # --plot_diffusion_steps

# python3 neural_lam/train_model.py --model diffusion --diffusion_model graph_fm --graph hierarchical-3 --pred_residual --border_condition --eval test --n_example_pred 0 --batch_size 18 --vertical_propnets 1 --processor_layers 1 --n_workers 16

# python3 neural_lam/train_model.py --model diffusion --diffusion_model graph_fm --graph hierarchical --pred_residual --border_condition --eval test --n_example_pred 0 --batch_size 18 --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/graph_fm_border_condition_400e-diffusion-6x128-12_09_07-5907/last.ckpt 
# python3 train_model.py --model diffusion --diffusion_model edm --pred_residual --batch_size 4 --wandb_run_name std_1_loss_diff_edm_no_border --val_interval 10 --n_workers 16 # --eval val --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/loss_diff_edm_no_border-diffusion-4x64-10_15_10-5635/last.ckpt
# python3 neural_lam/train_model.py --model graph_efm --graph hierarchical --pred_residual --batch_size 10 --border_condition --subset_ds --eval test 
# Sanity check 1: 10 min
# Sanity check 2: 
# Batch time training: 0.5 s / batch => 6 min / epoch
# Batch time validation: train * 19 * 39 = 19 * 39 * 0.5 s = 6 min => 169 * 6 min = 17 h / epoch, estimated 27 h for 1 epochs and 10 min per batch.

# Val size = 673 samples (ca 1/3 of test data)
# Test size = 2687 samples
# "thin" node 8 samples per batch => 86 batches
# "fat" node 16 samples per batch => 43 batches
# node (8 gpus) 128 samples per batch => 6 batches

# Diffussion model
# 2 s per forward pass => 80 s per ensemble member => 25 min per 19 step rollout
# 5 ensembles => 125 min (2h) per 19 step rollout
# 100 ensembles => 40 (h) per 19 step rollout

# Joel's model
# Test size 336 samples?
# 1.4s per ensemble member => 100 ensemble members => 140 s (2.3 min) per 19 step rollout
# 0.073 per forward pass?

# Max time 72 h
# Eval val - 48 h, batch size 10 "thin" node
# Eval test - 144 h, batch size 10 on "thin" node
# Eval test - 72 h batch size 10 on "fat" node

############################################################################################################
# Bash script to create a graph
############################################################################################################

# python3 create_mesh.py --graph hierarchical --hierarchical 1
# python3 neural_lam/build_rectangular_graph.py --output_dir graphs/hierarchical --archetype hierarchical
