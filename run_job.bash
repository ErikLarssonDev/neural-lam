#!/bin/bash
#SBATCH -J baseline_downscaling_1400e
#SBATCH -t 1-00:00:00
#SBATCH --gpus=4
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
MODEL="--model diffusion"
# DIFFUSION_MODEL="--diffusion_model graph_fm --graph hierarchical-3"
RUN_NAME="--wandb_run_name baseline_downscaling_1400e"

# Paths to saved models
EDM_200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/downscaling-diffusion-6x128-03_04_10-3876/last.ckpt"
EDM_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/baseline_downscaling_600e-diffusion-6x128-03_05_18-0610/last.ckpt"
EDM_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/baseline_downscaling_1000e-diffusion-6x128-03_06_09-0704/last.ckpt"
EDM_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/baseline_downscaling_1200e-diffusion-6x128-03_07_15-9081/last.ckpt"

# Execute Python script with arguments
# Train
# python3 neural_lam/train_model.py $MODEL $DIFFUSION_MODEL $RUN_NAME --n_workers 16 --pred_residual --border_condition --vertical_propnets 1 --batch_size 12  --processor_layers 2 --hidden_dim 128 --epochs 800 --load $level_3_600e --lr 0.0001
python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --pred_residual --epochs 1400 --batch_size 12 --lr 0.00001 --load $EDM_1200e # --lr 0.0000001 # --encoder_type residual # --noise_aug_prob 0.5

# Test
# python3 neural_lam/train_model.py $MODEL $DIFFUSION_MODEL $RUN_NAME --n_workers 2 --pred_residual --border_condition --vertical_propnets 1 --batch_size 18  --processor_layers 2 --hidden_dim 128 --eval val --n_example_pred 0 --ensemble_size 5 --load $level_3_600e
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME --n_workers 2 --pred_residual --eval val --n_example_pred 1 --batch_size 8 --ensemble_size 5 --load $EDM_1200e --sampler edm # --encoder_type residual

# Trial train
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --graph hierarchical-3 --n_workers 16 --pred_residual --border_condition --vertical_propnets 1 --batch_size 12  --processor_layers 2 --hidden_dim 128
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 16 --pred_residual --border_condition --batch_size 12 --epochs 600

# Trial test
# Batch size 18 for GraphFM
# Batch size (56 max but 32 more stable) for EDM
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 2 --pred_residual --eval test --n_example_pred 1 --batch_size 4 --subset_ds --ensemble_size 5 --sampler edm --sampler_steps 51 --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/baseline_downscaling_1200e-diffusion-6x128-03_07_15-9081/last.ckpt


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
