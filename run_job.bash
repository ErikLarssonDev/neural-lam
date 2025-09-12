#!/bin/bash
#SBATCH -J CRPS_AR_4_res_1600e
#SBATCH -t 03-00:00:00
#SBATCH --gpus=8
#SBATCH -C "fat"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#
# --gres=gpu:8
# --ntasks-per-node=8
# --cpus-per-task=16
# --dependency=afterany:JOB_ID

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate BZ31
wandb online

cd /proj/berzelius-2022-164/users/x_erila/neural-lam
git switch SI-forecast

# Standard arguments
DIFFUSION_MODEL="--diffusion_model graph_fm --graph hierarchical-3"
RUN_NAME="--wandb_run_name CRPS_AR_4_res_1600e"

# Paths to saved models
EDM_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_1200e-diffusion-6x128-01_14_08-9197/last.ckpt" # Best model

# The loss is not weighted based on the variable
SI_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_600e-SI-6x128-05_06_17-8464/last.ckpt"
SI_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_600e-SI-6x128-05_07_08-7772/last.ckpt" # Wierd name, forgot to change it
SI_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_1200e-SI-6x128-05_07_18-0323/last.ckpt"

# With weighted loss
SI_600e_weighted="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_600e_weighted-SI-6x128-05_20_18-5352/last.ckpt"
SI_1000e_weighted="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_1000e_weighted-SI-6x128-05_21_22-9484/last.ckpt"
SI_1200e_weighted="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_1200e_weighted-SI-6x128-05_22_11-4683/last.ckpt"

# With residual/remove_cond
SI_600e_res="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_res_600e-SI-6x128-07_16_21-1610/last.ckpt"
SI_1000e_res="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_res_1000e-SI-6x128-07_18_04-4341/last.ckpt"
SI_1200e_res="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_res_1200e-SI-6x128-07_24_04-0940/last.ckpt"

# tEDM, v=2
tEDM_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/tEDM_600e-tEDM-6x128-05_22_00-9316/last.ckpt"
tEDM_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/tEDM_1000e-tEDM-6x128-05_22_11-3238/last.ckpt"
tEDM_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/tEDM_1200e-tEDM-6x128-05_22_21-7260/last.ckpt" # Not tested

# tEDM, v=100 (Ablation to see if it works as the gaussian EDM)
tEDM_v100_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/tEDM_600e_v_100-tEDM-6x128-06_05_11-5420/last.ckpt"
tEDM_v100_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/tEDM_1000e_v_100-tEDM-6x128-06_09_17-3570/last.ckpt"
tEDM_v100_1200e=""

# FM
FM_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/FM_600e-FM-6x128-06_10_16-1838/last.ckpt"
FM_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/FM_1000e-FM-6x128-06_11_09-3948/last.ckpt"
FM_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/FM_1200e-FM-6x128-06_13_08-1627/last.ckpt"

# EDM with sigma_min=0.00002 and sigma_max=255 (Motivated by Climate in a bottle)

# CRPS res
CRPS_res_600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_600e_res-CRPS-6x128-07_25_09-6319/last.ckpt"
CRPS_res_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_1000e_res-CRPS-6x128-07_26_16-6370/last.ckpt"
CRPS_res_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_res_1200e-CRPS-6x128-07_27_10-5686/last.ckpt"

CRPS_AR_4_res_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_AR_4_res_1000e-CRPS-6x128-07_29_17-7794/last.ckpt" # 4 AR steps
CRPS_AR_7_res_1200e="" # 7 AR steps, can't fit 8 on one GPU

CRPS_AR_4_res_1504e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_AR_4_res_1600e-CRPS-6x128-08_01_17-9520/last.ckpt"
CRPS_AR_4_res_1600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_AR_4_res_1600e-CRPS-6x128-08_22_17-9414/last.ckpt"
CRPS_AR_7_res_1800e=""

# CRPS
CRPS_1000e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_600e-CRPS-6x128-07_24_12-2043/last.ckpt" # Trained 1000e with lr 0.001 by mistake (skipping stage 2 training)
CRPS_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_1200e-CRPS-6x128-07_27_10-1905/last.ckpt"

# Execute Python script with arguments
# Train
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME  --n_workers 16 --pred_residual --border_condition --epochs 2000 --batch_size 12 --load $EDM_1800e --lr 0.0000001 # --encoder_type residual # --noise_aug_prob 0.5
# python3 neural_lam/train_model.py --model tEDM --diffusion_model edm $RUN_NAME  --n_workers 16 --border_condition --epochs 600 --batch_size 12 --sigma_min 0.00002 --sigma_max 255
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME  --n_workers 16 --pred_residual --border_condition --epochs 1200 --batch_size 12 --load $SI_1000e_res --lr 0.00001
# python3 neural_lam/train_model.py --model FM --diffusion_model song_unet $RUN_NAME  --n_workers 16 --pred_residual --border_condition --epochs 1200 --batch_size 12 --load $FM_1000e --lr 0.00001
# python3 neural_lam/train_model.py --model CRPS $RUN_NAME  --n_workers 16 --pred_residual --border_condition --epochs 600 --batch_size 6 --noise_dim 64 # NOTE: Need lower batch size for CRPS as we are training with 2 predictions per sample
# python3 neural_lam/train_model.py --model CRPS $RUN_NAME  --n_workers 16 --pred_residual --border_condition --epochs 1600 --batch_size 1 --load $CRPS_AR_4_res_1504e --lr 0.0001 --ar_steps 4 # NOTE: Need lower batch size for CRPS as we are training with 2 predictions per sample
# python3 neural_lam/train_model.py --model CRPS $RUN_NAME  --n_workers 16 --pred_residual --border_condition --epochs 1800 --batch_size 1 --load $CRPS_AR_4_res_1600e --lr 0.0001 --ar_steps 7 # NOTE: Need lower batch size for CRPS as we are training with 2 predictions per sample

# Test
# python3 neural_lam/train_model.py $MODEL $DIFFUSION_MODEL $RUN_NAME --n_workers 2 --pred_residual --border_condition --vertical_propnets 1 --batch_size 18  --processor_layers 2 --hidden_dim 128 --eval val --n_example_pred 0 --ensemble_size 5 --load $level_3_600e
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm $RUN_NAME --n_workers 2 --pred_residual --border_condition --eval test --n_example_pred 1 --batch_size 8 --ensemble_size 5 --load $EDM_1200e --sampler edm --sampler_steps 10 # --encoder_type residual
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet $RUN_NAME --n_workers 2 --pred_residual --border_condition --eval val --n_example_pred 1 --batch_size 8 --ensemble_size 5 --load $SI_1200e_res --sampler_steps 100
# python3 neural_lam/train_model.py --model tEDM --diffusion_model edm $RUN_NAME --n_workers 2 --pred_residual --border_condition --eval val --n_example_pred 1 --batch_size 8 --ensemble_size 5 --load $tEDM_1000e
# python3 neural_lam/train_model.py --model FM --diffusion_model song_unet $RUN_NAME --n_workers 2 --pred_residual --border_condition --eval val --n_example_pred 1 --batch_size 8 --ensemble_size 25 --load $FM_1200e #  --sampler midpoint
python3 neural_lam/train_model.py --model CRPS $RUN_NAME  --n_workers 2 --pred_residual  --border_condition --batch_size 4 --load $CRPS_AR_4_res_1600e --eval val --n_example_pred 1 --ensemble_size 25

# Trial train
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --graph hierarchical-3 --n_workers 16 --pred_residual --border_condition --vertical_propnets 1 --batch_size 12  --processor_layers 2 --hidden_dim 128
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 16 --pred_residual --border_condition --resample_filter [1,3,3,1] --channel_mult [2, 2, 2, 2] --encoder_type standard --attn_resolutions [134, 68, 34, 18]
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 16 --pred_residual --border_condition --subset_ds
# python3 neural_lam/train_model.py --model tEDM --diffusion_model edm --n_workers 16 --border_condition --subset_ds
# python3 neural_lam/train_model.py --model FM --diffusion_model song_unet --n_workers 16 --border_condition --pred_residual --subset_ds
# python3 neural_lam/train_model.py --model CRPS --n_workers 16 --pred_residual --border_condition --epochs 2 --batch_size 6 --subset_ds --ar_steps 4 # Note max 7 steps for CRPS model, as we train with 2 predictions per sample, distribute ensemble members on different GPUs?

# Trial test
# Batch size 18 for GraphFM
# Batch size (56 max but 32 more stable) for EDM
# python3 neural_lam/train_model.py --model diffusion --diffusion_model graph_fm --graph hierarchical-3 --n_workers 2 --pred_residual --border_condition --vertical_propnets 1 --eval test --n_example_pred 0 --batch_size 18 --processor_layers 2 --hidden_dim 128
# python3 neural_lam/train_model.py --model diffusion --diffusion_model edm --n_workers 2 --pred_residual --border_condition --eval test --n_example_pred 1 --batch_size 1 --subset_ds --ensemble_size 2 --sampler ddpm --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/EDM_1200e-diffusion-6x128-01_14_08-9197/last.ckpt
# python3 neural_lam/train_model.py --model SI --diffusion_model song_unet --n_workers 2 --border_condition --eval test --n_example_pred 1 --batch_size 1 --subset_ds --ensemble_size 5 --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/SI_1200e-SI-6x128-05_07_18-0323/last.ckpt --sampler_steps 100
# python3 neural_lam/train_model.py --model tEDM --diffusion_model edm --n_workers 2 --pred_residual --border_condition --eval val --n_example_pred 1 --batch_size 1 --subset_ds --ensemble_size 2 --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/tEDM_1000e_v_100-tEDM-6x128-06_09_17-3570/last.ckpt
# python3 neural_lam/train_model.py --model FM --diffusion_model song_unet --n_workers 2 --border_condition --pred_residual --subset_ds --eval val --n_example_pred 1 --batch_size 1 --ensemble_size 5 --load /proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/FM_1200e-FM-6x128-06_13_08-1627/last.ckpt
# python3 neural_lam/train_model.py --model CRPS --n_workers 2 --border_condition --pred_residual --eval val --n_example_pred 1 --batch_size 1 --ensemble_size 25 --load $CRPS_AR_4_res_1504e --subset_ds
