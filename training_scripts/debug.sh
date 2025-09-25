# CRPS training script

# CHECKPOINT="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_600e_res-CRPS-6x128-07_25_09-6319/last.ckpt"
CHECKPOINT="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_1000e_res-CRPS-6x128-07_26_16-6370/last.ckpt"

python3 neural_lam/train_model.py --model CRPS --wandb_run_name DEBUG  --n_workers 16 --pred_residual --border_condition --epochs 1002 --batch_size 6 --subset_ds --load $CHECKPOINT