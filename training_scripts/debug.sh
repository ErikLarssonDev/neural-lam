# CRPS training script
python3 neural_lam/train_model.py --model CRPS --wandb_run_name DEBUG  --n_workers 16 --pred_residual --border_condition --epochs 10 --batch_size 6 --subset_ds