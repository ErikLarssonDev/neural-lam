RUN_NAME="--wandb_run_name CRPS_Spectra"
CRPS_res_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_res_1200e-CRPS-6x128-07_27_10-5686/last.ckpt"
CRPS_AR_4_res_1600e="/proj/berzelius-2022-164/users/x_erila/neural-lam/saved_models/CRPS_AR_4_res_1600e-CRPS-6x128-08_22_17-9414/last.ckpt"

python3 neural_lam/train_model.py \
        --model CRPS $RUN_NAME\
        --n_workers 16\
        --pred_residual\
        --border_condition\
        --batch_size 4\
        --load $CRPS_AR_4_res_1600e\
        --eval test\
        --n_example_pred 1\
        --ensemble_size 1\