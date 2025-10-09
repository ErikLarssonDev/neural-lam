RUN_NAME="--wandb_run_name EDM_Samples_80"
EDM_1200e="/proj/berzelius-2022-164/users/x_erila/neural-lam/paper_checkpoints/Diffusion-LAM/last.ckpt" # Best model

python3 neural_lam/train_model.py\
    --model diffusion\
    --diffusion_model edm\
     $RUN_NAME\
     --n_workers 16\
     --pred_residual\
     --border_condition\
     --eval test\
     --n_example_pred 0 \
     --batch_size 8 \
     --ensemble_size 25 \
     --load $EDM_1200e \
     --sampler heun \
    #  --subset_ds \
    #  --save_output \
