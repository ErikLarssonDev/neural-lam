RUN_NAME="--wandb_run_name EFM_Spectra"

python3 neural_lam/train_model.py\
    $RUN_NAME\
    --model graph_efm\
    --n_example_pred 1\
    --graph graph-efm-3\
    --n_workers 16\
    --hidden_dim 128\
    --processor_layers 1\
    --prior_processor_layers 1\
    --encoder_processor_layers 1\
    --output_std \
    --ensemble_size 1\
    --batch_size 4\
    --load /proj/berzelius-2022-164/users/x_erila/neural-lam/paper_checkpoints/Graph-EFM/graph_efm.ckpt\
    --eval test\
