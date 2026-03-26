RUN_NAME="--wandb_run_name SI_corrector"
SI_xsmall_25e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_xsmall-SI-6x32-01_24_10-5547/last.ckpt"

python3 neural_lam/train_model.py \
    --model SI \
    --diffusion_model song_unet \
    $RUN_NAME  \
    --n_workers 16 \
    --batch_size 1 \
    --epochs 50 \
    --lr 0.00001 \
    --load $SI_xsmall_25e \
    --eval val \
    --sampler euler \
    --sampler_steps 40 \
    --ensemble_size 5 \
    --conserve_mass_w 0.01 \
    --correction_steps 0 \
    --snr 0.3 \
    --corr_tmin 0.5 \
    --subset_ds \