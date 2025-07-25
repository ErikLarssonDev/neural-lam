export HDF5_USE_FILE_LOCKING=FALSE

RUN_NAME="--wandb_run_name SI_50e"

# Activate wandb
#apptainer exec ~/neural-lam.sif wandb off

# Saved models
SI_50e="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/SI_50e-SI-6x128-07_10_12-7283/last.ckpt"

# Define the local repository path
REPO_PATH=/mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam

# Run the container
apptainer exec \
  --bind ${REPO_PATH}:/opt/neural-lam \
  --pwd /opt/neural-lam \
  ~/neural-lam.sif python3 neural_lam/train_model.py \
  --model SI \
  --data_config neural_lam/clim_config_inference.yaml \
  --diffusion_model song_unet \
  --output_path output \
  --n_workers 16 \
  --eval test \
  --n_example_pred 99999 \
  --batch_size 1 \
  --ensemble_size 4 \
  --sampler_steps 10 \
  --load $SI_50e \
  --sampler euler_2 \
  --save_output