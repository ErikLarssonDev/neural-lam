#!/bin/bash
#SBATCH -J netcdf-write
#SBATCH -A NAISS2024-22-955 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-01:00:00

export HDF5_USE_FILE_LOCKING=FALSE

# Define the local repository path
REPO_PATH="/mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam"

pt_data_path="output/250725/exp4/pt"
output_path="output/250725/exp4"
ensemble_size=25
var_index=1
variable_name="tas"
variable_units="K"
n_workers=16

# Run the container
apptainer exec \
  --bind ${REPO_PATH}:/opt/neural-lam \
  --pwd /opt/neural-lam \
  ~/neural-lam.sif python3 neural_lam/convert_to_netCDF.py \
  --data_config neural_lam/clim_config_inference.yaml \
  --model SI \
  --pt_data_path "$pt_data_path" \
  --output_path "$output_path" \
  --n_workers $n_workers \
  --ensemble_size $ensemble_size \
  --var_index $var_index \
  --variable_name "$variable_name" \
  --variable_units "$variable_units" \
  --variable_standard_name "$variable_name" \
  --variable_long_name "$variable_name" \
  