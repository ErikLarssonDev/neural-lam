#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE

# Define the local repository path
REPO_PATH="/mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam"

n_workers=16
pt_data_path="/mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam/output/230126/test_SI"
output_path="/mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam/output/230126/testing/"
data_config="neural_lam/clim_config.yaml"
model="SI" # SI/CorrDiff/UNet
ensemble_size=20

var_index=0
variable_name="pr"
variable_units="kg m-2"

# Run the container
apptainer exec \
  --bind ${REPO_PATH}:/opt/neural-lam \
  --pwd /opt/neural-lam \
  ~/neural-lam.sif python3 neural_lam/convert_to_netCDF.py \
  --data_config "$data_config" \
  --model "$model" \
  --pt_data_path "$pt_data_path" \
  --output_path "$output_path" \
  --n_workers $n_workers \
  --ensemble_size $ensemble_size \
  --var_index $var_index \
  --variable_name "$variable_name" \
  --variable_units "$variable_units" \
  --variable_standard_name "$variable_name" \
  --variable_long_name "$variable_name" \
  #--deterministic

var_index=1
variable_name="tas"
variable_units="K" 

apptainer exec \
  --bind ${REPO_PATH}:/opt/neural-lam \
  --pwd /opt/neural-lam \
  ~/neural-lam.sif python3 neural_lam/convert_to_netCDF.py \
  --data_config "$data_config" \
  --model "$model" \
  --pt_data_path "$pt_data_path" \
  --output_path "$output_path" \
  --n_workers $n_workers \
  --ensemble_size $ensemble_size \
  --var_index $var_index \
  --variable_name "$variable_name" \
  --variable_units "$variable_units" \
  --variable_standard_name "$variable_name" \
  --variable_long_name "$variable_name" \
  #--deterministic
  