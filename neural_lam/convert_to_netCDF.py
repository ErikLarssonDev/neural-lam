import os
import argparse
from datetime import datetime, date

import numpy as np
import netCDF4
import pandas as pd
import torch
from tqdm import tqdm

from neural_lam import config
from neural_lam.netCDF_dataset import NetCDFDataset

def destandardize(
    sample,
    pr_stats_path='/mimer/NOBACKUP/groups/mlhighres/projects/detex/HCLIM_EC-Earth3-Veg/standardized_new/mean_std_remapped.pr_EUR-11_EC-Earth3-Veg_historical_r1i1p1f1_HCLIMcom-SMHI_HCLIM43-ALADIN_v1-r1_day_1951-2014_mm_day_noleap.npy', 
    tas_stats_path='/mimer/NOBACKUP/groups/mlhighres/projects/detex/HCLIM_EC-Earth3-Veg/standardized_new/mean_std_remapped.tas_EUR-11_EC-Earth3-Veg_historical_r1i1p1f1_HCLIMcom-SMHI_HCLIM43-ALADIN_v1-r1_day_1951-2014_noleap.npy',
    std_dataset=False
):
    pr_mean, pr_std = np.load(pr_stats_path)
    tas_mean, tas_std = np.load(tas_stats_path)

    if std_dataset:
        return sample * np.array([pr_std, tas_std])
    else:
        return sample * np.array([pr_std, tas_std]) + np.array([pr_mean, tas_mean])

def load_pt_batch(
        pt_data_path, date_str, ensemble_size, size=(400, 550), n_vars=2):
    target = destandardize(
        torch.load(f'{pt_data_path}/target_{date_str}.pt').numpy().reshape(size[0], size[1], n_vars))
    ensemble_mean = destandardize(
        torch.load(f'{pt_data_path}/ens_mean_{date_str}.pt').numpy().reshape(size[0], size[1], n_vars))
    ensemble_std = destandardize(
        torch.load(f'{pt_data_path}/ens_std_{date_str}.pt').numpy().reshape(size[0], size[1], n_vars), std_dataset=True)
    ensemble_members = []
    for ensemble_index in range(ensemble_size):
        ensemble_member = destandardize(
            torch.load(f'{pt_data_path}/member_{ensemble_index}_{date_str}.pt').numpy().reshape(size[0], size[1], n_vars))
        ensemble_members.append(ensemble_member)

    return target, ensemble_mean, ensemble_std, ensemble_members

def write_to_netCDF4(data, netCDF4_dataset, variable_standard_name, var_index, time_idx, input_timestamps):
    for var_name in netCDF4_dataset.variables:
        var = netCDF4_dataset.variables[var_name]
        if var.name == variable_standard_name:
            var[time_idx, :, :] = data[:, :, var_index]
        if var_name == 'time':
            var[time_idx] = get_netCDF4_timestamp(
                input_timestamps, netCDF4_dataset, time_idx)
    netCDF4_dataset.sync()

def initialize_output(dataloader, output_path, start_date, end_date, basename, var_index):
    """Initializes a netCDF file for the output data."""
    dt = datetime.today().strftime("%Y-%m-%dT%H_%M")
    output_file_path = os.path.join(
        output_path, f"{basename}_{start_date}_{end_date}.nc")

    ground_truth_dataset = dataloader.dataset.ground_truth_datasets[var_index]
    ground_truth_timestep = ground_truth_dataset.isel(time=0)
    ground_truth_timestep = ground_truth_timestep.expand_dims("time")
    ground_truth_timestep.time.encoding["unlimited"] = True
    ground_truth_timestep.to_netcdf(output_file_path, mode='w')

    netCDF4_dataset = netCDF4.Dataset(output_file_path, mode='a')
    print(f"Writing netCDF data to {output_file_path}")

    return netCDF4_dataset

def edit_netCDF4_attributes(netCDF4_dataset, ground_truth_variable_name,
                            standard_name, long_name, units):
    """Modifies the netCDF4 attributes."""

    if ground_truth_variable_name != standard_name:
        netCDF4_dataset.renameVariable(ground_truth_variable_name, standard_name)
    netCDF4_dataset.variables[standard_name].long_name = long_name
    netCDF4_dataset.variables[standard_name].standard_name = standard_name

    if "units" in netCDF4_dataset.variables[standard_name].ncattrs():
        netCDF4_dataset.variables[standard_name].setncattr("units", units)

def get_netCDF4_timestamp(input_timestamps, netCDF4_dataset, time_idx):
    """Returns a netCDF4 timestamp from a given datetime timestamp."""
    time_var = netCDF4_dataset.variables["time"]
    time_units = time_var.units  # "days since 1950-01-01T00:00:00+00:00"
    time_calendar = getattr(time_var, "calendar", "standard")  # Use default "standard" if missing
    
    return netCDF4.date2num(
        pd.to_datetime(input_timestamps.data[time_idx]), 
        units=time_units, calendar=time_calendar)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_config',
        type=str,
        default="neural_lam/clim_config_inference.yaml",
        help="Path to data config file (default: neural_lam/clim_config.yaml)",
    )
    parser.add_argument(
        '--pt_data_path',
        type=str,
        help="Path to inference files in .pt format."
    )
    parser.add_argument(
        '--output_path',
        type=str,
        help="Path to output in netCDF format."
    )
    parser.add_argument(
        '--model_name',
        default="SI",
        type=str,
        help="Model name (only used for output name)."
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=25,
        help="Number of ensemble members during evaluation (default: 25)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--var_index",
        type=int,
        default=0,
        help="Output variable index (defalut: 0)",
    )
    parser.add_argument(
        '--variable_name',
        default="pr",
        type=str,
        help="Variable name (pr, tas, etc.)"
    )
    parser.add_argument(
        '--variable_standard_name',
        default="pr",
        type=str,
        help="Variable standard name (pr, tas, etc.)"
    )
    parser.add_argument(
        '--variable_long_name',
        default="pr",
        type=str,
        help="Variable long name (pr, tas, etc.)"
    )
    parser.add_argument(
        '--variable_units',
        default="kg m-2",
        type=str,
        help="Variable units (kg m-2, K, etc.)"
    )

    args = parser.parse_args()
    

    config_loader = config.Config.from_file(args.data_config)
    pt_data_path = args.pt_data_path
    output_path = args.output_path
    model_name = args.model_name
    start_date = config_loader.dataset.validation_start_date
    end_date = config_loader.dataset.validation_end_date
    ensemble_size = args.ensemble_size
    num_workers = args.n_workers
    var_index = args.var_index
    batch_size = 1

    variable_name = args.variable_name
    variable_standard_name = args.variable_standard_name
    variable_long_name = args.variable_long_name
    variable_units = args.variable_units

    inference_dataloader = torch.utils.data.DataLoader(
        NetCDFDataset(
            start_date=start_date,
            end_date=end_date,
            input_path=config_loader.dataset.input_path,
            input_files=config_loader.dataset.input_files,
            ground_truth_path=config_loader.dataset.ground_truth_path,
            ground_truth_files=config_loader.dataset.ground_truth_files,
            ground_truth_stats_path=config_loader.dataset.ground_truth_stats_path,
            levels=config_loader.dataset.levels,
            is_inference_dataset=True,
            normalize_ground_truth=config_loader.dataset.normalize_ground_truth,
            subset_ds=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    input_datasets = inference_dataloader.dataset.input_datasets
    input_timestamps = input_datasets[0]['time']

    netCDF4_dataset_ensemble_mean = initialize_output(
        inference_dataloader, output_path,
        start_date, end_date, f"{variable_name}_ensemble_mean_{model_name}", var_index
    )

    edit_netCDF4_attributes(
        netCDF4_dataset_ensemble_mean,
        variable_name,
        variable_standard_name,
        variable_long_name,
        variable_units)
    
    netCDF4_dataset_ensemble_std = initialize_output(
        inference_dataloader, output_path,
        start_date, end_date, f"{variable_name}_ensemble_std_{model_name}", var_index
    )

    edit_netCDF4_attributes(
        netCDF4_dataset_ensemble_std,
        variable_name,
        variable_standard_name,
        variable_long_name,
        variable_units)
    
    netCDF4_dataset_target = initialize_output(
        inference_dataloader, output_path,
        start_date, end_date, f"{variable_name}_target_{model_name}", var_index
    )

    edit_netCDF4_attributes(
        netCDF4_dataset_target,
        variable_name,
        variable_standard_name,
        variable_long_name,
        variable_units)
    
    netCDF4_dataset_ensemble_members = []
    for ensemble_index in range(ensemble_size):
        netCDF4_dataset_ensemble_member = initialize_output(
            inference_dataloader, output_path,
            start_date, end_date, f"{variable_name}_ensemble_member_{ensemble_index}_{model_name}", var_index
        )
        edit_netCDF4_attributes(
            netCDF4_dataset_ensemble_member,
            variable_name,
            variable_standard_name,
            variable_long_name,
            variable_units)

        netCDF4_dataset_ensemble_members.append(netCDF4_dataset_ensemble_member)

    for time_idx in tqdm(range(len(input_timestamps)), desc=f"Processing samples"):
        current_date = inference_dataloader.dataset.get_current_ordinal_date(time_idx)
        date_str = date.fromordinal(current_date).strftime("%Y-%m-%d")
        target, ensemble_mean, ensemble_std, ensemble_members = load_pt_batch(
            pt_data_path, date_str, ensemble_size
        )
        write_to_netCDF4(ensemble_mean, netCDF4_dataset_ensemble_mean, variable_standard_name, var_index, time_idx, input_timestamps)
        write_to_netCDF4(ensemble_std, netCDF4_dataset_ensemble_std, variable_standard_name, var_index, time_idx, input_timestamps)
        write_to_netCDF4(target, netCDF4_dataset_target, variable_standard_name, var_index, time_idx, input_timestamps)
        for ensemble_member_data, netCDF4_dataset_ensemble_member in zip(ensemble_members, netCDF4_dataset_ensemble_members):
            write_to_netCDF4(ensemble_member_data, netCDF4_dataset_ensemble_member, variable_standard_name, var_index, time_idx, input_timestamps)

    netCDF4_dataset_ensemble_mean.close()
    netCDF4_dataset_ensemble_std.close()
    netCDF4_dataset_target.close()
    for netCDF4_dataset_ensemble_member in netCDF4_dataset_ensemble_members:
        netCDF4_dataset_ensemble_member.close()
    print("Done!")

if __name__ == "__main__":
    main()