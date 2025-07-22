import os
import netCDF4
import pandas as pd
from datetime import datetime

def initialize_output(dataloader, output_path, start_date, end_date, basename):
    """Initializes a netCDF file for the output data."""
    dt = datetime.today().strftime("%Y-%m-%dT%H_%M")
    output_file_path = os.path.join(
        output_path, f"{basename}_{start_date}_{end_date}.nc")

    ground_truth_dataset = dataloader.dataset.ground_truth_datasets[0]
    ground_truth_timestep = ground_truth_dataset.isel(time=0)
    ground_truth_timestep = ground_truth_timestep.expand_dims("time")
    ground_truth_timestep.time.encoding["unlimited"] = True
    ground_truth_timestep.to_netcdf(output_file_path, mode='w')

    netCDF4_dataset = netCDF4.Dataset(output_file_path, mode='a')

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