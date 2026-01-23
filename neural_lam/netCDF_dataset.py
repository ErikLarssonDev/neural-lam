import os
import toml
import torch
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class NetCDFDataset(Dataset):
    """PyTorch Dataset for loading NetCDF files lazily."""

    def __init__(self, start_date, end_date,
                 input_path, input_files,
                 ground_truth_path, ground_truth_files,
                 ground_truth_stats_path,
                 levels=None, is_inference_dataset=False,
                 normalize_ground_truth=False,
                 subset_ds=False,
                 upscale_inputs=False,
                 static_fields_files=None,
                 interpolation_mode='bicubic',
                 provide_coordinates=False,
                 provide_day_of_year=False):
        print(f"input_files: {input_files}")
        self.input_paths = [os.path.join(input_path, file)
                            for file in input_files]
        self.ground_truth_paths = [os.path.join(
            ground_truth_path, file) for file in ground_truth_files]
        if static_fields_files and any(file.strip() for file in static_fields_files):
            self.static_fields_paths = [os.path.join(
                input_path, file) for file in static_fields_files if file.strip()]
        else:
            self.static_fields_paths = None
        self.start_date = start_date
        self.end_date = end_date
        self.levels = levels
        self.normalize_ground_truth = normalize_ground_truth
        if self.normalize_ground_truth:
            self.ground_truth_mean, self.ground_truth_std = load_ground_truth_stats(
                os.path.join(ground_truth_path, ground_truth_stats_path)
            )
        # Open datasets lazily with dask
        self.input_datasets = [self.load_filtered_dataset(
            fp) for fp in self.input_paths]
        self.ground_truth_datasets = [self.load_filtered_dataset(
            fp) for fp in self.ground_truth_paths]

        self.is_inference_dataset = is_inference_dataset
        if not self.is_inference_dataset:
            assert len(self.input_datasets[0].time) == len(
                self.ground_truth_datasets[0].time), "Time dimensions of the input and the ground truth do not match"

        self.subset_ds = subset_ds
        self.upscale_inputs = upscale_inputs
        self.interpolation_mode = interpolation_mode
        ground_truth_size = self.get_batch(
            self.ground_truth_datasets, 0).shape[1:]
        self.ground_truth_size = ground_truth_size

        input_size = self.get_batch(self.input_datasets, 0).shape
        self.input_size = input_size

        if self.static_fields_paths:
            static_fields = [self.load_static_field(
                fp) for fp in self.static_fields_paths]
            self.static_data = torch.cat(
                static_fields, dim=0)  # [C_static, H, W]
            print(f"Reading static fields from {static_fields_files}...")
        else:
            self.static_data = None
            print(f"No static fields are loaded.")

        self.coord_grid = None
        if provide_coordinates:
            print(f"Providing ground truth coordinate grid...")
            self.coord_grid = self.generate_lat_lon_grids()
        else:
            print(f"Ground truth coordinate grid is not provided.")

        self.time_array = None
        if provide_day_of_year:
            print(f"Providing day of year encodings...")
            self.time_array = self.get_time_array()
        else:
            print(f"Time encodings are not provided.")

    def detect_pressure_dim(self, ds):
        """Detects the pressure level dimension name dynamically."""
        possible_names = ["air_pressure", "plev", "level", "isobaric"]
        for name in possible_names:
            if name in ds.dims:
                return name
        return None  # No pressure level found

    def load_static_field(self, file_path):
        ds = xr.open_dataset(file_path, engine='netcdf4')

        # Keep only numeric variables
        numeric_vars = [
            var for var in ds.data_vars if ds[var].dtype.kind in 'if']
        ds = ds[numeric_vars]

        # Convert to array: shape will be [C, H, W]
        # .squeeze(0)  # shape: [num_vars, H, W]
        static_data = ds.to_array().values
        # print(f"Loading static data...")
        # print(f"Static data shape: {static_data.shape}\n")

        # Validate shape
        if self.ground_truth_size is not None:
            # print(self.ground_truth_size)
            # print(static_data.shape)
            assert static_data.shape[1:] == self.ground_truth_size, \
                f"Ground truth size {self.ground_truth_size} does not match static field shape {static_data.shape[1:]}"

        return torch.tensor(static_data, dtype=torch.float32)

    def generate_lat_lon_grids(self):
        """Generates 2D lat/lon grids from the ground truth dataset coordinates."""
        ds = self.ground_truth_datasets[0]
        lat = ds.coords['lat'].values
        lon = ds.coords['lon'].values

        # Create 2D meshgrid
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        lat_norm = (lat_grid - lat_grid.min()) / \
            (lat_grid.max() - lat_grid.min())
        lon_norm = (lon_grid - lon_grid.min()) / \
            (lon_grid.max() - lon_grid.min())

        # Convert to torch tensor with shape [1, H, W]
        lat_tensor = torch.tensor(
            lat_norm[np.newaxis, :, :], dtype=torch.float32)
        lon_tensor = torch.tensor(
            lon_norm[np.newaxis, :, :], dtype=torch.float32)

        return torch.cat([lat_tensor, lon_tensor], dim=0)  # Shape: [2, H, W]

    def get_time_array(self):
        return self.input_datasets[0].coords['time']

    def get_day_of_year_encodings(self, idx):
        day_of_year = pd.to_datetime(self.time_array.values[idx]).dayofyear
        return (
            np.sin(2 * np.pi * day_of_year / 365.0),
            np.cos(2 * np.pi * day_of_year / 365.0)
        )

    def get_day_of_year_tensor(self, idx, height, width):
        doy_sin, doy_cos = self.get_day_of_year_encodings(idx)
        sin_tensor = torch.full(
            (1, height, width), fill_value=doy_sin, dtype=torch.float32)
        cos_tensor = torch.full(
            (1, height, width), fill_value=doy_cos, dtype=torch.float32)

        return torch.cat([sin_tensor, cos_tensor], dim=0)  # Shape: [2, H, W]

    def load_filtered_dataset(self, file_path):
        """Selects the time slice and the specified pressure levels."""
        ds = xr.open_dataset(file_path, engine='netcdf4')
        ds = ds.sel(time=slice(self.start_date, self.end_date))

        # Remove non-numeric variables like 'time_bnds'
        numeric_vars = [
            var for var in ds.data_vars if ds[var].dtype.kind in 'if']
        ds = ds[numeric_vars]

        pressure_dim = self.detect_pressure_dim(ds)
        if self.levels and pressure_dim:
            # Dynamically select the correct dimension
            ds = ds.sel({pressure_dim: self.levels})

        return ds

    def get_batch(self, datasets, idx):
        """Converts a batch of data into torch tensor of the correct shape."""
        batch_data = []
        for ds in datasets:
            data_array = ds.isel(time=idx).to_array().values
            if data_array.ndim == 4:  # Case where shape is (1, 5, 40, 55)
                split_arrays = np.split(data_array.squeeze(
                    # Split into 5 x (1, 40, 55)
                    axis=0), data_array.shape[1], axis=0)
                batch_data.extend(split_arrays)
            else:
                batch_data.append(data_array)

        batch_tensor = torch.tensor(np.concatenate(
            batch_data, axis=0), dtype=torch.float32)  # Ensure proper stacking
        return batch_tensor  # Shape: [variables, x, y]

    def __len__(self):
        """A required DataLoader method returning the length of the dataset."""
        if self.subset_ds:
            return 2
        else:
            # Assuming all files have the same time dimension
            return len(self.input_datasets[0].time)

    def __getitem__(self, idx):
        """Required DataLoader method that serves a batch of data."""
        input_batch_data = self.get_batch(self.input_datasets, idx)
        if self.upscale_inputs:
            # Add temporary batch dimension: [1, C, H, W]
            input_batch_data = input_batch_data.unsqueeze(0)
            input_batch_data = F.interpolate(
                input_batch_data, size=self.ground_truth_size, mode=self.interpolation_mode, align_corners=False)
            # Remove temporary batch dimension: [C, H_out, W_out]
            input_batch_data = input_batch_data.squeeze(0)
            # Concatenate the input data with static fields
            input_components = [input_batch_data]
            if self.static_data is not None:
                input_components.append(self.static_data)
            if self.coord_grid is not None:
                input_components.append(self.coord_grid)
            if self.time_array is not None:
                time_tensor = self.get_day_of_year_tensor(
                    idx, self.ground_truth_size[0], self.ground_truth_size[1])
                input_components.append(time_tensor)
            # print(f"Input data shape: {input_batch_data.shape}")
            # print(f"Static data shape: {self.static_data.shape}")
            # print(f"Coordinate grid shape: {self.coord_grid.shape}")
            # print(f"Time tensor shape: {time_tensor.shape}")
            input_batch_data = torch.cat(input_components, dim=0)
        if self.is_inference_dataset:
            return input_batch_data
        ground_truth_batch_data = self.get_batch(
            self.ground_truth_datasets, idx)
        if self.normalize_ground_truth and self.normalize_ground_truth:
            self.ground_truth_mean, self.ground_truth_std = self.ground_truth_stats
            ground_truth_batch_data = (
                ground_truth_batch_data - self.ground_truth_mean) / self.ground_truth_std

        # return input_batch_data, ground_truth_batch_data
        states = {
            # "LQ": F.interpolate(input_batch_data.unsqueeze(0), size=(400, 550), mode='bilinear', align_corners=False).squeeze(0), # [B, 23, 40, 55] -> [B, 23, 400, 550], NOTE: We need the low-res input on the same grid as the high-res output
            "LQ": input_batch_data,  # [B, n_input, 400, 550]
            "HQ": ground_truth_batch_data,  # [B, 2, 400, 550]
        }

        return states


def get_dataloader(input_path, input_files, ground_truth_path, ground_truth_files, ground_truth_stats_path,
                   start_date, end_date, levels=None, is_inference_dataset=False, normalize_ground_truth=False,
                   batch_size=4, shuffle=True, pin_memory=False, num_workers=4, prefetch_factor=None, upscale_inputs=False,
                   static_fields_files=None, interpolation_mode='bicubic', provide_coordinates=False, provide_day_of_year=False):
    """A function that creates the NetCDFDataset object."""

    dataset = NetCDFDataset(start_date, end_date, input_path, input_files,
                            ground_truth_path, ground_truth_files, ground_truth_stats_path,
                            levels, is_inference_dataset, normalize_ground_truth, upscale_inputs,
                            static_fields_files, interpolation_mode, provide_coordinates, provide_day_of_year)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, prefetch_factor=prefetch_factor)


def parse_config(config_keyword="input", config_path="config.toml"):
    """Parses and returns TOML input config.

    Example for input_config fields:

    path = input_config["path"]
    files = input_config["files"]
    levels = input_config["levels"]
    """
    try:
        with open(config_path, 'r') as config_file:
            config = toml.load(config_file)

        return config[config_keyword]
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        exit(1)
    except toml.TomlDecodeError:
        print(
            f"Error: Failed to parse the configuration file '{config_path}'. Ensure it's a valid TOML file.")
        exit(1)


def load_ground_truth_stats(path):
    mean, std = np.load(path)
    return mean, std
