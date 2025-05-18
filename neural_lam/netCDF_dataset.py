import os
import toml
import torch
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Dataset

class NetCDFDataset(Dataset):
    """PyTorch Dataset for loading NetCDF files lazily."""
    def __init__(self, start_date, end_date,
                 input_path, input_files,
                 ground_truth_path, ground_truth_files,
                 ground_truth_stats_path,
                 levels=None, is_inference_dataset=False,
                 normalize_ground_truth=False):
        self.input_paths = [os.path.join(input_path, file) for file in input_files]
        self.ground_truth_paths = [os.path.join(ground_truth_path, file) for file in ground_truth_files]
        self.start_date = start_date
        self.end_date = end_date
        self.levels = levels
        self.normalize_ground_truth = normalize_ground_truth
        if self.normalize_ground_truth:
            self.ground_truth_mean, self.ground_truth_std = load_ground_truth_stats(
                os.path.join(ground_truth_path, ground_truth_stats_path)
            )
        # Open datasets lazily with dask
        self.input_datasets = [self.load_filtered_dataset(fp) for fp in self.input_paths]
        self.ground_truth_datasets = [self.load_filtered_dataset(fp) for fp in self.ground_truth_paths]

        self.is_inference_dataset = is_inference_dataset
        if not self.is_inference_dataset:
            assert len(self.input_datasets[0].time) == len(self.ground_truth_datasets[0].time), "Time dimensions of the input and the ground truth do not match"
    
    def detect_pressure_dim(self, ds):
        """Detects the pressure level dimension name dynamically."""
        possible_names = ["air_pressure", "plev", "level", "isobaric"]
        for name in possible_names:
            if name in ds.dims:
                return name
        return None  # No pressure level found

    def load_filtered_dataset(self, file_path):
        """Selects the time slice and the specified pressure levels."""
        ds = xr.open_dataset(file_path, engine='netcdf4')
        ds = ds.sel(time=slice(self.start_date, self.end_date))

        # Remove non-numeric variables like 'time_bnds'
        numeric_vars = [var for var in ds.data_vars if ds[var].dtype.kind in 'if']
        ds = ds[numeric_vars]
        
        pressure_dim = self.detect_pressure_dim(ds)
        if self.levels and pressure_dim:
            ds = ds.sel({pressure_dim: self.levels})  # Dynamically select the correct dimension

        return ds

    def get_batch(self, datasets, idx):
        """Converts a batch of data into torch tensor of the correct shape."""
        batch_data = []
        for ds in datasets:
            data_array = ds.isel(time=idx).to_array().values
            if data_array.ndim == 4:  # Case where shape is (1, 5, 40, 55)
                split_arrays = np.split(data_array.squeeze(axis=0), data_array.shape[1], axis=0)  # Split into 5 x (1, 40, 55)
                batch_data.extend(split_arrays)
            else:
                batch_data.append(data_array)
        
        batch_tensor = torch.tensor(np.concatenate(batch_data, axis=0), dtype=torch.float32)  # Ensure proper stacking
        return batch_tensor  # Shape: [variables, x, y]
 
    def __len__(self):
        """A required DataLoader method returning the length of the dataset."""
        return len(self.input_datasets[0].time)  # Assuming all files have the same time dimension
  
    def __getitem__(self, idx):
        """Required DataLoader method that serves a batch of data."""
        input_batch_data = self.get_batch(self.input_datasets, idx)
        if self.is_inference_dataset:
            return input_batch_data
        ground_truth_batch_data = self.get_batch(self.ground_truth_datasets, idx)
        if self.normalize_ground_truth:
            ground_truth_batch_data = (ground_truth_batch_data - self.ground_truth_mean) / self.ground_truth_std
        
        # return input_batch_data, ground_truth_batch_data

        states = {
            "LQ": input_batch_data,
            "HQ": ground_truth_batch_data,
        }

        return states

        
def get_dataloader(input_path, input_files, ground_truth_path, ground_truth_files, ground_truth_stats_path,
                   start_date, end_date, levels=None, is_inference_dataset=False, normalize_ground_truth=False,
                   batch_size=4, shuffle=True, pin_memory=False, num_workers=4, prefetch_factor=None):
    """A function that creates the NetCDFDataset object."""

    dataset = NetCDFDataset(start_date, end_date, input_path, input_files, 
                            ground_truth_path, ground_truth_files, ground_truth_stats_path, 
                            levels, is_inference_dataset, normalize_ground_truth)
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
        print(f"Error: Failed to parse the configuration file '{config_path}'. Ensure it's a valid TOML file.")
        exit(1)

def load_ground_truth_stats(path):
    mean, std = np.load(path)
    return mean, std

def compute_quantile(dataset, var='pr', quantile=0.95, dim='time'):
    return dataset[var].quantile(quantile, dim=dim)

def prepare_quantile_maps(evaluation_config, data_config, dataset):
    path = os.path.join(data_config['path'], data_config['quantile_path'])
    variable = data_config['quantile_var_name']
    if not os.path.isfile(path):
        print("Could not open statistics file, computing percentiles...")
        quantile_map = compute_quantile(
            dataset,
            data_config['variable_name'],
            evaluation_config['quantile'] 
        ).values
    else:
        print(f"Reading statistics file {path}")
        ds = xr.open_dataset(path)

        if variable not in ds:
            raise ValueError(f"Variable '{variable}' not found in the dataset. Available: {list(ds.data_vars)}")
        
        quantile_map = ds[variable].values

        return torch.tensor(quantile_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def load_and_normalize_topography(path, variable='orog', normalize='none'):
    """
    Reads topography from NetCDF, returns a torch tensor with shape (1, H, W).

    Parameters:
    - path: str, full path to the NetCDF file
    - variable: str, name of the variable in the NetCDF file (usually 'orog')
    - normalize: str, normalization method ('meanstd', 'minmax', or 'none')

    Returns:
    - topo_tensor: torch.Tensor of shape (1, H, W)
    """
    if not os.path.isfile(path):
        print("No topography is loaded.")
        return
    
    print(f"Loading topography from {path}")
    ds = xr.open_dataset(path)

    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in the dataset. Available: {list(ds.data_vars)}")

    topo = ds[variable].values  # Shape: (H, W) or (1, H, W)

    # Convert to tensor
    topo_tensor = torch.tensor(topo, dtype=torch.float32)

    # Ensure shape is (1, H, W)
    if topo_tensor.ndim == 2:
        topo_tensor = topo_tensor.unsqueeze(0)

    # Normalize if requested
    if normalize == 'minmax':
        topo_tensor = (topo_tensor - topo_tensor.min()) / (topo_tensor.max() - topo_tensor.min())
    elif normalize == 'meanstd':
        topo_tensor = (topo_tensor - topo_tensor.mean()) / topo_tensor.std()
    elif normalize == 'none':
        pass
    else:
        raise ValueError(f"Unsupported normalization method: {normalize}")

    return topo_tensor