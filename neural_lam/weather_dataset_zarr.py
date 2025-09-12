import os
import zarr
import torch
import numpy as np

from neural_lam import constants
import matplotlib.pyplot as plt

class WeatherDataset(torch.utils.data.Dataset):
    """
    For the MEPS dataset:
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    N_x = 268
    N_y = 238
    N_grid = 268x238 = 63784
    d_features = 17 (d_features' = 18)
    d_forcing = 5
    """
    def __init__(self, dataset_name, pred_length=19, split="train", data_path="data"):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        sample_dir_path = os.path.join(data_path, dataset_name,
                "samples", split)

        # Open zarr:s
        self.forecast_zarr = zarr.open(os.path.join(sample_dir_path, "forecasts.zarr"),
                mode="r") # (N_samples, N_subsamples, N_t, N_grid, d_features)
        self.forcing_zarr = zarr.open(os.path.join(sample_dir_path, "forcing.zarr"),
                mode="r") # (N_samples, N_subsamples, N_t-2, N_grid, d_features*3)
        self.batch_static_zarr = zarr.open(os.path.join(sample_dir_path,
            "batch_static.zarr"), mode="r") # (N_samples, N_grid, d_static=1)
        self.num_samples, self.num_subsamples, self.original_sample_length, _, _ =\
            self.forecast_zarr.shape

        self.sample_length = pred_length + 2 # 2 init states
        assert self.sample_length <= self.original_sample_length, (
                "Requesting too long time series samples")
        self.sample_subsample = split == "train"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.sample_subsample:
            subsample_i = np.random.randint(self.num_subsamples)
        else:
            subsample_i = 0
        full_forecast = self.forecast_zarr[idx, subsample_i]

        # Sample what time point to start from
        init_id = np.random.randint(1+self.original_sample_length - self.sample_length)

        # Forecast
        init_conditions = full_forecast[init_id:(init_id + 2)] # (2, N_grid, d_features)
        target = full_forecast[(init_id + 2):(init_id + self.sample_length)]
        # (pred_length, N_grid, d_features)

        # Forcing
        # Note: Forcing is 2 offset from forecast, so forcing i=0 is forecast i=2
        forcing = self.forcing_zarr[idx, subsample_i,
                (init_id):(init_id + self.sample_length - 2)]
        # (pred_length, N_grid, d_forcing)

        # Batch-static features
        batch_static = self.batch_static_zarr[idx]
        # (N_grid, d_static)

        # Convert all to torch tensors
        init_conditions_torch = torch.tensor(init_conditions, dtype=torch.float32)
        target_torch = torch.tensor(target, dtype=torch.float32)
        forcing_torch = torch.tensor(forcing, dtype=torch.float32)
        batch_static_torch = torch.tensor(batch_static, dtype=torch.float32)

        return init_conditions_torch, target_torch, batch_static_torch, forcing_torch
