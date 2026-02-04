import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import silu
import matplotlib.pyplot as plt
import numpy as np
import wandb
import copy
import math
import time
import datetime
import os

from neural_lam.models.ar_model import ARModel
from neural_lam import constants, metrics, utils, vis

from neural_lam.models.edm_networks_2 import EDMPrecond


class Diffusion(ARModel):
    """
    An EDM diffusion model for downscaling
    """

    def __init__(self, args):
        super().__init__(args)
        if args.model == 'CorrDiff':
            # Diffusion noise + mean conditioning
            extra_dims = len(self.config_loader.dataset.downscaling_idx)*2
        else:
            # Diffusion noise
            extra_dims = len(self.config_loader.dataset.downscaling_idx)
        self.grid_dim = (
            self.config_loader.num_data_vars()
            + self.config_loader.dataset.num_static_features
            + self.config_loader.dataset.num_forcing_features
            + extra_dims
        )

        # Some dimensionalities that can be useful to have stored
        self.ensemble_size = args.ensemble_size
        self.sigma_min = args.sigma_min
        self.sigma_max = 88
        self.sigma_data = 1
        self.rho = 7
        self.sampler = args.sampler
        self.output_path = args.output_path
        self.save_output = args.save_output
        self.save_output_wandb = args.save_output_wandb
        self.sampler_steps = args.sampler_steps
        self.save_steps = args.save_steps
        self.data_std = 1  # We don't rescale the data when plotting, so we can use 1 here
        self.data_mean = 0  # We don't rescale the data when plotting, so we can use 0 here

        print(
            f"Diffusion model {args.diffusion_model} with grid_dim {self.grid_dim} and grid_output_dim {self.grid_output_dim}")

        if args.diffusion_model == 'edm':
            self.model = EDMPrecond(img_resolution=torch.as_tensor(self.config_loader.dataset.FULL_GRID_SHAPE),
                                    in_channels=self.grid_dim,  # We have noise and LQ as input
                                    out_channels=self.grid_output_dim,
                                    model_type='SongUNet',
                                    embedding_type=args.noise_embedding,
                                    sigma_data=self.sigma_data,
                                    sigma_min=self.sigma_min,
                                    sigma_max=self.sigma_max,
                                    resample_filter=args.resample_filter,
                                    channel_mult=args.channel_mult,
                                    encoder_type=args.encoder_type,
                                    attn_resolutions=args.attn_resolutions,
                                    )
        else:
            raise ValueError(
                f"Diffusion model {args.diffusion_model} not recognized")

        # Whether to predict the residual instead of the next state
        self.pred_residual = args.pred_residual
        self.diffusion_model = args.diffusion_model

        self.test_metrics = {
            "ens_mae": [],
            "ens_mse": [],
            "crps_ens": [],
            "spread_squared": [],
        }

        self.val_metrics.update(
            {
                "ens_mae": [],
                "ens_mse": [],
                "crps_ens": [],
                "spread_squared": [],
            }
        )

    def predict_step(self, LQ):
        """
        Downscaling weather state
        LQ: (B, d_f, X, Y)

        Returns:
        next_state: (B, N_grid, d_state)
        pred_std: None
        """

        # torch.cat((prev_state, prev_prev_state, forcing), dim=-1) # (B, N_grid, d_input)
        input_grid = LQ
        latents = torch.randn([LQ.shape[0], self.grid_output_dim, *
                              # (B, N_grid, d_state)
                               constants.FULL_GRID_SHAPE], device=LQ.device)

        # Run through sampler
        if self.sampler == "heun":
            next_state, diff_states = self.heun_sampler(
                latents=latents, class_labels=input_grid, sigma_min=self.sigma_min*1.5, num_steps=self.sampler_steps)
        elif self.sampler == "edm":
            next_state, diff_states = self.edm_sampler(
                latents=latents, class_labels=input_grid, sigma_min=self.sigma_min*1.5, num_steps=self.sampler_steps)
        elif self.sampler == "ddpm":
            next_state, diff_states = self.ddpm_sampler(
                latents=latents, class_labels=input_grid, sigma_min=self.sigma_min*1.5, num_steps=self.sampler_steps)

        if self.save_steps:
            # Save the diffusion steps
            self.plot_diffusion_steps(diff_states, LQ)

        # Add residual if needed
        if self.pred_residual and self.args.model != 'CorrDiff':
            next_state = next_state + \
                LQ[:, self.config_loader.dataset.downscaling_idx, ...]
            #     print(f"Pred residual is not supported as we don't have the residual/normalization in the training data")
            # next_state = (next_state * self.step_diff_std[constants.USED_PARAMS].view(1, len(constants.USED_PARAMS), 1, 1)) + self.step_diff_mean[constants.USED_PARAMS].view(1, len(constants.USED_PARAMS), 1, 1) # Unormalize residual
            # next_state = LQ + next_state

        return next_state.permute(0, 2, 3, 1).flatten(1, 2), None

    def predict_step_train(self, LQ, HQ):
        """
        Downscaling weather state
        LQ: (B, d_f, X, Y)
        HQ: (B, d_f, X, Y)

        Returns:
        downscaled_state: (B, N_grid, d_state)
        pred_std: None 
        """
        # Sample from F inverse
        rnd_uniform = torch.rand([HQ.shape[0], 1, 1, 1], device=HQ.device)
        rho_inv = 1 / self.rho
        sigma_max_rho = self.sigma_max ** rho_inv
        sigma_min_rho = self.sigma_min ** rho_inv
        sigma = (sigma_max_rho + rnd_uniform *
                 (sigma_min_rho - sigma_max_rho)) ** self.rho

        # torch.cat((prev_state, prev_prev_state, forcing), dim=-1)
        input_grid = LQ
        y = HQ

        # TODO: Pred residual is not supported as we don't have the residual/normalization in the training data
        # Make y residual if needed
        if self.pred_residual and self.args.model != 'CorrDiff':
            y = HQ - LQ[:, self.config_loader.dataset.downscaling_idx, ...]
        # if self.pred_residual:
        #     raise NotImplementedError(
        #         "Pred residual is not supported as we don't have the residual/normalization in the training data")
        #     y = HQ - LQ
        #     y = (y - self.step_diff_mean[constants.USED_PARAMS].view(1, len(constants.USED_PARAMS), 1, 1)) / self.step_diff_std[constants.USED_PARAMS].view(1, len(constants.USED_PARAMS), 1, 1) # Normalize residual

        n = torch.randn_like(y) * sigma
        noisy_input = y+n

        # Shape (B, d_state, N_x, N_y)
        next_state = self.forward(noisy_input, sigma, input_grid)

        # Add residual if needed
        if self.pred_residual and self.args.model != 'CorrDiff':
            next_state = next_state + \
                LQ[:, self.config_loader.dataset.downscaling_idx, ...]

        weight = (sigma ** 2 + self.sigma_data ** 2) / \
            (sigma * self.sigma_data) ** 2

        return next_state.permute(0, 2, 3, 1).flatten(1, 2), None, weight

    def unroll_prediction(self, LQ):
        """
        Roll out prediction taking multiple autoregressive steps with model
        LQ: (B, d_f, X, Y)

        Returns:
        prediction: (B, pred_steps, d_f, X, Y)
        """

        prediction_list = []
        pred_std_list = []
        pred_steps = 1

        for i in range(pred_steps):
            pred_state, pred_std = self.predict_step(LQ)

            prediction_list.append(pred_state)
            if self.output_std:
                pred_std_list.append(pred_std)

        prediction = torch.stack(
            prediction_list, dim=1
        )
        if self.output_std:
            # Using the same weights for all variables
            pred_std = torch.tensor(1, device=LQ.device)
        else:
            pred_std = self.per_var_std

        return prediction, pred_std

    def unroll_prediction_train(self, LQ, HQ):
        """
        Roll out prediction taking multiple autoregressive steps with model
        LQ: (B, d_f, X, Y)
        HQ: (B, d_f, X, Y)

        Returns:
        prediction: (B, pred_steps, N_grid, d_f)
        pred_std: (B, pred_steps, d_f)
        weight: (B, pred_steps) ?
        """
        prediction_list = []
        pred_std_list = []
        pred_steps = 1
        weight_list = []

        for i in range(pred_steps):
            pred_state, pred_std, weight = self.predict_step_train(LQ, HQ)

            prediction_list.append(pred_state)
            weight_list.append(weight)
            if self.output_std:
                pred_std_list.append(pred_std)

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        weight = torch.stack(
            weight_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        if self.output_std:
            # Using the same weights for all variables
            pred_std = torch.tensor(1, device=LQ.device)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std, weight

    def common_step_train(self, batch):
        """
        Predict on single batch
        batch consists of:
        LQ: (B, grid, d_f)
        HQ: (B, d_f, X, Y, d_f)

        Returns:
        prediction: (B, pred_steps, N_grid, d_f)
        target: (B, pred_steps, N_grid, d_f)
        pred_std: (B, pred_steps, d_f)
        weight: (B, pred_steps) ?
        """
        LQ, HQ = batch["LQ"], batch["HQ"]
        prediction, pred_std, weight = self.unroll_prediction_train(
            LQ, HQ
        )

        target = HQ.permute(0, 2, 3, 1).contiguous().flatten(
            1, 2).unsqueeze(1)  # (B, pred_steps, N_grid, d_f)

        return prediction, target, pred_std, weight

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, pred_std, weight = self.common_step_train(batch)

        # Compute loss
        batch_loss = torch.mean(
            self.loss(prediction, target, pred_std, weight=weight)
        )  # mean over unrolled times and batch

        batch_mse = torch.mean(
            metrics.mse(
                prediction, target, pred_std
            )
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss, "train_mse": batch_mse}
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return batch_loss

    def sample_trajectories(
        self,
        LQ,
        num_traj,
    ):
        """
        LQ: (B, d_f, X, Y)
        num_traj: S, number of trajectories to sample

        Returns
        traj_means: (B, S, d_f, X, Y)
        traj_stds: (d_f)
        """
        unroll_func = self.unroll_prediction

        traj_list = []
        for i in range(num_traj):
            traj = unroll_func(
                LQ,
            )

            traj_list.append(traj)

        # List of tuples, each containing
        # mean: (B, d_f, X, Y) and
        # std: (d_f,)

        traj_means = torch.stack(
            [pred_pair[0] for pred_pair in traj_list], dim=1
        )
        if self.output_std:
            traj_stds = torch.stack(
                [pred_pair[1] for pred_pair in traj_list], dim=1
            )
        else:
            traj_stds = self.per_var_std

        return traj_means, traj_stds

    # Not the best function, just used for quick debugging
    def plot_diffusion_steps(self, diff_states, LQ):
        # Save slices to wandb
        var_names = [self.config_loader.dataset.var_names[i]
                     for i in self.config_loader.dataset.downscaling_idx]

        LQ = LQ.permute(0, 2, 3, 1).contiguous().flatten(1, 2).unsqueeze(1)

        for i, diff_state in enumerate(diff_states):
            diff_state = diff_state.permute(0, 2, 3, 1).contiguous().flatten(
                1, 2).unsqueeze(1)  # (B, N_grid, d_f)
            for var_idx, var_name in enumerate(var_names):
                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(13, 7),
                )
                fig.suptitle(
                    f"Diffusion step {i+1} of {len(diff_states)}", fontsize=16)
                axes[0].set_title("Latent", size=15)
                vis.plot_on_axis(axes[0], diff_state[0, ..., var_idx])
                axes[1].set_title("LQ", size=15)
                vis.plot_on_axis(
                    axes[1], LQ[0, ..., self.config_loader.dataset.downscaling_idx[var_idx]])

                os.makedirs(
                    f"output/diffusion_steps/{var_name}", exist_ok=True)
                plt.savefig(
                    f"output/diffusion_steps/{var_name}/step_{i+1}.png")
                plt.close(fig)

    def plot_examples(self, batch, n_examples, prediction=None):
        """
        Plot ensemble forecast + mean and std
        """
        LQ, HQ = batch["LQ"], batch["HQ"]
        if prediction is None:
            print(f"Sampling new trajectories for plotting!")
            trajectories, _ = self.sample_trajectories(
                LQ,
                self.ensemble_size,
            )
        else:
            trajectories = prediction
        # (B, S, pred_steps, num_grid_nodes, d_f)

        initial_states = LQ.permute(0, 2, 3, 1).contiguous().flatten(
            1, 2).unsqueeze(1)  # (B, 1, num_grid_nodes, d_f)
        target_states = HQ.permute(
            0, 2, 3, 1).contiguous().flatten(1, 2).unsqueeze(1)

        # Rescale to original data scale
        # * self.data_std[constants.USED_PARAMS] + self.data_mean[constants.USED_PARAMS]
        traj_rescaled = trajectories
        # * self.data_std[constants.USED_PARAMS] + self.data_mean[constants.USED_PARAMS]
        target_rescaled = target_states
        # * self.data_std[constants.USED_PARAMS] + self.data_mean[constants.USED_PARAMS]
        initial_states_rescaled = initial_states

        # Compute mean and std of ensemble
        ens_mean = torch.mean(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        # Iterate over the examples
        for init_slice, traj_slice, target_slice, ens_mean_slice, ens_std_slice in zip(
            initial_states_rescaled[:n_examples],
            traj_rescaled[:n_examples],
            target_rescaled[:n_examples],
            ens_mean[:n_examples],
            ens_std[:n_examples],
        ):
            # traj_slice is (S, pred_steps, num_grid_nodes, d_f)
            # others are (pred_steps, num_grid_nodes, d_f)

            self.plotted_examples += 1  # Increment already here

            # Save slices to wandb
            output_dir = f"output/{wandb.run.name}"
            os.makedirs(output_dir, exist_ok=True)

            # TODO: Check that the saving is correct, we want to save one sample and not the entire batch
            # Save predictions to the output folder
            if self.save_output:
                torch.save(
                    ens_mean_slice[0], f"{output_dir}/example_ens_mean_{self.plotted_examples}.pt")
                torch.save(
                    ens_std_slice[0], f"{output_dir}/example_ens_std_{self.plotted_examples}.pt")
                torch.save(
                    traj_slice[0], f"{output_dir}/example_ens_members_{self.plotted_examples}.pt")
                torch.save(
                    target_slice[0], f"{output_dir}/example_target_{self.plotted_examples}.pt")

                # Save files to wandb
                if self.save_output_wandb:
                    wandb.save(
                        f"output/example_ens_mean_{self.plotted_examples}.pt")
                    wandb.save(
                        f"output/example_ens_std_{self.plotted_examples}.pt")
                    wandb.save(
                        f"output/example_ens_members_{self.plotted_examples}.pt")
                    wandb.save(
                        f"output/example_target_{self.plotted_examples}.pt")

            # Note: min and max values can not be in ensemble mean
            var_vmin = (
                torch.minimum(
                    traj_slice.flatten(0, 2).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                    # init_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    traj_slice.flatten(0, 2).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                    # init_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Get only the variables we want to downscale and plot
            var_names = [self.config_loader.dataset.var_names[i]
                         for i in self.config_loader.dataset.downscaling_idx]
            var_units = [self.config_loader.dataset.var_units[i]
                         for i in self.config_loader.dataset.downscaling_idx]
            init_slice = init_slice[:, :,
                                    self.config_loader.dataset.downscaling_idx]

            # Iterate over prediction horizon time steps
            for t_i, (samples_t, init_t, target_t, ens_mean_t, ens_std_t) in enumerate(
                zip(
                    # (n_ens, pred_steps, num_grid_nodes, d_f)
                    traj_slice.transpose(0, 1),
                    # (pred_steps, S, num_grid_nodes, d_f)
                    init_slice,
                    target_slice,
                    ens_mean_slice,
                    ens_std_slice,
                ),
                start=1,
            ):
                time_title_part = f"t={t_i} ({self.step_length*t_i} h)"
                # Create one figure per variable at this time step

                var_figs = [
                    vis.plot_ensemble_prediction(
                        init_t[:, var_i],
                        samples_t[:, :, var_i],
                        target_t[:, var_i],
                        ens_mean_t[:, var_i],
                        ens_std_t[:, var_i],
                        title=f"{var_name} ({var_unit}), {time_title_part}",
                        # vrange=var_vrange,
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            var_names,
                            var_units,
                            var_vranges,
                        )
                    )
                ]

                example_title = f"example_{self.plotted_examples}"
                wandb.log(
                    {
                        f"{var_name}_{example_title}": wandb.Image(fig)
                        for var_name, fig in zip(
                            var_names, var_figs
                        )
                    }
                )
                plt.close(
                    "all"
                )  # Close all figs for this time step, saves memory

    def ensemble_common_step(self, batch):
        """
        Perform ensemble forecast and compute basic metrics.
        Common step done during both evaluation and testing

        batch: tuple of tensors, batch to perform ensemble forecast on

        Returns:
        trajectories: (B, n_ens, d_f, X, Y)
        traj_stds: (B, n_ens, d_f, X, Y)
        target_states: (B, d_f, X, Y)
        spread_squared_batch: (B, d_f)
        ens_mse_batch: (B, d_f)
        """
        # Compute and store metrics for ensemble forecast
        LQ, HQ, date_ordinal = batch["LQ"], batch["HQ"], batch["date"]

        target_states = HQ.permute(0, 2, 3, 1).contiguous().flatten(
            1, 2).unsqueeze(1)  # (B, pred_steps, d_f)

        trajectories, traj_stds = self.sample_trajectories(
            LQ,
            self.ensemble_size,
        )

        spread_squared_batch = metrics.spread_squared(
            trajectories,
            target_states,
            traj_stds,
            sum_vars=False,
        )

        ens_mean = torch.mean(
            trajectories, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        ens_mse_batch = metrics.mse(
            ens_mean,
            target_states,
            None,
            sum_vars=False,
        )

        return (
            trajectories,
            traj_stds,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
            date_ordinal
        )

    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, pred_std, weight = self.common_step_train(
            batch)

        loss = self.loss(
            prediction, target, pred_std, weight=weight  # mask=self.interior_mask_bool
        )
        mean_loss = torch.mean(loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_mean_loss": mean_loss
        }
        self.log_dict(
            val_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )

        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            target,
            pred_std,
            # mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)

        # We only get probabilistic metrics for the first batch to save time
        if batch_idx == 0:
            # Get Probabilistic Metrics
            (
                trajectories,
                traj_stds,
                target_states,
                spread_squared_batch,
                ens_mse_batch,
                date_ordinal
            ) = self.ensemble_common_step(batch)

            self.val_metrics["spread_squared"].append(spread_squared_batch)
            self.val_metrics["ens_mse"].append(ens_mse_batch)

            # Compute additional ensemble metrics
            ens_mean = torch.mean(
                trajectories, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
            ens_std = torch.std(trajectories, dim=1)
            # (B, pred_steps, num_grid_nodes, d_f)

            # Compute MAE for ensemble mean + ensemble CRPS
            ens_maes = metrics.mae(
                ens_mean,
                target_states,
                ens_std,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.val_metrics["ens_mae"].append(ens_maes)
            crps_batch = metrics.crps_ens(
                trajectories,
                target_states,
                None,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.val_metrics["crps_ens"].append(crps_batch)

            # Plot example predictions (on rank 0 only)
            if self.trainer.is_global_zero:
                self.plot_examples(
                    batch,
                    1,
                    prediction=trajectories,
                    # Only plot 1 step ahead during validation
                    # lead_times_to_plot=constants.VAL_PLOT_STEPS,
                )
                # Decrease counter, we don't want to increase it in the validation step
                self.plotted_examples -= 1

    def log_spsk_ratio(self, metric_vals, prefix):
        """
        Compute the mean spread-skill ratio for logging in evaluation

        metric_vals: dict with all metric values
        prefix: string, prefix to use for logging
        """
        # Compute mean spsk_ratio
        spread_squared_tensor = self.all_gather_cat(
            torch.cat(metric_vals["spread_squared"], dim=0)
        )  # (N_eval, pred_steps, d_f)
        ens_mse_tensor = self.all_gather_cat(
            torch.cat(metric_vals["ens_mse"], dim=0)
        )  # (N_eval, pred_steps, d_f)

        # Do not log during sanity check?
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            # Note that spsk_ratio is scale-invariant, so do not have to rescale
            spread = torch.sqrt(torch.mean(spread_squared_tensor, dim=0))
            skill = torch.sqrt(torch.mean(ens_mse_tensor, dim=0))
            # Both (pred_steps, d_f)

            # Include finite sample correction
            spsk_ratios = np.sqrt(
                (self.ensemble_size + 1) / self.ensemble_size
            ) * (
                spread / skill
            )  # (pred_steps, d_f)
            log_dict = self.create_metric_log_dict(
                spsk_ratios, prefix, "spsk_ratio"
            )

            log_dict[f"{prefix}_mean_spsk_ratio"] = torch.mean(
                spsk_ratios
            )  # log mean
            wandb.log(log_dict)

    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        Include metrics computation for ensemble mean prediction
        """
        # super().test_step(batch, batch_idx) # TODO: Remove, takes a lot of time!

        (
            trajectories,
            traj_stds,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
            date_ordinal
        ) = self.ensemble_common_step(batch)
        self.test_metrics["spread_squared"].append(spread_squared_batch)
        self.test_metrics["ens_mse"].append(ens_mse_batch)

        # Compute additional ensemble metrics
        ens_mean = torch.mean(
            trajectories, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(trajectories, dim=1)
        # (B, pred_steps, num_grid_nodes, d_f)

        # Compute MAE for ensemble mean + ensemble CRPS
        ens_maes = metrics.mae(
            ens_mean,
            target_states,
            ens_std,
            # mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["ens_mae"].append(ens_maes)

        crps_batch = metrics.crps_ens(  # CRPS want the shape [B, n_ens, pred_steps, dim, d_f]
            trajectories,
            target_states,
            traj_stds,  # d_f
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["crps_ens"].append(crps_batch)

        if self.save_output:
            dates = []
            for date in date_ordinal:
                dates.append(datetime.date.fromordinal(date).strftime("%Y-%m-%d"))
            
            if self.trainer.is_global_zero:
                os.makedirs(self.output_path, exist_ok=True)

            # Iterate over the examples
            for traj_slice, target_slice, ens_mean_slice, ens_std_slice, date in zip(
                trajectories,
                target_states,
                ens_mean,
                ens_std,
                dates
            ):
                # Save predictions to the output folder
                print(f"Saving sample from {date} to {self.output_path}")
                print(f"Shape of ens_mean_slice: {ens_mean_slice.shape}")
                torch.save(ens_mean_slice.detach().cpu().contiguous(), f"{self.output_path}/ens_mean_{date}.pt")
                torch.save(ens_std_slice.detach().cpu().contiguous(), f"{self.output_path}/ens_std_{date}.pt")

                for ensemble_member in range(len(traj_slice)):
                    tensor_to_save = traj_slice[ensemble_member].detach().cpu().contiguous()
                    torch.save(tensor_to_save, f"{self.output_path}/member_{ensemble_member}_{date}.pt")

                torch.save(target_slice.detach().cpu().contiguous(), f"{self.output_path}/target_{date}.pt")

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                trajectories.shape[0], self.n_example_pred -
                self.plotted_examples
            )

            self.plot_examples(
                batch, n_additional_examples, prediction=trajectories
            )

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        # super().on_test_epoch_end()
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")
        self.log_spsk_ratio(self.test_metrics, "test")

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models"."""

    # ----------------------------------------------------------------------------
    # Proposed EDM sampler (Algorithm 2).

    def edm_sampler(
        self, latents, class_labels=None, boundary_forcing=None, randn_like=torch.randn_like,
        num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
        S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05,
    ):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
                   * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps, device=latents.device), torch.zeros_like(
            t_steps[:1], device=latents.device)])  # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        # 0, ..., N-1
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            # diff_steps.append(x_cur)

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) -
                        1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(
                t_cur + gamma * t_cur, device=latents.device)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * \
                S_noise * randn_like(x_cur, device=latents.device)

            # Euler step.
            denoised = self.forward(x_hat, t_hat, class_labels=class_labels)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.forward(
                    x_next, t_next, class_labels=class_labels)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * \
                    (0.5 * d_cur + 0.5 * d_prime)

        return x_next, None

    # ----------------------------------------------------------------------------
    # Proposed Heun sampler (Algorithm 1).
    def heun_sampler(
        self, latents, class_labels=None, boundary_forcing=None, randn_like=torch.randn_like,
        num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
    ):
        diffusion_steps = []

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
                   * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps, device=latents.device), torch.zeros_like(
            t_steps[:1], device=latents.device)])  # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        diffusion_steps.append(x_next)

        # 0, ..., N-1
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            denoised = self.forward(x_cur, t_cur, class_labels=class_labels)
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.forward(
                    x_next, t_next, class_labels=class_labels)
                d_prime = (x_next - denoised) / t_next
                x_next = x_cur + (t_next - t_cur) * \
                    (0.5 * d_cur + 0.5 * d_prime)

            diffusion_steps.append(x_next)

        return x_next, diffusion_steps

# ----------------------------------------------------------------------------

    # Sampler used in GenCast
    def ddpm_sampler(
        self, latents, class_labels=None, boundary_forcing=None, randn_like=torch.randn_like,
        num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
        S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05, r=0.5,
    ):

        time_steps = torch.arange(
            0, num_steps, device=latents.device) / (num_steps - 1)
        sigmas = (sigma_max ** (1 / rho) + time_steps *
                  (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # initialize noise
        x = sigmas[0] * latents

        for i in range(len(sigmas) - 1):
            # stochastic churn from Karras et al. (Alg. 2)
            gamma = (
                min(S_churn / num_steps, math.sqrt(2) - 1)
                if S_min <= sigmas[i] <= S_max
                else 0.0
            )
            # noise inflation from Karras et al. (Alg. 2)
            noise = S_noise * randn_like(latents, device=latents.device)

            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                x = x + (sigma_hat**2 - sigmas[i] ** 2) ** 0.5 * noise
            denoised = self.forward(x, sigma_hat, class_labels=class_labels)

            if i == len(sigmas) - 2:
                # final Euler step
                d = (x - denoised) / sigma_hat
                x = x + d * (sigmas[i + 1] - sigma_hat)
            else:
                # DPMSolver++2S  step (Alg. 1 in Lu et al.) with alpha_t=1.
                # t_{i-1} is t_hat because of stochastic churn!
                lambda_hat = -torch.log(sigma_hat)
                lambda_next = -torch.log(sigmas[i + 1])
                h = lambda_next - lambda_hat
                lambda_mid = lambda_hat + r * h
                sigma_mid = torch.exp(-lambda_mid)

                u = sigma_mid / sigma_hat * x - \
                    (torch.exp(-r * h) - 1) * denoised
                denoised_2 = self.forward(
                    u, sigma_mid, class_labels=class_labels)
                D = (1 - 1 / (2 * r)) * denoised + 1 / (2 * r) * denoised_2
                x = sigmas[i + 1] / sigma_hat * x - (torch.exp(-h) - 1) * D

        return x, None

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        return self.model(x, sigma, class_labels=class_labels, **model_kwargs)
