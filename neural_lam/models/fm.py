# Standard library
import copy
import math
import os
import time

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.nn.functional import silu

# First-party
from neural_lam import constants, metrics, utils, vis
from neural_lam.models.ar_model import ARModel
from neural_lam.models.edm_networks_2 import SongUNet


def bad(x):
    return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))


class FM(ARModel):
    """
    A new auto-regressive weather forecasting model
    """

    def __init__(self, args):
        super().__init__(args)

        # Some dimensionalities that can be useful to have stored
        self.border_condition = args.border_condition
        self.ensemble_size = args.ensemble_size
        self.sampler = args.sampler
        self.save_output = args.save_output
        self.save_output_wandb = args.save_output_wandb
        self.sampler_steps = args.sampler_steps
        self.save_steps = args.save_steps  # TODO: Fix later
        self.GT = None
        self.output_std = args.output_std  # Whether to use the variable weights or not

        # grid_dim from data + static
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape  # 63784 = 268x238

        num_states = 3

        self.grid_dim = (
            num_states * self.config_loader.num_data_vars()
            + grid_static_dim
            + self.config_loader.dataset.num_forcing_features
        )

        if args.diffusion_model == 'song_unet':
            self.model = SongUNet(img_resolution=torch.as_tensor(constants.FULL_GRID_SHAPE),
                                  in_channels=self.grid_dim,
                                  out_channels=self.config_loader.num_data_vars(),
                                  embedding_type=args.noise_embedding,
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

        self.t_min_sampling = 0.0
        self.t_max_sampling = 1.0

        self.test_metrics = {
            "ens_mae": [],
            "ens_mse": [],
            "crps_ens": [],
            "spread_squared": [],
        }

    def predict_step(self, prev_state, prev_prev_state, forcing, boundary_forcing):
        """
        Predict weather state one time step ahead
        X_{t-1}, X_t -> X_t+1

        prev_state: (B, N_grid, d_state), weather state X_t at time t
        prev_prev_state: (B, N_grid, d_state), weather state X_{t-1} at time t-1
        batch_static_features: (B, N_grid, batch_static_feature_dim), static forcing
        forcing: (B, N_grid, forcing_dim), dynamic forcing

        Returns:
        next_state: (B, N_grid, d_state), predicted weather state X_{t+1} at time t+1
        pred_std: None or (B, N_grid, d_state), predicted standard-deviations
                    (pred_std can be ignored by just returning None)
        """
        input_grid = torch.cat(
            (prev_state, prev_prev_state, forcing), dim=-1)  # (B, N_grid, d_input)
        latents = torch.randn_like(
            input_grid[:, :, :self.grid_output_dim])  # (B, N_grid, d_state)

        # Run through sampler
        if self.sampler == "heun":
            next_state, diff_states = self.heun_sampler(
                latents=latents, class_labels=input_grid, boundary_forcing=boundary_forcing, sigma_min=self.t_min_sampling, sigma_max=self.t_max_sampling)
        elif self.sampler == "midpoint":
            next_state, diff_states = self.midpoint(
                latents=latents, class_labels=input_grid, boundary_forcing=boundary_forcing, sigma_min=self.t_min_sampling, sigma_max=self.t_max_sampling)
        else:
            raise ValueError(f"Sampler {self.sampler} not recognized")

        # Add residual if needed
        if self.pred_residual:
            next_state = (next_state * self.step_diff_std[constants.USED_PARAMS]) + \
                self.step_diff_mean[constants.USED_PARAMS]  # Unormalize residual
            next_state = prev_state + next_state

        return next_state, None

    def predict_step_train(self, prev_state, prev_prev_state, forcing, true_state, boundary_forcing):
        """
        Predict weather state one time step ahead
        X_{t-1}, X_t -> X_t+1

        prev_state: (B, N_grid, d_state), weather state X_t at time t
        prev_prev_state: (B, N_grid, d_state), weather state X_{t-1} at time t-1
        batch_static_features: (B, N_grid, batch_static_feature_dim), static forcing
        forcing: (B, N_grid, forcing_dim), dynamic forcing

        Returns:
        next_state: (B, N_grid, d_state), predicted weather state X_{t+1} at time t+1
        pred_std: None or (B, N_grid, d_state), predicted standard-deviations
                    (pred_std can be ignored by just returning None)
        """

        # (B, N_grid, d_input), true_states[4, 19, n_grid, d_state], assuming 19 is for 19 rollouts
        y = true_state
        # Make y residual if needed
        if self.pred_residual:
            y = y - prev_state
            y = (y - self.step_diff_mean[constants.USED_PARAMS]) / \
                self.step_diff_std[constants.USED_PARAMS]  # Normalize residual

        input_grid = torch.cat((prev_state, prev_prev_state, forcing), dim=-1)

        # (B, N_grid, d_state)
        z0 = torch.randn_like(true_state, device=true_state.device)
        z1 = y

        t = torch.rand([prev_state.shape[0], 1, 1], device=prev_state.device)
        zt = (1 - t) * z0 + t * z1

        next_state = self.model(zt, t, input_grid, boundary_forcing)

        loss = (next_state - (z1-z0)) ** 2

        loss = torch.mean(loss / (self.per_var_std**2))

        # Add residual if needed
        if self.pred_residual:
            next_state = (next_state * self.step_diff_std[constants.USED_PARAMS]) + \
                self.step_diff_mean[constants.USED_PARAMS]  # Unormalize residual
            next_state = prev_state + next_state

        return next_state, None, loss.unsqueeze(0)

    def unroll_prediction(self, init_states, forcing_features, boundary_forcing):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            border_state = boundary_forcing[:, i]
            pred_state, pred_std = self.predict_step(
                prev_state, prev_prev_state, forcing, border_state
            )

            new_state = pred_state

            prediction_list.append(new_state)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        if self.output_std:
            # pred_std = torch.stack(
            #     pred_std_list, dim=1
            # )  # (B, pred_steps, num_grid_nodes, d_f)
            # Using the same weights for all variables
            pred_std = torch.tensor(1, device=init_states.device)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std

    def unroll_prediction_train(self, init_states, forcing_features, true_states, boundary_forcing):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]
        loss_list = []

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            border_state = boundary_forcing[:, i]
            true_state = true_states[:, i]

            pred_state, pred_std, loss = self.predict_step_train(
                prev_state, prev_prev_state, forcing, true_state, border_state
            )
            # state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            # Overwrite border with true state
            new_state = pred_state

            prediction_list.append(new_state)
            loss_list.append(loss)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        loss = torch.stack(
            loss_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
            # pred_std = self.per_var_std  # (d_f,)
        else:
            pred_std = self.per_var_std  # (d_f,)
            # pred_std = 1 # Testing equal weights

        return prediction, pred_std, loss

    def common_step_train(self, batch):
        """
        Predict on single batch
        batch consists of:
        init_states: (B, 2, num_grid_nodes, d_features)
        target_states: (B, pred_steps, num_grid_nodes, d_features)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        (init_states, target_states, forcing, boundary_forcing) = batch
        prediction, pred_std, loss = self.unroll_prediction_train(
            init_states, forcing, target_states, boundary_forcing
        )  # (B, pred_steps, num_grid_nodes, d_f)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        return prediction, target_states, pred_std, loss

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, pred_std, loss = self.common_step_train(batch)

        # Compute loss
        batch_loss = torch.mean(loss)  # mean over unrolled times and batch
        # batch_loss = torch.mean(
        #     self.loss(
        #         prediction, target, pred_std, weight=weight # mask=self.interior_mask_bool
        #     )
        # )  # mean over unrolled times and batch

        batch_mse = torch.mean(
            metrics.mse(
                prediction, target, pred_std,  # mask=self.interior_mask_bool
            )
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss, "train_mse": batch_mse}
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return batch_loss

    def sample_trajectories(
        self,
        init_states,
        forcing_features,
        boundary_forcing,
        num_traj,
        use_encoder=False,
    ):
        """
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        num_traj: S, number of trajectories to sample
        use_encoder: bool, if latent variables should be sampled from
            var. distribution

        Returns
        traj_means: (B, S, pred_steps, num_grid_nodes, d_f)
        traj_stds: (B, S, pred_steps, num_grid_nodes, d_f) or (d_f)
        """
        unroll_func = self.unroll_prediction

        traj_list = []
        for i in range(num_traj):
            # print(f"Starting trajectory {i + 1}/{num_traj}...")
            # start_time = time.time()

            traj = unroll_func(
                init_states,
                forcing_features,
                boundary_forcing,
            )

            traj_list.append(traj)

        # List of tuples, each containing
        # mean: (B, pred_steps, num_grid_nodes, d_f) and
        # std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        traj_means = torch.stack(
            [pred_pair[0] for pred_pair in traj_list], dim=1
        )
        if self.output_std:
            traj_stds = torch.stack(
                [pred_pair[1] for pred_pair in traj_list], dim=1
            )
        else:
            # TODO: Check if this is correct, self.per_var_std = self.step_diff_std / torch.sqrt(self.param_weights)
            traj_stds = self.per_var_std[constants.USED_PARAMS]

        return traj_means, traj_stds

    def plot_examples(self, batch, n_examples, prediction=None):
        """
        Plot ensemble forecast + mean and std
        """
        init_states, target_states, forcing_features, boundary_forcing = batch
        border = boundary_forcing[..., :len(constants.USED_PARAMS)]
        if prediction is None:
            print(f"Sampling new trajectories for plotting!")
            trajectories, _ = self.sample_trajectories(
                init_states,
                forcing_features,
                boundary_forcing,
                self.ensemble_size,
            )
        else:
            trajectories = prediction
        # (B, S, pred_steps, num_grid_nodes, d_f)

        # Rescale to original data scale
        traj_rescaled = trajectories * \
            self.data_std[constants.USED_PARAMS] + \
            self.data_mean[constants.USED_PARAMS]
        target_rescaled = target_states * \
            self.data_std[constants.USED_PARAMS] + \
            self.data_mean[constants.USED_PARAMS]
        border_rescaled = border * \
            self.data_std[constants.USED_PARAMS] + \
            self.data_mean[constants.USED_PARAMS]
        # Compute mean and std of ensemble
        ens_mean = torch.mean(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        # Iterate over the examples
        for traj_slice, target_slice, border_slice, ens_mean_slice, ens_std_slice in zip(
            traj_rescaled[:n_examples],
            target_rescaled[:n_examples],
            border_rescaled[:n_examples, ..., :len(constants.USED_PARAMS)],
            ens_mean[:n_examples],
            ens_std[:n_examples],
        ):
            # traj_slice is (S, pred_steps, num_grid_nodes, d_f)
            # others are (pred_steps, num_grid_nodes, d_f)

            self.plotted_examples += 1  # Increment already here

            # Save slices to wandb
            os.makedirs("output", exist_ok=True)

            # TODO: Check that the saving is correct, we want to save one sample and not the entire batch
            # Save predictions to the output folder
            if self.save_output:
                torch.save(
                    ens_mean_slice[0], f"output/example_ens_mean_{self.plotted_examples}.pt")
                torch.save(
                    ens_std_slice[0], f"output/example_ens_std_{self.plotted_examples}.pt")
                torch.save(
                    traj_slice[0], f"output/example_ens_members_{self.plotted_examples}.pt")
                torch.save(
                    target_slice[0], f"output/example_target_{self.plotted_examples}.pt")
                torch.save(
                    border_slice[0], f"output/example_border_{self.plotted_examples}.pt")

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
                    wandb.save(
                        f"output/example_border_{self.plotted_examples}.pt")

            # Note: min and max values can not be in ensemble mean
            var_vmin = (
                torch.minimum(
                    traj_slice.flatten(0, 2).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    traj_slice.flatten(0, 2).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, (samples_t, target_t, border_t, ens_mean_t, ens_std_t) in enumerate(
                zip(
                    traj_slice.transpose(0, 1),
                    # (pred_steps, S, num_grid_nodes, d_f)
                    target_slice,
                    border_slice,
                    ens_mean_slice,
                    ens_std_slice,
                ),
                start=1,
            ):
                time_title_part = f"t={t_i} ({self.step_length*t_i} h)"
                # Create one figure per variable at this time step
                var_figs = [
                    vis.plot_ensemble_prediction(
                        samples_t[:, :, var_i],
                        target_t[:, var_i],
                        border_t[:, var_i],
                        ens_mean_t[:, var_i],
                        ens_std_t[:, var_i],
                        self.interior_mask,
                        title=f"{var_name} ({var_unit}), {time_title_part}",
                        vrange=var_vrange,
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            constants.PARAM_NAMES_SHORT[constants.USED_PARAMS],
                            constants.PARAM_UNITS[constants.USED_PARAMS],
                            var_vranges,
                        )
                    )
                ]

                example_title = f"example_{self.plotted_examples}"
                wandb.log(
                    {
                        f"{var_name}_{example_title}": wandb.Image(fig)
                        for var_name, fig in zip(
                            constants.PARAM_NAMES_SHORT[constants.USED_PARAMS], var_figs
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
        trajectories: (B, S, pred_steps, num_grid_nodes, d_f)
        traj_stds: (B, S, pred_steps, num_grid_nodes, d_f)
        target_states: (B, pred_steps, num_grid_nodes, d_f)
        spread_squared_batch: (B, pred_steps, d_f)
        ens_mse_batch: (B, pred_steps, d_f)
        """
        # Compute and store metrics for ensemble forecast
        init_states, target_states, forcing_features, boundary_forcing = batch

        if self.save_steps:
            self.GT = target_states
        else:
            self.GT = None

        trajectories, traj_stds = self.sample_trajectories(
            init_states,
            forcing_features,
            boundary_forcing,
            self.ensemble_size,
        )
        # (B, S, pred_steps, num_grid_nodes, d_f)

        spread_squared_batch = metrics.spread_squared(
            trajectories,
            target_states,
            traj_stds,
            # mask=self.interior_mask_bool,
            sum_vars=False,
        )
        # (B, pred_steps, d_f)

        ens_mean = torch.mean(
            trajectories, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_mse_batch = metrics.mse(
            ens_mean,
            target_states,
            None,
            # mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)

        return (
            trajectories,
            traj_stds,
            target_states,
            spread_squared_batch,
            ens_mse_batch,
        )

    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, pred_std, loss = self.common_step_train(batch)

        time_step_loss = torch.mean(loss, dim=0)  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            # ONLY LOGGING FOR 1 STEP since logging diffusion steps for all steps is too much and not that informative
            for step in constants.VAL_STEP_LOG_ERRORS
        }
        val_log_dict["val_mean_loss"] = mean_loss
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

        if batch_idx == 0:
            # Get Probabilistic Metrics
            (
                trajectories,
                traj_stds,
                target_states,
                spread_squared_batch,
                ens_mse_batch,
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
            if self.trainer.is_global_zero and batch_idx == 0:
                self.plot_examples(
                    batch,
                    1,
                    prediction=trajectories,
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
        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            traj_stds,
            # mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["crps_ens"].append(crps_batch)

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

    def midpoint(self, latents, class_labels=None, boundary_forcing=None, num_steps=20, sigma_min=0.0, sigma_max=1.0):
        """
        Generate random images using the midpoint sampling technique.
        This is a placeholder for the midpoint sampling method.
        """
        t_steps = torch.linspace(
            sigma_min, sigma_max, num_steps, device=latents.device)  # t_0 = tmin, t_N = tmax
        x_t = latents
        # 0, ..., N-1
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # print(f"Midpoint sampling step {i+1}/{num_steps-1} with t_cur={t_cur}, t_next={t_next}")
            t_cur = t_cur.unsqueeze(0)  # (1)
            t_next = t_next.unsqueeze(0)
            half_step = (t_next - t_cur) / 2
            mid = half_step * \
                self.model(latents, t_cur, class_labels=class_labels,
                           boundary_forcing=boundary_forcing)
            x_t = x_t + (t_next - t_cur) * self.model(mid, t_cur+half_step,
                                                      class_labels=class_labels, boundary_forcing=boundary_forcing)

        return x_t, None

    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models"."""

    # ----------------------------------------------------------------------------
    # Proposed Heun sampler (Algorithm 1).
    def heun_sampler(
        self, latents, class_labels=None, boundary_forcing=None, num_steps=20, sigma_min=0.0, sigma_max=1.0
    ):
        tmin = 0.0
        tmax = 1.0

        # Time step discretization.
        t_steps = torch.linspace(tmin, tmax, num_steps, device=self.device)

        # Main sampling loop.
        x_next = latents  # * t_steps[0]
        # 0, ..., N-1
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            t_cur = t_cur.unsqueeze(0)  # (1)
            t_next = t_next.unsqueeze(0)
            d_cur = self.model(
                x_cur, t_cur, class_labels=class_labels, boundary_forcing=boundary_forcing)
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                d_prime = self.model(
                    x_next, t_next, class_labels=class_labels, boundary_forcing=boundary_forcing)
                x_next = x_cur + (t_next - t_cur) * \
                    (0.5 * d_cur + 0.5 * d_prime)

        return x_next, None

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def stochastic_sampler(
        self, latents, class_labels=None, boundary_forcing=None, randn_like=torch.randn_like,
            num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
    ):
        tmin = 0.0
        tmax = 1
        eps = 1.0

        # Time step discretization.
        ts = torch.linspace(tmin, tmax, num_steps+1, device=self.device)[:-1]
        dt = (tmax - tmin) / num_steps

        # Main sampling loop.
        zt = latents  # Initialize with noise
        for t in ts:
            alpha_t = t
            beta_t = 1 - t
            gamma_t = 1
            alpha_dot_t = 1
            beta_dot_t = -1
            eps_t = eps * beta_t

            b = self.forward(zt, t, class_labels=class_labels,
                             dropout=self.dropout)
            s = (alpha_t * b - alpha_dot_t * zt) / \
                (beta_t * gamma_t)  # s = (t * b - zt) / (1 - t)
            dz = b + eps_t * s

            dW = torch.randn_like(zt) * torch.sqrt(2*dt * eps_t)

            zt = zt + dz * dt + dW

        return zt  # (B, N_grid, d_state)
