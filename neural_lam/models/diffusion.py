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
import os

from neural_lam.models.ar_model import ARModel
from neural_lam import constants, metrics, utils, vis

from neural_lam.models.graph_fm import GraphFM
from neural_lam.models.graphcast import GraphCast
from neural_lam.models.edm_networks_2 import EDMPrecond

class Diffusion(ARModel):
    """
    A new auto-regressive weather forecasting model
    """
    def __init__(self, args):
        super().__init__(args)

        # Some dimensionalities that can be useful to have stored
        self.border_condition = args.border_condition
        self.ensemble_size = args.ensemble_size
        self.sigma_min = args.sigma_min
        self.sigma_max = 88
        self.sigma_data = 1
        self.rho = 7
        self.sampler = args.sampler
        self.noise_aug_prob = args.noise_aug_prob # Probability of augmenting with noise [0, 1]
        self.save_output = args.save_output
        self.save_output_wandb = args.save_output_wandb
        self.sampler_steps = args.sampler_steps

        if args.diffusion_model != 'edm':
            self.map_noise = NoiseEmbedding()
            
        if args.diffusion_model == 'graph_fm':
            self.model = GraphFM(args)
        elif args.diffusion_model == 'edm':
            self.model = EDMPrecond(img_resolution=torch.as_tensor(constants.FULL_GRID_SHAPE),
                                    in_channels=self.grid_dim,
                                    out_channels=self.grid_output_dim,
                                    model_type='SongUNet',
                                    embedding_type=args.noise_embedding,
                                    obs_mask=self.interior_mask,
                                    sigma_data=self.sigma_data,
                                    sigma_min=self.sigma_min,
                                    sigma_max=self.sigma_max,
                                    resample_filter=args.resample_filter,
                                    channel_mult=args.channel_mult,
                                    encoder_type=args.encoder_type,
                                    attn_resolutions=args.attn_resolutions,
                                    )
        else:
            raise ValueError(f"Diffusion model {args.diffusion_model} not recognized")
            
        self.pred_residual = args.pred_residual # Whether to predict the residual instead of the next state
        self.diffusion_model = args.diffusion_model

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
        input_grid = torch.cat((prev_state, prev_prev_state, forcing), dim=-1) # (B, N_grid, d_input)
        latents = torch.randn_like(input_grid[:, :, :self.grid_output_dim]) # (B, N_grid, d_state)

        # Run through sampler
        if self.sampler == "heun":
            next_state, diff_states = self.heun_sampler(latents=latents, class_labels=input_grid, boundary_forcing=boundary_forcing, sigma_min=self.sigma_min*1.5)
        elif self.sampler == "edm":
            next_state, diff_states = self.edm_sampler(latents=latents, class_labels=input_grid, boundary_forcing=boundary_forcing, sigma_min=self.sigma_min*1.5, num_steps=self.sampler_steps)
        elif self.sampler == "ddpm":
            next_state, diff_states = self.ddpm_sampler(latents=latents, class_labels=input_grid, boundary_forcing=boundary_forcing, sigma_min=self.sigma_min*1.5)

        # Add residual if needed
        if self.pred_residual:
            next_state = (next_state * self.step_diff_std[constants.USED_PARAMS]) + self.step_diff_mean[constants.USED_PARAMS] # Unormalize residual
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

        # Sample from F inverse
        rnd_uniform = torch.rand([prev_state.shape[0], 1, 1], device=prev_state.device)
        rho_inv = 1 / self.rho
        sigma_max_rho = self.sigma_max ** rho_inv
        sigma_min_rho = self.sigma_min ** rho_inv
        sigma = (sigma_max_rho + rnd_uniform * (sigma_min_rho - sigma_max_rho)) ** self.rho
        y = true_state # (B, N_grid, d_input), true_states[4, 19, n_grid, d_state], assuming 19 is for 19 rollouts

        # Noise augmentation
        if torch.rand(1) < self.noise_aug_prob:
            rnd_uniform_aug = torch.empty([prev_state.shape[0], 1, 1], device=prev_state.device).uniform_(0.9, 1) # Previously 0.75, 1
            sigma_aug = (sigma_max_rho + rnd_uniform_aug * (sigma_min_rho - sigma_max_rho)) ** self.rho
            prev_state += torch.randn_like(prev_state) * sigma_aug
            prev_prev_state += torch.randn_like(prev_prev_state) * sigma_aug
      
        input_grid = torch.cat((prev_state, prev_prev_state, forcing), dim=-1)

        # Make y residual if needed
        if self.pred_residual:
            y = y - prev_state
            y = (y - self.step_diff_mean[constants.USED_PARAMS]) / self.step_diff_std[constants.USED_PARAMS] # Normalize residual

        n = torch.randn_like(y) * sigma    
        noisy_input = y+n

        next_state = self.forward(noisy_input, sigma, input_grid, boundary_forcing) # Shape (B, d_state, N_x, N_y)

        # Add residual if needed
        if self.pred_residual:
            next_state = (next_state * self.step_diff_std[constants.USED_PARAMS]) + self.step_diff_mean[constants.USED_PARAMS] # Unormalize residual
            next_state = prev_state + next_state

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        return next_state, None, weight
    
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
                pred_std = torch.tensor(1, device=init_states.device) # Using the same weights for all variables
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
        weight_list = []

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            border_state = boundary_forcing[:, i]
            true_state = true_states[:, i]

            pred_state, pred_std, weight = self.predict_step_train(
                prev_state, prev_prev_state, forcing, true_state, border_state
            )
            # state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            # Overwrite border with true state
            new_state = pred_state

            prediction_list.append(new_state)
            weight_list.append(weight)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        weight = torch.stack(
            weight_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
            # pred_std = self.per_var_std  # (d_f,)
        else:
            pred_std = self.per_var_std  # (d_f,)
            # pred_std = 1 # Testing equal weights

        return prediction, pred_std, weight

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
        prediction, pred_std, weight = self.unroll_prediction_train(
            init_states, forcing, target_states, boundary_forcing
        )  # (B, pred_steps, num_grid_nodes, d_f)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        return prediction, target_states, pred_std, weight

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, pred_std, weight = self.common_step_train(batch)

        # Compute loss
        batch_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, weight=weight # mask=self.interior_mask_bool
            )
        )  # mean over unrolled times and batch

        batch_mse = torch.mean(
            metrics.mse(
                prediction, target, pred_std, # mask=self.interior_mask_bool
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
            traj_stds = self.per_var_std[constants.USED_PARAMS] # TODO: Check if this is correct, self.per_var_std = self.step_diff_std / torch.sqrt(self.param_weights)
        
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
        traj_rescaled = trajectories * self.data_std[constants.USED_PARAMS] + self.data_mean[constants.USED_PARAMS]
        target_rescaled = target_states * self.data_std[constants.USED_PARAMS] + self.data_mean[constants.USED_PARAMS]
        border_rescaled = border * self.data_std[constants.USED_PARAMS] + self.data_mean[constants.USED_PARAMS]
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
                torch.save(ens_mean_slice[0], f"output/example_ens_mean_{self.plotted_examples}.pt")
                torch.save(ens_std_slice[0], f"output/example_ens_std_{self.plotted_examples}.pt")
                torch.save(traj_slice[0], f"output/example_ens_members_{self.plotted_examples}.pt")
                torch.save(target_slice[0], f"output/example_target_{self.plotted_examples}.pt")
                torch.save(border_slice[0], f"output/example_border_{self.plotted_examples}.pt")

                # Save files to wandb
                if self.save_output_wandb:
                    wandb.save(f"output/example_ens_mean_{self.plotted_examples}.pt")
                    wandb.save(f"output/example_ens_std_{self.plotted_examples}.pt")
                    wandb.save(f"output/example_ens_members_{self.plotted_examples}.pt")
                    wandb.save(f"output/example_target_{self.plotted_examples}.pt")
                    wandb.save(f"output/example_border_{self.plotted_examples}.pt")

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

    def validation_step(self, batch, *args):
        """
        Run validation on single batch
        """
        print("No validation step implemented!")
        # super().validation_step(batch, *args)
        # prediction, target, pred_std, weight = self.common_step_train(batch)

        # time_step_loss = torch.mean(
        #     self.loss(
        #         prediction, target, pred_std, weight=weight # mask=self.interior_mask_bool
        #     ),
        #     dim=0,
        # )  # (time_steps-1)
        # mean_loss = torch.mean(time_step_loss)

        # # Log loss per time step forward and mean
        # val_log_dict = {
        #     f"val_loss_unroll{step}": time_step_loss[step - 1]
        #     for step in constants.VAL_STEP_LOG_ERRORS # ONLY LOGGING FOR 1 STEP since logging diffusion steps for all steps is too much and not that informative
        # }
        # val_log_dict["val_mean_loss"] = mean_loss
        # self.log_dict(
        #     val_log_dict, on_step=False, on_epoch=True, sync_dist=True
        # )

        # # Store MSEs
        # entry_mses = metrics.mse(
        #     prediction,
        #     target,
        #     pred_std,
        #     # mask=self.interior_mask_bool,
        #     sum_vars=False,
        # )  # (B, pred_steps, d_f)
        # self.val_metrics["mse"].append(entry_mses)

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
                trajectories.shape[0], self.n_example_pred - self.plotted_examples
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

    #----------------------------------------------------------------------------
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
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps, device=latents.device), torch.zeros_like(t_steps[:1], device=latents.device)]) # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next
            # diff_steps.append(x_cur)

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur, device=latents.device)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur, device=latents.device)

            # Euler step.
            denoised = self.forward(x_hat, t_hat, class_labels=class_labels, boundary_forcing=boundary_forcing)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.forward(x_next, t_next, class_labels=class_labels, boundary_forcing=boundary_forcing)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next, None

    #----------------------------------------------------------------------------
    # Proposed Heun sampler (Algorithm 1).
    def heun_sampler(
        self, latents, class_labels=None, boundary_forcing=None, randn_like=torch.randn_like,
        num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
    ):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps, device=latents.device), torch.zeros_like(t_steps[:1], device=latents.device)]) # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next
            denoised = self.forward(x_cur, t_cur, class_labels=class_labels, boundary_forcing=boundary_forcing)
            d_cur = (x_cur - denoised) / t_cur      
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.forward(x_next, t_next, class_labels=class_labels, boundary_forcing=boundary_forcing)
                d_prime = (x_next - denoised) / t_next   
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next, None

#----------------------------------------------------------------------------

    # Sampler used in GenCast
    def ddpm_sampler(
        self, latents, class_labels=None, boundary_forcing=None, randn_like=torch.randn_like,
        num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
        S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05, r=0.5,
    ):

        time_steps = torch.arange(0, num_steps, device=latents.device) / (num_steps - 1)
        sigmas = (sigma_max ** (1 / rho)+ time_steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

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
            denoised = self.forward(x, sigma_hat, class_labels=class_labels, boundary_forcing=boundary_forcing)

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

                u = sigma_mid / sigma_hat * x - (torch.exp(-r * h) - 1) * denoised
                denoised_2 = self.forward(u, sigma_mid, class_labels=class_labels, boundary_forcing=boundary_forcing)
                D = (1 - 1 / (2 * r)) * denoised + 1 / (2 * r) * denoised_2
                x = sigmas[i + 1] / sigma_hat * x - (torch.exp(-h) - 1) * D

        return x, None
        

    def forward(self, x, sigma, class_labels=None, boundary_forcing=None, force_fp32=False, **model_kwargs):
        if self.diffusion_model == "edm":
            return self.model(x, sigma, class_labels=class_labels, boundary_forcing=boundary_forcing, **model_kwargs)

        sigma = sigma.reshape(-1, 1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
      
        F_x = self.model_forward((c_in * x), c_noise.flatten(), class_labels=class_labels, boundary_forcing=boundary_forcing, **model_kwargs)
        D_x = c_skip * x + c_out * F_x
        return D_x
    
    def model_forward(self, x, noise_labels, class_labels, boundary_forcing):
        # Mapping.
        emb = self.map_noise(noise_labels).unsqueeze(1).expand(x.shape[0], 1, -1)

        next_state, _ = self.model.predict_step(x, class_labels[:, :, :len(constants.USED_PARAMS)*2], class_labels[:, :, len(constants.USED_PARAMS)*2:], boundary_forcing, emb)

        return next_state
    
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class NoiseLevelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=16):
        super(NoiseLevelMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, fourier_features):
        x = silu(self.fc1(fourier_features))
        x = silu(self.fc2(x))
        x = self.fc3(x)
        return x  # Output noise-level encoding (batch_size, output_dim)
class NoiseEmbedding(nn.Module):
    def __init__(self, num_frequencies=32, base_period=16):
        super(NoiseEmbedding, self).__init__()
        self.fourier_transform = FourierEmbedding(num_channels=num_frequencies, scale=base_period)
        self.mlp = NoiseLevelMLP(input_dim=num_frequencies, output_dim=16)

    def forward(self, log_noise_levels):
        fourier_features = self.fourier_transform(log_noise_levels)
        noise_level_encoding = self.mlp(fourier_features)
        return noise_level_encoding
    
#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
