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
import einops

from neural_lam.models.ar_model import ARModel
from neural_lam import constants, metrics, utils, vis

from neural_lam.models.unet import UNET
from neural_lam.models.diffusion import Diffusion


class CorrDiff(ARModel):
    """
    A downscaling model based on stochastic interpolants
    """

    def __init__(self, args):
        super().__init__(args)

        mean_ckpt_path = "/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_Static_50e-unet-6x128-12_12_16-6743/last.ckpt"
        # TODO: Should be loaded with checkpoint and frozen
        self.mean_model = UNET(args)
        self.ensemble_size = args.ensemble_size
        self.save_output = args.save_output
        self.save_output_wandb = args.save_output_wandb

        # Load into mean_model (allow partial loading)
        try:
            ckpt = torch.load(mean_ckpt_path, map_location="cpu")
            # extract state_dict if Lightning .ckpt style
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and all(
                isinstance(v, torch.Tensor) for v in ckpt.values()
            ):
                state_dict = ckpt
            else:
                state_dict = None

            if state_dict is not None:
                # strip common prefixes that appear in Lightning checkpoints
                stripped = {}
                for k, v in state_dict.items():
                    new_k = k
                    for p in ("model.", "mean_model.", "mean_model.module.", "module."):
                        if new_k.startswith(p):
                            new_k = new_k[len(p):]
                    stripped[new_k] = v

                self.mean_model.load_state_dict(stripped, strict=False)
                for p in self.mean_model.parameters():
                    p.requires_grad = False
                self.mean_model.eval()
                print(
                    f"Loaded mean model from '{mean_ckpt_path}' and froze it.")
            else:
                print(
                    f"No state_dict found in checkpoint '{mean_ckpt_path}'.")
        except Exception as e:
            print(
                f"Failed to load mean model checkpoint '{mean_ckpt_path}': {e}")

        # Freeze parameters and set eval mode
        for p in self.mean_model.parameters():
            p.requires_grad = False
        self.mean_model.eval()
        print(f"Loaded and froze mean model from: {mean_ckpt_path}")

        self.diffusion_model = Diffusion(args)

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

    # ----------------------------------------------------------------------------

    def predict_step(self, LQ):
        """
        Downscaling weather state
        LQ: (B, d_f, X, Y)

        Returns:
        next_state: (B, N_grid, d_state)
        pred_std: None
        """
        # Predict the mean
        # TODO: Don't need to predict the mean for each ensemble member
        mean, _ = self.mean_model.predict_step(LQ)  # (B, N_grid, d_f)

        # Reshape the mean to (B, d_f, X, Y)
        mean_grid = mean.reshape(
            LQ.shape[0], LQ.shape[2], LQ.shape[3], -1).permute(0, 3, 1, 2)

        # Predict the residual
        input_grid = torch.cat(
            [LQ, mean_grid], dim=1)
        residual, _ = self.diffusion_model.predict_step(
            input_grid)

        # Add the residual to the mean
        sample = mean + residual

        return sample, None

    def predict_step_train(self, LQ, HQ):
        """
        Downscaling weather state
        LQ: (B, d_f, X, Y)
        HQ: (B, d_f, X, Y)

        Returns:
        downscaled_state: (B, N_grid, d_state)
        pred_std: None 
        """
        # Predict the mean
        mean, _ = self.mean_model.predict_step(LQ)  # (B, N_grid, d_f)

        # Reshape the mean to (B, d_f, X, Y)
        mean_grid = mean.reshape(
            HQ.shape[0], HQ.shape[2], HQ.shape[3], HQ.shape[1]).permute(0, 3, 1, 2)

        # Predict the residual
        input_grid = torch.cat(
            [LQ, mean_grid], dim=1)

        # Call diffusion training step to obtain a prediction and a weight
        # sample: (B, N_grid, d_f), weight: broadcastable tensor
        sample, _, weight = self.diffusion_model.predict_step_train(
            input_grid, HQ)

        # Build target in same flattened layout: (B, N_grid, d_f)
        target = HQ.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # Our loss expects a time/unroll dimension (pred_steps). Add that dim
        pred = sample.unsqueeze(1)  # (B, 1, N_grid, d_f)
        targ = target.unsqueeze(1)  # (B, 1, N_grid, d_f)

        # Use the configured loss (inherited from ARModel) and include weight
        # weight from the diffusion model is already shaped for broadcasting
        loss = self.loss(pred, targ, self.per_var_std, weight=weight)

        return sample, None, loss

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
        loss: (B, pred_steps) ?
        """
        prediction_list = []
        pred_std_list = []
        pred_steps = 1
        loss_list = []

        for i in range(pred_steps):
            pred_state, pred_std, loss = self.predict_step_train(LQ, HQ)

            prediction_list.append(pred_state)
            loss_list.append(loss)
            if self.output_std:
                pred_std_list.append(pred_std)

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        loss = torch.stack(
            loss_list, dim=0
        )  # (B, pred_steps, num_grid_nodes, d_f)

        if self.output_std:
            # Using the same weights for all variables
            pred_std = torch.tensor(1, device=LQ.device)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std, loss

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
        prediction, pred_std, loss = self.unroll_prediction_train(
            LQ, HQ
        )

        target = HQ.permute(0, 2, 3, 1).contiguous().flatten(
            1, 2).unsqueeze(1)  # (B, pred_steps, N_grid, d_f)

        return prediction, target, pred_std, loss

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, pred_std, loss = self.common_step_train(batch)

        # Compute loss
        batch_loss = torch.mean(
            loss
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
                        vrange=var_vrange,
                        data_config=self.config_loader,
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
        LQ, HQ = batch["LQ"], batch["HQ"]

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
