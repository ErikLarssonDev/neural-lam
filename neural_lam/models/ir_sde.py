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

from neural_lam.models.graph_fm import GraphFM
from neural_lam.models.graphcast import GraphCast
from neural_lam.models.edm_networks_2 import SongUNet

class IR_SDE(ARModel):
    """
    A new auto-regressive weather forecasting model
    """
    def __init__(self, args):
        super().__init__(args)

        # Some dimensionalities that can be useful to have stored
        self.ensemble_size = args.ensemble_size
        self.sampler = args.sampler
        self.max_sigma = args.sigma_max
        self.noise_aug_prob = args.noise_aug_prob # Probability of augmenting with noise [0, 1]
        self.save_output = args.save_output
        self.save_output_wandb = args.save_output_wandb
        self.T = args.sampler_steps # TODO: Should probably rename this and investigate if we can get continuous time steps for training [0, 1]
        self.save_steps = args.save_steps
        self.loss_fn = MatchingLoss(loss_type="l2")

        if args.diffusion_model == 'song_unet':
            self.model = SongUNet(img_resolution=torch.as_tensor(constants.FULL_GRID_SHAPE),
                                    in_channels=self.grid_output_dim*2, # We have noise and LQ as input
                                    out_channels=self.grid_output_dim,
                                    embedding_type=args.noise_embedding,
                                    resample_filter=args.resample_filter,
                                    channel_mult=args.channel_mult,
                                    encoder_type=args.encoder_type,
                                    attn_resolutions=args.attn_resolutions,
                                    ir_sde=True,
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
        # Instantiate model + trainer
        if torch.cuda.is_available():
            self.device_name = "cuda"
            torch.set_float32_matmul_precision(
                "high"
            )  # Allows using Tensor Cores on A100s
        else:
            self.device_name = "cpu"

    ############################################################################
    # IR-SDE
        # self.max_sigma = args.sigma_max / 255 if args.sigma_max >= 1 else args.sigma_max # Is this only because of images?, still needed for good results.
        self._initialize(self.max_sigma, T=args.sampler_steps, schedule="cosine", eps=args.eps)

    def _initialize(self, max_sigma=10 / 255, T=100, schedule="cosine", eps=0.005): # Standard values from deblurring task
        def cosine_theta_schedule(timesteps, s = 0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2 # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=self.device_name)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2 # TODO: x/timesteps could be [0,1], does not need to be discrete
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas
        
        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
        
        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas
        self.sigmas = sigmas
        self.thetas_cumsum = thetas_cumsum
        self.sigma_bars = sigma_bars

    # set mu for different cases
    def set_mu(self, mu): # TODO: Should probably remove this and just send mu to the noise function
        self.mu = mu

    def mu_bar(self, x0, t):
        return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)
    
    def sigma_bar(self, t):
        return self.sigma_bars[t]
    
    def score_fn(self, x, t, **kwargs):
        noise_level = (t-1) / (self.T-1) # Normalize the diffusion time to [0, 1], TODO: How should we send the noise level to the model?
        noise = self.model(x, noise_level.reshape(x.shape[0]), self.mu, **kwargs)
        return self.get_score_from_noise(noise, t)
    
    def noise_fn(self, x, t, **kwargs):
        noise_level = (t-1) / (self.T-1) # Normalize the diffusion time to [0, 1], TODO: How should we send the noise level to the model?
        return self.model(x, noise_level.reshape(x.shape[0]), self.mu, **kwargs)
    
    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)
        
    # Sample states for training
    def generate_random_states(self, x0, mu):
        x0 = x0
        mu = mu

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1), device=x0.device).long()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean, device=x0.device)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32)
    
    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - self.sigmas[t]**2 * score) * self.dt

    def reverse_sde_step_mean(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t)
    
    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu
    
    def feed_data(self, state, LQ, GT=None):
        self.state = state # noisy_state
        self.condition = LQ # LQ
        if GT is not None:
            self.state_0 = GT # GT

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x, device=self.device_name) * math.sqrt(self.dt))

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)
    
    def reverse_sde(self, xt, T=-1, save_states=False, save_dir='diffusion_steps', GT=None, **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in reversed(range(1, T + 1)):
            idx = t
            t = torch.tensor(t, device=x.device)
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

            if self.save_steps:
                for var_idx, var_name in enumerate(constants.PARAM_NAMES_SHORT):
                    os.makedirs(f"{save_dir}/{var_name}", exist_ok=True) # TODO: Make the saving work for multiple fields, preferably in a subfolders
                    print(f"Saving to {save_dir}/{var_name}/state_{idx}.png")

                    vmin = GT[0, var_idx, ...].min().item()
                    vmax = GT[0, var_idx, ...].max().item()

                    fig, axes = plt.subplots(
                        1,
                        2,
                        figsize=(12, 8),
                        subplot_kw={"projection": constants.LAMBERT_PROJ},
                    )

                    axes[0].coastlines()  # Add coastline outlines
                    im = axes[0].imshow(
                        x[0, var_idx, ...].data.cpu().numpy(),
                        origin="lower",
                        vmin=vmin, # Should have the same vmin and vmax for all images, but at least the same for GT and pred, maybe GT sets the same for all?
                        vmax=vmax,
                        cmap="plasma",
                        # extent=grid_limits,
                    )
                    axes[0].set_title(f"X_t", size=15)
                    
                    axes[1].coastlines()  # Add coastline outlines
                    im = axes[1].imshow(
                        GT[0, var_idx, ...].data.cpu().numpy(),
                        origin="lower",
                        vmin=vmin,
                        vmax=vmax,
                        cmap="plasma",
                        # extent=grid_limits,
                    )
                    axes[1].set_title(f"Ground Truth", size=15)
                    # fig.colorbar(im, ax=axes, orientation="horizontal")
                    fig.suptitle(f"{var_name} at time {t}", size=20)
                    fig.savefig(f"{save_dir}/{var_name}/state_{idx}.png", bbox_inches='tight') # TODO: Should log to wandb
                    plt.close(fig)
                
                self.save_steps = False # Only save the first time

        return x
    
    def noise_state(self, tensor):
        return tensor + torch.randn_like(tensor) * self.max_sigma
    
    # # TODO: Implement interpolate function

    #----------------------------------------------------------------------------

    def predict_step(self, LQ):
        """
        Downscaling weather state
        LQ: (B, d_f, X, Y)

        Returns:
        next_state: (B, N_grid, d_state)
        pred_std: None
        """
        noisy_state = self.noise_state(LQ)
        self.feed_data(state=noisy_state, LQ=LQ, GT=None)
        self.set_mu(self.condition)

        # TODO: Implement Heun solver
        next_state = self.reverse_sde(self.state, save_states=False, GT=None) # TODO: Fix save states for plotting

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

        # TODO: Implement pred_residual for training

        # Generate random states
        timesteps, noisy_states = self.generate_random_states(HQ, LQ)
        self.feed_data(noisy_states, LQ, HQ)
        
        # Optimize parameters
        self.set_mu(self.condition)

        # Get noise and score
        noise = self.noise_fn(self.state, timesteps) # Changed to noise levels from timesteps
        score = self.get_score_from_noise(noise, timesteps)

        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = self.reverse_sde_step_mean(self.state, score, timesteps) # TODO: Can this be used to calculate the mse_step? Or is it something else than the prediction. 
        xt_1_optimum = self.reverse_optimum_step(self.state, self.state_0, timesteps)
        # TODO: Use exactly the same loss function as IR-SDE project and see if the loss is the same. If not, we need to investigate why. loss_fn in denoising_model.py leads to loss.py
        loss = self.loss_fn(xt_1_expection, xt_1_optimum) # metrics.mse(xt_1_expection.permute(0, 2, 3, 1).flatten(1, 2), xt_1_optimum.permute(0, 2, 3, 1).flatten(1, 2), pred_std=torch.tensor(1, device=LQ.device)) # Equal weights for all variables, maybe we should use the same weights as in the baseline model
        
        print(f"xt_1_exp: {xt_1_expection.shape}, xt_1_opt: {xt_1_optimum.shape}, loss: {loss}")

        return xt_1_expection.permute(0, 2, 3, 1).flatten(1, 2), None, loss
    
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
                pred_std = torch.tensor(1, device=LQ.device) # Using the same weights for all variables
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
        loss_list = []

        for i in range(pred_steps):
            # forcing = forcing_features[:, i]
            # border_state = boundary_forcing[:, i]
            # true_state = true_states[:, i]

            pred_state, pred_std, loss = self.predict_step_train(LQ, HQ)

            prediction_list.append(pred_state)
            loss_list.append(loss)
            if self.output_std:
                pred_std_list.append(pred_std)

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        loss = torch.stack(
            loss_list, dim=0 # Changed to dim=0
        )  # (B, pred_steps, num_grid_nodes, d_f)

        if self.output_std:
            # pred_std = torch.stack(
            #     pred_std_list, dim=1
            # )  # (B, pred_steps, num_grid_nodes, d_f)
            pred_std = torch.tensor(1, device=LQ.device) # Using the same weights for all variables
        else:
            pred_std = self.per_var_std # (d_f,)
            # pred_std = 1 # Testing equal weights

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

        target = HQ.permute(0, 2, 3, 1).contiguous().flatten(1, 2).unsqueeze(1) # (B, pred_steps, N_grid, d_f)

        return prediction, target, pred_std, loss

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, pred_std, loss = self.common_step_train(batch)
        print("Training step")
        print(f"Prediction: {prediction.shape}, Target: {target.shape}, Loss: {loss.shape}")

        # Compute loss
        batch_loss = torch.mean(loss)  # mean over unrolled times and batch

        batch_mse = torch.mean(
            metrics.mse(
                prediction, target, pred_std
            )
        )  # mean over unrolled times and batch

        print(f"Batch loss: {batch_loss}, Batch mse: {batch_mse}")

        log_dict = {"train_loss": batch_loss, "train_mse": batch_mse, "loss": batch_loss} # TODO: Remove "loss" from log_dict if we can see that we get the same results as before
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
            traj_stds = self.per_var_std[constants.USED_PARAMS]
        
        return traj_means, traj_stds
    
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

        target_states = HQ.permute(0, 2, 3, 1).contiguous().flatten(1, 2).unsqueeze(1)

        # Rescale to original data scale
        traj_rescaled = trajectories * self.data_std[constants.USED_PARAMS] + self.data_mean[constants.USED_PARAMS]
        target_rescaled = target_states * self.data_std[constants.USED_PARAMS] + self.data_mean[constants.USED_PARAMS]

        # Compute mean and std of ensemble
        ens_mean = torch.mean(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        ens_std = torch.std(
            traj_rescaled, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)

        # Iterate over the examples
        for traj_slice, target_slice, ens_mean_slice, ens_std_slice in zip(
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
                torch.save(ens_mean_slice[0], f"output/example_ens_mean_{self.plotted_examples}.pt")
                torch.save(ens_std_slice[0], f"output/example_ens_std_{self.plotted_examples}.pt")
                torch.save(traj_slice[0], f"output/example_ens_members_{self.plotted_examples}.pt")
                torch.save(target_slice[0], f"output/example_target_{self.plotted_examples}.pt")

                # Save files to wandb
                if self.save_output_wandb:
                    wandb.save(f"output/example_ens_mean_{self.plotted_examples}.pt")
                    wandb.save(f"output/example_ens_std_{self.plotted_examples}.pt")
                    wandb.save(f"output/example_ens_members_{self.plotted_examples}.pt")
                    wandb.save(f"output/example_target_{self.plotted_examples}.pt")

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
            for t_i, (samples_t, target_t, ens_mean_t, ens_std_t) in enumerate(
                zip(
                    traj_slice.transpose(0, 1),
                    # (pred_steps, S, num_grid_nodes, d_f)
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
                        samples_t[:, :, var_i],
                        target_t[:, var_i],
                        ens_mean_t[:, var_i],
                        ens_std_t[:, var_i],
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
        trajectories: (B, n_ens, d_f, X, Y)
        traj_stds: (B, n_ens, d_f, X, Y)
        target_states: (B, d_f, X, Y)
        spread_squared_batch: (B, d_f)
        ens_mse_batch: (B, d_f)
        """
        # Compute and store metrics for ensemble forecast
        LQ, HQ = batch["LQ"], batch["HQ"]

        target_states = HQ.permute(0, 2, 3, 1).contiguous().flatten(1, 2).unsqueeze(1) # (B, pred_steps, d_f)

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

    def validation_step(self, batch, *args):
        """
        Run validation on single batch
        """
        print("No validation step implemented!")

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

        crps_batch = metrics.crps_ens( # CRPS want the shape [B, n_ens, pred_steps, dim, d_f]
            trajectories,
            target_states,
            traj_stds, # d_f
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
            denoised = self.forward(x_hat, t_hat, class_labels=class_labels)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.forward(x_next, t_next, class_labels=class_labels)
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
            denoised = self.forward(x_cur, t_cur, class_labels=class_labels)
            d_cur = (x_cur - denoised) / t_cur      
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.forward(x_next, t_next, class_labels=class_labels)
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

                u = sigma_mid / sigma_hat * x - (torch.exp(-r * h) - 1) * denoised
                denoised_2 = self.forward(u, sigma_mid, class_labels=class_labels)
                D = (1 - 1 / (2 * r)) * denoised + 1 / (2 * r) * denoised_2
                x = sigmas[i + 1] / sigma_hat * x - (torch.exp(-h) - 1) * D

        return x, None
        

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        return self.model(x, sigma, class_labels=class_labels, **model_kwargs)
    

# Loss function from IR-SDE paper
class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()
