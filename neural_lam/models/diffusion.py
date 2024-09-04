import torch
from torch.nn.functional import silu

from neural_lam.models.ar_model import ARModel
from neural_lam import constants
from neural_lam.models.EDM_networks import EDMPrecond, GroupNorm, PositionalEmbedding, FourierEmbedding, Linear
from neural_lam.models.hi_lam import HiLAM
from neural_lam.models.graph_lam import GraphLAM

class Diffusion(ARModel):
    """
    A new auto-regressive weather forecasting model
    """
    def __init__(self, args):
        super().__init__(args)

        # Some dimensionalities that can be useful to have stored
        self.border_condition = args.border_condition
        self.input_dim = 3*constants.grid_state_dim + constants.grid_forcing_dim +\
            constants.batch_static_feature_dim
        self.output_dim = constants.grid_state_dim
        
        self.sigma_min = 0.02
        self.sigma_max = 88
        self.sigma_data = 1
        self.use_fp16 = False

        if args.diffusion_model == 'EDM':
            raise ValueError(f"Diffusion model {args.diffusion_model} is not implemented")
            # self.model = EDMPrecond(img_resolution=268, in_channels=self.input_dim, out_channels=self.output_dim, \
            #                         embedding_type='fourier', encoder_type='standard', decoder_type='standard', \
            #                         channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=32, channel_mult=[2,4,8], \
            #                         attn_resolutions=[32,], sigma_data=1, sigma_min=0.02, sigma_max=88, label_dropout=0,
            #                         model_type='SongUNet', use_fp16=True) # TODO: Check out time_emb = 1 in Martins repo
        elif args.diffusion_model == 'hi_lam':
            self.model = HiLAM(args)
        
        elif args.diffusion_model == 'graph_lam':
            self.model = GraphLAM(args)
        else:
            raise ValueError(f"Diffusion model {args.diffusion_model} not recognized")
            
        self.available_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_residual = args.pred_residual # Whether to predict the residual instead of the next state

    # TODO add predict_step for val/test, generate_ensamble_from_batch
    def predict_step(self, prev_state, prev_prev_state, forcing, border_state):
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

        latents = torch.randn_like(input_grid[:, :, :17]).to(self.available_device)

        # Add border condition
        if self.border_condition:
            latents = self.border_mask * border_state + self.interior_mask * latents

        # Run through sampler
        next_state = self.heun_sampler(self, latents, input_grid)

        # Add residual if needed
        if self.pred_residual:
            next_state = prev_state + next_state
        
        return next_state, None

    def predict_step_train(self, prev_state, prev_prev_state, forcing, true_states=None):
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

        # Reshape 1d grid to 2d image
        input_grid = torch.cat((prev_state, prev_prev_state, forcing), dim=-1)

        # Sample from F inverse
        rnd_uniform = torch.rand([true_states.shape[0], 1, 1], device=true_states.device)
        rho = 7
        sigma_min = 0.02
        sigma_max = 88
        rho_inv = 1 / rho
        sigma_max_rho = sigma_max ** rho_inv
        sigma_min_rho = sigma_min ** rho_inv
        sigma = (sigma_max_rho + rnd_uniform * (sigma_min_rho - sigma_max_rho)) ** rho
        y = true_states # (B, 1, N_grid, d_input)
        y = y.flatten(1,2) # (B, N_grid, d_input)
  
        n = torch.randn_like(y) * sigma
        # Add border condition
        if self.border_condition:
            noisy_input = y + self.interior_mask * n # Only add noise inside of the border
        else:
            noisy_input = y+n

        next_state = self.forward(noisy_input, sigma, input_grid) # Shape (B, d_state, N_x, N_y)

        # Add residual if needed
        if self.pred_residual:
            next_state = prev_state + next_state

        return next_state, None
    
    def unroll_prediction(self, init_states, forcing_features, true_states):
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
                border_state = true_states[:, i]

                pred_state, pred_std = self.predict_step(
                    prev_state, prev_prev_state, forcing, border_state*self.border_mask
                )
                # state: (B, num_grid_nodes, d_f)
                # pred_std: (B, num_grid_nodes, d_f) or None

                # Overwrite border with true state
                new_state = (
                    self.border_mask * border_state
                    + self.interior_mask * pred_state
                )

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
                pred_std = torch.stack(
                    pred_std_list, dim=1
                )  # (B, pred_steps, num_grid_nodes, d_f)
            else:
                pred_std = self.per_var_std  # (d_f,)

            return prediction, pred_std


    def unroll_prediction_train(self, init_states, forcing_features, true_states):
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
            border_state = true_states[:, i]

            pred_state, pred_std = self.predict_step_train(
                prev_state, prev_prev_state, forcing, true_states
            )
            # state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            # Overwrite border with true state
            new_state = (
                self.border_mask * border_state
                + self.interior_mask * pred_state
            )

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
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std

    def common_step_train(self, batch):
        """
        Predict on single batch
        batch consists of:
        init_states: (B, 2, num_grid_nodes, d_features)
        target_states: (B, pred_steps, num_grid_nodes, d_features)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        (
            init_states,
            target_states,
            forcing_features,
        ) = batch

        prediction, pred_std = self.unroll_prediction_train(
            init_states, forcing_features, target_states
        )  # (B, pred_steps, num_grid_nodes, d_f)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        return prediction, target_states, pred_std

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, pred_std = self.common_step_train(batch)

        # Compute loss
        batch_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            )
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return batch_loss
    
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

    def edm_sampler(self,
        net, latents, class_labels=None, time_labels=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    ):
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = net(x_hat, t_hat, class_labels, time_labels)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = net(x_next, t_next, class_labels, time_labels)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    #----------------------------------------------------------------------------
    # Proposed Heun sampler (Algorithm 1).
    def heun_sampler(self,
        net, latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    ):
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        if self.border_condition: # Only add noise inside the border
            latents = latents * self.border_mask + latents * t_steps[0] * self.interior_mask
        else:
            latents = latents * t_steps[0]

        x_next = latents
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Euler step.
            denoised = net.forward(x_cur, t_cur, class_labels)
            d_cur = (x_cur - denoised) / t_cur
            if self.border_condition: # Only add noise inside the border
                x_next = x_cur * self.border_mask + (x_cur + (t_next - t_cur) * d_cur) * self.interior_mask
            else:       
                x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = net.forward(x_next, t_next, class_labels)
                d_prime = (x_next - denoised) / t_next

            if self.border_condition: # Only add noise inside the border
                x_next = x_cur * self.border_mask + (x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)) * self.interior_mask
            else:       
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

#----------------------------------------------------------------------------

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        model_input = x * self.border_mask + (c_in * x) * self.interior_mask # Only add noise inside the border
        F_x = self.model_forward((model_input).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x
    
    def model_forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        # emb = self.map_noise(noise_labels)
        # emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        # if self.map_label is not None:
        #     tmp = class_labels
        #     if self.training and self.label_dropout:
        #         tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
        #     emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        # if self.map_augment is not None and augment_labels is not None:
        #     emb = emb + self.map_augment(augment_labels)
        # emb = silu(self.map_layer0(emb))
        # emb = silu(self.map_layer1(emb))

        # # In each UNET Block UNET(x, emb)
        # params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        # if self.adaptive_scale:
        #     scale, shift = params.chunk(chunks=2, dim=1)
        #     x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        # else:
        #     x = silu(self.norm1(x.add_(params)))
        #__________________________________________
        # model_input = torch.cat((x, class_labels), dim=1)
        output, _ = self.model.predict_step(x, class_labels[:, :, :34], class_labels[:, :, 34:])
        return output
    
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
