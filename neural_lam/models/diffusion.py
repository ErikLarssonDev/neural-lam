import torch

from neural_lam.models.ar_model import ARModel
from neural_lam import constants
from neural_lam.models.EDM_networks import EDMPrecond

class Diffusion(ARModel):
    """
    A new auto-regressive weather forecasting model
    """
    def __init__(self, args):
        super().__init__(args)

        # Some dimensionalities that can be useful to have stored
        self.input_dim = 2*constants.grid_state_dim + constants.grid_forcing_dim +\
            constants.batch_static_feature_dim
        self.output_dim = constants.grid_state_dim

        # TODO: Define modules as members here that will be used in predict_step
        self.model = EDMPrecond(img_resolution=64, in_channels=self.output_dim, out_channels=self.output_dim, \
                                embedding_type='fourier', encoder_type='standard', decoder_type='standard', \
                                channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=32, channel_mult=[2,4,8], \
                                attn_resolutions=[32,], sigma_data=1, sigma_min=0.02, sigma_max=88, label_dropout=0,
                                model_type='SongUNet') # TODO: Check out time_emb = 1 in Martins repo
        

        # self.model = EDMPrecond(img_resolution=268, in_channels=self.output_dim, out_channels=self.output_dim,
        #                         model_type='SongUNet')

        # Martins model, use diffusion_networks.py
        # self.model = EDMPrecond(filters=32, img_channels=self.output_dim, out_channels=self.output_dim,
        #                         img_resolution = 268,
        #                         time_emb=1,
        #                         model_type="attention",
        #                         sigma_data=1,
        #                         sigma_min=0.02,
        #                         sigma_max=88,
        #                         label_dropout=0)

   
    # Autoencoder --> 256x128 --> EDM --> Autoencoder --> 268x238     
        

    # TODO add predict_step for val/test, generate_ensamble_from_batch
    def predict_step(self, prev_state, prev_prev_state, forcing):
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
        print("predict_step!!!")
        return prev_state, None

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
        print(f"prev_state: {prev_state.shape}")
        print(f"prev_prev_state: {prev_prev_state.shape}")
        print(f"forcing: {forcing.shape}")
        print(f"true_states: {true_states.shape}")
        input_flat = torch.cat((prev_state, prev_prev_state, forcing), dim=-1) # (B, N_grid, d_input)
        print(f"input_flat: {input_flat.shape}")
        input_grid = torch.reshape(input_flat, (-1, *constants.grid_shape,
            input_flat.shape[2])) # (B, N_x, N_y, d_input)
        print(f"input_grid: {input_grid.shape}")
        # Most computer vision methods in torch want channel dimension first
        input_grid = input_grid.permute((0,3,1,2)).contiguous() # (B, d_input, N_x, N_y)

        # Sample from F inverse
        rnd_uniform = torch.rand([true_states.shape[0], 1, 1, 1], device=true_states.device)
        rho = 7
        sigma_min = 0.02
        sigma_max = 88
        rho_inv = 1 / rho
        sigma_max_rho = sigma_max ** rho_inv
        sigma_min_rho = sigma_min ** rho_inv
        sigma = (sigma_max_rho + rnd_uniform * (sigma_min_rho - sigma_max_rho)) ** rho
        y = true_states # (B, 1, N_grid, d_input)
        y = y.flatten(1,2) # (B, N_grid, d_input)
        y = torch.reshape(y, (-1, *constants.grid_shape, y.shape[2])) # (B, N_x, N_y, d_input)
        y = y.permute((0,3,1,2)).contiguous() # (B, d_input, N_x, N_y)
        y = y[:, :, :256, :128] # For some reason it does only work with multiples of 2 for the resolution params which 268x238 is not
        n = torch.randn_like(y) * sigma
        class_labels = input_grid
        print(f"y: {y.shape}")
        print(f"n: {n.shape}")
        print(f"sigma: {sigma.shape}")
        print(f"class_labels: {class_labels.shape}")

        # TODO Run y, x_0, x_(t-1) through encoder to 64x64 dimension
        

        # TODO: Feed input_grid through some model to predict output_grid
        output_grid = self.model(y+n, sigma) # Shape (B, d_state, N_x, N_y) TODO: Add class labels to condition on
        print(f"output_grid: {output_grid.shape}")
        # Reshape back from 2d to flattened grid dimension
        output_grid = output_grid.permute((0,2,3,1)) # (B, N_x, N_y, d_state)
        next_state = output_grid.flatten(1,2) # (B, N_grid, d_state)

        return next_state, None

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