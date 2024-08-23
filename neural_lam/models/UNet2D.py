import torch

from neural_lam.models.ar_model import ARModel
from neural_lam import constants
from diffusers import UNet2DModel, schedulers

class UNet2D(ARModel):
    """
    A new auto-regressive weather forecasting model
    """
    def __init__(self, args, in_channels=2*constants.grid_state_dim + constants.grid_forcing_dim + constants.batch_static_feature_dim, out_channels=constants.grid_state_dim):
        super().__init__(args)

        # Some dimensionalities that can be useful to have stored
        self.input_dim = 2*constants.grid_state_dim + constants.grid_forcing_dim +\
            constants.batch_static_feature_dim
        self.output_dim = constants.grid_state_dim

        # TODO: Define modules as members here that will be used in predict_step
        self.layer = UNet2DModel()# (in_channels=50, out_channels=50)
        self.scheduler = schedulers.DDPMScheduler()

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

        # Reshape 1d grid to 2d image
        input_flat = torch.cat((prev_state, prev_prev_state, forcing), dim=-1) # (B, N_grid, d_input)
        input_grid = torch.reshape(input_flat, (-1, *constants.grid_shape,
            input_flat.shape[2])) # (B, N_x, N_y, d_input)
        # Most computer vision methods in torch want channel dimension first
        input_grid = input_grid.permute((0,3,1,2)).contiguous() # (B, d_input, N_x, N_y)
      
        # TODO: Feed input_grid through some model to predict output_grid
        noise = torch.randn_like(input_grid)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (input_grid.size(0),), device=self.device)
        print(f"steps: {steps}")
        noisy_grid = self.scheduler.add_noise(input_grid, noise, steps)
        print(f"Noisy grid shape: {noisy_grid.shape}")
        output_grid = self.layer(noisy_grid, steps).sample # Shape (B, d_state, N_x, N_y)
        print(f"Output grid shape: {output_grid.shape}")
        print(f"Output grid: {output_grid}")

        # Reshape back from 2d to flattened grid dimension
        output_grid = output_grid.permute((0,2,3,1)) # (B, N_x, N_y, d_state)
        next_state = output_grid.flatten(1,2) # (B, N_grid, d_state)

        return next_state, None
