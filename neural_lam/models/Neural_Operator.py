import torch
import numpy as np
import pandas as pd

from neural_lam.models.ar_model import ARModel
from neural_lam import constants
from neuralop.models import FNO, SFNO, UNO

class Neural_Operator(ARModel):
    """
    A new auto-regressive weather forecasting model
    """
    def __init__(self, args):
        super().__init__(args)

        # Some dimensionalities that can be useful to have stored
        self.border_condition = args.border_condition
        num_states = 3 if self.border_condition else 2
        self.input_dim = num_states*constants.grid_state_dim + constants.grid_forcing_dim +\
            constants.batch_static_feature_dim
        self.output_dim = constants.grid_state_dim
        
        if args.neural_operator == 'FNO':
            print("Using FNO")
            self.operator = FNO(n_modes=(16, 16), hidden_channels=64,
                    in_channels=self.input_dim, out_channels=self.output_dim, layers=4)
        elif args.neural_operator == 'SFNO':
            print("Using SFNO")
            self.operator = SFNO(n_modes=(16, 16), hidden_channels=64,
                    in_channels=self.input_dim, out_channels=self.output_dim)
        elif args.neural_operator == 'UNO':
            print("Using UNO")
            self.operator = UNO(hidden_channels=64, in_channels=self.input_dim,
                    out_channels=self.output_dim)
        
        # Whether to predict the residual instead of the next state
        self.pred_residual = args.pred_residual


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

        # Reshape 1d grid to 2d image
        if self.border_condition:
            input_flat = torch.cat((prev_state, prev_prev_state, forcing, border_state), dim=-1)
        else:
            input_flat = torch.cat((prev_state, prev_prev_state, forcing), dim=-1) # (B, N_grid, d_input)
        input_grid = torch.reshape(input_flat, (-1, *constants.grid_shape,
            input_flat.shape[2])) # (B, N_x, N_y, d_input)
        # Most computer vision methods in torch want channel dimension first
        input_grid = input_grid.permute((0,3,1,2)).contiguous() # (B, d_input, N_x, N_y)

        # TODO: Feed input_grid through some model to predict output_grid
        output_grid = self.operator(input_grid) # Shape (B, d_state, N_x, N_y)

        # Reshape back from 2d to flattened grid dimension
        output_grid = output_grid.permute((0,2,3,1)) # (B, N_x, N_y, d_state)
        next_state = output_grid.flatten(1,2) # (B, N_grid, d_state)

        # print(f"forcing shape: {forcing.shape}")
        # df = pd.DataFrame(torch.reshape(forcing, (-1, *constants.grid_shape[::-1],
        #     forcing.shape[2])).permute((0,3,1,2)).contiguous().cpu()[0, 0, :, :])
        # print(f"df shape: {df.shape}")
        # df.to_csv('forcing.csv', index=False)

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
