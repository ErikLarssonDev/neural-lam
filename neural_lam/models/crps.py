from neural_lam.models.ar_prob_model import ARProbModel
from neural_lam.models.edm_networks_2 import SongUNet

class CRPS(ARProbModel):
    """
    A crps based probabilistic auto-regressive weather forecasting model
    """
    def __init__(self, args):
        super().__init__(args)

    # TODO: Define backbone model architecture here
        self.model = SongUNet(
            img_resolution=torch.as_tensor(constants.FULL_GRID_SHAPE),
            in_channels=self.grid_dim,
            out_channels=self.grid_output_dim,
            embedding_type="linear",
            resample_filter=args.resample_filter,
            channel_mult=args.channel_mult,
            encoder_type=args.encoder_type,
            attn_resolutions=args.attn_resolutions,
            noise_dim=args.noise_dim,
        )

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

        x = torch.cat((prev_state, prev_prev_state, forcing), dim=-1) # (B, N_grid, d_input)

        z = torch.randn(prev_state.shape[0], self.model.noise_dim, device=prev_state.device)
        next_state = self.model(x, z, boundary_forcing=boundary_forcing)
        
        return next_state, None
    
    def training_step(self, batch):
        """
        Train on single batch
        """
        init_states, target_states, forcing_features = batch

        # Sample trajectories using prior
        pred_traj_means, pred_traj_stds = self.sample_trajectories(
            init_states,
            forcing_features,
            target_states,
            2, # TODO: Always sample 2 trajectories?
        )
        # (B, S=2, pred_steps, num_grid_nodes, d_f), always 2 samples

        # Compute CRPS
        crps_estimate = metrics.crps_ens( # TODO: Should we use this CRPS version?
            pred_traj_means,
            target_states,
            pred_traj_stds,
            # mask=self.interior_mask_bool,
        )  # (B, pred_steps)
        loss = torch.mean(crps_estimate)


        log_dict["train_loss"] = loss
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
