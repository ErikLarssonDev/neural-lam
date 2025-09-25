# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import pytorch_lightning as pl
import os

# First-party
from neural_lam import constants, metrics, utils, vis, config
from neural_lam.models.ar_model import ARModel
from neural_lam.models.constant_latent_encoder import ConstantLatentEncoder
from neural_lam.models.graph_latent_decoder import GraphLatentDecoder
from neural_lam.models.graph_latent_encoder import GraphLatentEncoder
from neural_lam.models.hi_graph_latent_decoder import HiGraphLatentDecoder
from neural_lam.models.hi_graph_latent_encoder import HiGraphLatentEncoder
from neural_lam.interaction_net_old import make_gnn_seq


class GraphEFM(pl.LightningModule):
    """
    Graph-based Ensemble Forecasting Model
    """

    def __init__(self, args):
        super().__init__()
        self.save_output = args.save_output

        # ARModel
        self.save_hyperparameters()
        self.lr = args.lr
        self.config_loader = config.Config.from_file(args.data_config)

        # Load static features for grid/data
        static_data_dict = utils.load_static_data_old(
            self.config_loader.dataset.name, self.config_loader.dataset.data_path
        )
        for static_data_name, static_data in static_data_dict.items():
            if isinstance(static_data, torch.Tensor):
                self.register_buffer(
                    static_data_name, static_data, persistent=False
                )
            else:
                # Non-tensor static can not and should not be buffers
                setattr(self, static_data_name, static_data)

        # Double grid output dim. to also output std.-dev.
        self.output_std = bool(args.output_std)
        if self.output_std:
            self.grid_output_dim = (
                2 * constants.GRID_STATE_DIM
            )  # Pred. dim. in grid cell
        else:
            self.grid_output_dim = (
                constants.GRID_STATE_DIM
            )  # Pred. dim. in grid cell

            # Store constant per-variable std.-dev. weighting
            # Note that this is the inverse of the multiplicative weighting
            # in wMSE/wMAE
            self.register_buffer(
                "per_var_std",
                self.step_diff_std / torch.sqrt(self.param_weights),
                persistent=False,
            )

        # grid_dim from data + static
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape  # 63784 = 268x238

        print(f"Loaded grid with {self.num_grid_nodes} nodes"
              f" and {grid_static_dim} static features")
        self.grid_dim = (
            2 * constants.GRID_STATE_DIM
            + grid_static_dim
            + constants.GRID_FORCING_DIM
        )

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

        # Pre-compute interior mask for use in loss function
        self.register_buffer(
            "interior_mask", 1.0 - self.border_mask, persistent=False
        )

        self.step_length = args.step_length  # Number of hours per pred. step
        self.val_metrics = {
            "mse": [],
        }
        self.test_metrics = {
            "mse": [],
            "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

        # Graph-EFM
        assert (
            args.n_example_pred <= args.batch_size
        ), "Can not plot more examples than batch size in GraphEFM"
        self.sample_obs_noise = bool(args.sample_obs_noise)
        self.ensemble_size = args.ensemble_size
        self.kl_beta = args.kl_beta
        self.crps_weight = args.crps_weight

        # Load graph with static features
        self.hierarchical_graph, graph_ldict = utils.load_graph(args.graph)
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        # grid_dim from data + static
        grid_current_dim = self.grid_dim + constants.GRID_STATE_DIM
        g2m_dim = self.g2m_features.shape[1]
        m2g_dim = self.m2g_features.shape[1]

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.grid_prev_embedder = utils.make_mlp_old(
            [self.grid_dim] + self.mlp_blueprint_end
        )  # For states up to t-1
        self.grid_current_embedder = utils.make_mlp_old(
            [grid_current_dim] + self.mlp_blueprint_end
        )  # For states including t
        # Embedders for mesh
        self.g2m_embedder = utils.make_mlp_old(
            [g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp_old(
            [m2g_dim] + self.mlp_blueprint_end)
        if self.hierarchical_graph:
            # Print some useful info
            print("Loaded hierarchical graph with structure:")
            level_mesh_sizes = [
                mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
            ]
            self.num_mesh_nodes = level_mesh_sizes[-1]
            num_levels = len(self.mesh_static_features)
            for level_index, level_mesh_size in enumerate(level_mesh_sizes):
                same_level_edges = self.m2m_features[level_index].shape[0]
                print(
                    f"level {level_index} - {level_mesh_size} nodes, "
                    f"{same_level_edges} same-level edges"
                )

                if level_index < (num_levels - 1):
                    up_edges = self.mesh_up_features[level_index].shape[0]
                    down_edges = self.mesh_down_features[level_index].shape[0]
                    print(f"  {level_index}<->{level_index+1}")
                    print(f" - {up_edges} up edges, {down_edges} down edges")
            # Embedders
            # Assume all levels have same static feature dimensionality
            mesh_dim = self.mesh_static_features[0].shape[1]
            m2m_dim = self.m2m_features[0].shape[1]
            mesh_up_dim = self.mesh_up_features[0].shape[1]
            mesh_down_dim = self.mesh_down_features[0].shape[1]

            # Separate mesh node embedders for each level
            self.mesh_embedders = torch.nn.ModuleList(
                [
                    utils.make_mlp_old([mesh_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels)
                ]
            )
            self.mesh_up_embedders = torch.nn.ModuleList(
                [
                    utils.make_mlp_old([mesh_up_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels - 1)
                ]
            )
            self.mesh_down_embedders = torch.nn.ModuleList(
                [
                    utils.make_mlp_old([mesh_down_dim] +
                                       self.mlp_blueprint_end)
                    for _ in range(num_levels - 1)
                ]
            )
            # If not using any processor layers, no need to embed m2m
            self.embedd_m2m = (
                max(
                    args.prior_processor_layers,
                    args.encoder_processor_layers,
                    args.processor_layers,
                )
                > 0
            )
            if self.embedd_m2m:
                self.m2m_embedders = torch.nn.ModuleList(
                    [
                        utils.make_mlp_old([m2m_dim] + self.mlp_blueprint_end)
                        for _ in range(num_levels)
                    ]
                )
        else:
            self.num_mesh_nodes, mesh_static_dim = (
                self.mesh_static_features.shape
            )
            print(
                f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes}"
                f"nodes ({self.num_grid_nodes} grid, "
                f"{self.num_mesh_nodes} mesh)"
            )
            mesh_static_dim = self.mesh_static_features.shape[1]
            self.mesh_embedder = utils.make_mlp_old(
                [mesh_static_dim] + self.mlp_blueprint_end
            )
            m2m_dim = self.m2m_features.shape[1]
            self.m2m_embedder = utils.make_mlp_old(
                [m2m_dim] + self.mlp_blueprint_end
            )

        latent_dim = (
            args.latent_dim if args.latent_dim is not None else args.hidden_dim
        )
        # Prior
        if args.learn_prior:
            if self.hierarchical_graph:
                self.prior_model = HiGraphLatentEncoder(
                    latent_dim,
                    self.g2m_edge_index,
                    self.m2m_edge_index,
                    self.mesh_up_edge_index,
                    args.hidden_dim,
                    args.prior_processor_layers,
                    hidden_layers=args.hidden_layers,
                    output_dist=args.prior_dist,
                )
            else:
                self.prior_model = GraphLatentEncoder(
                    latent_dim,
                    self.g2m_edge_index,
                    self.m2m_edge_index,
                    args.hidden_dim,
                    args.prior_processor_layers,
                    hidden_layers=args.hidden_layers,
                    output_dist=args.prior_dist,
                )
        else:
            self.prior_model = ConstantLatentEncoder(
                latent_dim,
                self.num_mesh_nodes,
                output_dist=args.prior_dist,
            )

        # Enc. + Dec.
        if self.hierarchical_graph:
            # Encoder
            self.encoder = HiGraphLatentEncoder(
                latent_dim,
                self.g2m_edge_index,
                self.m2m_edge_index,
                self.mesh_up_edge_index,
                args.hidden_dim,
                args.encoder_processor_layers,
                hidden_layers=args.hidden_layers,
                output_dist="diagonal",
            )
            # Decoder
            self.decoder = HiGraphLatentDecoder(
                self.g2m_edge_index,
                self.m2m_edge_index,
                self.m2g_edge_index,
                self.mesh_up_edge_index,
                self.mesh_down_edge_index,
                args.hidden_dim,
                latent_dim,
                args.processor_layers,
                hidden_layers=args.hidden_layers,
                output_std=bool(args.output_std),
            )
        else:
            # Encoder
            self.encoder = GraphLatentEncoder(
                latent_dim,
                self.g2m_edge_index,
                self.m2m_edge_index,
                args.hidden_dim,
                args.encoder_processor_layers,
                hidden_layers=args.hidden_layers,
                output_dist="diagonal",
            )
            # Decoder
            self.decoder = GraphLatentDecoder(
                self.g2m_edge_index,
                self.m2m_edge_index,
                self.m2g_edge_index,
                args.hidden_dim,
                latent_dim,
                args.processor_layers,
                hidden_layers=args.hidden_layers,
                output_std=bool(args.output_std),
            )

        # Add lists for val and test errors of ensemble prediction
        self.val_metrics.update(
            {
                "spread_squared": [],
                "ens_mse": [],
            }
        )
        self.test_metrics.update(
            {
                "ens_mae": [],
                "ens_mse": [],
                "crps_ens": [],
                "spread_squared": [],
            }
        )

        self.spectra_metrics = {
            "spectra": [],  # For each time step and variable
            "spectra_gt": [],  # For each time step and variable
        }

    # ARModel
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95)
        )
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        return opt

    @property
    def interior_mask_bool(self):
        """
        Get the interior mask as a boolean (N,) mask.
        """
        return self.interior_mask[:, 0].to(torch.bool)

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

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
                prev_state, prev_prev_state, forcing
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

    def common_step(self, batch):
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

        prediction, pred_std = self.unroll_prediction(
            init_states, forcing_features, target_states
        )  # (B, pred_steps, num_grid_nodes, d_f)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        return prediction, target_states, pred_std

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0
        (instead of stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        return self.all_gather(tensor_to_gather).flatten(0, 1)

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Put together a dict with everything to log for one metric.
        Also saves plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging
        metric_name: string, name of the metric

        Return:
        log_dict: dict with everything to log for given metric
        """
        log_dict = {}
        metric_fig = vis.plot_error_map(
            metric_tensor, self.config_loader, step_length=self.step_length
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = wandb.Image(metric_fig)

        if prefix == "test":
            # Save pdf
            metric_fig.savefig(
                os.path.join(wandb.run.dir, f"{full_log_name}.pdf")
            )
            # Save errors also as csv
            np.savetxt(
                os.path.join(wandb.run.dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        # Check if metrics are watched, log exact values for specific vars
        if full_log_name in constants.METRICS_WATCH:
            for var_i, timesteps in constants.VAR_LEADS_METRICS_WATCH.items():
                var = constants.PARAM_NAMES_SHORT[var_i]
                log_dict.update(
                    {
                        f"{full_log_name}_{var}_step_{step}": metric_tensor[
                            step - 1, var_i
                        ]  # 1-indexed in constants
                        for step in timesteps
                    }
                )

        return log_dict

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        """
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )  # (N_eval, pred_steps, d_f)

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # (pred_steps, d_f)

                # Take square root after averaging to change squared metrics
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")
                elif metric_name.endswith("_squared"):
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name[: -len("_squared")]

                # Note: we here assume rescaling for all metrics is linear
                metric_rescaled = metric_tensor_averaged * self.data_std
                # (pred_steps, d_f)
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            wandb.log(log_dict)  # Log all
            plt.close("all")  # Close all figs

    def on_load_checkpoint(self, checkpoint):
        """
        Perform any changes to state dict before loading checkpoint
        """
        loaded_state_dict = checkpoint["state_dict"]
        # model_state_dict = self.state_dict()

        # # Print first 10 keys from checkpoint state dict
        # print(
        #     f"First 10 checkpoint keys: {list(loaded_state_dict.keys())[:10]}")
        # print(f"Total checkpoint keys: {len(loaded_state_dict)}")

        # # Print first 10 keys from model state dict
        # print(f"First 10 model keys: {list(model_state_dict.keys())[:10]}")
        # print(f"Total model keys: {len(model_state_dict)}")

        # # Print keys that are in model but not in checkpoint
        # missing_keys = [k for k in model_state_dict.keys()
        #                 if k not in loaded_state_dict]
        # print(f"First 10 missing keys: {missing_keys[:10]}")
        # print(f"Total missing keys: {len(missing_keys)}")

        # # Print keys that are in checkpoint but not in model
        # unexpected_keys = [
        #     k for k in loaded_state_dict.keys() if k not in model_state_dict]
        # print(f"First 10 unexpected keys: {unexpected_keys[:10]}")
        # print(f"Total unexpected keys: {len(unexpected_keys)}")

        # # Look for pattern differences in the keys
        # if len(missing_keys) > 0 and len(unexpected_keys) > 0:
        #     example_missing = missing_keys[0]
        #     example_unexpected = unexpected_keys[0]
        #     print(f"Example missing key: {example_missing}")
        #     print(f"Example unexpected key: {example_unexpected}")

        #     # Try to identify pattern differences
        #     if "mlp_layers" in example_missing and not "mlp_layers" in example_unexpected:
        #         print("The model uses 'mlp_layers' structure but checkpoint doesn't")
        #     elif not "mlp_layers" in example_missing and "mlp_layers" in example_unexpected:
        #         print("The checkpoint uses 'mlp_layers' structure but model doesn't")

        # Fix for loading older models after IneractionNet refactoring, where
        # the grid MLP was moved outside the encoder InteractionNet class
        if "g2m_gnn.grid_mlp.0.weight" in loaded_state_dict:
            replace_keys = list(
                filter(
                    lambda key: key.startswith("g2m_gnn.grid_mlp"),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace(
                    "g2m_gnn.grid_mlp", "encoding_grid_mlp"
                )
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]

    # Graph-EFM
    def sample_next_state(self, pred_mean, pred_std):
        """
        Sample state at next time step given Gaussian distribution.
        If self.sample_obs_noise is False, only return mean.

        pred_mean: (B, num_grid_nodes, d_state)
        pred_std: (B, num_grid_nodes, d_state) or
            None (if not output_std)

        Return:
        next_state: (B, num_grid_nodes, d_state)
        """
        if not self.output_std:
            pred_std = self.per_var_std  # (d_f,)

        if self.sample_obs_noise:
            return torch.distributions.Normal(pred_mean, pred_std).rsample()
            # (B, num_grid_nodes, d_state)

        return pred_mean  # (B, num_grid_nodes, d_state)

    def embedd_current(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        current_state,
    ):
        """
        embed grid representation including current (target) state. Used as
        input to the encoder, which is conditioned also on the target.

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        current_state: (B, num_grid_nodes, feature_dim), X_{t+1}

        Returns:
        current_emb: (B, num_grid_nodes, d_h)
        """
        batch_size = prev_state.shape[0]

        grid_current_features = torch.cat(
            (
                prev_prev_state,
                prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
                current_state,
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_current_dim)

        return self.grid_current_embedder(
            grid_current_features
        )  # (B, num_grid_nodes, d_h)

    def embedd_all(self, prev_state, prev_prev_state, forcing):
        """
        embed all node and edge representations

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)

        Returns:
        grid_emb: (B, num_grid_nodes, d_h)
        graph_embedding: dict with entries of shape (B, *, d_h)
        """
        batch_size = prev_state.shape[0]

        grid_features = torch.cat(
            (
                prev_prev_state,
                prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_dim)

        grid_emb = self.grid_prev_embedder(grid_features)
        # (B, num_grid_nodes, d_h)

        # Graph embedding
        graph_emb = {
            "g2m": self.expand_to_batch(
                self.g2m_embedder(self.g2m_features), batch_size
            ),  # (B, M_g2m, d_h)
            "m2g": self.expand_to_batch(
                self.m2g_embedder(self.m2g_features), batch_size
            ),  # (B, M_m2g, d_h)
        }

        if self.hierarchical_graph:
            graph_emb["mesh"] = [
                self.expand_to_batch(emb(node_static_features), batch_size)
                for emb, node_static_features in zip(
                    self.mesh_embedders,
                    self.mesh_static_features,
                )
            ]  # each (B, num_mesh_nodes[l], d_h)

            if self.embedd_m2m:
                graph_emb["m2m"] = [
                    self.expand_to_batch(emb(edge_feat), batch_size)
                    for emb, edge_feat in zip(
                        self.m2m_embedders, self.m2m_features
                    )
                ]
            else:
                # Need a placeholder otherwise, just use raw features
                graph_emb["m2m"] = list(self.m2m_features)

            graph_emb["mesh_up"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_up_embedders, self.mesh_up_features
                )
            ]
            graph_emb["mesh_down"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_down_embedders, self.mesh_down_features
                )
            ]
        else:
            graph_emb["mesh"] = self.expand_to_batch(
                self.mesh_embedder(self.mesh_static_features), batch_size
            )  # (B, num_mesh_nodes, d_h)
            graph_emb["m2m"] = self.expand_to_batch(
                self.m2m_embedder(self.m2m_features), batch_size
            )  # (B, M_m2m, d_h)

        return grid_emb, graph_emb

    def compute_step_loss(
        self,
        prev_states,
        current_state,
        forcing_features,
    ):
        """
        Perform forward pass and compute loss for one time step

        prev_states: (B, 2, num_grid_nodes, d_features), X^{t-p}, ..., X^{t-1}
        current_state: (B, num_grid_nodes, d_features) X^t
        forcing_features: (B, num_grid_nodes, d_forcing) corresponding to
            index 1 of prev_states
        """
        # embed all features
        grid_prev_emb, graph_emb = self.embedd_all(
            prev_states[:, 1],
            prev_states[:, 0],
            forcing_features,
        )
        # embed also including current grid state, for encoder
        grid_current_emb = self.embedd_current(
            prev_states[:, 1],
            prev_states[:, 0],
            forcing_features,
            current_state,
        )  # (B, num_grid_nodes, d_h)

        # Compute variational approximation (encoder)
        var_dist = self.encoder(
            grid_current_emb, graph_emb=graph_emb
        )  # Gaussian, (B, num_mesh_nodes, d_latent)

        # Compute likelihood
        last_state = prev_states[:, -1]
        likelihood_term, pred_mean, pred_std = self.estimate_likelihood(
            var_dist, current_state, last_state, grid_prev_emb, graph_emb
        )
        if self.kl_beta > 0:
            # Compute prior
            prior_dist = self.prior_model(
                grid_prev_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)

            # Compute KL
            kl_term = torch.sum(
                torch.distributions.kl_divergence(var_dist, prior_dist),
                dim=(1, 2),
            )  # (B,)
        else:
            # If beta=0, do not need to even compute prior nor KL
            kl_term = None  # Set to None to crash if erroneously used

        return likelihood_term, kl_term, pred_mean, pred_std

    def estimate_likelihood(
        self, latent_dist, current_state, last_state, grid_prev_emb, graph_emb
    ):
        """
        Estimate (masked) likelihood using given distribution over
        latent variables

        latent_dist: distribution, (B, num_mesh_nodes, d_latent)
        current_state: (B, num_grid_nodes, d_state)
        last_state: (B, num_grid_nodes, d_state)
        grid_prev_emb: (B, num_grid_nodes, d_state)
        g2m_emb: (B, M_g2m, d_h)
        m2m_emb: (B, M_m2m, d_h)
        m2g_emb: (B, M_m2g, d_h)

        Returns:
        likelihood_term: (B,)
        pred_mean: (B, num_grid_nodes, d_state)
        pred_std: (B, num_grid_nodes, d_state) or (d_state,)
        """
        # Sample from variational distribution
        latent_samples = latent_dist.rsample()  # (B, num_mesh_nodes, d_latent)

        # Compute reconstruction (decoder)
        pred_mean, model_pred_std = self.decoder(
            grid_prev_emb, latent_samples, last_state, graph_emb
        )  # both (B, num_grid_nodes, d_state)

        if self.output_std:
            pred_std = model_pred_std  # (B, num_grid_nodes, d_state)
        else:
            # Use constant set std.-devs.
            pred_std = self.per_var_std  # (d_f,)

        # Compute likelihood (negative loss, exactly likelihood for nll loss)
        # Note: There are some round-off errors here due to float32
        # and large values
        entry_likelihoods = -self.loss(
            pred_mean,
            current_state,
            pred_std,
            mask=self.interior_mask_bool,
            average_grid=False,
            sum_vars=False,
        )  # (B, num_grid_nodes', d_state)
        likelihood_term = torch.sum(entry_likelihoods, dim=(1, 2))  # (B,)
        return likelihood_term, pred_mean, pred_std

    def training_step(self, batch):
        """
        Train on single batch

        batch, containing:
        init_states: (B, 2, num_grid_nodes, d_state)
        target_states: (B, pred_steps, num_grid_nodes, d_state)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing), where
            index 0 corresponds to index 1 of init_states
        """
        init_states, target_states, forcing_features = batch

        prev_prev_state = init_states[:, 0]  # (B, num_grid_nodes, d_state)
        prev_state = init_states[:, 1]  # (B, num_grid_nodes, d_state)
        pred_steps = forcing_features.shape[1]

        loss_like_list = []
        loss_kl_list = []

        for i in range(pred_steps):
            forcing = forcing_features[:, i]  # (B, num_grid_nodes, d_forcing)
            target_state = target_states[:, i]  # (B, num_grid_nodes, d_state)

            prev_states_stacked = torch.stack(
                (prev_prev_state, prev_state), dim=1
            )  # (B, 2, num_grid_nodes, d_state)
            loss_like_term, loss_kl_term, pred_mean, pred_std = (
                self.compute_step_loss(
                    prev_states_stacked,
                    target_state,
                    forcing,
                )
            )
            # (B,), (B,), (B, num_grid_nodes, d_state),
            # pred_std is (B, num_grid_nodes, d_state) or (d_state)

            loss_like_list.append(loss_like_term)
            loss_kl_list.append(loss_kl_term)

            # Get predicted next state (sample or mean)
            predicted_state = self.sample_next_state(pred_mean, pred_std)

            # Overwrite border with true state
            new_state = (
                self.border_mask * target_state
                + self.interior_mask * predicted_state
            )

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        # Compute final ELBO and loss, sum over time, mean over batch
        per_sample_likelihood = torch.sum(
            torch.stack(loss_like_list, dim=1), dim=1
        )  # (B,)
        mean_likelihood = torch.mean(per_sample_likelihood)
        log_dict = {
            "elbo_likelihood": mean_likelihood,
        }

        if self.kl_beta > 0:
            # Only compute full KL + ELBO if beta > 0
            per_sample_kl = torch.sum(
                torch.stack(loss_kl_list, dim=1), dim=1
            )  # (B,)
            mean_kl = torch.mean(per_sample_kl)
            elbo = mean_likelihood - mean_kl
            loss = -mean_likelihood + self.kl_beta * mean_kl

            log_dict["elbo"] = elbo
            log_dict["elbo_kl"] = mean_kl
        else:
            # Pure auto-encoder training
            loss = -mean_likelihood

        # Optionally sample trajectories and compute CRPS loss
        if self.crps_weight > 0:
            # Sample trajectories using prior
            pred_traj_means, pred_traj_stds = self.sample_trajectories(
                init_states,
                forcing_features,
                target_states,
                2,
            )
            # (B, S=2, pred_steps, num_grid_nodes, d_f), always 2 samples

            # Compute CRPS
            crps_estimate = metrics.crps_ens(
                pred_traj_means,
                target_states,
                pred_traj_stds,
                mask=self.interior_mask_bool,
            )  # (B, pred_steps)
            crps_loss = torch.mean(crps_estimate)

            # Add onto loss
            loss = loss + self.crps_weight * crps_loss
            log_dict["crps_loss"] = crps_loss

        log_dict["train_loss"] = loss
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss

    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Sample one time step prediction

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)

        Returns:
        new_state: (B, num_grid_nodes, feature_dim)
        """
        # embed all features
        grid_prev_emb, graph_emb = self.embedd_all(
            prev_state, prev_prev_state, forcing
        )

        # Compute prior
        prior_dist = self.prior_model(
            grid_prev_emb, graph_emb=graph_emb
        )  # (B, num_mesh_nodes, d_latent)

        # Sample from prior
        latent_samples = prior_dist.rsample()
        # (B, num_mesh_nodes, d_latent)

        # Compute reconstruction (decoder)
        last_state = prev_state
        pred_mean, pred_std = self.decoder(
            grid_prev_emb, latent_samples, last_state, graph_emb
        )  # (B, num_grid_nodes, d_state)

        return self.sample_next_state(pred_mean, pred_std), pred_std

    def sample_trajectories(
        self,
        init_states,
        forcing_features,
        true_states,
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
        unroll_func = (
            self.unroll_prediction_vi if use_encoder else self.unroll_prediction
        )
        traj_list = [
            unroll_func(
                init_states,
                forcing_features,
                true_states,
            )
            for _ in range(num_traj)
        ]
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
            traj_stds = self.per_var_std

        return traj_means, traj_stds

    def unroll_prediction_vi(self, init_states, forcing_features, true_states):
        """
        Roll out prediction, sampling latent var. from variational
        encoder distribution

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
            # Compute 1-step prediction, but using encoder
            forcing = forcing_features[:, i]
            current_state = true_states[:, i]

            # embed all features
            grid_prev_emb, graph_emb = self.embedd_all(
                prev_state, prev_prev_state, forcing
            )

            # embed also including current grid state, for encoder
            grid_current_emb = self.embedd_current(
                prev_state,
                prev_prev_state,
                forcing,
                current_state,
            )

            # Compute variational distribution
            var_dist = self.encoder(
                grid_current_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)

            # Sample from var. dist.
            latent_samples = var_dist.rsample()
            # (B, num_mesh_nodes, d_latent)

            # Compute reconstruction (decoder)
            pred_mean, pred_std = self.decoder(
                grid_prev_emb, latent_samples, prev_state, graph_emb
            )  # (B, num_grid_nodes, d_state)

            pred_state = self.sample_next_state(pred_mean, pred_std)
            # pred_state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            # Overwrite border with true state
            new_state = (
                self.border_mask * current_state
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

    def plot_examples(self, batch, n_examples, prediction=None):
        """
        Plot ensemble forecast + mean and std
        """
        init_states, target_states, forcing_features = batch

        trajectories, _ = self.sample_trajectories(
            init_states,
            forcing_features,
            target_states,
            self.ensemble_size,
        )
        # (B, S, pred_steps, num_grid_nodes, d_f)

        # Rescale to original data scale
        traj_rescaled = trajectories * self.data_std + self.data_mean
        target_rescaled = target_states * self.data_std + self.data_mean

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
                # Save predictions to the output folder with wandb run ID
                if self.save_output:
                    # Get wandb run name/ID if available
                    run_id = wandb.run.id if wandb.run is not None else "no-wandb-run"
                    
                    # Create output directory with run ID
                    output_dir = f"output/{run_id}"
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(
                        ens_mean_slice[0], f"{output_dir}/example_ens_mean_{self.plotted_examples}.pt")
                    torch.save(
                        ens_std_slice[0], f"{output_dir}/example_ens_std_{self.plotted_examples}.pt")
                    torch.save(
                        traj_slice[0], f"{output_dir}/example_ens_members_{self.plotted_examples}.pt")
                    torch.save(
                        target_slice[0], f"{output_dir}/example_target_{self.plotted_examples}.pt")

                time_title_part = f"t={t_i} ({self.step_length*t_i} h)"
                # Create one figure per variable at this time step
                var_figs = [
                    vis.plot_ensemble_prediction(
                        samples_t[:, :, var_i],
                        target_t[:, var_i],
                        None,
                        ens_mean_t[:, var_i],
                        ens_std_t[:, var_i],
                        self.interior_mask[:, 0],
                        title=f"{var_name} ({var_unit}), {time_title_part}",
                        vrange=var_vrange,
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            constants.PARAM_NAMES_SHORT,
                            constants.PARAM_UNITS,
                            var_vranges,
                        )
                    )
                ]

                example_title = f"example_{self.plotted_examples}"
                wandb.log(
                    {
                        f"{var_name}_{example_title}": wandb.Image(fig)
                        for var_name, fig in zip(
                            constants.PARAM_NAMES_SHORT, var_figs
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
        init_states, target_states, forcing_features = batch

        trajectories, traj_stds = self.sample_trajectories(
            init_states,
            forcing_features,
            target_states,
            self.ensemble_size,
        )
        # (B, S, pred_steps, num_grid_nodes, d_f)

        spread_squared_batch = metrics.spread_squared(
            trajectories,
            target_states,
            traj_stds,
            mask=self.interior_mask_bool,
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
            mask=self.interior_mask_bool,
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
        prediction, target, pred_std = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            ),
            dim=0,
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
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
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)
        batch_idx = args[0]

        # Run ensemble forecast
        prior_trajectories, _, _, spread_squared_batch, ens_mse_batch = (
            self.ensemble_common_step(batch)
        )
        self.val_metrics["spread_squared"].append(spread_squared_batch)
        self.val_metrics["ens_mse"].append(ens_mse_batch)

        # Plot some example predictions using prior and encoder
        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.n_example_pred > 0
        ):
            # Roll out trajectories using variational distribution (encoder)
            (
                init_states,
                target_states,
                forcing_features,
            ) = batch
            # Only create ens. forecast for as many examples as needed
            init_states = init_states[: self.n_example_pred]
            target_states = target_states[: self.n_example_pred]
            forcing_features = forcing_features[: self.n_example_pred]

            # Sample trajectories using variational dist. for latent var.
            enc_trajectories, _ = self.sample_trajectories(
                init_states,
                forcing_features,
                target_states,
                self.ensemble_size,
                use_encoder=True,
            )

            # Only need n_example_pred prior trajectories
            prior_trajectories = prior_trajectories[: self.n_example_pred]

            # Plot samples
            log_plot_dict = {}
            for example_i, (prior_traj, enc_traj, target_traj) in enumerate(
                zip(prior_trajectories, enc_trajectories, target_states),
                start=1,
            ):
                # prior_traj and enc traj are
                # (S, pred_steps, num_grid_nodes, d_f)

                for var_i, timesteps in constants.VAL_PLOT_VARS.items():
                    var_name = constants.PARAM_NAMES_SHORT[var_i]
                    var_unit = constants.PARAM_UNITS[var_i]
                    for step in timesteps:
                        prior_states = prior_traj[
                            :, step - 1, :, var_i
                        ]  # (S, num_grid_nodes)
                        enc_states = enc_traj[
                            :, step - 1, :, var_i
                        ]  # (S, num_grid_nodes)
                        target_state = target_traj[
                            step - 1, :, var_i
                        ]  # (num_grid_nodes,)

                        plot_title = (
                            f"{var_name} ({var_unit}), t={step} "
                            f"({self.step_length*step} h)"
                        )

                        # Make plots
                        log_plot_dict[
                            f"prior_{var_name}_step_{step}_ex{example_i}"
                        ] = vis.plot_ensemble_prediction(
                            prior_states,
                            target_state,
                            prior_states.mean(dim=0),
                            prior_states.std(dim=0),
                            self.interior_mask[:, 0],
                            title=f"{plot_title} (prior)",
                        )
                        log_plot_dict[
                            f"vi_{var_name}_step_{step}_ex{example_i}"
                        ] = vis.plot_ensemble_prediction(
                            enc_states,
                            target_state,
                            enc_states.mean(dim=0),
                            enc_states.std(dim=0),
                            self.interior_mask[:, 0],
                            title=f"{plot_title} (vi)",
                        )

            # Sample latent variable and plot
            # embed all features
            grid_prev_emb, graph_emb = self.embedd_all(
                init_states[:, 1],
                init_states[:, 0],
                forcing_features[:, 0],
            )  # (B, num_grid_nodes, d_h)
            # embed also including current grid state, for encoder
            grid_current_emb = self.embedd_current(
                init_states[:, 1],
                init_states[:, 0],
                forcing_features[:, 0],
                target_states[:, 0],
            )  # (B, num_grid_nodes, d_h)

            # Create latent variable samples
            prior_dist = self.prior_model(
                grid_prev_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)
            prior_samples = prior_dist.rsample(
                (constants.LATENT_SAMPLES_PLOT,)
            ).transpose(
                0, 1
            )  # (B, samples, num_mesh_nodes, d_latent)

            vi_dist = self.encoder(
                grid_current_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)
            vi_samples = vi_dist.rsample(
                (constants.LATENT_SAMPLES_PLOT,)
            ).transpose(
                0, 1
            )  # (B, samples, num_mesh_nodes, d_latent)

            # Make plot for each example
            for example_i, (prior_ex_samples, vi_ex_samples) in enumerate(
                zip(prior_samples, vi_samples), start=1
            ):
                log_plot_dict[f"latent_samples_ex{example_i}"] = (
                    vis.plot_latent_samples(prior_ex_samples, vi_ex_samples)
                )

            if not self.trainer.sanity_checking:
                # Log all plots to wandb
                wandb.log(log_plot_dict)

            plt.close("all")

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

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Must log before super call, as metric lists are cleared at end of step
        self.log_spsk_ratio(self.val_metrics, "val")

        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        # Clear lists with validation metrics values
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        Include metrics computation for ensemble mean prediction
        """
        prediction, target, pred_std = self.common_step(batch)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        time_step_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            ),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in constants.VAL_STEP_LOG_ERRORS
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(
            test_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )

        # Compute all evaluation metrics for error maps
        # Note: explicitly list metrics here, as test_metrics can contain
        # additional ones, computed differently, but that should be aggregated
        # on_test_epoch_end
        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)
            batch_metric_vals = metric_func(
                prediction,
                target,
                pred_std,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.test_metrics[metric_name].append(batch_metric_vals)

        if self.output_std:
            # Store output std. per variable, spatially averaged
            mean_pred_std = torch.mean(
                pred_std[..., self.interior_mask_bool, :], dim=-2
            )  # (B, pred_steps, d_f)
            self.test_metrics["output_std"].append(mean_pred_std)

        # Save per-sample spatial loss for specific times
        spatial_loss = self.loss(
            prediction, target, pred_std, average_grid=False
        )  # (B, pred_steps, num_grid_nodes)
        log_spatial_losses = spatial_loss[:, constants.VAL_STEP_LOG_ERRORS - 1]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_grid_nodes)

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                prediction.shape[0], self.n_example_pred -
                self.plotted_examples
            )

            self.plot_examples(
                batch, n_additional_examples, prediction=prediction
            )

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
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["ens_mae"].append(ens_maes)
        crps_batch = metrics.crps_ens(
            trajectories,
            target_states,
            traj_stds,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.test_metrics["crps_ens"].append(crps_batch)

        # TODO: Make this into a general function for ARModel
        B, N, T, d_g, d_f = trajectories.shape
        trajectories_reshaped = trajectories[..., self.interior_mask_bool, :].permute(
            0, 1, 2, 4, 3)
        target_states_reshaped = target_states[..., self.interior_mask_bool, :].permute(
            0, 1, 3, 2)

        target_states_2d = target_states_reshaped.reshape(
            -1, T, d_f, *constants.GRID_SHAPE)
        trajectories_2d = trajectories_reshaped.reshape(
            -1, T, d_f, *constants.GRID_SHAPE)

        # Spectra for each time step and variable
        # TODO: This can be done more efficiently
        target_spectra = metrics.calculate_energy_spectra(
            target_states_2d)

        radial_target_spectra = metrics.radial_average(target_spectra)
        self.spectra_metrics["spectra_gt"].append(
            radial_target_spectra.cpu())  # (B, t, n_freqs)

        traj_spectra = metrics.calculate_energy_spectra(
            trajectories_2d)
        radial_traj_spectra = metrics.radial_average(traj_spectra)
        self.spectra_metrics["spectra"].append(
            radial_traj_spectra.cpu())  # (B, t, n_freqs)

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        # Plot spatial loss maps
        spatial_loss_tensor = self.all_gather_cat(
            torch.cat(self.spatial_loss_maps, dim=0)
        )  # (N_test, N_log, num_grid_nodes)

        # if self.trainer.is_global_zero:
        #     mean_spatial_loss = torch.mean(
        #         spatial_loss_tensor, dim=0
        #     )  # (N_log, num_grid_nodes)

        #     loss_map_figs = [
        #         vis.plot_spatial_error(
        #             loss_map,
        #             self.interior_mask[:, 0],
        #             title=f"Test loss, t={t_i} ({self.step_length*t_i} h)",
        #         )
        #         for t_i, loss_map in zip(
        #             constants.VAL_STEP_LOG_ERRORS, mean_spatial_loss
        #         )
        #     ]

        #     # log all to same wandb key, sequentially
        #     for fig in loss_map_figs:
        #         wandb.log({"test_loss": wandb.Image(fig)})

        #     # also make without title and save as pdf
        #     pdf_loss_map_figs = [
        #         vis.plot_spatial_error(loss_map, self.interior_mask[:, 0])
        #         for loss_map in mean_spatial_loss
        #     ]
        #     pdf_loss_maps_dir = os.path.join(
        #         wandb.run.dir, "spatial_loss_maps")
        #     os.makedirs(pdf_loss_maps_dir, exist_ok=True)
        #     for t_i, fig in zip(
        #         constants.VAL_STEP_LOG_ERRORS, pdf_loss_map_figs
        #     ):
        #         fig.savefig(os.path.join(
        #             pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
        #     # save mean spatial loss as .pt file also
        #     torch.save(
        #         mean_spatial_loss.cpu(),
        #         os.path.join(wandb.run.dir, "mean_spatial_loss.pt"),
        #     )

        self.spatial_loss_maps.clear()
        self.log_spsk_ratio(self.test_metrics, "test")

        spectra_tensor = self.all_gather_cat(
            torch.cat(self.spectra_metrics["spectra"], dim=0).cpu()
        ).mean(dim=0).cpu()  # (pred_steps, d_f, R)

        spectra_gt_tensor = self.all_gather_cat(
            torch.cat(self.spectra_metrics["spectra_gt"], dim=0).cpu()
        ).mean(dim=0).cpu()  # (pred_steps, d_f, R)

        if self.trainer.is_global_zero:
            torch.save(spectra_tensor.cpu(), os.path.join(
                wandb.run.dir, f"model_spectra.pt"))
            torch.save(spectra_gt_tensor.cpu(), os.path.join(
                wandb.run.dir, f"ground_truth_spectra.pt"))
            # Plot spectra
            for (t_i, (spectra_gt, spectra)) in enumerate(zip(spectra_gt_tensor, spectra_tensor), start=1):
                # Both (pred_steps, n_freqs)
                time_title_part = f"t={t_i}"
                # Create one figure per variable at this time step
                var_figs = [
                    vis.plot_energy_spectra(
                        spectra_gt[var_i].numpy(),
                        spectra[var_i].numpy(),
                        var=var_i,
                        title=f"{var_name} ({var_unit}), {time_title_part}",)
                    for var_i, (var_name, var_unit) in enumerate(
                        zip(
                            constants.PARAM_NAMES_SHORT[constants.USED_PARAMS],
                            constants.PARAM_UNITS[constants.USED_PARAMS],
                        )
                    )
                ]
                example_title = f"spectra"
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
        self.spectra_metrics["spectra"].clear()
        self.spectra_metrics["spectra_gt"].clear()
