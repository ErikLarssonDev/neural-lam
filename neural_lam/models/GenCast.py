import torch
import numpy as np
import math

from neural_lam.models.ar_model import ARModel
from neural_lam import constants
from graph_weather.models.gencast.utils.noise import generate_isotropic_noise, sample_noise_level


class GenCast(ARModel):
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
        self.grid_lat = np.arange(-90, 90, 1) # (0, 238, 1)
        self.grid_lon = np.arange(0, 360, 1) # (0, 268, 1)
        self.num_lon = len(self.grid_lon)
        self.num_lat = len(self.grid_lat)
        self.model = Denoiser(graph=args.graph, grid_lon=self.grid_lon,
            grid_lat=self.grid_lat,
            input_features_dim=self.input_dim,
            output_features_dim=self.output_dim)
        self.sampler = Sampler()

        self.border_condition = args.border_condition

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
        if self.border_condition:
            input_flat = torch.cat((prev_state, prev_prev_state, forcing, border_state), dim=-1) # (B, N_grid, d_input)
        else:
            input_flat = torch.cat((prev_state, prev_prev_state, forcing), dim=-1) # (B, N_grid, d_input)
        
        input_grid = torch.reshape(input_flat, (-1, *constants.grid_shape,
            input_flat.shape[2])) # (B, N_x, N_y, d_input)
        
        # Most computer vision methods in torch want channel dimension first
        # input_grid = input_grid.permute((0,3,1,2)).contiguous() # (B, d_input, N_x, N_y)

        # Run through sampler
        output_grid = self.sampler.sample(self.model, input_grid)

        # Reshape back from 2d to flattened grid dimension
        # output_grid = output_grid.permute((0,2,3,1)) # (B, N_x, N_y, d_state)
        next_state = output_grid.flatten(1,2) # (B, N_grid, d_state)

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
        if self.border_condition:
            input_flat = torch.cat((prev_state, prev_prev_state, forcing, true_states.squeeze(1)*self.border_mask), dim=-1) # (B, N_grid, d_input)
        else:
            input_flat = torch.cat((prev_state, prev_prev_state, forcing), dim=-1)
        
        input_grid = torch.reshape(input_flat, (-1, *constants.grid_shape,
            input_flat.shape[2])) # (B, N_x, N_y, d_input)
        
        # Most computer vision methods in torch want channel dimension first
        # input_grid = input_grid.permute((0,3,1,2)).contiguous() # (B, d_input, N_x, N_y)

        y = true_states # (B, 1, N_grid, d_input)
        y = y.flatten(1,2) # (B, N_grid, d_input)
        y = torch.reshape(y, (-1, *constants.grid_shape, y.shape[2])) # (B, N_x, N_y, d_input)
        # y = y.permute((0,3,1,2)).contiguous() # (B, d_input, N_x, N_y)
    
        # Corrupt targets with noise
        noise_levels = np.zeros((y.shape[0], 1), dtype=np.float32)
        corrupted_targets = np.zeros_like(y, dtype=np.float32)
        for b in range(y.shape[0]):
            noise_level = sample_noise_level()
            noise = generate_isotropic_noise(
                num_lon=self.num_lon,
                num_lat=self.num_lat,
                num_samples=y.shape[-1],
                isotropic=False, # Isotropic noise requires grid's shape to be 2N x N or 2N x (N+1)
            )
            corrupted_targets[b] = y[b] + noise_level * noise # TODO: Only create noise inside the border
            noise_levels[b] = noise_level
        output_grid = self.model(corrupted_targets, input_grid, noise_levels) # (B, N_x, N_y, d_state)

        # Reshape back from 2d to flattened grid dimension
        # output_grid = output_grid.permute((0,2,3,1)) # (B, N_x, N_y, d_state)
        next_state = output_grid.flatten(1,2) # (B, N_grid, d_state)

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
    
    """Denoiser.

The denoiser takes as inputs the previous two timesteps, the corrupted target residual, and the
noise level, and outputs the denoised predictions. It performs the following tasks:
1. Initializes the graph, encoder, processor, and decoder.
2. Computes f_theta as the combination of encoder, processor, and decoder.
3. Preconditions f_theta on the noise levels using the parametrization from Karras et al. (2022).
"""

import einops
import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin

from graph_weather.models.gencast.graph.graph_builder import GraphBuilder
from graph_weather.models.gencast.layers.decoder import Decoder
from graph_weather.models.gencast.layers.encoder import Encoder
from graph_weather.models.gencast.layers.processor import Processor
from graph_weather.models.gencast.utils.batching import batch, hetero_batch
from graph_weather.models.gencast.utils.noise import Preconditioner
from neural_lam import utils


class Denoiser(torch.nn.Module, PyTorchModelHubMixin):
    """GenCast's Denoiser."""

    def __init__(
        self,
        graph,
        grid_lon: np.ndarray,
        grid_lat: np.ndarray,
        input_features_dim: int,
        output_features_dim: int,
        hidden_dims: list[int] = [512, 512],
        num_blocks: int = 16,
        num_heads: int = 4,
        splits: int = 6,
        num_hops: int = 6,
        device: torch.device = torch.device("cpu"),
        sparse: bool = False,
        use_edges_features: bool = True,
        scale_factor: float = 1.0,
    ):
        """Initialize the Denoiser.

        Args:
            grid_lon (np.ndarray): array of longitudes.
            grid_lat (np.ndarray): array of latitudes.
            input_features_dim (int): dimension of the input features for a single timestep.
            output_features_dim (int): dimension of the target features.
            hidden_dims (list[int], optional): list of dimensions for the hidden layers in the MLPs
                used in GenCast. This also determines the latent dimension. Defaults to [512, 512].
            num_blocks (int, optional): number of transformer blocks in Processor. Defaults to 16.
            num_heads (int, optional): number of heads for each transformer. Defaults to 4.
            splits (int, optional): number of time to split the icosphere during graph building.
                Defaults to 6.
            num_hops (int, optional): the transformes will attention to the (2^num_hops)-neighbours
                of each node. Defaults to 6.
            device (torch.device, optional): device on which we want to build graph.
                Defaults to torch.device("cpu").
            sparse (bool): if true the processor will apply Sparse Attention using DGL backend.
                Defaults to False.
            use_edges_features (bool): if true use mesh edges features inside the Processor.
                Defaults to True.
            scale_factor (float):  in the Encoder the message passing output is multiplied by the
                scale factor. This is important when you want to fine-tune a pretrained model to a
                higher resolution. Defaults to 1.
        """
        super().__init__()
        self.num_lon = len(grid_lon)
        self.num_lat = len(grid_lat)
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.use_edges_features = use_edges_features

        # Initialize graph
        # TODO: Add support for graph for LAM and not spherical graph
        self.graphs = GraphBuilder(
            grid_lon=grid_lon,
            grid_lat=grid_lat,
            splits=splits,
            num_hops=num_hops,
            device=device,
            add_edge_features_to_khop=use_edges_features,
        )

        self._register_graph()

        # Own graph representation
        self.hierarchical, graph_ldict = utils.load_graph(graph)
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            print(f"Registering {name}, attribute value: {torch.tensor(attr_value).shape}")
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        self.num_mesh_nodes, _ = self.get_num_mesh()
        print(
            f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes} "
            f"nodes ({self.num_grid_nodes} grid, {self.num_mesh_nodes} mesh)"
        )

        # grid_dim from data + static
        self.g2m_edges, g2m_dim = self.g2m_features.shape
        self.m2g_edges, m2g_dim = self.m2g_features.shape

        # Initialize Encoder
        self.encoder = Encoder(
            grid_dim=output_features_dim + input_features_dim, # + self.graphs.grid_nodes_dim, # 2*input_features_dim, chagned to input_features_dim
            mesh_dim=self.graphs.mesh_nodes_dim,
            edge_dim=self.graphs.g2m_edges_dim,
            hidden_dims=hidden_dims,
            activation_layer=torch.nn.SiLU,
            use_layer_norm=True,
            scale_factor=scale_factor,
        )

        # Initialize Processor
        if sparse and use_edges_features:
            raise ValueError("Sparse processor don't support edges features.")

        self.processor = Processor(
            latent_dim=hidden_dims[-1],
            edges_dim=self.graphs.mesh_edges_dim if use_edges_features else None,
            hidden_dims=hidden_dims,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_frequencies=32,
            base_period=16,
            noise_emb_dim=16,
            activation_layer=torch.nn.SiLU,
            use_layer_norm=True,
            sparse=sparse,
        )

        # Initialize Decoder
        self.decoder = Decoder(
            edges_dim=self.graphs.m2g_edges_dim,
            output_dim=output_features_dim,
            hidden_dims=hidden_dims,
            activation_layer=torch.nn.SiLU,
            use_layer_norm=True,
        )

        # Initialize preconditioning functions
        self.precs = Preconditioner(sigma_data=1.0)

    def _check_shapes(self, corrupted_targets, prev_inputs, noise_levels):
        batch_size = prev_inputs.shape[0]
        exp_inputs_shape = (batch_size, self.num_lon, self.num_lat, self.input_features_dim)
        exp_targets_shape = (batch_size, self.num_lon, self.num_lat, self.output_features_dim)
        exp_noise_shape = (batch_size, 1)

        if not all(
            [
                corrupted_targets.shape == exp_targets_shape,
                prev_inputs.shape == exp_inputs_shape,
                noise_levels.shape == exp_noise_shape,
            ]
        ):
            raise ValueError(
                "The shapes of the input tensors don't match with the initialization parameters: "
                f"expected {exp_inputs_shape} got {prev_inputs.shape} for prev_inputs, {exp_targets_shape} got {corrupted_targets.shape} for targets and "
                f"{exp_noise_shape} got {noise_levels.shape} for noise_levels."
            )

    def _run_encoder(self, grid_features):
        # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
        batch_size = grid_features.shape[0]
        batched_senders, batched_receivers, batched_edge_index, batched_edge_attr = hetero_batch(
            self.g2m_grid_nodes,
            self.g2m_mesh_nodes,
            self.g2m_edge_index,
            self.g2m_edge_attr,
            batch_size,
        )
        # load features.
        grid_features = einops.rearrange(grid_features, "b n f -> (b n) f")
        input_grid_nodes = torch.cat([grid_features, batched_senders], dim=-1)
        input_mesh_nodes = batched_receivers
        input_edge_attr = batched_edge_attr
        edge_index = batched_edge_index

        # run the encoder.
        latent_grid_nodes, latent_mesh_nodes = self.encoder(
            input_grid_nodes=input_grid_nodes,
            input_mesh_nodes=input_mesh_nodes,
            input_edge_attr=input_edge_attr,
            edge_index=edge_index,
        )

        # restore nodes dimension: [b, n, f]
        latent_grid_nodes = einops.rearrange(latent_grid_nodes, "(b n) f -> b n f", b=batch_size)
        latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "(b n) f -> b n f", b=batch_size)

        assert not torch.isnan(latent_grid_nodes).any()
        assert not torch.isnan(latent_mesh_nodes).any()
        return latent_grid_nodes, latent_mesh_nodes

    def _run_decoder(self, latent_mesh_nodes, latent_grid_nodes):
        # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
        batch_size = latent_mesh_nodes.shape[0]
        _, _, batched_edge_index, batched_edge_attr = hetero_batch(
            self.m2g_mesh_nodes,
            self.m2g_grid_nodes,
            self.m2g_edge_index,
            self.m2g_edge_attr,
            batch_size,
        )

        # load features.
        input_mesh_nodes = einops.rearrange(latent_mesh_nodes, "b n f -> (b n) f")
        input_grid_nodes = einops.rearrange(latent_grid_nodes, "b n f -> (b n) f")
        input_edge_attr = batched_edge_attr
        edge_index = batched_edge_index

        # run the decoder.
        output_grid_nodes = self.decoder(
            input_mesh_nodes=input_mesh_nodes,
            input_grid_nodes=input_grid_nodes,
            input_edge_attr=input_edge_attr,
            edge_index=edge_index,
        )

        # restore nodes dimension: [b, n, f]
        output_grid_nodes = einops.rearrange(output_grid_nodes, "(b n) f -> b n f", b=batch_size)

        assert not torch.isnan(output_grid_nodes).any()
        return output_grid_nodes

    def _run_processor(self, latent_mesh_nodes, noise_levels):
        # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
        batch_size = latent_mesh_nodes.shape[0]
        num_nodes = latent_mesh_nodes.shape[1]
        _, batched_edge_index, batched_edge_attr = batch(
            self.khop_mesh_nodes,
            self.khop_mesh_edge_index,
            self.khop_mesh_edge_attr if self.use_edges_features else None,
            batch_size,
        )

        # load features.
        latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "b n f -> (b n) f")
        input_edge_attr = batched_edge_attr
        edge_index = batched_edge_index

        # repeat noise levels for each node.
        noise_levels = einops.repeat(noise_levels, "b f -> (b n) f", n=num_nodes)

        # run the processor.
        latent_mesh_nodes = self.processor.forward(
            latent_mesh_nodes=latent_mesh_nodes,
            input_edge_attr=input_edge_attr,
            edge_index=edge_index,
            noise_levels=noise_levels,
        )

        # restore nodes dimension: [b, n, f]
        latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "(b n) f -> b n f", b=batch_size)

        assert not torch.isnan(latent_mesh_nodes).any()
        return latent_mesh_nodes

    def _f_theta(self, grid_features, noise_levels):
        # run encoder, processor and decoder.
        latent_grid_nodes, latent_mesh_nodes = self._run_encoder(grid_features)
        latent_mesh_nodes = self._run_processor(latent_mesh_nodes, noise_levels)
        output_grid_nodes = self._run_decoder(latent_mesh_nodes, latent_grid_nodes)
        return output_grid_nodes

    def forward(
        self, corrupted_targets: torch.Tensor, prev_inputs: torch.Tensor, noise_levels: torch.Tensor
    ) -> torch.Tensor:
        """Compute the denoiser output.

        The denoiser is a version of the (encoder, processor, decoder)-model (called f_theta),
        preconditioned on the noise levels, as described below:

        D(Z, X, sigma) := c_skip(sigma)Z + c_out(sigma) * f_theta(c_in(sigma)Z, X, c_noise(sigma)),

        where Z is the corrupted target, X is the previous two timesteps concatenated and sigma is
        the noise level used for Z's corruption.

        Args:
            corrupted_targets (torch.Tensor): the target residuals corrupted by noise.
            prev_inputs (torch.Tensor): the previous two timesteps concatenated across the features'
                dimension.
            noise_levels (torch.Tensor): the noise level used for corruption.
        """
        # check shapes and noise.
        self._check_shapes(corrupted_targets, prev_inputs, noise_levels)
        if not (noise_levels > 0).any():
            raise ValueError("All the noise levels must be strictly positive.")

        # flatten lon/lat dimensions.
        prev_inputs = einops.rearrange(prev_inputs, "b lon lat f -> b (lon lat) f")
        corrupted_targets = einops.rearrange(corrupted_targets, "b lon lat f -> b (lon lat) f")

        # apply preconditioning functions to target and noise.
        scaled_targets = self.precs.c_in(noise_levels)[:, :, None] * corrupted_targets
        scaled_noise_levels = self.precs.c_noise(noise_levels)

        # concatenate inputs and targets across features dimension.
        grid_features = torch.cat((scaled_targets, prev_inputs), dim=-1)

        # run the model.
        preds = self._f_theta(grid_features, scaled_noise_levels)

        # add skip connection.
        out = (
            self.precs.c_skip(noise_levels)[:, :, None] * corrupted_targets
            + self.precs.c_out(noise_levels)[:, :, None] * preds
        )

        # restore lon/lat dimensions.
        out = einops.rearrange(out, "b (lon lat) f -> b lon lat f", lon=self.num_lon)
        return out

    def _register_graph(self):
        # we need to register all the tensors associated with the graph as buffers. In this way they
        # will move to the same device of the model. These tensors won't be part of the state since
        # persistent is set to False.

        self.register_buffer(
            "g2m_grid_nodes", self.graphs.g2m_graph["grid_nodes"].x, persistent=False
        )
        self.register_buffer(
            "g2m_mesh_nodes", self.graphs.g2m_graph["mesh_nodes"].x, persistent=False
        )
        self.register_buffer(
            "g2m_edge_attr",
            self.graphs.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_attr,
            persistent=False,
        )
        self.register_buffer(
            "g2m_edge_index",
            self.graphs.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_index,
            persistent=False,
        )

        self.register_buffer("mesh_nodes", self.graphs.mesh_graph.x, persistent=False)
        self.register_buffer("mesh_edge_attr", self.graphs.mesh_graph.edge_attr, persistent=False)
        self.register_buffer("mesh_edge_index", self.graphs.mesh_graph.edge_index, persistent=False)

        self.register_buffer("khop_mesh_nodes", self.graphs.khop_mesh_graph.x, persistent=False)
        self.register_buffer(
            "khop_mesh_edge_attr", self.graphs.khop_mesh_graph.edge_attr, persistent=False
        )
        self.register_buffer(
            "khop_mesh_edge_index", self.graphs.khop_mesh_graph.edge_index, persistent=False
        )

        self.register_buffer(
            "m2g_grid_nodes", self.graphs.m2g_graph["grid_nodes"].x, persistent=False
        )
        self.register_buffer(
            "m2g_mesh_nodes", self.graphs.m2g_graph["mesh_nodes"].x, persistent=False
        )
        self.register_buffer(
            "m2g_edge_attr",
            self.graphs.m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_attr,
            persistent=False,
        )
        self.register_buffer(
            "m2g_edge_index",
            self.graphs.m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_index,
            persistent=False,
        )

"""Diffusion sampler"""

class Sampler:
    """Sampler for the denoiser.

    The sampler consists in the second-order DPMSolver++2S solver (Lu et al., 2022), augmented with
    the stochastic churn (again making use of the isotropic noise) and noise inflation techniques
    used in Karras et al. (2022) to inject further stochasticity into the sampling process. In
    conditioning on previous timesteps it follows the Conditional Denoising Estimator approach
    outlined and motivated by Batzolis et al. (2021).
    """

    def __init__(
        self,
        S_noise: float = 1.05,
        S_tmin: float = 0.75,
        S_tmax: float = 80.0,
        S_churn: float = 2.5,
        r: float = 0.5,
        sigma_max: float = 80.0,
        sigma_min: float = 0.03,
        rho: float = 7,
        num_steps: int = 20,
    ):
        """Initialize the sampler.

        Args:
            S_noise (float): noise inflation parameter. Defaults to 1.05.
            S_tmin (float): minimum noise for sampling. Defaults to 0.75.
            S_tmax (float): maximum noise for sampling. Defaults to 80.
            S_churn (float): stochastic churn rate. Defaults to 2.5.
            r (float): _description_. Defaults to 0.5.
            sigma_max (float): maximum value of sigma for sigma's distribution. Defaults to 80.
            sigma_min (float): minimum value of sigma for sigma's distribution. Defaults to 0.03.
            rho (float): exponent of the sigma's distribution. Defaults to 7.
            num_steps (int): number of timesteps during sampling. Defaults to 20.
        """
        self.S_noise = S_noise
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_churn = S_churn
        self.r = r
        self.num_steps = num_steps

        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho

    def _sigmas_fn(self, u):
        return (
            self.sigma_max ** (1 / self.rho)
            + u * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho

    @torch.no_grad()
    def sample(self, denoiser: Denoiser, prev_inputs: torch.Tensor):
        """Generate a sample from random noise for the given inputs.

        Args:
            denoiser (Denoiser): the denoiser model.
            prev_inputs (torch.Tensor): previous two timesteps.

        Returns:
            torch.Tensor: normalized residuals predicted.
        """
        device = prev_inputs.device

        time_steps = torch.arange(0, self.num_steps).to(device) / (self.num_steps - 1)
        sigmas = self._sigmas_fn(time_steps).to(device)

        batch_ones = torch.ones(prev_inputs.shape[0], 1).to(device)
        x = torch.zeros((prev_inputs.shape[0], denoiser.num_lon, denoiser.num_lat, denoiser.output_features_dim)).to(device)
        # initialize noise
        for b in range(x.shape[0]):
            x[b] = sigmas[0] * torch.tensor(
                generate_isotropic_noise(
                    num_lon=denoiser.num_lon,
                    num_lat=denoiser.num_lat,
                    num_samples=denoiser.output_features_dim,
                    isotropic=False, # Isotropic noise requires grid's shape to be 2N x N or 2N x (N+1)
                )
            ).unsqueeze(0).to(device)

        for i in range(len(sigmas) - 1):
            # stochastic churn from Karras et al. (Alg. 2)
            gamma = (
                min(self.S_churn / self.num_steps, math.sqrt(2) - 1)
                if self.S_tmin <= sigmas[i] <= self.S_tmax
                else 0.0
            )
            noise = torch.zeros_like(x).to(device)
            for b in range(x.shape[0]):
                # noise inflation from Karras et al. (Alg. 2)
                noise[b] = self.S_noise * torch.tensor(
                    generate_isotropic_noise(
                        num_lon=denoiser.num_lon,
                        num_lat=denoiser.num_lat,
                        num_samples=denoiser.output_features_dim,
                        isotropic=False, # Isotropic noise requires grid's shape to be 2N x N or 2N x (N+1)
                    )
                ).to(device)

            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                x = x + (sigma_hat**2 - sigmas[i] ** 2) ** 0.5 * noise
            denoised = denoiser(x, prev_inputs, sigma_hat * batch_ones)

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
                lambda_mid = lambda_hat + self.r * h
                sigma_mid = torch.exp(-lambda_mid)

                u = sigma_mid / sigma_hat * x - (torch.exp(-self.r * h) - 1) * denoised
                denoised_2 = denoiser(u, prev_inputs, sigma_mid * batch_ones)
                D = (1 - 1 / (2 * self.r)) * denoised + 1 / (2 * self.r) * denoised_2
                x = sigmas[i + 1] / sigma_hat * x - (torch.exp(-h) - 1) * D

        return x