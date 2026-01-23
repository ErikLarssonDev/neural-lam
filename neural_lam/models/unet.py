import torch

from neural_lam import constants
from neural_lam.models.ar_model import ARModel
from neural_lam.models.edm_networks_2 import SongUNet


class UNET(ARModel):
    """
    A deterministic UNET model for downscaling
    """

    def __init__(self, args):
        super().__init__(args)

        self.model = SongUNet(img_resolution=torch.as_tensor(constants.FULL_GRID_SHAPE),
                              in_channels=self.grid_dim,
                              out_channels=self.grid_output_dim,
                              embedding_type=None,
                              resample_filter=args.resample_filter,
                              channel_mult=args.channel_mult,
                              encoder_type=args.encoder_type,
                              attn_resolutions=args.attn_resolutions,
                              ir_sde=False,
                              target_idx=None,
                              )

    def predict_step(self, LQ):
        """
        Downscaling weather state
        LQ: (B, d_f, X, Y)

        Returns:
        next_state: (B, N_grid, d_state)
        pred_std: None
        """

        z = torch.ones((LQ.shape[0], 1, 1, 1), device=LQ.device)
        sample = self.model(LQ, z)

        return sample.permute(0, 2, 3, 1).flatten(1, 2), None
