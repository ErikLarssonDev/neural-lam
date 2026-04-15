import os
import torch
import wandb
import datetime

from neural_lam import constants
from neural_lam.models.ar_model import ARModel
from neural_lam.models.edm_networks_2 import SongUNet

# Local
from .. import config, metrics, utils, vis, constants


class UNET(ARModel):
    """
    A deterministic UNET model for downscaling
    """

    def __init__(self, args):
        super().__init__(args)

        self.model = SongUNet(img_resolution=torch.as_tensor(self.config_loader.dataset.FULL_GRID_SHAPE),
                              in_channels=self.grid_dim,
                              out_channels=self.grid_output_dim,
                              model_channels=args.hidden_dim,
                              embedding_type=None,
                              resample_filter=args.resample_filter,
                              channel_mult=args.channel_mult,
                              encoder_type=args.encoder_type,
                              attn_resolutions=args.attn_resolutions,
                              ir_sde=False,
                              target_idx=None,
                              )

        self.save_output = args.save_output
        self.output_path = args.output_path
        self.save_output_wandb = args.save_output_wandb

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

    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target, pred_std, date_ordinal = self.common_step(batch)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                target,  # (B, 1, num_grid_nodes, d_f)
                pred_std,
            ),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # print(f"Time step loss: {time_step_loss.shape}")

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
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
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.test_metrics[metric_name].append(batch_metric_vals)

        if self.output_std:
            # Store output std. per variable, spatially averaged
            mean_pred_std = torch.mean(
                pred_std, dim=-2)  # (B, pred_steps, d_f)
            self.test_metrics["output_std"].append(mean_pred_std)

        # Save per-sample spatial loss for specific times
        spatial_loss = self.loss(
            prediction, target, pred_std, average_grid=False
        )  # (B, pred_steps, num_grid_nodes)
        log_spatial_losses = spatial_loss[
            :, [step - 1 for step in self.args.val_steps_to_log]
        ]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_grid_nodes)

        if self.save_output:
            dates = []
            for date in date_ordinal:
                dates.append(datetime.date.fromordinal(
                    date).strftime("%Y-%m-%d"))

            if self.trainer.is_global_zero:
                os.makedirs(self.output_path, exist_ok=True)

            for prediction_slice, target_slice, date in zip(
                prediction, target, dates
            ):
                if self.save_output:
                    print(f"Saving sample from {date} to {self.output_path}")
                    print(f"Shape of the prediction: {prediction.shape}")
                    print(f"Shape of the target: {target.shape}")
                    torch.save(prediction_slice.detach().cpu().contiguous(
                    ), f"{self.output_path}/ens_mean_{date}.pt")
                    torch.save(target_slice.detach().cpu().contiguous(),
                               f"{self.output_path}/target_{date}.pt")

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        print("On test epoch end")
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")
