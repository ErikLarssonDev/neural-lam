# Standard library
import json
import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profilers import AdvancedProfiler

# First-party
from neural_lam import constants, utils, config
from neural_lam.weather_dataset import WeatherDataset
from neural_lam.downscaling_dataset import DownscalingDataset
from neural_lam.netCDF_dataset import NetCDFDataset
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.models.graph_fm import GraphFM
from neural_lam.models.graphcast import GraphCast
from neural_lam.models.diffusion import Diffusion
from neural_lam.models.ir_sde import IR_SDE
from neural_lam.models.stochastic_interpolants import SI
from neural_lam.models.unet import UNET
from neural_lam.models.CorrDiff import CorrDiff
import matplotlib.pyplot as plt
import numpy as np

import torch
from tqdm import tqdm
import os

def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def main(input_args=None):
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/clim_config.yaml",
        help="Path to data config file (default: neural_lam/clim_config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_lam",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )
    parser.add_argument(
        "--subset_ds",
        action="store_true",
        help="Use only a small subset of the dataset, for debugging"
        "(default: false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        action="store_true",
        help="If optimizer state should be restored with model "
        "(default: false)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        help="Numerical precision to use for model (32/16/bf16/bf16-mixed) (default: bf16-mixed)",
    )

    # Model architecture
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Graph to load and use in graph-based model "
        "(default: multiscale)",
    )
    parser.add_argument(
        "--diffusion_model",
        type=str,
        default="edm",
        help="Model to use in the diffusion model"
        "(default: edm)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Dimensionality of all hidden representations (default: 64)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Dimensionality of latent R.V. at each node (if different than"
        " hidden_dim) (default: None (same as hidden_dim))",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in all MLPs (default: 1)",
    )
    parser.add_argument(
        "--processor_layers",
        type=int,
        default=6,
        help="Number of GNN layers in processor GNN (for prob. model: in "
        "decoder) (default: 6)",
    )
    parser.add_argument(
        "--encoder_processor_layers",
        type=int,
        default=1,
        help="Number of on-mesh GNN layers in encoder GNN (default: 2)",
    )
    parser.add_argument(
        "--prior_processor_layers",
        type=int,
        default=1,
        help="Number of on-mesh GNN layers in prior GNN (default: 2)",
    )
    parser.add_argument(
        "--mesh_aggr",
        type=str,
        default="sum",
        help="Aggregation to use for m2m processor GNN layers (sum/mean) "
        "(default: sum)",
    )
    parser.add_argument(
        "--output_std",
        action="store_true",
        help="If models should additionally output std.-dev. per "
        "output dimensions "
        "(default: False (no))",
    )
    parser.add_argument(
        "--shared_grid_embedder",
        action="store_true",  # Default to separate embedders
        help="If the same embedder MLP should be used for interior and boundary"
        " grid nodes. Note that this requires the same dimensionality for "
        "both kinds of grid inputs. (default: False (no))",
    )
    parser.add_argument(
        "--prior_dist",
        type=str,
        default="isotropic",
        help="Structure of Gaussian distribution in prior network output "
        "(isotropic/diagonal) (default: isotropic)",
    )
    parser.add_argument(
        "--learn_prior",
        type=int,
        default=1,
        help="If the prior should be learned as a mapping from previous state "
        "and forcing, otherwise static with mean 0 (default: 1 (yes))",
    )
    parser.add_argument(
        "--vertical_propnets",
        type=int,
        default=1,  # TODO: Change to 1 as it is used in the paper
        help="If PropagationNets should be used for all vertical message "
        "passing (g2m, m2g, up in hierarchy), in deterministic models."
        "(default: 1 (Yes))",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="edm",
        help="The sampler to use when generating trajectories with a diffusion model"
        "(heun/edm) (default: heun)",
    )

    # Training options
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss (1-19) "
        "(default: 1)",
    )
    parser.add_argument(
        "--control_only",
        action="store_true",
        help="Train only on control member of ensemble data "
        "(default: False)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wmse",
        help="Loss function to use, see metric.py (default: wmse)",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=3,
        help="Step length in hours to consider single time step 1-3 "
        "(default: 3)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=1.0,
        help="Beta weighting in front of kl-term in ELBO (default: 1)",
    )
    parser.add_argument(
        "--crps_weight",
        type=float,
        default=0,
        help="Weighting for CRPS term of loss, not computed if = 0. CRPS is "
        "computed based on trajectories sampled using prior distribution. "
        "(default: 0)",
    )
    parser.add_argument(
        "--sample_obs_noise",
        type=int,
        default=0,
        help="If observation noise should be sampled during rollouts (both "
        "training and eval), or just mean prediction used "
        "(default: 0 (no))",
    )
    parser.add_argument(
        "--border_condition",
        action="store_true",
        help="If border condition should be used in diffusion model ",
    )

    parser.add_argument(
        "--pred_residual",
        action="store_true",
        help="If the model should predict residuals instead of absolute values",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for training. (default: 0.01)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        help="Learning rate scheduler to use, supported (cosine), (default: None)",
    )

    # IR-SDE Options
    parser.add_argument(
        "--sigma_max",
        type=float,
        # To get the same sigma max as the paper (IR-SDE), normalize by 255 to get it into the image domain. TODO: Experiment with this for better results for atmospheric data
        default=10 / 255,
        help="Sigma max for training. (default: 10)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.005,
        help="Eps for IR-SDE. (default: 0.005)",
    )
    parser.add_argument(
        "--keep_cond",
        action="store_true",
        help="If the conditioning should be kept during training of IR-SDE/SI "
        "(default: False)",
    )

    # EDM Options
    # resample_filter=args.resample_filter,
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,  # TODO: Do we need lower sigma_min for atmospheric data?
        help="Sigma min for training. (default: 0.002)",
    )
    parser.add_argument(
        "--resample_filter",
        type=list_of_ints,
        default="1,1",
        help="Resample filter for edm model (default: 1,1 or 1,3,3,1)",
    )
    # channel_mult=args.channel_mult,
    parser.add_argument(
        "--channel_mult",
        type=list_of_ints,
        default="1,2,2,2",
        help="Channel multiplier for edm model (depth and width of UNET) (default: 1,2,2,2)",
    )

    # encoder_type=args.encoder_type,
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="standard",
        help="Type of encoder to use in edm model (standard/residual/skip)"
        "(default: 'standard')",
    )

    # attn_resolutions=args.attn_resolutions
    parser.add_argument(
        "--attn_resolutions",
        type=list_of_ints,
        default="1",
        help="Resolutions to apply attention to in edm model (default: '1')",
    )

    parser.add_argument(
        "--noise_embedding",
        type=str,
        default="fourier",
        help="Type of encoder to use in edm model (positional/fourier)"
        "(default: 'fourier')",
    )

    # SI Options
    parser.add_argument(
        "--correction_steps",
        type=int,
        default=0,
        help="Number of correction steps to use in EDM sampler (default: 0)",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=0.3,
        help="SNR value for EDM (default: 0.3)",
    )
    parser.add_argument(
        "--corr_tmin",
        type=float,
        default=0.5,
        help="Minimum time value for applying correction steps in EDM "
        "(default: 0.5)",
    )
    parser.add_argument(
        "--conserve_mass_w",
        type=float,
        default=0,
        help="Guidance strength for mass conservation guidance (default: 0)"
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during val/test "
        "(default: 1)",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of ensemble members during evaluation (default: 5)",
    )
    parser.add_argument(
        "--sampler_steps",
        type=int,
        default=20,
        help="Number of sampling steps during inference (default: 20)",
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        help="If the model output should be saved to the output folder (default: False)",
    )
    parser.add_argument(
        "--save_steps",
        action="store_true",
        help="If the diffusion steps output of 1 sample should be saved to the folder diffusion_steps (default: False)",
    )
    parser.add_argument(
        "--beta_fn",
        type=str,
        default="t^2",
        help="Beta function to use in diffusion model (linear/t^2) (default: t^2)",
    )
    parser.add_argument(
        "--sigma_coef",
        type=float,
        default=1,
        help="Sigma coefficient for stochatic interpolants (default: 1)",
    )
    parser.add_argument(
        "--sigma_coef_sampling",
        type=float,
        default=1,
        help="Sigma coefficient for stochatic interpolants during sampling (default: 1)",
    )
    parser.add_argument(
        "--diffusion_fn",
        type=str,
        default=None,
        help="Diffusion function to use in stochastic interpolants during sampling (g_sigma/g_sigma_pow4) (default: None (use trained one))",
    )

    # CorrDiff options
    parser.add_argument(
        "--residual_model",
        type=str,
        default="EDM",
        help="Model to use for residual prediction in CorrDiff (EDM/SI) (default: EDM)",
    )
    parser.add_argument(
        "--mean_model_ckpt_path",
        type=str,
        default="/mimer/NOBACKUP/groups/mlhighres/users/erifh/neural-lam/saved_models/UNET_Static_50e-unet-6x128-12_12_16-6743/last.ckpt",
        help="Path to checkpoint of mean model to load in CorrDiff (default: '')",
    )



    # Logger Settings
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="clim-downscaling",
        help="Wandb run project (default: 'neural-lam-downscaling')",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="Wandb run name (default: '')",
    )
    parser.add_argument(
        "--val_steps_to_log",
        type=list,
        default=[1],
        help="Steps to log val loss for (default: [1])",
    )
    parser.add_argument(
        "--metrics_watch",
        nargs="+",
        default=[],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )
    parser.add_argument(
        "--var_leads_metrics_watch",
        type=str,
        default="{}",
        help="""JSON string with variable-IDs and lead times to log watched
             metrics (e.g. '{"1": [1, 2], "3": [3, 4]}')""",
    )
    parser.add_argument(
        "--save_output_wandb",
        action="store_true",
        help="If the model output should be saved to wandb (save_output has to be enabled)",
    )

    args = parser.parse_args(input_args)
    args.var_leads_metrics_watch = {
        int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    }

    config_loader = config.Config.from_file(args.data_config)

    train_loader = torch.utils.data.DataLoader(
        NetCDFDataset(
            start_date=config_loader.dataset.train_start_date,
            end_date=config_loader.dataset.train_end_date,
            input_path=config_loader.dataset.input_path,
            input_files=config_loader.dataset.input_files,
            ground_truth_path=config_loader.dataset.ground_truth_path,
            ground_truth_files=config_loader.dataset.ground_truth_files,
            ground_truth_stats_path=config_loader.dataset.ground_truth_stats_path,
            levels=config_loader.dataset.levels,
            is_inference_dataset=False,
            normalize_ground_truth=config_loader.dataset.normalize_ground_truth,
            subset_ds=args.subset_ds,
            upscale_inputs=config_loader.dataset.upscale_inputs,
            static_fields_files=config_loader.dataset.static_fields_files,
            interpolation_mode=config_loader.dataset.interpolation_mode,
            provide_coordinates=config_loader.dataset.provide_coordinates,
            provide_day_of_year=config_loader.dataset.provide_day_of_year
        ),
        args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )

    os.makedirs("plots_data", exist_ok=True)
    os.makedirs("stats", exist_ok=True)


    lq_means, hq_means = [], []
    lq_sums, hq_sums = [], []

    for batch in tqdm(train_loader, desc="Computing field means"):
        LQ = batch['LQ'][:, config_loader.dataset.downscaling_idx,...].cpu()
        HQ = batch['HQ'].cpu()

        lq_means.append(LQ.mean(dim=(-2,-1)))
        hq_means.append(HQ.mean(dim=(-2,-1)))
        lq_sums.append(LQ.sum(dim=(-2,-1)))
        hq_sums.append(HQ.sum(dim=(-2,-1)))

    lq_means = torch.cat(lq_means, dim=0)
    hq_means = torch.cat(hq_means, dim=0)
    lq_sums = torch.cat(lq_sums, dim=0)
    hq_sums = torch.cat(hq_sums, dim=0)

    print("LQ mean per channel:", lq_means.mean(dim=0))
    print("HQ mean per channel:", hq_means.mean(dim=0))
    print("LQ sum per channel:", lq_sums.mean(dim=0))
    print("HQ sum per channel:", hq_sums.mean(dim=0))

    for c in range(lq_means.shape[1]):
        plt.figure(figsize=(10,4))
        plt.hist(lq_means[:, c].numpy(), bins=50, alpha=0.5, label="LQ")
        plt.hist(hq_means[:, c].numpy(), bins=50, alpha=0.5, label="HQ")
        plt.title(f"Channel {c} mean distribution")
        plt.xlabel("Spatial mean")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(f'plots_data/channel_{c}_mean_dist.png')
        plt.close()

    for c in range(lq_means.shape[1]):
        plt.figure(figsize=(10,4))
        plt.hist(lq_sums[:, c].numpy(), bins=50, alpha=0.5, label="LQ")
        plt.hist(hq_sums[:, c].numpy(), bins=50, alpha=0.5, label="HQ")
        plt.title(f"Channel {c} sum distribution")
        plt.xlabel("Spatial mean")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(f'plots_data/channel_{c}_sum_dist.png')
        plt.close()

    # save to files
    torch.save({
        'lq_means': lq_means,
        'hq_means': hq_means,
        'lq_sums': lq_sums,
        'hq_sums': hq_sums
    }, "stats/field_stats.pt")

    # Optionally save as numpy for general use
    np.savez("stats/field_stats.npz",
            lq_means=lq_means.numpy(),
            hq_means=hq_means.numpy(),
            lq_sums=lq_sums.numpy(),
            hq_sums=hq_sums.numpy())

if __name__ == "__main__":
    main()