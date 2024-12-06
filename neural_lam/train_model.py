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

# First-party
from neural_lam import constants, utils, config
from neural_lam.models.graph_efm import GraphEFM
from neural_lam.models.graph_fm import GraphFM
from neural_lam.models.graphcast import GraphCast
from neural_lam.models.diffusion import Diffusion
from neural_lam.models.swin_u_trans_2 import SwinUTrans2
from neural_lam.weather_dataset import WeatherDataset

MODELS = {
    "graphcast": GraphCast,
    "graph_fm": GraphFM,
    "graph_efm": GraphEFM,
    "diffusion": Diffusion,
    "swin_u2": SwinUTrans2,
}


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
        default="neural_lam/data_config.yaml",
        help="Path to data config file (default: neural_lam/data_config.yaml)",
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
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
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
        default="graphcast",
        help="Model to use in the diffusion model"
        "(default: graphcast)",
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
        default=2,
        help="Number of on-mesh GNN layers in encoder GNN (default: 2)",
    )
    parser.add_argument(
        "--prior_processor_layers",
        type=int,
        default=2,
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
        default=0,
        help="If PropagationNets should be used for all vertical message "
        "passing (g2m, m2g, up in hierarchy), in deterministic models."
        "(default: 0 (no))",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="heun",
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
        default=0,
        help="Weight decay for training. (default: 0)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        help="Learning rate scheduler to use, supported (cosine), (default: None)",
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
        "--plot_diffusion_steps",
        action="store_true",
        help="If the diffusion steps should be saved, only one time step is saved",
    )


    # Logger Settings
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural-lam_prob",
        help="Wandb run name (default: 'neural-lam_prob')",
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
        default=[1, 2, 3, 5, 10, 15, 19],
        help="Steps to log val loss for (default: [1, 2, 3, 5, 10, 15, 19])",
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
    args = parser.parse_args(input_args)
    args.var_leads_metrics_watch = {
        int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    }
    config_loader = config.Config.from_file(args.data_config)

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.step_length <= 3, "Too high step length"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    # Load data
    train_loader = torch.utils.data.DataLoader(
        WeatherDataset(
            config_loader.dataset.name,
            pred_length=args.ar_steps,
            split="train",
            subsample_step=args.step_length,
            subset=args.subset_ds,
            control_only=args.control_only,
            model_name=args.diffusion_model,
            border_condition=args.border_condition,
        ),
        args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )
    max_pred_length = (65 // args.step_length) - 2  # 19
    if args.plot_diffusion_steps:
        max_pred_length = 1
    
    if args.model == "diffusion":
        max_pred_length_val = 1
    else:
        max_pred_length_val = max_pred_length
    val_loader = torch.utils.data.DataLoader(
        WeatherDataset(
            config_loader.dataset.name,
            pred_length=max_pred_length_val,
            split="val",
            subsample_step=args.step_length,
            subset=args.subset_ds,
            control_only=args.control_only,
            model_name=args.diffusion_model,
            border_condition=args.border_condition,
        ),
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    model = model_class(args)

    prefix = "subset-" if args.subset_ds else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"
    
    prefix = f"{args.wandb_run_name}-{prefix}" if args.wandb_run_name else prefix
    run_name = (
        f"{prefix}{args.model}-{args.processor_layers}x{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    # Callbacks for saving model checkpoint
    callbacks = []
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="min_val_loss",
            monitor="val_mean_loss",
            mode="min",
            save_last=True,
        )
    )
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    # Save checkpoints for minimum loss at specific lead times
    for unroll_time in constants.VAL_STEP_CHECKPOINTS:
        metric_name = f"val_loss_unroll{unroll_time}"
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=f"saved_models/{run_name}",
                filename=f"min_{metric_name}",
                monitor=metric_name,
                mode="min",
            )
        )
    logger = pl.loggers.WandbLogger(
        project=args.wandb_project, name=run_name, config=args
    )

    # Training strategy
    # If doing pure autoencoder training (kl_beta = 0), the prior network is not
    # used at all in producing the loss. This is desired, but DDP complains.
    strategy = "ddp" if args.kl_beta > 0 else "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy=strategy,
        accelerator=device_name,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(
            logger, args.val_steps_to_log
        )  # Do after wandb.init

    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else:  # Test
            eval_loader = torch.utils.data.DataLoader(
                WeatherDataset(
                    config_loader.dataset.name,
                    pred_length=max_pred_length,
                    split="test",
                    subsample_step=args.step_length,
                    subset=bool(args.subset_ds),
                    model_name=args.diffusion_model,
                    border_condition=args.border_condition,
                ),
                args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
            )
    
        print(f"Running evaluation on {args.eval}")
        trainer.test(model=model, dataloaders=eval_loader, ckpt_path=args.load)
    else:
        # Train model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.load,
        )


if __name__ == "__main__":
    main()
