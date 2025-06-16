import numpy as np
import wandb

# First-party
from neural_lam import constants, utils, config

# Load static features for grid/data
static_data_dict = utils.load_static_data("meps")
data_std = static_data_dict["data_std"].numpy()
config_loader = config.Config.from_file("/proj/berzelius-2022-164/users/x_erila/neural-lam/neural_lam/data_config.yaml")
WANDB_PROJECT = "neural-lam_prob_eval"

PREFIX = "val" # "test" or "val"
RUN_NAME = f"FM_1200e_{PREFIX}_5_ens"
METRIC_NAMES = ["ens_rmse", "crps_ens", "spsk_ratio"]  # or "ens_rmse" "crps_ens", "spsk_ratio" "spread", etc.

# CSV_PATH = f"/proj/berzelius-2022-164/users/x_erila/neural-lam/neural_lam/lam_plotting/lam_eval/prob_hi_spskr.csv"
# RUN_NAME = f"Graph-EFM_{PREFIX}_25_ens"
wandb.init(project=WANDB_PROJECT, name=RUN_NAME)

for METRIC_NAME in METRIC_NAMES:
    CSV_PATH = f"/proj/berzelius-2022-164/users/x_erila/neural-lam/neural_lam/lam_plotting/lam_eval/FM_1200_{PREFIX}_5_{METRIC_NAME}.csv"
    full_log_name = f"{PREFIX}_{METRIC_NAME}"

    # Read metrics from csv file
    metric_np = np.genfromtxt(CSV_PATH, delimiter=",")

    # Get mean for the metric over all variables
    metric_mean = np.mean(metric_np / data_std, axis=1)  # (pred_steps,)

    # Add the mean to the log dict and the metric name "Mean"
    var_names = config_loader.dataset.var_names + ["Mean"]
    metric_np = np.column_stack(
        [metric_np, metric_mean]
    )

    # Initialize wandb run

    # Logging all of the metrics as line plots
    for i, varname in enumerate(var_names):  # adjust var names as needed
        wandb.log({f"{full_log_name}_{varname}_lineplot": wandb.plot.line_series(
            xs=list(range(metric_np.shape[0])),
            ys=[metric_np[:, i].tolist()],
            keys=[RUN_NAME],
            title=f"{full_log_name} {varname}",
            xname="Time Step"
        )})

wandb.finish()
