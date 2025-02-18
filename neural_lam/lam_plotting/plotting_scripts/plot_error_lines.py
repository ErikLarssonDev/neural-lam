# Standard library
import glob
import os

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xa
from tueplots import figsizes
import torch

# First-party
from neural_lam import constants, utils

PLOT_DIR_NAME = "line_plots"

# Load static features for grid/data
static_data_dict = utils.load_static_data("meps_example")

@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_lines(
    error_dir_path,
    model_lookup,
    metric_name_lookup,
    unitless_metrics=None,
    line_width=0.8,
    step_length=6,  # in hours
    plot_ylabel=False,
    separate_legend=False,
    var_names=constants.PARAM_NAMES_SHORT,
    var_units=constants.PARAM_UNITS,
    print_metrics=None,  # Dict of var: (index) to print
    legend_cols = 6,  # Max number that fit in width
):
    """
    Make lineplots with metrics from files stored in given directory.
    """
    model_order_index = {m: i for i, m in enumerate(model_lookup.keys())}
    print(f"Model order: {model_order_index}")

    var_names = np.concatenate([var_names, np.array(["Mean"])])
    var_units = np.concatenate([var_units, np.array([""])])


    # Load all data
    error_file_paths = {
        metric_name: glob.glob(
            os.path.join(error_dir_path, f"*_{metric_name}.csv")
        )
        for metric_name in metric_name_lookup.keys()
    }

    # Check for csv error arrays
    error_arrays = {}
    for metric_name, file_paths in error_file_paths.items():
        if file_paths:  # Only for metrics that exist
            # Some files with this error exists
            err_array_dict = {}
                                            
            for file_path in file_paths:
                base_file_name = os.path.basename(file_path)
                model_name = "_".join(base_file_name.split("_")[:-1])

                err_array = np.genfromtxt(file_path, delimiter=",")
                mean_error = np.mean(err_array / static_data_dict["data_std"].numpy(), axis=1, keepdims=True)
                err_array_dict[model_name] = np.concatenate([err_array, mean_error], axis=1)

            error_arrays[metric_name] = err_array_dict

    # Check for netcdf results arrays
    xa_results = {}
    nc_paths = glob.glob(os.path.join(error_dir_path, "*.nc"))
    for path in nc_paths:
        model_name = os.path.basename(path)[:-3]
        raw_xds = xa.open_dataset(path)

        # Extract global region if necessary
        if "region" in raw_xds.coords:
            # Global region
            raw_xds = raw_xds.sel(region="global")

        # Change to variables used in our code (short names in constants.py)
        split_xds = utils.variable_split_xarray(raw_xds)
        xa_results[model_name] = split_xds

    assert error_arrays or xa_results, "No matching error files found"

    # Create plot directory
    plot_save_dir = os.path.join(error_dir_path, PLOT_DIR_NAME)
    os.makedirs(plot_save_dir, exist_ok=True)

    if error_arrays:
        num_time_steps = err_array.shape[0]
    else:
        num_time_steps = split_xds.lead_time.shape[0]

    # Make and save plots
    default_lead_times = step_length * (np.arange(num_time_steps) + 1)
    # Use whole days as ticks
    xticks = list(filter(lambda lt: lt % 24 == 0, default_lead_times))
    xtick_labels = list(map(lambda lt: lt // 24, xticks))
    for metric_name in metric_name_lookup.keys():
        plotted_legend_size = 0

        print(f"Plotting metric: {metric_name}")
        metric_arrays = (
            error_arrays[metric_name] if metric_name in error_arrays else {}
        )
        for var_i, (var_name, unit) in enumerate(zip(var_names, var_units)):
            png_save_dir = os.path.join(plot_save_dir, metric_name, "png")
            pdf_save_dir = os.path.join(plot_save_dir, metric_name, "pdf")
            os.makedirs(png_save_dir, exist_ok=True)
            os.makedirs(pdf_save_dir, exist_ok=True)

            tue_width, tue_height = figsizes.neurips2023()["figure.figsize"]
            # These are set arbitrarily
            fig, ax = plt.subplots(figsize=(tue_width / 3, 0.8 * tue_height))

            # Add error curves from metric_arrays
            model_curves = {}
            for model_name, errors in metric_arrays.items():
                model_curves[model_name] = (
                    errors[:, var_i],
                    default_lead_times,
                )
               
            # Add error curves from xa_results
            for model_name, res_xds in xa_results.items():
                if var_name in res_xds:
                    var_res_xda = res_xds[var_name]

                    lead_times = res_xds.lead_time.to_numpy().astype(
                        "timedelta64[h]"
                    )
                    if metric_name == "rmse":
                        # Handle RMSE as special case
                        if "ensemble_mean_mse" in var_res_xda.metric:
                            mse_vals = var_res_xda.sel(
                                metric="ensemble_mean_mse"
                            ).to_numpy()
                            model_curves[model_name] = (
                                np.sqrt(mse_vals),
                                lead_times,
                            )
                        elif "mse" in var_res_xda.metric:
                            mse_vals = var_res_xda.sel(metric="mse").to_numpy()
                            model_curves[model_name] = (
                                np.sqrt(mse_vals),
                                lead_times,
                            )
                    elif metric_name in var_res_xda.metric:
                        metric_vals = var_res_xda.sel(
                            metric=metric_name
                        ).to_numpy()
                        model_curves[model_name] = (metric_vals, lead_times)
                    elif metric_name == "crps" and "mae" in var_res_xda.metric:
                        # Use MAE as CRPS for det. models
                        mae_vals = var_res_xda.sel(metric="mae").to_numpy()
                        model_curves[model_name] = (
                            mae_vals,
                            lead_times,
                        )
                    # Else do not add curve from model

            # Plot all curves
            lines = []
            for model_name in sorted(
                (m for m in model_curves if m in model_order_index), 
                key=lambda m: model_order_index[m]
            ):

                errors, lead_times = model_curves[model_name]

                # model dict lookup
                label_model_name, model_color, model_ls = model_lookup[
                    model_name
                ]

                lines.append(
                    ax.plot(
                        lead_times,
                        errors,
                        label=label_model_name,
                        lw=line_width,
                        color=model_color,
                        ls=model_ls,
                    )[0]
                )

                if print_metrics and var_name in print_metrics:
                    for lead_index in print_metrics[var_name]:
                        lead_h = lead_times[lead_index]
                        metric_val = errors[lead_index]
                        print(
                            f"{label_model_name}, {var_name}, lead {lead_h}, "
                            f"{metric_name}: {metric_val}"
                        )

            # Style figure
            if not separate_legend:
                ax.legend(handlelength=1.5, loc="lower right", ncol=2, columnspacing=0.6)
            # ax.set_title(f"{var_name}")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlabel("Lead time (Days)")
            ax.set_xlim(0, default_lead_times[-1])

            plt.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            ax.figure.canvas.draw()

            if plot_ylabel:
                yaxis_label = metric_name_lookup[metric_name]
                if (
                    unitless_metrics is None
                    or metric_name not in unitless_metrics
                ):
                    offset = ax.yaxis.get_major_formatter().get_offset()
                    yaxis_label = (
                        yaxis_label
                        # + f" ({offset}{' ' if offset else ''}{unit})"
                    )
                ax.set_ylabel(yaxis_label)

            # Constrain y-axis for spsk-r
            if metric_name == "spskr":
                ax.set_ylim(0, 1.3)

            # Save
            fig.savefig(
                os.path.join(pdf_save_dir, f"{metric_name}_{var_name}.pdf")
            )
            fig.savefig(
                os.path.join(png_save_dir, f"{metric_name}_{var_name}.png"),
                dpi=500,
            )

            # Potentially plot legend separately
            legend_entries = len(lines)
            if legend_entries > plotted_legend_size:
                plotted_legend_size = legend_entries

                # Make legend on separate figure
                legend_width = tue_width * (
                    min(legend_entries, legend_cols) / legend_cols
                )
                legend_height = 0.2 * (legend_entries // legend_cols)
                legend_fig, legend_ax = plt.subplots(
                    figsize=(legend_width, legend_height)
                )
                legend_artist = legend_ax.legend(
                    handles=lines,
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=legend_fig.transFigure,
                    frameon=False,
                    fancybox=None,
                    shadow=False,
                    ncol=min(legend_cols, legend_entries),
                    # mode="expand",
                    loc="center"
                )
                legend_ax.axis("off")

                # Save legend
                legend_fig.savefig(
                    os.path.join(pdf_save_dir, f"{metric_name}_legend.pdf"),
                    bbox_inches="tight",
                    bbox_extra_artists=[legend_artist],
                )
                legend_fig.savefig(
                    os.path.join(png_save_dir, f"{metric_name}_legend.png"),
                    bbox_inches="tight",
                    bbox_extra_artists=[legend_artist],
                    dpi=500,
                )

            plt.close()
