# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# Local
from . import config, constants, utils, metrics


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, data_config, title=None, step_length=3):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1)  # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in hours
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (h)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [
        f"{name} ({unit})"
        for name, unit in zip(
            data_config.dataset.var_names, data_config.dataset.var_units
        )
    ]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig

# @matplotlib.rc_context(utils.fractional_plot_bundle(1))
# def plot_diffusion(pred_steps, pred, target, obs_mask=None, title=None, vrange=None):
#     """
#     Plot diffusion steps, prediction and ground truth.
#     Each has shape (N_grid,)
#     """
#     # Get common scale for values
#     if vrange is None:
#         vmin = min(vals.min().cpu().item() for vals in (pred, target))
#         vmax = max(vals.max().cpu().item() for vals in (pred, target))
#     else:
#         vmin, vmax = vrange
#     # Set up masking of border region
#     mask_reshaped = obs_mask.reshape(*constants.GRID_SHAPE)
#     pixel_alpha = (
#         mask_reshaped.clamp(0.7, 1).cpu().numpy()
#     )  # Faded border region

#     fig, axes = plt.subplots(
#         1, 2, figsize=(13, 7), subplot_kw={"projection": constants.LAMBERT_PROJ}
#     )

#     # Plot pred and target
#     for ax, data in zip(axes, (target, pred)):
#         ax.coastlines()  # Add coastline outlines
#         data_grid = data.reshape(*constants.GRID_SHAPE).cpu().numpy()
#         im = ax.imshow(
#             data_grid,
#             origin="lower",
#             extent=constants.GRID_LIMITS,
#             alpha=pixel_alpha,
#             vmin=vmin,
#             vmax=vmax,
#             cmap="plasma",
#         )

#     # Ticks and labels
#     axes[0].set_title("Ground Truth", size=15)
#     axes[1].set_title("Prediction", size=15)
#     cbar = fig.colorbar(im, aspect=30)
#     cbar.ax.tick_params(labelsize=10)

#     if title:
#         fig.suptitle(title, size=20)

#     return fig


def plot_on_axis(
    ax,
    data,
    border_data=None,
    data_config=None,
    obs_mask=None,
    vmin=None,
    vmax=None,
    ax_title=None,
    cmap="plasma",
    grid_limits=None,
):
    """
    Plot weather state on given axis
    """
    # if data_config is None:
    #     data_config = config.Config.from_file("/proj/berzelius-2022-164/users/x_erila/neural-lam/neural_lam/data_config.yaml")
    # Set up masking of border region
    if obs_mask is None:
        pixel_alpha = 1
        data_grid = data.reshape(*constants.GRID_SHAPE).cpu().numpy()
    elif border_data is None:
        mask_reshaped = obs_mask.reshape(*constants.FULL_GRID_SHAPE)
        pixel_alpha = (
            mask_reshaped.clamp(0.7, 1).cpu().numpy()
        )  # Faded border region
        data_grid = data.reshape(*constants.FULL_GRID_SHAPE).cpu().numpy()
    else:
        mask_reshaped = obs_mask.reshape(*constants.FULL_GRID_SHAPE)
        pixel_alpha = (
            mask_reshaped.clamp(0.7, 1).cpu().numpy()
        )  # Faded border region
        # Create a blank array for the full image
        reconstructed_image = np.zeros(
            constants.FULL_GRID_SHAPE[0] * constants.FULL_GRID_SHAPE[1])

        # Fill in the interior and boundary regions
        reconstructed_image[obs_mask.cpu().numpy()] = data.cpu().numpy()
        reconstructed_image[~obs_mask.cpu().numpy()
                            ] = border_data.cpu().numpy()

        # Reshape to 2D for plotting
        data_grid = reconstructed_image.reshape(*constants.FULL_GRID_SHAPE)

    ax.coastlines()  # Add coastline outlines
    im = ax.imshow(
        data_grid,
        origin="lower",
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=grid_limits,
    )

    if ax_title:
        ax.set_title(ax_title, size=15)
    return im


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(
    pred,
    target,
    border,
    data_config,
    obs_mask=None,
    title=None,
    vrange=None,
    grid_limits=None,
):
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        subplot_kw={"projection": data_config.coords_projection},
    )

    # Plot pred and target
    for ax, data, border_data in zip(axes, (target, pred), (border, border)):
        im = plot_on_axis(
            ax, data, border_data, data_config, obs_mask, vmin, vmax, grid_limits=grid_limits
        )

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_ensemble_prediction(
    samples, target, border, ens_mean, ens_std, obs_mask, title=None, vrange=None
):
    """
    Plot example predictions, ground truth, mean and std.-dev.
    from ensemble forecast

    samples: (S, N_grid,)
    target: (N_grid,)
    ens_mean: (N_grid,)
    ens_std: (N_grid,)
    obs_mask: (N_grid,)
    (optional) title: title of plot
    (optional) vrange: tuple of length with common min and max of values
        (not for std.)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (samples, target))
        vmax = max(vals.max().cpu().item() for vals in (samples, target))
    else:
        vmin, vmax = vrange

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15, 15),
        subplot_kw={"projection": constants.LAMBERT_PROJ},
    )
    axes = axes.flatten()

    # Plot target, ensemble mean and std.
    gt_im = plot_on_axis(
        axes[0],
        target,
        border,
        obs_mask=obs_mask,
        vmin=vmin,
        vmax=vmax,
        ax_title="Ground Truth",
    )
    plot_on_axis(
        axes[1],
        ens_mean,
        border,
        obs_mask=obs_mask,
        vmin=vmin,
        vmax=vmax,
        ax_title="Ens. Mean",
    )

    if border is not None:
        border_std = border*0  # Zero std in border region
    else:
        border_std = None
    std_im = plot_on_axis(
        axes[2],
        ens_std,
        border_std,
        obs_mask=obs_mask,
        ax_title="Ens. Std."
    )  # Own vrange

    # Plot samples
    for member_i, (ax, member) in enumerate(
        zip(axes[3:], samples[:6]), start=1
    ):
        plot_on_axis(
            ax,
            member,
            border,
            obs_mask=obs_mask,
            vmin=vmin,
            vmax=vmax,
            ax_title=f"Member {member_i}",
        )

    # Turn off unused axes
    for ax in axes[(3 + samples.shape[0]):]:
        ax.axis("off")

    # Add colorbars
    values_cbar = fig.colorbar(
        gt_im, ax=axes[:2], aspect=60, location="bottom", shrink=0.9
    )
    values_cbar.ax.tick_params(labelsize=10)
    std_cbar = fig.colorbar(std_im, aspect=30, location="bottom", shrink=0.9)
    std_cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(
    error, data_config, obs_mask=None, title=None, vrange=None, grid_limits=None
):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    fig, ax = plt.subplots(
        figsize=(5, 4.8),
        subplot_kw={"projection": constants.LAMBERT_PROJ},
    )

    im = plot_on_axis(
        ax,
        error,
        data_config,
        obs_mask=obs_mask,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
        grid_limits=grid_limits,
    )

    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_latent_samples(prior_samples, vi_samples, title=None):
    """
    Plot samples of latent variable drawn from prior and
    variational distribution

    prior_samples: (samples, N_mesh, d_latent)
    vi_samples: (samples, N_mesh, d_latent)

    Returns:
    fig: the plot figure
    """
    num_samples, num_mesh_nodes, latent_dim = prior_samples.shape
    plot_dims = min(latent_dim, 3)  # Plot first 3 dimensions
    img_side_size = int(np.sqrt(num_mesh_nodes))
    assert img_side_size**2 == num_mesh_nodes, (
        "Number of mesh nodes is not a "
        "square number, can not plot latent samples as images"
    )

    # Get common scale for values
    vmin = min(
        vals[..., :plot_dims].min().cpu().item()
        for vals in (prior_samples, vi_samples)
    )
    vmax = max(
        vals[..., :plot_dims].max().cpu().item()
        for vals in (prior_samples, vi_samples)
    )

    # Create figure
    fig, axes = plt.subplots(num_samples, 2 * plot_dims, figsize=(20, 16))

    # Plot samples
    for row_i, (axes_row, prior_sample, vi_sample) in enumerate(
        zip(axes, prior_samples, vi_samples)
    ):

        for dim_i in range(plot_dims):
            prior_sample_reshaped = (
                prior_sample[:, dim_i]
                .reshape(img_side_size, img_side_size)
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            vi_sample_reshaped = (
                vi_sample[:, dim_i]
                .reshape(img_side_size, img_side_size)
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            # Plot every other as prior and vi
            prior_ax = axes_row[2 * dim_i]
            vi_ax = axes_row[2 * dim_i + 1]
            prior_ax.imshow(prior_sample_reshaped, vmin=vmin, vmax=vmax)
            vi_im = vi_ax.imshow(vi_sample_reshaped, vmin=vmin, vmax=vmax)

            if row_i == 0:
                # Add titles at top of columns
                prior_ax.set_title(f"d{dim_i} (prior)", size=15)
                vi_ax.set_title(f"d{dim_i} (vi)", size=15)

    # Remove ticks from all axes
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(vi_im, ax=axes, aspect=60, location="bottom")
    cbar.ax.tick_params(labelsize=15)

    if title:
        fig.suptitle(title, size=20)

    return fig


def radial_average(psd2D):
    """
    Radially average 2D power spectrum.
    psd2D shape: (B, H, W)
    returns: (B, R) radial profiles, where R = max(H,W)//2
    """
    B, H, W = psd2D.shape
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r.to(torch.int64)

    R = r.max().item() + 1
    radial_profiles = []
    for b in range(B):
        tbin = torch.bincount(
            r.flatten(), weights=psd2D[b].flatten(), minlength=R)
        nr = torch.bincount(r.flatten(), minlength=R)
        radial_profiles.append(tbin / torch.clamp(nr, min=1))
    return torch.stack(radial_profiles)  # (B, R)


def get_energy_spectra(power_spectrum):

    # Convert to numpy if tensor
    is_tensor = isinstance(power_spectrum, torch.Tensor)
    if is_tensor:
        power_spectrum = power_spectrum.detach().cpu().numpy()

    dx = 10 * 1000  # 10 km in meters
    dy = 10 * 1000  # 10 km in meters
    ny, nx = power_spectrum.shape[-2:]

    # Rest of the function remains the same
    # Get wavenumbers
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)

    # Create 2D wavenumber grid
    kxx, kyy = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kxx**2 + kyy**2)

    # Create wavenumber bins for azimuthal averaging
    k_bins = np.logspace(
        np.log10(k_mag[k_mag > 0].min()), np.log10(k_mag.max()), num=50
    )

    # Perform azimuthal averaging
    k_averaged = []
    power_averaged = []

    for i in range(len(k_bins) - 1):
        k_mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        if k_mask.any():
            k_averaged.append(np.mean(k_mag[k_mask]))
            power_averaged.append(np.mean(power_spectrum[k_mask]))

    # Convert to arrays
    k_averaged = np.array(k_averaged)
    power_averaged = np.array(power_averaged)

    return k_averaged, power_averaged  # (R,), (R,)


def plot_energy_spectra(spectra_gt, spectra_ml, var, title=None, show_legend=False):
    """Plot energy spectra comparison using pre-calculated spectra.

    Parameters
    ----------
    spectra_cache : dict
        Cache containing pre-calculated spectra
    var : str
        Variable name to plot
    level : float, optional
        Vertical level in hPa
    show_legend : bool, optional
        Whether to show the legend (default: False)
    """

    # k_gt, spec_gt = get_energy_spectra(spectra_gt)  # (R,), (R,)
    # k_ml, spec_ml = get_energy_spectra(spectra_ml)  # (R,), (R,)

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    x = np.arange(0, spectra_gt.shape[-1])

    # Plot ground truth spectrum
    ax.loglog(
        x,
        spectra_gt,
        label="Ground Truth",
    )

    # Plot ML spectrum
    ax.loglog(
        x,
        spectra_ml,
        label="Prediction",
    )

    # Add LSD metric
    add_lsd_to_plot(ax, spectra_gt, spectra_ml)

    # Customize plot
    ax.set_xlabel("Wavenumber (1/m)")
    unit = constants.PARAM_UNITS[constants.USED_PARAMS][var]
    ax.set_ylabel(f"Energy Density (({unit})² * m)")
    var_name = constants.PARAM_NAMES_SHORT[constants.USED_PARAMS][var]
    title = title if title is not None else f"Energy Spectra Comparison for {var_name}"
    ax.set_title(title)
    if show_legend:
        ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    return fig


def add_lsd_to_plot(ax, true_spectrum, ml_spectrum):
    """
    Add LSD metric as text box to spectrum plot
    """
    lsd_ml = metrics.calculate_log_spectral_distance(
        true_spectrum, ml_spectrum
    )
    textstr = f"LSD = {lsd_ml:.4f}"

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.1,
        textstr,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )
