#!/usr/bin/env python3
import os, glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ------------------ CONFIG ------------------
BASE_DIR = "/mimer/NOBACKUP/groups/mlhighres/users/sm_ramfu/output/evaluation/long_exp"
OUT_DIR  = os.path.join(BASE_DIR, "ec_earth_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# Inputs
ECE_MASK_COARSE = os.path.join(BASE_DIR, "pr_gt_p95_r2_mask.nc")  # 0/1 exceedance on coarse grid
HCLIM_R2_PR = "/mimer/NOBACKUP/groups/mlhighres/projects/detex/HCLIM_EC-Earth3-Veg/remapped.pr_EUR-11_EC-Earth3-Veg_historical_r2i1p1f1_HCLIMcom-SMHI_HCLIM43-ALADIN_v1-r1_day_1951-2014_mm_day_noleap.nc"
NLAM_TILES_GLOB = os.path.join(BASE_DIR, "bias_exceedance_r2", "tiles", "pr", "frac", "pr_p95_exceedance_frac_*.nc")

# Time window
START, END = "1985-01-01", "2014-12-31"

# Threshold for NLAM fraction -> predicted daily exceedance
THRESH = 0.10

# Map crop (almost original Europe, but 2° less from east & south)
LON_MIN, LON_MAX = -16, 35
LAT_MIN, LAT_MAX =  33, 71.5

# Color limits
MCC_VMIN, MCC_VMAX = -1.0, 1.0
# --------------------------------------------

def safe_var(ds, preferred):
    if preferred in ds: return ds[preferred]
    if len(ds.data_vars) == 1: return next(iter(ds.data_vars.values()))
    for v in ds.data_vars:
        if preferred in v: return ds[v]
    return next(iter(ds.data_vars.values()))

def open_hclim_grid():
    pr = xr.open_dataset(HCLIM_R2_PR)
    return safe_var(pr, "pr").sel(time=slice(START, END))

def load_nlam_frac_daily():
    files = sorted(glob.glob(NLAM_TILES_GLOB))
    if not files:
        raise FileNotFoundError(f"No tiles at {NLAM_TILES_GLOB}")
    tmpl = xr.open_dataset(files[0])
    latn = "lat" if "lat" in tmpl.coords else "latitude"
    lonn = "lon" if "lon" in tmpl.coords else "longitude"
    lat_ref, lon_ref = tmpl[latn].values, tmpl[lonn].values
    def preprocess(ds):
        ds = ds.assign_coords({latn: (tmpl[latn].dims, lat_ref),
                               lonn: (tmpl[lonn].dims, lon_ref)}).sortby("time")
        ds[latn].attrs.pop("bounds", None); ds[lonn].attrs.pop("bounds", None)
        return ds
    ds = xr.open_mfdataset(files, combine="by_coords", preprocess=preprocess)
    varname = next(v for v in ds.data_vars if v.endswith("_p95_exceedance_frac"))
    return ds[varname].sel(time=slice(START, END))

def interpolate_to_hclim(ece_mask_coarse, hclim_sample):
    lat_h = "lat" if "lat" in hclim_sample.coords else "latitude"
    lon_h = "lon" if "lon" in hclim_sample.coords else "longitude"
    rename = {}
    if "latitude" in ece_mask_coarse.coords and lat_h == "lat":   rename["latitude"]  = "lat"
    if "longitude" in ece_mask_coarse.coords and lon_h == "lon":  rename["longitude"] = "lon"
    src = ece_mask_coarse.rename(rename)
    out = src.interp({lat_h: hclim_sample[lat_h], lon_h: hclim_sample[lon_h]}, method="nearest")
    return (out > 0.5).astype(np.uint8)

def daily_frac_to_mask(frac, thr=0.1):
    return (frac >= thr).astype(np.uint8).rename("nlam_pred_mask")

def compute_hclim_r2_daily_mask(hclim_pr):
    p95 = hclim_pr.quantile(0.95, dim="time")
    return (hclim_pr > p95).astype(np.uint8).rename("hclim_r2_p95_exceed_mask")

def align_times(*das):
    tmin = max([np.min(da.time.values) for da in das])
    tmax = min([np.max(da.time.values) for da in das])
    das = [da.sel(time=slice(str(np.datetime64(tmin, 'D')),
                             str(np.datetime64(tmax, 'D')))) for da in das]
    ref_t = das[0].time
    return [da.reindex(time=ref_t) for da in das]

def per_grid_confusion(pred: xr.DataArray, truth: xr.DataArray):
    pv, tv = pred.values, truth.values
    tp = np.logical_and(pv == 1, tv == 1).sum(axis=0)
    fp = np.logical_and(pv == 1, tv == 0).sum(axis=0)
    fn = np.logical_and(pv == 0, tv == 1).sum(axis=0)
    tn = np.logical_and(pv == 0, tv == 0).sum(axis=0)
    return tp, fp, fn, tn

def per_grid_mcc(tp, fp, fn, tn):
    tp = tp.astype(float); fp = fp.astype(float)
    fn = fn.astype(float); tn = tn.astype(float)
    num = tp*tn - fp*fn
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return np.where(den > 0, num/den, np.nan)

def global_mcc(tp, fp, fn, tn):
    # Use extended precision to prevent overflow for large domain totals
    tpG = np.longdouble(tp.sum())
    fpG = np.longdouble(fp.sum())
    fnG = np.longdouble(fn.sum())
    tnG = np.longdouble(tn.sum())
    num = tpG*tnG - fpG*fnG
    den = np.sqrt((tpG+fpG)*(tpG+fnG)*(tnG+fpG)*(tnG+fnG))
    mcc = float(num/den) if den > 0 else np.nan
    return float(np.clip(mcc, -1.0, 1.0)) if np.isfinite(mcc) else np.nan

def to_da(arr, like: xr.DataArray, name: str):
    base = like.isel(time=0, drop=True)
    return xr.DataArray(arr, dims=base.dims,
                        coords={d: base.coords[d] for d in base.dims},
                        name=name)

def main():
    # HCLIM grid/time
    hclim_pr = open_hclim_grid()

    # EC-Earth r2 mask (coarse) -> HCLIM grid
    ece_ds = xr.open_dataset(ECE_MASK_COARSE)
    ece_mask_coarse = None
    for cand in ("pr", "mask", "exceed", "gt_mask"):
        if cand in ece_ds.data_vars:
            ece_mask_coarse = ece_ds[cand]; break
    if ece_mask_coarse is None:
        ece_mask_coarse = next(iter(ece_ds.data_vars.values()))
    ece_mask_coarse = ece_mask_coarse.sel(time=slice(START, END))
    ece_mask = interpolate_to_hclim(ece_mask_coarse, hclim_pr)

    # Neural-LAM predicted mask (frac ≥ THRESH)
    frac = load_nlam_frac_daily()
    nlam_mask = daily_frac_to_mask(frac, THRESH)

    # HCLIM r2 daily mask (> its own p95)
    hclim_mask = compute_hclim_r2_daily_mask(hclim_pr)

    # Align time
    nlam_mask, hclim_mask, ece_mask = align_times(nlam_mask, hclim_mask, ece_mask)

    # Confusion & MCCs
    tp_a, fp_a, fn_a, tn_a = per_grid_confusion(nlam_mask,  ece_mask)  # NLAM vs ECE
    tp_b, fp_b, fn_b, tn_b = per_grid_confusion(hclim_mask, ece_mask)  # HCLIM vs ECE

    mcc_a = per_grid_mcc(tp_a, fp_a, fn_a, tn_a)
    mcc_b = per_grid_mcc(tp_b, fp_b, fn_b, tn_b)
    mcc_a_da = to_da(mcc_a, nlam_mask, "MCC_NLAM_vs_ECEr2")
    mcc_b_da = to_da(mcc_b, nlam_mask, "MCC_HCLIMr2_vs_ECEr2")

    mccG_a = global_mcc(tp_a, fp_a, fn_a, tn_a)
    mccG_b = global_mcc(tp_b, fp_b, fn_b, tn_b)
    print(f"Global MCC — Neural-LAM vs EC-Earth r2: {mccG_a:.3f}")
    print(f"Global MCC — HCLIM r2 vs EC-Earth r2 : {mccG_b:.3f}")

    # Plot (shared colorbar, fixed limits, cropped extent)
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0),
                             subplot_kw={"projection": proj},
                             constrained_layout=True)

    cmap = "coolwarm"; vmin, vmax = MCC_VMIN, MCC_VMAX
    ims = []
    for ax, da, title, gval in [
        (axes[0], mcc_a_da, "MCC: Neural-LAM vs EC-Earth r2", mccG_a),
        (axes[1], mcc_b_da, "MCC: HCLIM r2 vs EC-Earth r2",   mccG_b),
    ]:
        im = da.plot.pcolormesh(ax=ax, transform=proj, cmap=cmap,
                                vmin=vmin, vmax=vmax, add_colorbar=False)
        ims.append(im)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
        ax.set_title(f"{title}\nGlobal MCC = {gval:.2f}", fontsize=12, pad=8)

    cbar = fig.colorbar(ims[0], ax=axes, orientation="horizontal", fraction=0.06, pad=0.12)
    cbar.set_label("MCC")

    out_png = os.path.join(OUT_DIR, "fig_MCC_NeuralLAM_and_HCLIM_vs_ECEr2_sharedcbar_cropped.png")
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"🖼️  Saved {out_png}")

if __name__ == "__main__":
    # HPC-friendly defaults
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    main()

