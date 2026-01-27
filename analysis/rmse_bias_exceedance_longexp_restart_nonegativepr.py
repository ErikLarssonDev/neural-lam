#!/usr/bin/env python3
import os
import xarray as xr
import numpy as np
import pandas as pd
import traceback
from typing import List, Tuple, Optional

# =========================
# CONFIG
# =========================
# New, separate output directory (so we recompute regardless of existing outputs elsewhere)
output_dir = "/mimer/NOBACKUP/groups/mlhighres/users/sm_ramfu/output/evaluation/long_exp/bias_exceedance_r2/pr_recheck_noneg"

# Ground truth (unchanged)
gt_path_pr  = "/mimer/NOBACKUP/groups/mlhighres/projects/detex/HCLIM_EC-Earth3-Veg/remapped.pr_EUR-11_EC-Earth3-Veg_historical_r2i1p1f1_HCLIMcom-SMHI_HCLIM43-ALADIN_v1-r1_day_1951-2014_mm_day_noleap.nc"

# Long experiment location (members & chunks)
exp_dir = "/mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam/output/120825/exp9_long/netcdf"
num_members = 100

# Period to evaluate (GT stops end of 2014)
start_date = "1985-01-01"
end_date   = "2014-12-31"

# Chunk bounds on disk (as provided)
chunk_bounds = [
    ("1985-01-01", "1990-01-01"),
    ("1990-01-01", "1995-01-01"),
    ("1995-01-01", "2000-01-01"),
    ("2000-01-01", "2005-01-01"),
    ("2005-01-01", "2010-01-01"),
    ("2010-01-01", "2015-01-01"),
]

ENGINE_WRITE = "h5netcdf"
COMPRESS = {"zlib": True, "complevel": 1}

plt_dpi = 120  # if you decide to add quick plots later

# =========================
# HELPERS
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def maybe_save_da(da: xr.DataArray, path: str):
    ensure_dir(os.path.dirname(path))
    name = da.name or "data"
    enc = {name: {**COMPRESS}}
    da.to_netcdf(path, engine=ENGINE_WRITE, encoding=enc)

def list_member_chunk_files(member_idx: int) -> List[str]:
    return [os.path.join(exp_dir, f"pr_ensemble_member_{member_idx}_SI_{s}_{e}.nc")
            for (s,e) in chunk_bounds]

def list_ensmean_chunk_files() -> List[str]:
    return [os.path.join(exp_dir, f"pr_ensemble_mean_SI_{s}_{e}.nc")
            for (s,e) in chunk_bounds]

def safe_load_pr(ds: xr.Dataset) -> xr.DataArray:
    if "pr" in ds.data_vars:
        return ds["pr"]
    elif len(ds.data_vars) == 1:
        return next(iter(ds.data_vars.values()))
    raise KeyError(f"Could not find 'pr' in dataset; found {list(ds.data_vars)}")

def open_concat_pr(paths: List[str]) -> xr.DataArray:
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(paths))
    ds = xr.open_mfdataset(existing, combine="by_coords")
    da = safe_load_pr(ds)
    return da.sel(time=slice(start_date, end_date))

def month_slices(start: str, end: str):
    months = pd.date_range(start=start, end=end, freq="MS")
    out = []
    for i, m in enumerate(months):
        m_end = pd.to_datetime(end) if i == len(months)-1 else months[i+1] - pd.Timedelta(days=1)
        out.append((m, m_end))
    return out

def month_tag(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m")

def tile_path(kind: str, tag: str) -> str:
    # kind ∈ {"gt_exc", "frac", "joint"}
    base = os.path.join(output_dir, "tiles", "pr", kind)
    ensure_dir(base)
    if kind == "gt_exc":
        fn = f"pr_p95_gt_exceed_mask_{tag}.nc"
    elif kind == "frac":
        fn = f"pr_p95_exceedance_frac_{tag}.nc"
    elif kind == "joint":
        fn = f"pr_p95_joint_gtfrac_{tag}.nc"
    else:
        raise ValueError("unknown kind")
    return os.path.join(base, fn)

# =========================
# SCAN FOR NEGATIVES
# =========================
def scan_negatives():
    """
    Scan all long-exp pr files (ensmean + members) and report min and count of negatives per file.
    Save CSV report for traceability.
    """
    rows = []

    # Ensemble mean
    for fp in list_ensmean_chunk_files():
        if not os.path.exists(fp): 
            rows.append({"file": fp, "exists": False, "min": np.nan, "n_negative": np.nan})
            continue
        try:
            ds = xr.open_dataset(fp)
            pr = safe_load_pr(ds)
            mn = float(pr.min().compute().values)
            nneg = int((pr < 0).sum().compute().values)
            rows.append({"file": fp, "exists": True, "min": mn, "n_negative": nneg})
            ds.close()
        except Exception as e:
            rows.append({"file": fp, "exists": True, "min": np.nan, "n_negative": np.nan, "error": str(e)})

    # Members
    for m in range(1, num_members+1):
        for fp in list_member_chunk_files(m):
            if not os.path.exists(fp):
                rows.append({"file": fp, "exists": False, "min": np.nan, "n_negative": np.nan})
                continue
            try:
                ds = xr.open_dataset(fp)
                pr = safe_load_pr(ds)
                mn = float(pr.min().compute().values)
                nneg = int((pr < 0).sum().compute().values)
                rows.append({"file": fp, "exists": True, "min": mn, "n_negative": nneg})
                ds.close()
            except Exception as e:
                rows.append({"file": fp, "exists": True, "min": np.nan, "n_negative": np.nan, "error": str(e)})

    df = pd.DataFrame(rows)
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "negative_scan_pr.csv")
    df.to_csv(csv_path, index=False)
    print(f"🔎 Negative-scan report saved: {csv_path}")

# =========================
# EVALUATION (recomputed, clamping negatives in memory)
# =========================
def evaluate_pr():
    print("\n🔄 Recomputing precipitation evaluation (with clamp-to-zero, fresh output dir)")

    # Load GT (1985–2014) and compute p95 once
    gt_pr = xr.open_dataset(gt_path_pr)["pr"].sel(time=slice(start_date, end_date))
    p95 = gt_pr.quantile(0.95, dim="time").astype("float32").rename("pr_p95")
    maybe_save_da(p95, os.path.join(output_dir, "pr_p95_gt.nc"))

    # Load ensemble mean (and clamp to zero)
    pr_em = open_concat_pr(list_ensmean_chunk_files())
    pr_em = pr_em.clip(min=0)
    pr_em.name = "pr"

    # Basic ENS-mean metrics vs GT (corr/bias/rmse) — monthly streaming
    sums = {"n": 0, "sx": None, "sy": None, "sxx": None, "syy": None, "sxy": None, "sse": None}
    def acc(sums, x, y):
        y = y.reindex_like(x)
        n = int(x.sizes.get("time", 0))
        if n == 0: return
        sums["n"] += n
        sx  = x.sum("time").compute().astype("float64").values
        sy  = y.sum("time").compute().astype("float64").values
        sxx = (x*x).sum("time").compute().astype("float64").values
        syy = (y*y).sum("time").compute().astype("float64").values
        sxy = (x*y).sum("time").compute().astype("float64").values
        sse = ((y - x)**2).sum("time").compute().astype("float64").values
        if sums["sx"] is None:
            sums["sx"],sums["sy"],sums["sxx"],sums["syy"],sums["sxy"],sums["sse"] = sx,sy,sxx,syy,sxy,sse
        else:
            sums["sx"]+=sx; sums["sy"]+=sy; sums["sxx"]+=sxx; sums["syy"]+=syy; sums["sxy"]+=sxy; sums["sse"]+=sse

    freq_gt_acc  = None
    freq_sim_acc = None

    for (m_start, m_end) in month_slices(start_date, end_date):
        t0, t1 = str(m_start.date()), str(m_end.date())
        gt = gt_pr.sel(time=slice(t0, t1))
        em = pr_em.sel(time=slice(t0, t1))
        if gt.sizes.get("time", 0) == 0:
            continue

        # write GT exceedance tiles (fresh dir => recompute)
        gt_exc = (gt > p95).astype(np.uint8).rename("pr_p95_gt_exceed_mask")
        maybe_save_da(gt_exc, tile_path("gt_exc", month_tag(m_start)))

        # accumulate stats and frequencies
        acc(sums, gt, em)
        gt_cnt  = gt_exc.sum("time").compute().astype("int32").values
        sim_cnt = (em > p95).sum("time").compute().astype("int32").values
        if freq_gt_acc is None:
            freq_gt_acc, freq_sim_acc = gt_cnt, sim_cnt
        else:
            freq_gt_acc  += gt_cnt
            freq_sim_acc += sim_cnt

    # finalize and save ens-mean maps
    n    = float(sums["n"]) if sums["n"] else 1.0
    sx   = sums["sx"]; sy = sums["sy"]; sxx = sums["sxx"]; syy = sums["syy"]; sxy = sums["sxy"]; sse = sums["sse"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov  = sxy - (sx*sy)/n
        varx = sxx - (sx*sx)/n
        vary = syy - (sy*sy)/n
        corr = cov / np.sqrt(varx * vary)
        bias = (sy - sx) / n
        rmse = np.sqrt(sse / n)

    tmpl0 = gt_pr.isel(time=0, drop=True)
    dims = tmpl0.dims
    coords = {d: tmpl0.coords[d] for d in dims}

    corr_da = xr.DataArray(corr, dims=dims, coords=coords, name="pr_corr_ensmean")
    bias_da = xr.DataArray(bias, dims=dims, coords=coords, name="pr_mean_diff_ensmean")
    rmse_da = xr.DataArray(rmse, dims=dims, coords=coords, name="pr_rmse_ensmean")
    maybe_save_da(corr_da, os.path.join(output_dir, "pr_corr_ensmean.nc"))
    maybe_save_da(bias_da, os.path.join(output_dir, "pr_mean_diff_ensmean.nc"))
    maybe_save_da(rmse_da, os.path.join(output_dir, "pr_rmse_ensmean.nc"))

    freq_gt_DA  = xr.DataArray(freq_gt_acc,  dims=dims, coords=coords, name="pr_freq_gt")
    freq_sim_DA = xr.DataArray(freq_sim_acc, dims=dims, coords=coords, name="pr_freq_sim")
    maybe_save_da(freq_gt_DA,  os.path.join(output_dir, "pr_freq_gt.nc"))
    maybe_save_da(freq_sim_DA, os.path.join(output_dir, "pr_freq_sim.nc"))

    # ----- Exceedance fractions across members (monthly tiles), with clamp -----
    print("  ▶ Writing exceedance fraction/joint monthly tiles (members, clamped)…")
    for (m_start, m_end) in month_slices(start_date, end_date):
        tag = month_tag(m_start)
        frac_path  = tile_path("frac",  tag)
        joint_path = tile_path("joint", tag)

        # if you want to always recompute in the fresh dir, do not skip
        # (fresh output_dir means files won't exist anyway)
        t0, t1 = str(m_start.date()), str(m_end.date())
        gt = gt_pr.sel(time=slice(t0, t1))
        if gt.sizes.get("time", 0) == 0:
            continue

        T = int(gt.sizes["time"])
        tmpl = gt.isel(time=0)
        space_dims   = tmpl.dims
        space_coords = {d: tmpl.coords[d] for d in space_dims}
        acc = np.zeros((T,) + tuple(int(tmpl.sizes[d]) for d in space_dims), dtype=np.uint16)

        used_members = 0
        for member_idx in range(1, num_members+1):
            files = list_member_chunk_files(member_idx)
            if not any(os.path.exists(f) for f in files):
                continue
            try:
                mm = open_concat_pr(files).sel(time=slice(t0, t1))
            except FileNotFoundError:
                continue
            # clamp negatives to zero
            mm = mm.clip(min=0)

            if int(mm.sizes.get("time", 0)) != T:
                mm = mm.reindex(time=gt.time)

            exc = (mm > p95).astype(np.uint8).compute().values
            acc += exc.astype(np.uint16)
            used_members += 1

        if used_members == 0:
            continue

        frac = (acc.astype("float32") / float(used_members))
        frac_da = xr.DataArray(
            frac,
            dims=("time",) + space_dims,
            coords={"time": gt.time, **space_coords},
            name="pr_p95_exceedance_frac"
        )
        maybe_save_da(frac_da, frac_path)

        gt_exc = (gt > p95).astype(np.uint8)
        joint = (frac_da * gt_exc).astype("float32").rename("pr_p95_joint_gtfrac")
        maybe_save_da(joint, joint_path)

    print("✅ Recompute complete (fresh outputs, clamp applied).")

# =========================
# MAIN
# =========================
def main():
    try:
        ensure_dir(output_dir)
        print("🔎 Scanning long-exp precipitation files for negatives …")
        scan_negatives()
        evaluate_pr()
        print(f"\nAll outputs written under:\n  {output_dir}\n")
        print("NB: Raw long-exp files were NOT modified; clamp-to-zero was applied in-memory during evaluation.")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # HPC niceties
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    main()

