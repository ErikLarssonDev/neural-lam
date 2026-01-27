#!/usr/bin/env python3
import os
import xarray as xr
import numpy as np
import pandas as pd
import traceback
from typing import List, Tuple, Optional

"""
Resumable, Alvis-safe streaming evaluation for one long experiment (1985–2014).

- Checks what already exists and only computes what's missing.
- Continues only with ensemble members that still need outputs.
- Monthly tiles are written once per month (no appends).
"""

# -----------------------------
# CONFIG
# -----------------------------
output_dir = "/mimer/NOBACKUP/groups/mlhighres/users/sm_ramfu/output/evaluation/long_exp/bias_exceedance_r2/fix"

gt_paths = {
    "pr":  "/mimer/NOBACKUP/groups/mlhighres/projects/detex/HCLIM_EC-Earth3-Veg/remapped.pr_EUR-11_EC-Earth3-Veg_historical_r2i1p1f1_HCLIMcom-SMHI_HCLIM43-ALADIN_v1-r1_day_1951-2014_mm_day_noleap.nc",
    "tas": "/mimer/NOBACKUP/groups/mlhighres/projects/detex/HCLIM_EC-Earth3-Veg/remapped.tas_EUR-12_EC-Earth3-Veg_historical_r2i1p1f1_HCLIMcom-SMHI_HCLIM43-ALADIN_v1-r1_day_1951-2014_noleap.nc",
}

variables = ["pr", "tas"]

exp_dir = "/mimer/NOBACKUP/groups/mlhighres/users/mikhaili/neural-lam/output/120825/exp9_long/netcdf"

start_date = "1998-12-01"
end_date   = "1998-12-31"

num_members = 100

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

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def maybe_save_da(da: xr.DataArray, path: str, mode: str = "w"):
    ensure_dir(os.path.dirname(path))
    name = da.name or "data"
    enc = {name: {**COMPRESS}}
    da.to_netcdf(path, mode=mode, engine=ENGINE_WRITE, encoding=enc)

def safe_load_var(ds: xr.Dataset, preferred_name: str) -> xr.DataArray:
    if preferred_name in ds:
        return ds[preferred_name]
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars.values()))
    base = preferred_name.split("_")[0]
    if base in ds:
        return ds[base]
    raise KeyError(f"No variable named {preferred_name} in dataset. Found: {list(ds.data_vars)}")

def list_chunk_files(var: str, kind: str, member_idx: Optional[int] = None) -> List[str]:
    files = []
    for (s,e) in chunk_bounds:
        if kind == "ensemble_mean":
            fn = f"{var}_ensemble_mean_SI_{s}_{e}.nc"
        else:
            assert member_idx is not None
            fn = f"{var}_ensemble_member_{member_idx}_SI_{s}_{e}.nc"
        files.append(os.path.join(exp_dir, fn))
    return files

def open_concat(var: str, kind: str, member_idx: Optional[int] = None) -> xr.DataArray:
    files = list_chunk_files(var, kind, member_idx)
    existing = [f for f in files if os.path.exists(f)]
    if not existing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(files))
    ds = xr.open_mfdataset(existing, combine="by_coords")
    da = safe_load_var(ds, var)
    return da.sel(time=slice(start_date, end_date))

def month_slices(start: str, end: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    months = pd.date_range(start=start, end=end, freq="MS")
    out = []
    for i, m in enumerate(months):
        m_end = pd.to_datetime(end) if i == len(months)-1 else months[i+1] - pd.Timedelta(days=1)
        out.append((m, m_end))
    return out

# Streaming stats helpers
def _accumulate_sums(sums: dict, da_x: xr.DataArray, da_y: xr.DataArray):
    da_y = da_y.reindex_like(da_x)
    n = int(da_x.sizes.get("time", 0))
    if n == 0: return
    sums["n"] += n
    sx  = da_x.sum("time").compute().astype("float64").values
    sy  = da_y.sum("time").compute().astype("float64").values
    sxx = (da_x*da_x).sum("time").compute().astype("float64").values
    syy = (da_y*da_y).sum("time").compute().astype("float64").values
    sxy = (da_x*da_y).sum("time").compute().astype("float64").values
    sse = ((da_y - da_x)**2).sum("time").compute().astype("float64").values
    if sums["sx"] is None:
        sums["sx"],sums["sy"],sums["sxx"],sums["syy"],sums["sxy"],sums["sse"] = sx,sy,sxx,syy,sxy,sse
    else:
        sums["sx"] += sx; sums["sy"] += sy; sums["sxx"] += sxx; sums["syy"] += syy; sums["sxy"] += sxy; sums["sse"] += sse

def _finalize_corr_bias_rmse(template: xr.DataArray, sums: dict) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    n    = float(sums["n"]) if sums["n"] else 1.0
    sx   = sums["sx"]; sy = sums["sy"]; sxx = sums["sxx"]; syy = sums["syy"]; sxy = sums["sxy"]; sse = sums["sse"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov  = sxy - (sx*sy)/n
        varx = sxx - (sx*sx)/n
        vary = syy - (sy*sy)/n
        corr = cov / np.sqrt(varx * vary)
        bias = (sy - sx) / n
        rmse = np.sqrt(sse / n)
    tmpl0 = template.isel(time=0, drop=True)
    dims = tmpl0.dims
    coords = {d: tmpl0.coords[d] for d in dims}
    return (
        xr.DataArray(corr, dims=dims, coords=coords, name=f"{template.name}_corr"),
        xr.DataArray(bias, dims=dims, coords=coords, name=f"{template.name}_mean_diff"),
        xr.DataArray(rmse, dims=dims, coords=coords, name=f"{template.name}_rmse"),
    )

# Tiling utils
def _month_tag(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m")

def _tile_path(kind: str, var: str, tag: str) -> str:
    base = os.path.join(output_dir, "tiles", var, kind)
    ensure_dir(base)
    if kind == "gt_exc":
        fn = f"{var}_p95_gt_exceed_mask_{tag}.nc"
    elif kind == "frac":
        fn = f"{var}_p95_exceedance_frac_{tag}.nc"
    elif kind == "joint":
        fn = f"{var}_p95_joint_gtfrac_{tag}.nc"
    else:
        raise ValueError("unknown kind")
    return os.path.join(base, fn)

def _member_done(var: str, member_idx: int) -> bool:
    c = os.path.join(output_dir, f"{var}_corr_member{member_idx}.nc")
    b = os.path.join(output_dir, f"{var}_mean_diff_member{member_idx}.nc")
    return os.path.exists(c) and os.path.exists(b)

def _remaining_members(var: str) -> List[int]:
    return [i for i in range(1, num_members+1) if not _member_done(var, i)]

# -----------------------------
# MAIN
# -----------------------------
def evaluate_long_run_streaming():
    # HPC niceties
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    ensure_dir(output_dir)

    for var in variables:
        try:
            print(f"\n🔄 Evaluating (streaming, resumable) {var} 1985–2014")

            # Non-time output paths
            p95_file       = f"{output_dir}/{var}_p95_gt.nc"
            corr_ens_file  = f"{output_dir}/{var}_corr_ensmean.nc"
            bias_ens_file  = f"{output_dir}/{var}_mean_diff_ensmean.nc"
            rmse_ens_file  = f"{output_dir}/{var}_rmse_ensmean.nc"
            freq_gt_file   = f"{output_dir}/{var}_freq_gt.nc"
            freq_sim_file  = f"{output_dir}/{var}_freq_sim.nc"

            # Load GT and ensemble mean
            gt_full  = xr.open_dataset(gt_paths[var])[var].sel(time=slice(start_date, end_date))
            ens_mean = open_concat(var, "ensemble_mean").rename(var)

            # p95 (compute once)
            if not os.path.exists(p95_file):
                print("  ▶ Computing GT p95 once …")
                p95 = gt_full.quantile(0.95, dim="time").astype("float32").rename(f"{var}_p95")
                maybe_save_da(p95, p95_file, mode="w")
            p95 = safe_load_var(xr.open_dataset(p95_file), f"{var}_p95")

            # ----------------- Monthly GT exceedance tiles (resume-aware)
            print("  ▶ Writing monthly GT exceedance tiles (skip existing) …")
            for (m_start, m_end) in month_slices(start_date, end_date):
                tag = _month_tag(m_start)
                tile_path = _tile_path("gt_exc", var, tag)
                if os.path.exists(tile_path):
                    continue
                t0, t1 = str(m_start.date()), str(m_end.date())
                gt = gt_full.sel(time=slice(t0, t1))
                if gt.sizes.get("time", 0) == 0:
                    continue
                gt_exc = (gt > p95).astype(np.uint8).rename(f"{var}_p95_gt_exceed_mask")
                maybe_save_da(gt_exc, tile_path, mode="w")

            # ----------------- Ensemble-mean metrics (resume-aware)
            need_ens_stats = not (os.path.exists(corr_ens_file) and
                                  os.path.exists(bias_ens_file) and
                                  os.path.exists(rmse_ens_file) and
                                  os.path.exists(freq_gt_file) and
                                  os.path.exists(freq_sim_file))
            if need_ens_stats:
                print("  ▶ Computing ensemble-mean correlation/bias/rmse + frequencies …")
                sums = {"n": 0, "sx": None, "sy": None, "sxx": None, "syy": None, "sxy": None, "sse": None}
                freq_gt_acc  = None
                freq_sim_acc = None
                for (m_start, m_end) in month_slices(start_date, end_date):
                    t0, t1 = str(m_start.date()), str(m_end.date())
                    gt = gt_full.sel(time=slice(t0, t1))
                    em = ens_mean.sel(time=slice(t0, t1))
                    if gt.sizes.get("time", 0) == 0:
                        continue
                    _accumulate_sums(sums, gt, em)
                    gt_cnt  = (gt > p95).sum("time").compute().astype("int32").values
                    sim_cnt = (em > p95).sum("time").compute().astype("int32").values
                    if freq_gt_acc is None:
                        freq_gt_acc, freq_sim_acc = gt_cnt, sim_cnt
                    else:
                        freq_gt_acc += gt_cnt; freq_sim_acc += sim_cnt

                corr_da, bias_da, rmse_da = _finalize_corr_bias_rmse(gt_full, sums)
                corr_da = corr_da.rename(f"{var}_corr_ensmean")
                bias_da = bias_da.rename(f"{var}_mean_diff_ensmean")
                rmse_da = rmse_da.rename(f"{var}_rmse_ensmean")
                maybe_save_da(corr_da,  corr_ens_file)
                maybe_save_da(bias_da,  bias_ens_file)
                maybe_save_da(rmse_da,  rmse_ens_file)

                freq_gt_DA  = xr.DataArray(freq_gt_acc,  dims=rmse_da.dims, coords=rmse_da.coords, name=f"{var}_freq_gt")
                freq_sim_DA = xr.DataArray(freq_sim_acc, dims=rmse_da.dims, coords=rmse_da.coords, name=f"{var}_freq_sim")
                maybe_save_da(freq_gt_DA,  freq_gt_file)
                maybe_save_da(freq_sim_DA, freq_sim_file)
            else:
                print("  ▶ Ensemble-mean metrics already present — skipping.")

            # ----------------- Per-member correlation & bias (resume-aware)
            todo = _remaining_members(var)
            if todo:
                print(f"  ▶ Members remaining for {var}: {todo[0]}..{todo[-1]} ({len(todo)})")
            else:
                print(f"  ▶ All member metrics already present for {var}.")

            for member_idx in todo:
                try:
                    member_all = open_concat(var, "ensemble_member", member_idx).rename(var)
                except FileNotFoundError:
                    print(f"    ⚠️ Missing files for member {member_idx} ({var}). Skipping.")
                    continue

                sums_m = {"n": 0, "sx": None, "sy": None, "sxx": None, "syy": None, "sxy": None, "sse": None}
                for (m_start, m_end) in month_slices(start_date, end_date):
                    t0, t1 = str(m_start.date()), str(m_end.date())
                    gt = gt_full.sel(time=slice(t0, t1))
                    mm = member_all.sel(time=slice(t0, t1))
                    if gt.sizes.get("time", 0) == 0:
                        continue
                    _accumulate_sums(sums_m, gt, mm)

                corr_m, bias_m, _ = _finalize_corr_bias_rmse(gt_full, sums_m)
                corr_m = corr_m.rename(f"{var}_corr_member{member_idx}")
                bias_m = bias_m.rename(f"{var}_mean_diff_member{member_idx}")
                maybe_save_da(corr_m, f"{output_dir}/{var}_corr_member{member_idx}.nc")
                maybe_save_da(bias_m, f"{output_dir}/{var}_mean_diff_member{member_idx}.nc")
                print(f"    ✅ Member {member_idx} done ({var})")

            # ----------------- Exceedance fraction + joint tiles (resume-aware per month)
            print("  ▶ Exceedance fraction/joint monthly tiles (skip existing months) …")
            for (m_start, m_end) in month_slices(start_date, end_date):
                tag = _month_tag(m_start)
                frac_path  = _tile_path("frac",  var, tag)
                joint_path = _tile_path("joint", var, tag)
                if os.path.exists(frac_path) and os.path.exists(joint_path):
                    continue

                t0, t1 = str(m_start.date()), str(m_end.date())
                gt = gt_full.sel(time=slice(t0, t1))
                if gt.sizes.get("time", 0) == 0:
                    continue
                T = int(gt.sizes["time"])
                tmpl = gt.isel(time=0)
                space_dims = tmpl.dims
                space_coords = {d: tmpl.coords[d] for d in space_dims}

                acc = np.zeros((T,) + tuple(int(tmpl.sizes[d]) for d in space_dims), dtype=np.uint16)
                used_members = 0
                for member_idx in range(1, num_members+1):
                    try:
                        mm = open_concat(var, "ensemble_member", member_idx).sel(time=slice(t0, t1))
                    except FileNotFoundError:
                        continue
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
                    name=f"{var}_p95_exceedance_frac"
                )
                maybe_save_da(frac_da, frac_path)

                gt_exc = (gt > p95).astype(np.uint8)
                joint = (frac_da * gt_exc).astype("float32").rename(f"{var}_p95_joint_gtfrac")
                maybe_save_da(joint, joint_path)

            print(f"✅ Done (resumable) {var}")

        except Exception as e:
            print(f"❌ Error in {var}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    evaluate_long_run_streaming()

