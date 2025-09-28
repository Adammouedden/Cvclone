# evaluate_hurricane_accuracy.py
# Uses the same math as your forecast script to compute accuracy and hit rates within a radius.
import math, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import GRUSeq2Seq  # your model class

# ----------------------------
# Helpers (same math as training/inference)
# ----------------------------
def unwrap_lon(arr):
    return np.rad2deg(np.unwrap(np.deg2rad(arr.astype(np.float64)))).astype(np.float32)

def to_hours(ns):  # int ns -> float hours
    return (ns.astype(np.float64) / 3_600_000_000_000.0).astype(np.float32)

def equirect_inv(lat0, lon0, x, y):
    R = 6371.0088
    lat0r = math.radians(float(lat0))
    lat = np.degrees(y / R + math.radians(float(lat0)))
    lon = np.degrees(x / (R * math.cos(lat0r)) + math.radians(float(lon0)))
    return lat.astype(np.float32), lon.astype(np.float32)

def rewrap_180(lon):
    lw = (lon + 180.0) % 360.0 - 180.0
    lw[lw == 180.0] = -180.0
    return lw

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1 = np.radians(lat1.astype(np.float64))
    p2 = np.radians(lat2.astype(np.float64))
    dlat = p2 - p1
    dlon = np.radians(lon2.astype(np.float64) - lon1.astype(np.float64))
    a = np.sin(dlat/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2.0)**2
    return (2*R*np.arcsin(np.sqrt(a))).astype(np.float32)

def match_truth_by_time(gt_times_ns: np.ndarray,
                        gt_lat: np.ndarray,
                        gt_lon: np.ndarray,
                        target_times_ns: np.ndarray,
                        tol_ns: int):
    n_targets = len(target_times_ns)
    lat_t = np.empty(n_targets, dtype=np.float32)
    lon_t = np.empty(n_targets, dtype=np.float32)
    ok = np.zeros(n_targets, dtype=bool)
    if gt_times_ns is None or len(gt_times_ns) == 0:
        return lat_t, lon_t, ok
    j = 0
    for i, t in enumerate(target_times_ns):
        while j+1 < len(gt_times_ns) and abs(gt_times_ns[j+1]-t) <= abs(gt_times_ns[j]-t):
            j += 1
        if abs(gt_times_ns[j] - t) <= tol_ns:
            lat_t[i] = gt_lat[j]
            lon_t[i] = gt_lon[j]
            ok[i] = True
    return lat_t, lon_t, ok

# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Evaluate hurricane trajectory forecasts.")
    ap.add_argument("--csv_in", type=str, default="hurricane_points.csv",
                    help="Observed points CSV with columns [sid,time_ns,lat,lon].")
    ap.add_argument("--ckpt", type=str, default="saved_gru_seq2seq_best.pt",
                    help="Path to model checkpoint.")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--tin", type=int, default=6)
    ap.add_argument("--tout", type=int, default=8)
    ap.add_argument("--dt_norm", type=float, default=3.0)
    ap.add_argument("--match_tol_mult", type=float, default=1.75,
                    help="Tolerance multiplier × median cadence for time matching.")
    ap.add_argument("--radius_km", type=float, default=50.0,
                    help="Primary radius to report hit rate (%% within R km).")
    ap.add_argument("--radius_curve", type=str, default="25,50,100,150,200",
                    help="Comma-separated list of radii (km) for a hit-rate curve.")
    ap.add_argument("--out_csv", type=str, default="accuracy_report.csv",
                    help="Per-lead metrics CSV.")
    ap.add_argument("--out_json", type=str, default="accuracy_summary.json",
                    help="Overall summary JSON.")
    args = ap.parse_args()

    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    TIN, TOUT, DT_NORM = args.tin, args.tout, args.dt_norm
    MATCH_TOL_NS_MULTIPLIER = args.match_tol_mult
    primary_R = float(args.radius_km)
    radii = [float(r.strip()) for r in args.radius_curve.split(",") if r.strip()]

    # Load observations
    df = pd.read_csv(args.csv_in).dropna(subset=["sid","time_ns","lat","lon"])
    df = df.sort_values(["sid","time_ns"])

    # Load model
    model = GRUSeq2Seq(in_dim=3, hid=64, num_layers=1, out_steps=TOUT).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.eval()

    # Collect matched errors along with lead time index (1..TOUT)
    err_list = []         # km
    persist_list = []     # km
    lead_idx_list = []    # 1..TOUT

    for sid, g in df.groupby("sid", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < TIN:
            continue

        lat = g["lat"].to_numpy(np.float32)
        lon = unwrap_lon(g["lon"].to_numpy(np.float32))
        ts  = g["time_ns"].to_numpy(np.int64)

        # context
        lat_ctx = lat[-TIN:]; lon_ctx = lon[-TIN:]; ts_ctx = ts[-TIN:]

        # cadence
        th   = to_hours(ts_ctx)
        dth  = np.diff(th, prepend=th[0])
        dt_h = np.maximum(np.median(dth[1:]) if len(dth)>1 else DT_NORM, 1e-3)
        dts_out = np.full((TOUT,1), dt_h/DT_NORM, dtype=np.float32)

        # x_in = [x, y, Δt_norm]
        R = 6371.0088
        lat0, lon0 = float(lat_ctx[0]), float(lon_ctx[0])
        lat0r = math.radians(lat0)
        x = (np.radians(lon_ctx) - math.radians(lon0)) * math.cos(lat0r) * R
        y = (np.radians(lat_ctx) - math.radians(lat0)) * R
        dth_ctx = np.diff(th, prepend=th[0]); dth_ctx[0] = max(dth_ctx[0], DT_NORM)
        x_in = np.stack([x, y, (dth_ctx/DT_NORM)], axis=-1).astype(np.float32)[None, ...]

        pred_xy = model(torch.from_numpy(x_in).to(DEVICE),
                        torch.from_numpy(dts_out[None,...]).to(DEVICE),
                        teacher=None).cpu().numpy()[0]  # [TOUT,2]
        x_pred, y_pred = pred_xy[:,0], pred_xy[:,1]
        lat_pred, lon_pred = equirect_inv(lat0, lon0, x_pred, y_pred)
        lon_pred = rewrap_180(lon_pred)

        # future timestamps
        step_ns = int(dt_h * 3_600_000_000_000)
        t0 = int(ts_ctx[-1])
        times_ns = np.array([t0 + (i+1)*step_ns for i in range(TOUT)], dtype=np.int64)

        # ground truth after t0
        mask_future = ts > t0
        gt_times = ts[mask_future]
        gt_lat   = lat[mask_future]
        gt_lon   = rewrap_180(lon[mask_future])

        # match tolerance from cadence
        full_dth = np.diff(th)
        median_cad_ns = (np.median(full_dth[full_dth>0]) * 3_600_000_000_000.0) if np.any(full_dth>0) else (DT_NORM * 3_600_000_000_000.0)
        tol_ns = int(MATCH_TOL_NS_MULTIPLIER * median_cad_ns)

        lat_t, lon_t, ok = match_truth_by_time(gt_times, gt_lat, gt_lon, times_ns, tol_ns)
        if not np.any(ok):
            continue

        # compute errors for matched lead steps
        lat_p = lat_pred[ok]; lon_p = lon_pred[ok]
        err_km = haversine_km(lat_p, lon_p, lat_t[ok], lon_t[ok])

        lat_last = float(lat_ctx[-1])
        lon_last = float(rewrap_180(np.array([lon_ctx[-1]], dtype=np.float32))[0])
        persist_km = haversine_km(np.full_like(lat_t[ok], lat_last),
                                  np.full_like(lon_t[ok], lon_last),
                                  lat_t[ok], lon_t[ok])

        # lead-time indices for matched slots
        matched_idx = np.where(ok)[0]  # 0..TOUT-1
        err_list.extend(err_km.tolist())
        persist_list.extend(persist_km.tolist())
        lead_idx_list.extend((matched_idx + 1).tolist())  # make it 1..TOUT

    if len(err_list) == 0:
        print("No accuracy computed (no ground-truth matches found).")
        return

    # Aggregate overall
    err = np.asarray(err_list, dtype=np.float32)
    persist = np.asarray(persist_list, dtype=np.float32)
    overall_mae = float(np.mean(np.abs(err)))
    overall_rmse = float(np.sqrt(np.mean(err**2)))
    persist_mae = float(np.mean(np.abs(persist)))
    skill = 1.0 - (overall_mae / max(1e-6, persist_mae))
    hit_primary = float(np.mean(err <= primary_R))  # fraction within R

    # Lead-time breakdown
    lead_idx = np.asarray(lead_idx_list, dtype=np.int32)
    rows = []
    for k in range(1, TOUT+1):
        m = (lead_idx == k)
        if not np.any(m):
            continue
        e = err[m]
        p = persist[m]
        row = {
            "lead_index": k,
            "n": int(m.sum()),
            "mae_km": float(np.mean(np.abs(e))),
            "rmse_km": float(np.sqrt(np.mean(e**2))),
            "persist_mae_km": float(np.mean(np.abs(p))),
            "skill_vs_persist": float(1.0 - (np.mean(np.abs(e)) / max(1e-6, np.mean(np.abs(p))))),
            f"hit_rate_<=_{int(primary_R)}km": float(np.mean(e <= primary_R)),
        }
        # radius curve
        for Rk in radii:
            row[f"hit_rate_<=_{int(Rk)}km"] = float(np.mean(e <= Rk))
        rows.append(row)

    # Save per-lead CSV
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    # Save overall JSON
    summary = {
        "n_matched": int(len(err)),
        "overall_mae_km": overall_mae,
        "overall_rmse_km": overall_rmse,
        "persistence_mae_km": persist_mae,
        "skill_vs_persistence": skill,
        "hit_rate_within_radius": {
            f"<= {int(primary_R)} km": hit_primary,
            **{f"<= {int(Rk)} km": float(np.mean(err <= Rk)) for Rk in radii}
        },
        "tin": TIN,
        "tout": TOUT,
        "dt_norm": DT_NORM,
        "match_tolerance_multiplier": MATCH_TOL_NS_MULTIPLIER,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote per-lead metrics to {args.out_csv}")
    print(f"Wrote overall summary to {args.out_json}")

if __name__ == "__main__":
    main()
