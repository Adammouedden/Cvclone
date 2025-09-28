# forecast_csv.py
import math, numpy as np, pandas as pd, torch
import folium
from folium import FeatureGroup, LayerControl
from model import GRUSeq2Seq
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
TIN       = 6
TOUT      = 8
DT_NORM   = 3.0
FIRST_N   = 10                  # how many storms to include on combined map
OUT_CSV   = "hurricane_forecast.csv"
OUT_DIR   = "maps_forecast"     # folder for html maps
# How close (ns) a ground-truth point must be to a forecast time to count for accuracy
MATCH_TOL_NS_MULTIPLIER = 1.75  # times the median cadence; tweak if needed

# ----------------------------
# Helpers (same math as training)
# ----------------------------
def unwrap_lon(arr):
    return np.rad2deg(np.unwrap(np.deg2rad(arr.astype(np.float64)))).astype(np.float32)

def to_hours(ns):  # int ns -> float hours
    return (ns.astype(np.float64) / 3_600_000_000_000.0).astype(np.float32)

def equirect_inv(lat0, lon0, x, y):
    """Inverse of equirect_km -> back to lat/lon (deg)."""
    R = 6371.0088
    lat0r = math.radians(float(lat0))
    lat = np.degrees(y / R + math.radians(float(lat0)))
    lon = np.degrees(x / (R * math.cos(lat0r)) + math.radians(float(lon0)))
    return lat.astype(np.float32), lon.astype(np.float32)

def rewrap_180(lon):
    """Wrap to [-180,180)."""
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
    """
    For each target time, pick the closest ground-truth index within tol_ns.
    Returns (lat_truth, lon_truth, ok_mask). Works even if gt is empty.
    """
    n_targets = len(target_times_ns)
    lat_t = np.empty(n_targets, dtype=np.float32)
    lon_t = np.empty(n_targets, dtype=np.float32)
    ok = np.zeros(n_targets, dtype=bool)

    # If no ground truth, nothing to match
    if gt_times_ns is None or len(gt_times_ns) == 0:
        return lat_t, lon_t, ok

    # two-pointer search over sorted times
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
# Mapping
# ----------------------------
def add_storm_layers(m: folium.Map, sid: str, obs_coords, fcst_coords):
    """
    obs_coords, fcst_coords: list of [lat, lon]
    """
    g_obs  = FeatureGroup(name=f"{sid} • Observed", show=True)
    g_fcst = FeatureGroup(name=f"{sid} • Forecast", show=True)

    if len(obs_coords) >= 2:
        folium.PolyLine(obs_coords, weight=3, color="#000000",
                        tooltip=f"{sid} observed").add_to(g_obs)
        folium.CircleMarker(obs_coords[0], radius=4, color="#000000",
                            fill=True, fill_opacity=1.0,
                            tooltip=f"{sid} start (obs)").add_to(g_obs)
        folium.CircleMarker(obs_coords[-1], radius=4, color="#000000",
                            fill=True, fill_opacity=1.0,
                            tooltip=f"{sid} last obs").add_to(g_obs)

    if len(fcst_coords) >= 1:
        # connect last obs → first forecast for continuity
        line = ([obs_coords[-1]] if obs_coords else []) + fcst_coords
        folium.PolyLine(line, weight=3, color="#ff7f0e",
                        tooltip=f"{sid} forecast").add_to(g_fcst)
        folium.CircleMarker(fcst_coords[-1], radius=4, color="#ff7f0e",
                            fill=True, fill_opacity=1.0,
                            tooltip=f"{sid} forecast end").add_to(g_fcst)

    g_obs.add_to(m); g_fcst.add_to(m)

# ----------------------------
# Main
# ----------------------------
@torch.no_grad()
def main(csv_in="hurricane_points.csv", ckpt="gru_seq2seq_best.pt",
         out_csv=OUT_CSV, out_dir=OUT_DIR, first_n=FIRST_N):
    # Load observations
    df = pd.read_csv(csv_in).dropna(subset=["sid","time_ns","lat","lon"])
    df = df.sort_values(["sid","time_ns"])

    # Load model
    model = GRUSeq2Seq(in_dim=3, hid=64, num_layers=1, out_steps=TOUT).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    # Accuracy accumulators
    all_err_km, all_persist_km = [], []

    # Forecast all SIDs
    out_rows = []
    preview_sids = []
    for sid, g in df.groupby("sid", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < TIN:
            continue

        lat = g["lat"].to_numpy(np.float32)
        lon = unwrap_lon(g["lon"].to_numpy(np.float32))
        ts  = g["time_ns"].to_numpy(np.int64)

        # last TIN steps as context
        lat_ctx = lat[-TIN:]; lon_ctx = lon[-TIN:]; ts_ctx = ts[-TIN:]

        # decoder Δt: use median observed cadence
        th   = to_hours(ts_ctx)
        dth  = np.diff(th, prepend=th[0])
        dt_h = np.maximum(np.median(dth[1:]) if len(dth)>1 else DT_NORM, 1e-3)
        dts_out = np.full((TOUT,1), dt_h/DT_NORM, dtype=np.float32)

        # anchor and build x_in = [x,y,Δt_norm]
        R = 6371.0088
        lat0, lon0 = float(lat_ctx[0]), float(lon_ctx[0])
        lat0r = math.radians(lat0)
        x = (np.radians(lon_ctx) - math.radians(lon0)) * math.cos(lat0r) * R
        y = (np.radians(lat_ctx) - math.radians(lat0)) * R
        dth_ctx = np.diff(th, prepend=th[0]); dth_ctx[0] = max(dth_ctx[0], DT_NORM)
        x_in = np.stack([x, y, (dth_ctx/DT_NORM)], axis=-1).astype(np.float32)[None, ...]

        # predict
        pred_xy = model(torch.from_numpy(x_in).to(DEVICE),
                        torch.from_numpy(dts_out[None,...]).to(DEVICE),
                        teacher=None).cpu().numpy()[0]  # [TOUT,2]
        x_pred, y_pred = pred_xy[:,0], pred_xy[:,1]
        lat_pred, lon_pred = equirect_inv(lat0, lon0, x_pred, y_pred)
        lon_pred = rewrap_180(lon_pred)

        # future timestamps (nominal)
        step_ns = int(dt_h * 3_600_000_000_000)
        t0 = int(ts_ctx[-1])
        times_ns = np.array([t0 + (i+1)*step_ns for i in range(TOUT)], dtype=np.int64)
        times_iso = [pd.Timestamp(int(t)).isoformat() for t in times_ns]

        for i in range(TOUT):
            out_rows.append((sid, times_iso[i], int(times_ns[i]),
                             float(lat_pred[i]), float(lon_pred[i]), "forecast"))

        # ----------------------------
        # Per-storm accuracy vs ground truth (if available)
        # ----------------------------
        # Truth after the context window only
        mask_future = ts > t0
        gt_times = ts[mask_future]
        gt_lat   = lat[mask_future]
        gt_lon   = rewrap_180(lon[mask_future])  # rewrap to compare

        # match tolerance based on dataset cadence
        th_full = to_hours(ts_ctx)
        full_dth = np.diff(th_full)
        median_cad_ns = (np.median(full_dth[full_dth>0]) * 3_600_000_000_000.0) if np.any(full_dth>0) else (DT_NORM * 3_600_000_000_000.0)
        tol_ns = int(MATCH_TOL_NS_MULTIPLIER * median_cad_ns)

        lat_t, lon_t, ok = match_truth_by_time(gt_times, gt_lat, gt_lon, times_ns, tol_ns)
        if np.any(ok):
            lat_p = lat_pred[ok]; lon_p = lon_pred[ok]
            err_km = haversine_km(lat_p, lon_p, lat_t[ok], lon_t[ok])
            mae_km = float(np.mean(np.abs(err_km)))
            rmse_km = float(np.sqrt(np.mean(err_km**2)))

            # Persistence baseline: hold last observed position flat
            lat_last = float(lat_ctx[-1])
            lon_last = float(rewrap_180(np.array([lon_ctx[-1]], dtype=np.float32))[0])
            persist_km = haversine_km(np.full_like(lat_t[ok], lat_last),
                                      np.full_like(lon_t[ok], lon_last),
                                      lat_t[ok], lon_t[ok])
            persist_mae = float(np.mean(np.abs(persist_km)))

            all_err_km.extend(err_km.tolist())
            all_persist_km.extend(persist_km.tolist())

            print(f"[{sid}] matched {int(ok.sum())}/{TOUT} steps  |  MAE: {mae_km:.2f} km  RMSE: {rmse_km:.2f} km  |  Persistence MAE: {persist_mae:.2f} km")
        else:
            print(f"[{sid}] no matching truth within tolerance (tol≈{tol_ns/3.6e12:.2f} h); accuracy skipped.")

        # stash a few SIDs for mapping
        if len(preview_sids) < first_n:
            preview_sids.append(sid)

    # Write forecast CSV
    out = pd.DataFrame(out_rows, columns=["sid","time_iso","time_ns","lat","lon","kind"])
    out.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} with {len(out):,} rows")

    # ----------------------------
    # Overall accuracy summary
    # ----------------------------
    if len(all_err_km) > 0:
        all_err = np.asarray(all_err_km, dtype=np.float32)
        all_persist = np.asarray(all_persist_km, dtype=np.float32)
        print("\n=== Overall accuracy (across all matched forecasts) ===")
        print(f"MAE:  {np.mean(np.abs(all_err)):.2f} km")
        print(f"RMSE: {np.sqrt(np.mean(all_err**2)):.2f} km")
        print(f"Persistence MAE (baseline): {np.mean(np.abs(all_persist)):.2f} km")
        skill = 1.0 - (np.mean(np.abs(all_err)) / max(1e-6, np.mean(np.abs(all_persist))))
        print(f"Skill vs persistence: {100.0*skill:.1f}%")
    else:
        print("\nNo accuracy computed (no ground-truth matches found).")

    # ----------------------------
    # Build interactive maps (observed vs forecast)
    # ----------------------------
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    combined_map = None

    # Combined map center: first preview sid first obs
    if preview_sids:
        g0 = df[df["sid"] == preview_sids[0]].iloc[0]
        combined_map = folium.Map(location=[float(g0["lat"]), float(g0["lon"])], zoom_start=4)

    for sid in preview_sids:
        g_obs = df[df["sid"] == sid].sort_values("time_ns")
        g_fc  = out[out["sid"] == sid].sort_values("time_ns")

        obs_coords  = g_obs[["lat","lon"]].to_numpy().tolist()
        fcst_coords = g_fc[["lat","lon"]].to_numpy().tolist()

        # Per-storm map
        center_lat = obs_coords[-1][0] if obs_coords else (fcst_coords[0][0] if fcst_coords else 0.0)
        center_lon = obs_coords[-1][1] if obs_coords else (fcst_coords[0][1] if fcst_coords else 0.0)
        m_sid = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        add_storm_layers(m_sid, sid, obs_coords, fcst_coords)
        LayerControl(collapsed=False).add_to(m_sid)
        out_html = Path(out_dir) / f"{sid}.html"
        m_sid.save(str(out_html))
        print(f"Wrote map: {out_html}")

        # Add to combined
        if combined_map is not None:
            add_storm_layers(combined_map, sid, obs_coords, fcst_coords)

    if combined_map is not None:
        LayerControl(collapsed=False).add_to(combined_map)
        combined_html = Path(out_dir) / f"combined_first_{len(preview_sids)}.html"
        combined_map.save(str(combined_html))
        print(f"Wrote combined map: {combined_html}")

if __name__ == "__main__":
    main()
