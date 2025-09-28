# spaghetti_spartan.py
# Classic 'spaghetti model' look (no colormaps, no circles, no multicolor).
# Observed + linearized forecast in the SAME color.

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import folium
from folium import FeatureGroup, LayerControl

# ======================
# ---- GLOBALS (edit) ---
# ======================
CSV_OBS            = r"C:\Users\adamm\Documents\HACKATHONS\ShellHacks\hurricane_points.csv"  # sid,time_ns,lat,lon
CKPT_PATH          = r"C:\Users\adamm\Documents\HACKATHONS\ShellHacks\saved_gru_seq2seq_best.pt"
OUT_DIR            = Path("maps_spaghetti_classic")
FL_NEAR_CSV        = OUT_DIR / "storms_near_florida.csv"   # permanent filtered CSV output
FIRST_N_SIDS       = 200                                   # how many storms to scan for FL proximity
TIN                = 6
TOUT               = 8
DT_NORM            = 3.0
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
TILES              = "cartodbpositron"

# Appearance (single color for BOTH observed and forecast)
TRACK_COLOR        = "#111111"   # one color only
OBS_WEIGHT         = 5
FC_WEIGHT          = 4           # forecast weight (same color, dashed)
OBS_ALPHA          = 0.98
FC_ALPHA           = 0.85
FC_DASH            = "8,6"       # dashed pattern (same color); set to None for solid

# ======================
# ---- Model import ----
# ======================
from model import GRUSeq2Seq  # your project class

# ======================
# ---- Geometry helpers ----
# ======================
def unwrap_lon(arr: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(arr.astype(np.float64)))).astype(np.float32)

def rewrap_180(lon: np.ndarray) -> np.ndarray:
    lw = (lon + 180.0) % 360.0 - 180.0
    lw[lw == 180.0] = -180.0
    return lw

def to_hours(ns: np.ndarray) -> np.ndarray:
    return (ns.astype(np.float64) / 3_600_000_000_000.0).astype(np.float32)

def equirect_inv(lat0, lon0, x, y):
    R = 6371.0088
    lat0r = math.radians(float(lat0))
    lat = np.degrees(y / R + math.radians(float(lat0)))
    lon = np.degrees(x / (R * math.cos(lat0r)) + math.radians(float(lon0)))
    return lat.astype(np.float32), lon.astype(np.float32)

def km_to_deg_latlon(lat_ref_deg: float, dx_km: float, dy_km: float) -> Tuple[float, float]:
    R = 6371.0088
    dlat = np.degrees(dy_km / R)
    dlon = np.degrees(dx_km / (R * math.cos(math.radians(lat_ref_deg)) + 1e-12))
    return float(dlat), float(dlon)

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p = math.pi/180.0
    dlat = (lat2-lat1)*p
    dlon = (lon2-lon1)*p
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(a))

# ======================
# ---- Linearize forecast path (less "spaghetti") ----
# ======================
def linearize_track(latlon_track: List[List[float]],
                    ref_lat: float,
                    ref_lon: float) -> List[List[float]]:
    """
    Make the forecast look more linear by fitting a 1st-order trend in a local tangent plane.
    Inputs:
      latlon_track: [[lat,lon], ...] length N
      ref_lat/lon : anchor (e.g., last observed point)
    Returns:
      [[lat_lin, lon_lin], ...] same length
    """
    if not latlon_track or len(latlon_track) < 2:
        return latlon_track

    # Project to local meters using equirectangular around ref
    R = 6371.0088
    lat0 = float(ref_lat)
    lon0 = float(unwrap_lon(np.array([ref_lon], np.float32))[0])  # ensure continuity near 180
    lat0r = math.radians(lat0)

    lat = np.array([p[0] for p in latlon_track], dtype=np.float64)
    lon = unwrap_lon(np.array([p[1] for p in latlon_track], dtype=np.float64))

    x = (np.radians(lon) - math.radians(lon0)) * math.cos(lat0r) * R
    y = (np.radians(lat) - math.radians(lat0)) * R

    # Fit linear trend x(i), y(i) vs step index i (degree=1)
    n = len(latlon_track)
    i = np.arange(n, dtype=np.float64)
    bx = np.polyfit(i, x, deg=1)  # x ≈ bx[0]*i + bx[1]
    by = np.polyfit(i, y, deg=1)  # y ≈ by[0]*i + by[1]

    x_fit = bx[0]*i + bx[1]
    y_fit = by[0]*i + by[1]

    # Back to lat/lon
    lat_lin, lon_lin = equirect_inv(lat0, lon0, x_fit.astype(np.float32), y_fit.astype(np.float32))
    lon_lin = rewrap_180(lon_lin)
    return [[float(lat_lin[k]), float(lon_lin[k])] for k in range(n)]

# ======================
# ---- Forecast core ----
# ======================
@torch.no_grad()
def forecast_tracks_from_obs(df_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Returns: DataFrame[sid,time_iso,time_ns,lat,lon,kind='forecast'] for all sids in df_obs.
    """
    model = GRUSeq2Seq(in_dim=3, hid=64, num_layers=1, out_steps=TOUT).to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    # Allow for nested state dicts or plain
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    out_rows = []
    for sid, g in df_obs.groupby("sid", sort=False):
        g = g.dropna(subset=["lat","lon","time_ns"]).sort_values("time_ns").reset_index(drop=True)
        if len(g) < TIN:
            continue

        lat = g["lat"].to_numpy(np.float32)
        lon = unwrap_lon(g["lon"].to_numpy(np.float32))
        ts  = g["time_ns"].to_numpy(np.int64)

        lat_ctx = lat[-TIN:]; lon_ctx = lon[-TIN:]; ts_ctx = ts[-TIN:]
        th      = to_hours(ts_ctx)
        dth     = np.diff(th, prepend=th[0])
        dt_h    = np.maximum(np.median(dth[1:]) if len(dth)>1 else DT_NORM, 1e-3)

        # Project context to local plane around first context point
        R = 6371.0088
        lat0, lon0 = float(lat_ctx[0]), float(lon_ctx[0])
        lat0r = math.radians(lat0)
        x = (np.radians(lon_ctx) - math.radians(lon0)) * math.cos(lat0r) * R
        y = (np.radians(lat_ctx) - math.radians(lat0)) * R

        dth_ctx = np.diff(th, prepend=th[0]); dth_ctx[0] = max(dth_ctx[0], DT_NORM)
        x_in = np.stack([x, y, (dth_ctx/DT_NORM)], axis=-1).astype(np.float32)[None, ...]
        dts_out = np.full((TOUT,1), dt_h/DT_NORM, dtype=np.float32)[None, ...]

        pred_xy = model(torch.from_numpy(x_in).to(DEVICE),
                        torch.from_numpy(dts_out).to(DEVICE),
                        teacher=None).cpu().numpy()[0]  # [TOUT,2]
        x_pred, y_pred = pred_xy[:,0], pred_xy[:,1]
        lat_pred, lon_pred = equirect_inv(lat0, lon0, x_pred, y_pred)
        lon_pred = rewrap_180(lon_pred)

        step_ns = int(dt_h * 3_600_000_000_000)
        t0 = int(ts_ctx[-1])
        times_ns = np.array([t0 + (i+1)*step_ns for i in range(TOUT)], dtype=np.int64)
        times_iso = [pd.Timestamp(int(t)).isoformat() for t in range(t0 + step_ns, t0 + step_ns*(TOUT+1), step_ns)]

        for i in range(TOUT):
            out_rows.append((sid, times_iso[i], int(times_ns[i]),
                             float(lat_pred[i]), float(lon_pred[i]), "forecast"))

    return pd.DataFrame(out_rows, columns=["sid","time_iso","time_ns","lat","lon","kind"])

# ======================
# ---- DRAW FUNCTION (single color) ----
# ======================
def draw_tracks_single_color(
    *,
    map_obj: folium.Map,
    storm_id: str,
    origin: List[float],                    # [lat, lon] last observation
    observed: Optional[List[List[float]]],  # list[[lat,lon]] or None
    forecast: List[List[float]],            # list[[lat,lon]] (already linearized)
    show_observed: bool = True,
    show_forecast: bool = True
):
    """
    Minimal layering:
      - observed: bold solid TRACK_COLOR
      - forecast: same color, dashed (or solid if FC_DASH=None)
    """
    g_obs = FeatureGroup(name=f"{storm_id} • Observed", show=show_observed)
    g_fc  = FeatureGroup(name=f"{storm_id} • Forecast (linearized)", show=show_forecast)

    if show_observed and observed and len(observed) >= 2:
        folium.PolyLine(observed,
                        weight=OBS_WEIGHT, color=TRACK_COLOR, opacity=OBS_ALPHA).add_to(g_obs)
        folium.CircleMarker(observed[-1], radius=4, color=TRACK_COLOR, fill=True, fill_opacity=1.0).add_to(g_obs)

    if show_forecast and forecast and len(forecast) >= 2:
        folium.PolyLine(forecast,
                        weight=FC_WEIGHT,
                        color=TRACK_COLOR,
                        opacity=FC_ALPHA,
                        dash_array=FC_DASH).add_to(g_fc)
        folium.CircleMarker(forecast[-1], radius=5, color=TRACK_COLOR, fill=True, fill_opacity=1.0).add_to(g_fc)

    # Origin pin
    folium.CircleMarker(origin, radius=6, color=TRACK_COLOR, fill=True, fill_opacity=1.0).add_to(g_obs if show_observed else map_obj)

    g_obs.add_to(map_obj)
    g_fc.add_to(map_obj)

# ======================
# ---- Florida filter ----
# ======================
def storms_near_florida(df: pd.DataFrame,
                        max_storms: int = 200,
                        box_margin_deg: float = 0.7,
                        max_distance_km: float = 550.0) -> pd.DataFrame:
    """
    Heuristic filter:
      1) take first `max_storms` unique sids (by first appearance in the CSV)
      2) keep a storm if ANY track point lies within an expanded FL bounding box OR
         if the last observation falls within `max_distance_km` of a FL centroid.
    Returns a subset of df containing only selected storms (all their rows).
    """
    # Florida rough bounding box (deg): (lat, lon) ~ 24.5..31.5, -87.7..-80.0
    lat_min, lat_max = 24.5 - box_margin_deg, 31.5 + box_margin_deg
    lon_min, lon_max = -87.7 - box_margin_deg, -80.0 + box_margin_deg
    fl_center = (27.8, -81.7)

    # First N unique sids
    sids_ordered = df.drop_duplicates(subset=["sid"], keep="first")["sid"].tolist()
    pick_sids = set(sids_ordered[:max_storms])

    keep = []
    for sid, g in df[df["sid"].isin(pick_sids)].groupby("sid", sort=False):
        g = g.sort_values("time_ns")
        lats = g["lat"].to_numpy(float)
        lons = g["lon"].to_numpy(float)

        in_box = ((lats >= lat_min) & (lats <= lat_max) &
                  (lons >= lon_min) & (lons <= lon_max)).any()

        near_centroid = False
        if len(g) > 0:
            lat_last, lon_last = float(lats[-1]), float(lons[-1])
            near_centroid = (haversine_km(lat_last, lon_last, fl_center[0], fl_center[1]) <= max_distance_km)

        if in_box or near_centroid:
            keep.append(sid)

    return df[df["sid"].isin(keep)].copy()

# ======================
# ---- Main workflow ----
# ======================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load observations
    df_obs = pd.read_csv(CSV_OBS).dropna(subset=["sid","time_ns","lat","lon"])
    df_obs["sid"] = df_obs["sid"].astype(str)
    df_obs = df_obs.sort_values(["sid","time_ns"])

    # 2) Filter storms near Florida among first N sids and save permanently
    df_fl = storms_near_florida(df_obs, max_storms=FIRST_N_SIDS)
    df_fl.to_csv(FL_NEAR_CSV, index=False)
    print(f"[OK] Saved storms near Florida → {FL_NEAR_CSV}  (storms: {df_fl['sid'].nunique()})")

    # 3) Make forecasts (mean) just for selected storms
    df_fc = forecast_tracks_from_obs(df_fl)

    if df_fl.empty:
        print("[WARN] No storms matched the Florida filter.")
        return

    # Combined map centered at last available observation in the subset
    g0 = df_fl.iloc[-1]
    combined_map = folium.Map(location=[float(g0["lat"]), float(g0["lon"])], zoom_start=5, tiles=TILES)

    for sid, g_obs in df_fl.groupby("sid", sort=False):
        g_obs = g_obs.sort_values("time_ns")
        g_fc  = df_fc[df_fc["sid"] == sid].sort_values("time_ns")

        obs_coords  = g_obs[["lat","lon"]].to_numpy().tolist()
        fc_coords   = g_fc[["lat","lon"]].to_numpy().tolist()

        if len(obs_coords) == 0 or len(fc_coords) < 2:
            continue

        origin = obs_coords[-1]

        # Linearize the forecast to reduce wiggles; anchor around the last observed point
        fc_coords_lin = linearize_track(fc_coords, ref_lat=origin[0], ref_lon=origin[1])

        # Per-storm map
        m_sid = folium.Map(location=[origin[0], origin[1]], zoom_start=5, tiles=TILES)

        draw_tracks_single_color(
            map_obj=m_sid,
            storm_id=str(sid),
            origin=origin,
            observed=obs_coords,
            forecast=fc_coords_lin,
            show_observed=True,
            show_forecast=True
        )
        LayerControl(collapsed=False).add_to(m_sid)
        out_html = OUT_DIR / f"{sid}_spaghetti.html"
        m_sid.save(str(out_html))
        print(f"[OK] Wrote map: {out_html}")

        # Add to combined map as well
        draw_tracks_single_color(
            map_obj=combined_map,
            storm_id=str(sid),
            origin=origin,
            observed=obs_coords,
            forecast=fc_coords_lin,
            show_observed=True,
            show_forecast=True
        )

    LayerControl(collapsed=False).add_to(combined_map)
    combined_html = OUT_DIR / f"spaghetti_combined.html"
    combined_map.save(str(combined_html))
    print(f"[OK] Wrote combined map: {combined_html}")

if __name__ == "__main__":
    main()
