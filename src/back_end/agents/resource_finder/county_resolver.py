# back_end/agents/resource_finder/county_resolver.py
from __future__ import annotations
import json, math, os, time
from typing import Dict, Any, List, Optional, Tuple
import urllib.request

# Load once: a tiny JSON of county centroids (name, state, fips, lat, lon)
# You can generate this from TIGER/Line or grab a prebuilt CSV/JSON.
COUNTY_CENTROIDS_PATH = os.getenv("COUNTY_CENTROIDS_PATH", "./back_end/agents/resource_finder/us_county_centroids.json")
with open(COUNTY_CENTROIDS_PATH, "r", encoding="utf-8") as f:
    COUNTY_POINTS = json.load(f)  # list of {name, state, fips, lat, lon}

def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    R = 6371.0
    phi1, phi2 = map(math.radians, [a_lat, b_lat])
    dphi = math.radians(b_lat - a_lat)
    dlambda = math.radians(b_lon - a_lon)
    h = (math.sin(dphi/2)**2 +
         math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2)
    return 2*R*math.asin(math.sqrt(h))

def _nearest_counties(lat: float, lon: float, radius_km: float) -> List[Dict[str, Any]]:
    out = []
    for c in COUNTY_POINTS:
        d = _haversine_km(lat, lon, c["lat"], c["lon"])
        if d <= radius_km:
            out.append({**c, "distance_km": round(d, 2)})
    out.sort(key=lambda x: x["distance_km"])
    return out

def _fcc_primary_county(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    # FCC API: returns FIPS and county name quickly, no key needed.
    url = f"https://geo.fcc.gov/api/census/area?lat={lat}&lon={lon}&format=json"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # areas is usually non-empty; pick first with county FIPS (state-county)
        for area in data.get("results", []):
            if area.get("county_fips"):
                return {
                    "name": area.get("county_name"),
                    "state": area.get("state_code"),
                    "fips": area.get("county_fips")
                }
    except Exception:
        return None
    return None

def resolve_counties(lat: float, lon: float, radius_km: int) -> Dict[str, Any]:
    # Primary via offline nearest (0 km) or FCC fallback
    # 1) exact nearest centroid as primary
    nearest = _nearest_counties(lat, lon, radius_km=max(radius_km, 5))
    primary = nearest[0] if nearest else None

    if not primary:
        fallback = _fcc_primary_county(lat, lon)
        if fallback:
            primary = fallback

    nearby = [c for c in _nearest_counties(lat, lon, radius_km)]
    # remove the primary from nearby
    nearby = [c for c in nearby if not primary or c["fips"] != primary["fips"]]

    return {
        "primary_county": primary,
        "nearby_counties": nearby,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "offline_centroids" if primary else "fcc_fallback"
    }
