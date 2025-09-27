# Cvclone/src/back_end/agents/forecast/agent.py
"""
Forecast Agent (Option B: MCP-backed via HTTP tool server)

Responsibilities:
- Call the NWS tool server for:
  - getForecast(lat, lon, hourly=False)  -> periods (day/night)
  - getForecast(lat, lon, hourly=True)   -> periods (hourly)
  - getAlerts(lat, lon)                  -> CAP GeoJSON features
- Normalize to the frozen schema:
  {
    "source": "nws",
    "fetched_at": "...Z",
    "location": {"lat": ..., "lon": ...},
    "current": {...},
    "hourly": [ ... ],
    "daily":  [ ... ],
    "alerts": [ ... ],
    "sources": [ ... ]
  }
- Provide a tiny in-memory TTL cache (default ~120s)
- (Optional) expose a small HTTP endpoint for A2A-style consumption
"""
import os, time, requests
from flask import Flask, request, jsonify

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from __future__ import annotations
import os, time, math, re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from flask import Flask, request, jsonify  # OPTIONAL: only for local HTTP exposure

# IMPORTANT: use your HTTP client that talks to the tool server
# DO NOT import Python functions from the tool server.
from back_end.mcp.client.mcp_client import call_tool

# ----------------------------
# Config / constants
# ----------------------------

CACHE_TTL_SECONDS = int(os.getenv("FORECAST_CACHE_TTL", "120"))
ROUND_COORDS = int(os.getenv("FORECAST_COORD_ROUND", "4"))  # helps cache reuse
# If you’d like to skip daily/hourly on demand, the input allows flags.

# ----------------------------
# Tiny in-memory TTL cache
# ----------------------------

_cache: Dict[Tuple[float, float, bool, bool], Tuple[float, Dict[str, Any]]] = {}

def _cache_get(key: Tuple[float, float, bool, bool]) -> Optional[Dict[str, Any]]:
    entry = _cache.get(key)
    if not entry:
        return None
    ts, value = entry
    if (time.time() - ts) <= CACHE_TTL_SECONDS:
        return value
    # expired
    _cache.pop(key, None)
    return None

def _cache_set(key: Tuple[float, float, bool, bool], value: Dict[str, Any]) -> None:
    _cache[key] = (time.time(), value)

# ----------------------------
# Unit helpers
# ----------------------------

def f_to_c(temp_f: Optional[float]) -> Optional[float]:
    if temp_f is None:
        return None
    return (temp_f - 32.0) * 5.0 / 9.0

def mph_to_mps(mph: Optional[float]) -> Optional[float]:
    if mph is None:
        return None
    return mph * 0.44704

_wind_re = re.compile(r"(\d+)(?:\s*to\s*(\d+))?\s*mph", re.IGNORECASE)

def parse_wind_speed_to_mps(wind_speed: Optional[str]) -> Optional[float]:
    """
    NWS windSpeed examples: "5 mph", "5 to 10 mph"
    Return average mph converted to m/s, or None.
    """
    if not wind_speed or not isinstance(wind_speed, str):
        return None
    m = _wind_re.search(wind_speed)
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else a
    avg_mph = (a + b) / 2.0
    return mph_to_mps(avg_mph)

def parse_mph_string_to_mps(s: str | None) -> Optional[float]:
    if not s: return None
    m = _wind_re.search(s)
    if not m: return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else a
    return mph_to_mps((a+b)/2)

def percent_to_fraction(p: Optional[float]) -> Optional[float]:
    if p is None:
        return None
    # NWS uses 0..100; we want 0..1
    return max(0.0, min(1.0, p / 100.0))

def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------
# Normalization helpers
# ----------------------------

def normalize_hourly(periods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in periods or []:
        # Fields seen in NWS hourly periods:
        # startTime, endTime, temperature (+ Unit), windSpeed (string), windDirection, probabilityOfPrecipitation.value
        ts = p.get("startTime")  # ISO string
        temp = p.get("temperature")
        unit = p.get("temperatureUnit")
        temp_c = f_to_c(temp) if unit == "F" else temp  # some offices might use C; handle gracefully

        wind_mps = parse_wind_speed_to_mps(p.get("windSpeed"))
        # gust often not present in hourly; leave None if absent
        gust_mps = parse_mph_string_to_mps(p.get("windGust"))

        pop_frac = None
        pop = p.get("probabilityOfPrecipitation") or {}
        if isinstance(pop, dict):
            pop_frac = percent_to_fraction(pop.get("value"))

        out.append({
            "ts": ts,
            "temp_c": temp_c,
            "wind_mps": wind_mps,
            "gust_mps": gust_mps,
            "pop": pop_frac,
            "qpf_mm": None,  # NWS periods typically omit QPF; can be added later via gridpoints
            "condition": p.get("shortForecast"),
        })
    return out

def _date_from_iso_local(iso_str: str) -> str:
    """
    Given an ISO timestamp (with zone offset), return local calendar date as YYYY-MM-DD.
    Python's datetime.fromisoformat parses offsets like ...-04:00.
    """
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        # Use the date in its own offset (local to the forecast period)
        return dt.date().isoformat()
    except Exception:
        # Fallback: just take split
        return iso_str[:10]

def normalize_daily(periods: list[dict], hourly: list[dict] | None = None) -> list[dict]:
    """
    Non-hourly 'forecast' returns day/night periods (Today/Tonight/Mon,...).
    We group by calendar date and compute tmin/tmax. PoP rule: take max PoP for the date.
    """
    by_date: Dict[str, Dict[str, Any]] = {}
    for p in periods or []:
        start = p.get("startTime")
        if not start:
            continue
        dkey = _date_from_iso_local(start)

        # Temperature
        temp = p.get("temperature")
        unit = p.get("temperatureUnit")
        temp_c = f_to_c(temp) if unit == "F" else temp

        # PoP
        pop_frac = None
        pop = p.get("probabilityOfPrecipitation") or {}
        if isinstance(pop, dict):
            pop_frac = percent_to_fraction(pop.get("value"))

        # Init bucket
        bucket = by_date.setdefault(dkey, {
            "tmin_c": math.inf,
            "tmax_c": -math.inf,
            "pop": 0.0,  # we'll take max observed
        })

        # Update min/max
        if temp_c is not None:
            if temp_c < bucket["tmin_c"]:
                bucket["tmin_c"] = temp_c
            if temp_c > bucket["tmax_c"]:
                bucket["tmax_c"] = temp_c

        # Update PoP (max strategy)
        if pop_frac is not None:
            bucket["pop"] = max(bucket["pop"], pop_frac)

    # Build rows with reasonable defaults
    out = []
    for dkey in sorted(by_date.keys()):
        b = by_date[dkey]
        tmin = None if b["tmin_c"] is math.inf else b["tmin_c"]
        tmax = None if b["tmax_c"] is -math.inf else b["tmax_c"]

        # fallback using hourly if needed
        if (tmin is None or tmax is None or abs((tmax or 0)-(tmin or 0)) < 0.01) and hourly:
            hs = [h for h in hourly if (h.get("ts") or "")[:10] == dkey and h.get("temp_c") is not None]
            if hs:
                temps = [h["temp_c"] for h in hs]
                tmin = min(temps)
                tmax = max(temps)

        out.append({
            "date": dkey,
            "tmin_c": tmin,
            "tmax_c": tmax,
            "pop": b["pop"] if b["pop"] > 0 else None,
            "qpf_mm": None,
        })
    return out

def synthesize_current_from_hourly(hourly: list[dict]) -> dict | None:
    if not hourly:
        return None
    now = datetime.now(timezone.utc)
    def to_dt(s: str | None):
        if not s: return None
        # handle "Z" or "-04:00"
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    # find period where start <= now < end
    for p in hourly:
        st, et = to_dt(p.get("ts")), None
        # hourly has 1h buckets; derive end by +1h if not present
        # or rely on next element; simplest: match by equality/ordering
        if not st: 
            continue
        # consider it “current” if within 90 minutes of now
        if abs((now - st).total_seconds()) <= 90*60:
            return {
                "ts": p.get("ts"),
                "temp_c": p.get("temp_c"),
                "wind_mps": p.get("wind_mps"),
                "gust_mps": p.get("gust_mps"),
                "rh": None,
                "pressure_hpa": None,
                "condition": p.get("condition"),
            }
    # fallback
    first = hourly[0]
    return {
        "ts": first.get("ts"),
        "temp_c": first.get("temp_c"),
        "wind_mps": first.get("wind_mps"),
        "gust_mps": first.get("gust_mps"),
        "rh": None,
        "pressure_hpa": None,
        "condition": first.get("condition"),
    }

def normalize_alerts(alert_geojson: dict, active_only: bool = True) -> list[dict]:
    out = []
    feats = (alert_geojson or {}).get("features") or []
    now = datetime.now(timezone.utc)
    def to_dt(s: str | None):
        return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None
    for f in feats:
        props = (f or {}).get("properties") or {}
        exp = to_dt(props.get("expires"))
        if active_only and exp and exp < now:
            continue
        cap_id = props.get("id") or props.get("@id") or f.get("id")
        out.append({
            "event": props.get("event"),
            "severity": props.get("severity"),
            "onset": props.get("onset"),
            "expires": props.get("expires"),
            "headline": props.get("headline"),
            "cap_id": cap_id,
        })
    return out


# ----------------------------
# Core agent function
# ----------------------------

def get_snapshot(lat: float,
                 lon: float,
                 hourly: bool = True,
                 daily: bool = True) -> Dict[str, Any]:
    """
    Main entry: assemble a normalized forecast snapshot using the tool server.
    """
    # Cache key uses rounded coords to improve reuse for nearby clicks
    key = (round(float(lat), ROUND_COORDS),
           round(float(lon), ROUND_COORDS),
           bool(hourly),
           bool(daily))
    cached = _cache_get(key)
    if cached:
        return cached

    # Call tools (each returns `data` already, thanks to your client unwrap)
    sources: List[str] = []
    hourly_periods: List[Dict[str, Any]] = []
    daily_periods: List[Dict[str, Any]] = []
    alerts_raw: Dict[str, Any] = {}

    def _safe_tool(name, payload):
        try:
            return call_tool("nws", name, payload)
        except Exception as e:
            return None

    # usage
    hf = _safe_tool("getForecast", {"lat": lat, "lon": lon, "hourly": True}) if hourly else None
    df = _safe_tool("getForecast", {"lat": lat, "lon": lon, "hourly": False}) if daily else None
    ar = _safe_tool("getAlerts", {"lat": lat, "lon": lon}) or {"features": []}

    # 1) Hourly periods (optional)
    if hourly:
        hf = _safe_tool("getForecast", {"lat": lat, "lon": lon, "hourly": True}) if hourly else None
        # Keep track of the actual URL we fetched for transparency
        # The tool returns the full forecast JSON; its 'id' is the canonical URL,
        # or you can store the `@id`/`forecastGenerator` as metadata. If absent,
        # we skip adding a source here.
        src_url = hf.get("id") or hf.get("@id") or ((hf.get("_meta") or {}).get("resolved_url"))
        if isinstance(src_url, str):
            sources.append(src_url)
        hourly_periods = (hf.get("properties") or {}).get("periods") or []

    # 2) Daily periods (optional)
    if daily:
        df = _safe_tool("getForecast", {"lat": lat, "lon": lon, "hourly": False}) if daily else None
        src_url = df.get("id") or df.get("@id") or ((df.get("_meta") or {}).get("resolved_url"))
        if isinstance(src_url, str):
            sources.append(src_url)
        daily_periods = (df.get("properties") or {}).get("periods") or []

    # 3) Alerts (always nice to have)
    ar = _safe_tool("getAlerts", {"lat": lat, "lon": lon}) or {"features": []}

    # The alerts GeoJSON has its own `@context`/`type`; store the query URL instead.
    # Since our tool server constructed it, we don't have the raw URL here;
    # you can synthesize it for transparency:
    sources.append(f"https://api.weather.gov/alerts?point={lat},{lon}&status=actual&message_type=alert,update")
    alerts_raw = ar

    # Normalize structures
    HOURS = int(os.getenv("FORECAST_HOURLY_HOURS", "24"))
    hourly_rows = normalize_hourly(hourly_periods)[:HOURS] if hourly else []
    daily_rows = normalize_daily(daily_periods, hourly_rows) if daily else []
    current_obj = synthesize_current_from_hourly(hourly_rows)

    def _round1(x):
        return None if x is None else round(x, 1)

    def _round2(x):
        return None if x is None else round(x, 2)

    for h in hourly_rows:
        h["temp_c"]  = _round1(h.get("temp_c"))
        h["wind_mps"] = _round1(h.get("wind_mps"))
        h["gust_mps"] = _round1(h.get("gust_mps"))  # you’re parsing gusts now, so round it too
        # keep pop as 0..1; if you want cleaner UI numbers later, round at the frontend

    for d in daily_rows:
        d["tmin_c"] = _round1(d.get("tmin_c"))
        d["tmax_c"] = _round1(d.get("tmax_c"))
        d["pop"]   = _round2(d.get("pop"))  # optional—UI could also handle this

    result: Dict[str, Any] = {
        "source": "nws",
        "fetched_at": iso_utc_now(),
        "location": {"lat": float(lat), "lon": float(lon)},
        "current": current_obj,
        "hourly": hourly_rows,
        "daily": daily_rows,
        "alerts": normalize_alerts(alerts_raw, active_only=True),
        "sources": list(dict.fromkeys(sources)),  # de-duplicate, keep order
    }

    if result["current"]:
        c = result["current"]
        c["temp_c"] = _round1(c["temp_c"])
        c["wind_mps"] = _round1(c["wind_mps"])

    _cache_set(key, result)
    
    return result

# ----------------------------
# (Optional) Tiny HTTP wrapper for A2A-style consumption
# If you already have a proper A2A server in your stack, you can skip this.
# ----------------------------

app = Flask(__name__)

@app.get("/.well-known/agent-card.json")
def agent_card():
    card = {
        "id": "forecast-nws",
        "name": "NWS Forecast Agent",
        "version": "0.1.0",
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [{
            "name": "get_snapshot",
            "description": "Return normalized forecast + alerts for a coordinate.",
            "args": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "hourly": {"type": "boolean", "default": True},
                    "daily": {"type": "boolean", "default": True}
                },
                "required": ["lat", "lon"]
            },
            "returns": {"type": "object"}
        }]
    }
    return jsonify(card)

@app.post("/skills/get_snapshot")
def skill_get_snapshot():
    b = request.get_json(force=True) or {}
    lat, lon = b.get("lat"), b.get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "lat/lon required"}), 400
    hourly = bool(b.get("hourly", True))
    daily = bool(b.get("daily", True))
    try:
        data = get_snapshot(lat=float(lat), lon=float(lon), hourly=hourly, daily=daily)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 502

if __name__ == "__main__":
    # Run this only if you need a quick local HTTP endpoint for your orchestrators.
    # Otherwise, integrate this module into your main A2A server.
    port = int(os.getenv("FORECAST_AGENT_PORT", "8081"))
    print(f"🌤️ NWS Forecast Agent up on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
