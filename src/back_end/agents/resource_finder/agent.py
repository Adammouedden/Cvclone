# ================================
# back_end/agents/resource_finder/agent.py
# MVP: simple search → fetch → parse → assemble
# ================================

from __future__ import annotations
from back_end.agents.resource_finder.county_resolver import resolve_counties
from back_end.agents.resource_finder.cse_fetcher import cse_discover_urls

# load .env early
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os, time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from urllib.parse import urlparse

# calls the resources tool server through your shared client
from back_end.mcp.client.mcp_client import call_tool

# ----------------------------
# Config
# ----------------------------
CACHE_TTL = int(os.getenv("RESOURCE_CACHE_TTL", "120"))
DEFAULT_RADIUS_KM = int(os.getenv("RESOURCE_RADIUS_KM", "25"))

# ----------------------------
# Tiny in-memory cache
# ----------------------------
_cache: Dict[Tuple[float, float, int, str], Tuple[float, Dict[str, Any]]] = {}

def _cache_get(key: Tuple[float, float, int, str]) -> Optional[Dict[str, Any]]:
    entry = _cache.get(key)
    if not entry:
        return None
    ts, value = entry
    if (time.time() - ts) <= CACHE_TTL:
        return value
    _cache.pop(key, None)
    return None

def _cache_set(key: Tuple[float, float, int, str], value: Dict[str, Any]) -> None:
    _cache[key] = (time.time(), value)

# ----------------------------
# Helpers
# ----------------------------
def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # distance in km; no external deps
    from math import radians, sin, cos, asin, sqrt
    R = 6371.0
    a1, o1, a2, o2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = a2 - a1
    dlon = o2 - o1
    h = sin(dlat/2)**2 + cos(a1) * cos(a2) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(h))

def _round2(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 2)

# ----------------------------
# Core
# ----------------------------
def get_snapshot(
    lat: float,
    lon: float,
    radius_km: int = DEFAULT_RADIUS_KM,
    types: Optional[List[str]] = None,
    curated_urls: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    MVP flow:
      for each type -> searchResources (with optional curated seed URLs)
                    -> for each URL: fetchUrl -> parseResources
      assemble items; compute distance if coordinates present
      sort by: items with distance first (nearest), then .gov preference
    """
    types = types or ["shelter", "sandbags", "food"]
    key = (round(float(lat), 4), round(float(lon), 4), int(radius_km), ",".join(types))
    cached = _cache_get(key)
    if cached:
        return cached

    sources: List[str] = []
    items: List[Dict[str, Any]] = []

    geo = resolve_counties(lat, lon, radius_km)
    candidate_counties = [c for c in [geo.get("primary_county")] if c] + geo.get("nearby_counties", [])

    # Build a per-type URL pool using CSE
    cse_pool: Dict[str, List[str]] = {t: [] for t in types}
    for rtype in types:
        for c in candidate_counties:
            try:
                discovered = cse_discover_urls(c, rtype)
                cse_pool[rtype].extend(discovered)
            except Exception:
                continue
    # de-dupe
    for rtype in types:
        cse_pool[rtype] = list(dict.fromkeys(cse_pool[rtype]))
    
    print("CSE cfg:", os.getenv("GOOGLE_CSE_KEY") is not None, os.getenv("GOOGLE_CSE_CX"))
    print("counties:", [c.get("name") for c in candidate_counties])

    for t in types: print("CSE discovered", t, len(cse_pool[t]))

    # 1) discover URLs for each resource type
    for rtype in types:
        seeds = []
        if curated_urls and rtype in curated_urls:
            seeds = curated_urls[rtype] or []
        search = call_tool("resources", "searchResources", {"type": rtype, "seed_urls": seeds}) or {}
        search_urls = (search.get("data") or {}).get("urls") or search.get("urls") or []
        urls = search_urls + cse_pool[rtype]
        # de-dupe preserving order
        seen, merged = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u); merged.append(u)
        urls = merged
        if not urls:
            continue

        # 2) fetch + 3) parse per URL
        def _publisher_from_url(u: str) -> Optional[str]:
            try:
                netloc = urlparse(u if "://" in u else f"https://{u}").netloc
                return netloc or None
            except Exception:
                return None

        for url in urls:
            try:
                fetched = call_tool("resources", "fetchUrl", {"url": url}) or {}
                fdata = fetched.get("data") or fetched  # support both shapes
                parsed = call_tool("resources", "parseResources", {
                    "url": fdata.get("url") or url,
                    "content": fdata.get("content") or "",
                    "content_type": fdata.get("content_type") or "text/html",
                    "type": rtype
                }) or {}
                pdata = parsed.get("data") or parsed  # if your parser also wraps

                for it in (pdata.get("items") or []):
                    it.setdefault("type", rtype)
                    it.setdefault("link", url)
                    it.setdefault("publisher", _publisher_from_url(url))
                    items.append(it)

                sources.append(url)
            except Exception:
                continue

    # 4) distance calc (if coordinates provided by parsers)
    for it in items:
        coords = it.get("coordinates") or {}
        la, lo = coords.get("lat"), coords.get("lon")
        if isinstance(la, (int, float)) and isinstance(lo, (int, float)):
            it["distance_km"] = _round2(_haversine_km(lat, lon, float(la), float(lo)))
        else:
            it["distance_km"] = None  # still include; MVP doesn’t geocode

    # 5) (MVP) simple dedupe by (name, address, link)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for it in items:
        key2 = (it.get("name"), it.get("address"), it.get("link"))
        if key2 in seen:
            continue
        seen.add(key2)
        deduped.append(it)

    # 6) filter: if coords exist, enforce radius; otherwise keep (MVP keeps possibly useful items)
    in_radius: List[Dict[str, Any]] = []
    for it in deduped:
        if it.get("distance_km") is None:
            # keep unknown-distance items in MVP so users see them
            in_radius.append(it)
        elif it["distance_km"] <= float(radius_km):
            in_radius.append(it)
        diag = {
        "counties": {"primary": geo.get("primary_county"),
                    "nearby": len(geo.get("nearby_counties", []))},
        "cse_urls": {t: len(cse_pool.get(t, [])) for t in types},
        "sources_count": len(sources),
        "items_before_dedupe": len(items),
    }

    result = {
        "source": "local_resources",
        "fetched_at": iso_utc_now(),
        "location": {"lat": float(lat), "lon": float(lon), "radius_km": int(radius_km)},
        "resources": in_radius,
        "sources": sorted(set(sources)),
    }
    if not in_radius:
        result["debug"] = diag

    # 7) sort: items with distance first (nearest), prefer .gov domains
    OFFICIAL_HINTS = ("redcross.org", "feedingamerica.org", ".fl.us", "fema.gov", "ready.gov", "disasterassistance.gov", "foodpantries.org")
    def _score(it):
        has_dist = 0 if it.get("distance_km") is None else 1
        dist = it.get("distance_km") or 9999.0
        pub = (it.get("publisher") or "")
        gov_bonus = -1000 if pub.endswith(".gov") else 0
        hint_bonus = -200 if any(pub.endswith(h) or pub.endswith("." + h) for h in OFFICIAL_HINTS) else 0
        return (-has_dist, dist + gov_bonus + hint_bonus)


    in_radius.sort(key=_score)

    result: Dict[str, Any] = {
        "source": "local_resources",
        "fetched_at": iso_utc_now(),
        "location": {"lat": float(lat), "lon": float(lon), "radius_km": int(radius_km)},
        "resources": in_radius,
        "sources": sorted(set(sources)),
    }

    _cache_set(key, result)
    return result

# ----------------------------
# Tiny HTTP wrapper (A2A)
# ----------------------------
app = Flask(__name__)

@app.get("/.well-known/agent-card.json")
def agent_card():
    return jsonify({
        "id": "resource-finder",
        "name": "Resource Finder Agent (MVP)",
        "version": "0.1.0",
        "skills": [{
            "name": "get_snapshot",
            "description": "Returns local resource items (shelter, sandbags, food) near a coordinate.",
            "args": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "radius_km": {"type": "number", "default": DEFAULT_RADIUS_KM},
                    "types": {"type": "array", "items": {"type": "string"}},
                    "curated_urls": {"type": "object"}  # optional per-type seed list for MVP
                },
                "required": ["lat", "lon"]
            },
            "returns": {"type": "object"}
        }]
    })

@app.post("/skills/get_snapshot")
def skill_get_snapshot():
    b = request.get_json(force=True) or {}
    lat, lon = b.get("lat"), b.get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "lat/lon required"}), 400
    radius_km = int(b.get("radius_km", DEFAULT_RADIUS_KM))
    types = b.get("types")
    curated = b.get("curated_urls")  # optional
    try:
        data = get_snapshot(float(lat), float(lon), radius_km, types, curated)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 502

# Optional debug output if you want to confirm env loading
if os.getenv("DEBUG_ENV") == "1":
    print("Loaded env values:")
    print("  MCP_BASE_NWS        =", os.getenv("MCP_BASE_NWS"))
    print("  MCP_BASE_RESOURCES  =", os.getenv("MCP_BASE_RESOURCES"))
    print("  RESOURCE_CACHE_TTL  =", os.getenv("RESOURCE_CACHE_TTL"))
    print("  RESOURCE_RADIUS_KM  =", os.getenv("RESOURCE_RADIUS_KM"))
    print("  RESOURCES_UA        =", os.getenv("RESOURCES_UA"))
    print("---")

if __name__ == "__main__":
    port = int(os.getenv("RESOURCE_AGENT_PORT", "8082"))
    print(f"🧭 Resource Finder Agent (MVP) on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
