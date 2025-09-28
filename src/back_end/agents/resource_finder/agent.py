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

import os, time, sys
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from collections import deque
from urllib.parse import urlparse
import requests

# calls the resources tool server through your shared client
from back_end.mcp.client.mcp_client import call_tool

# ==================== PASTE THE COPIED CODE HERE ====================
UA = os.getenv("RESOURCES_UA", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TIMEOUT = float(os.getenv("RESOURCES_FETCH_TIMEOUT", "10"))

def fetch_once(url: str) -> dict:
    headers = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
    }
    r = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower() or "text/html"
    return {
        "url": r.url,
        "status": r.status_code,
        "content_type": ctype,
        "content": r.text if "text" in ctype or "json" in ctype else "",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
# =====================================================================

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
    # ==================== DEMO MODE FALLBACK ====================
    if os.getenv("DEMO_MODE") == "1":
        print("--- RUNNING IN DEMO MODE: RETURNING HARDCODED DATA ---", file=sys.stderr)
        demo_data = {
            "source": "local_resources_demo",
            "fetched_at": iso_utc_now(),
            "location": {"lat": lat, "lon": lon, "radius_km": 35},
            "resources": [
                {
                    "type": "sandbags",
                    "name": "Miami-Dade Public Works Distribution Site",
                    "address": "123 Main St, Miami, FL",
                    "notes": "Limit 10 bags per vehicle. Proof of residency required.",
                    "link": "https://www.miamidade.gov/global/emergency/hurricane/sandbags.page",
                    "publisher": "miamidade.gov",
                    "distance_km": 5.2
                },
                {
                    "type": "shelter",
                    "name": "American Red Cross Shelter - South Florida",
                    "address": "456 Oak Ave, Coral Gables, FL",
                    "notes": "Pets welcome in carriers. Please bring your own bedding.",
                    "link": "https://www.redcross.org/local/florida/south-florida.html",
                    "publisher": "redcross.org",
                    "distance_km": 8.1
                },
                {
                    "type": "food",
                    "name": "Feeding South Florida - Emergency Pantry",
                    "address": "789 Palm Blvd, Homestead, FL",
                    "notes": "Drive-thru distribution. Open 9 AM - 1 PM.",
                    "link": "https://feedingsouthflorida.org/",
                    "publisher": "feedingsouthflorida.org",
                    "distance_km": 22.5
                }
            ],
            "sources": ["demo_data.json"]
        }
        return demo_data
    # =============================================================

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

    # --- PHASE 1: DISCOVER ALL URLs FOR ALL TYPES ---
    urls_by_type: Dict[str, List[str]] = {t: [] for t in types}

    for rtype in types:
        # Get curated/seed URLs
        seeds = (curated_urls.get(rtype) if curated_urls and rtype in curated_urls else []) or []
        
        # Get URLs from the search tool
        search_result = call_tool("resources", "searchResources", {"type": rtype, "seed_urls": seeds}) or {}
        search_urls = (search_result.get("data") or {}).get("urls") or search_result.get("urls") or []
        
        # Combine with URLs discovered from CSE
        all_urls = search_urls + cse_pool.get(rtype, [])
        
        # De-dupe while preserving order
        seen, merged = set(), []
        for u in all_urls:
            if u not in seen:
                seen.add(u)
                merged.append(u)
        urls_by_type[rtype] = merged

    # --- PHASE 2: PROCESS URLs TYPE BY TYPE ---
    MAX_DEPTH = int(os.getenv("RESOURCE_PARSE_MAX_DEPTH", "2"))
    MAX_FOLLOWS_PER_TYPE = int(os.getenv("RESOURCE_PARSE_MAX_FOLLOWS", "12"))

    def _publisher_from_url(u: str) -> Optional[str]:
        try:
            netloc = urlparse(u if "://" in u else f"https://{u}").netloc
            return netloc or None
        except Exception:
            return None

    for rtype, urls in urls_by_type.items():
        if not urls:
            continue

        # Bounded BFS crawl for this type
        queue = deque([(u, 0) for u in urls])
        visited = set()
        follows_used = 0

        while queue:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                # Call the local function directly, bypassing the tool server
                fdata = fetch_once(url) 
                content = fdata.get("content") or ""

                # Ensure you have content to parse
                if not content:
                    continue

                parsed = call_tool("resources", "parseResources", {
                    "url": fdata.get("url") or url,
                    "content": content,
                    "content_type": fdata.get("content_type") or "text/html",
                    "type": rtype  # Use the correct rtype for this loop
                }) or {}
                pdata = parsed.get("data") or parsed

                for it in (pdata.get("items") or []):
                    it.setdefault("type", rtype)
                    it.setdefault("link", url)
                    it.setdefault("publisher", _publisher_from_url(url))
                    items.append(it)
                    sources.append(url)

                # Follow-up links
                if depth < MAX_DEPTH and follows_used < MAX_FOLLOWS_PER_TYPE:
                    for nu in (pdata.get("next_urls") or []):
                        if nu not in visited:
                            queue.append((nu, depth + 1))
                            follows_used += 1
                            if follows_used >= MAX_FOLLOWS_PER_TYPE:
                                break
            except Exception as e:
                # ==================== ADD THIS PRINT ====================
                print(f"!!! ERROR: Failed to fetch/parse URL '{url}'. Reason: {e}", file=sys.stderr)
                # ========================================================
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
            in_radius.append(it)  # keep unknown-distance items
        elif it["distance_km"] <= float(radius_km):
            in_radius.append(it)

    # 7) sort: items with distance first (nearest), then official-ish domains
    OFFICIAL_HINTS = (
        "redcross.org", "feedingamerica.org", ".fl.us", "fema.gov",
        "ready.gov", "disasterassistance.gov", "foodpantries.org"
    )
    def _score(it):
        has_dist = 0 if it.get("distance_km") is None else 1
        dist = it.get("distance_km") or 9999.0
        pub = (it.get("publisher") or "")
        gov_bonus = -1000 if pub.endswith(".gov") else 0
        hint_bonus = -200 if any(pub.endswith(h) or pub.endswith("." + h) for h in OFFICIAL_HINTS) else 0
        return (-has_dist, dist + gov_bonus + hint_bonus)

    in_radius.sort(key=_score)

    # Prepare diagnostics BEFORE building the final result
    diag = {
        "counties": {
            "primary": geo.get("primary_county"),
            "nearby": len(geo.get("nearby_counties", [])),
        },
        "cse_urls": {t: len(cse_pool.get(t, [])) for t in types},
        "sources_count": len(sources),
        "items_before_dedupe": len(items),
    }

    # Final response (build once)
    result: Dict[str, Any] = {
        "source": "local_resources",
        "fetched_at": iso_utc_now(),
        "location": {"lat": float(lat), "lon": float(lon), "radius_km": int(radius_km)},
        "resources": in_radius,
        "sources": sorted(set(sources)),
    }
    if not in_radius:
        result["debug"] = diag

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
