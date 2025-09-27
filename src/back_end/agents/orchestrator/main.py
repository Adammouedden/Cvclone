# Cvclone/src_back_end/agents/orchestrator/main.py
from __future__ import annotations
import os, math, time
from typing import Any, Dict, List, Optional
from flask import Flask, request, jsonify
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

app = Flask(__name__)

# --- Agent endpoints (config via env for portability) ---
FORECAST_URL       = os.getenv("FORECAST_URL",       "http://localhost:8081/skills/get_snapshot")
SUPPLY_PLANNER_URL = os.getenv("SUPPLY_PLANNER_URL", "http://localhost:8082/skills/build_checklist")
STORE_ROUTING_URL  = os.getenv("STORE_ROUTING_URL",  "http://localhost:8083/skills/source_supplies")
RESOURCE_FINDER_URL= os.getenv("RESOURCE_FINDER_URL", "")  # optional

HTTP_TIMEOUT = float(os.getenv("ORCH_HTTP_TIMEOUT", "8"))
MAX_WORKERS  = int(os.getenv("ORCH_MAX_WORKERS", "6"))
DEFAULT_HOURS = int(os.getenv("DEFAULT_HOURS", "72"))

def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def post_json(url: str, payload: dict, timeout: float = HTTP_TIMEOUT) -> tuple[Optional[dict], Optional[str]]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"{url}: {e}"

def essentials_from(checklist: Dict[str,Any], hours: int) -> List[Dict[str,Any]]:
    names = {
        "Water (liters)",
        "Shelf-stable food (calories)",
        "Prescription meds (days)",
        "Batteries AA/AAA (count)",
        "First aid kit"
    }
    items = [it for it in (checklist.get("items") or []) if it.get("name") in names]
    # ensure meds & first aid
    H = max(1, math.ceil(hours/24))
    if not any(it["name"]=="Prescription meds (days)" for it in items):
        items.append({"name":"Prescription meds (days)","qty": max(H,7)})
    if not any(it["name"]=="First aid kit" for it in items):
        items.append({"name":"First aid kit","qty": 1})
    # only keep with qty>0
    return [ {"name":it["name"], "qty":it.get("qty",0)} for it in items if it.get("qty",0) > 0 ]

@app.get("/.well-known/agent-card.json")
def card():
    return jsonify({
        "id": "orchestrator",
        "name": "Orchestrator Agent",
        "version": "0.1.0",
        "skills": [{
            "name": "run_scenario",
            "description": "Aggregate forecast, alerts, supply plan, and store routing.",
            "args": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                    "horizon": {"type": "integer", "maximum": 24, "default": 24},
                    "hours": {"type": "integer", "default": DEFAULT_HOURS},
                    "household": {"type": "object"}
                },
                "required": ["lat","lon"]
            },
            "returns": {"type":"object"}
        }]
    })

@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "ts": iso_utc_now()})

@app.post("/skills/run_scenario")
def run_scenario():
    b = request.get_json(force=True) or {}
    lat, lon = b.get("lat"), b.get("lon")
    if lat is None or lon is None:
        return jsonify({"error":"lat/lon required"}), 400
    horizon  = int(b.get("horizon", 24))
    if horizon > 24: horizon = 24
    hours    = int(b.get("hours", DEFAULT_HOURS))
    household= b.get("household") or {"adults":2,"children":1,"infants":0,"pets":1,"special_needs":[], "diet":[]}

    started = time.time()
    sources: List[str] = []
    errors:  List[str] = []

    # Step 1: call forecast first (alerts live there too)
    forecast_payload = {"lat": float(lat), "lon": float(lon), "hourly": True, "daily": True}
    forecast, err = post_json(FORECAST_URL, forecast_payload)
    if err: errors.append(err)
    if forecast and isinstance(forecast.get("sources"), list):
        sources.extend(forecast["sources"])

    # Step 2 & 3 in parallel: supply planner and (optionally) resource finder
    checklist: Dict[str,Any] = {}
    civic_resources: Dict[str,Any] = {}  # placeholder for resource finder outputs

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        futures[ex.submit(post_json, SUPPLY_PLANNER_URL, {"household": household, "hours": hours})] = "supply"
        if RESOURCE_FINDER_URL:
            futures[ex.submit(post_json, RESOURCE_FINDER_URL, {"lat": lat, "lon": lon})] = "resources"
        for fut in as_completed(futures):
            tag = futures[fut]
            res, e = fut.result()
            if e: errors.append(e)
            if tag == "supply" and isinstance(res, dict):
                checklist = res
            if tag == "resources" and isinstance(res, dict):
                civic_resources = res

    # Step 4: store routing (depends on checklist essentials)
    shops = []; routes = []
    if checklist:
        essentials = essentials_from(checklist, hours)
        routing_payload = {
            "items": essentials,
            "user_loc": {"lat": float(lat), "lon": float(lon)},
            "max_radius_km": 25
        }
        routing, err = post_json(STORE_ROUTING_URL, routing_payload)
        if err: errors.append(err)
        if isinstance(routing, dict):
            shops  = routing.get("shops")  or []
            routes = routing.get("routes") or []
            if isinstance(routing.get("sources"), list):
                sources.extend(routing["sources"])

    # Assemble result
    result = {
        "source": "orchestrator",
        "fetched_at": iso_utc_now(),
        "inputs": {"lat": float(lat), "lon": float(lon), "horizon": horizon, "hours": hours},
        "forecast": forecast or {},
        "alerts": (forecast or {}).get("alerts") or [],
        "plan": {
            "checklist": checklist or {},
            "shops": shops,
            "routes": routes
        },
        "civic_resources": civic_resources or {},   # optional section (sandbags, shelters, etc.)
        "cone": None,                               # placeholder for trajectory/cone viz
        "news": [],                                 # placeholder until News agent is ready
        "sources": list(dict.fromkeys(sources)),    # de-dupe keep order
        "metrics": {"elapsed_ms": int((time.time()-started)*1000)},
        "errors": errors
    }
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.getenv("ORCHESTRATOR_PORT","8080"))
    print(f"🧩 Orchestrator on http://127.0.0.1:{port}")
    app.run("0.0.0.0", port, debug=False)
