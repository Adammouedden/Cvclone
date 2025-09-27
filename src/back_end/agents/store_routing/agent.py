from __future__ import annotations
import os, math, json
from typing import List, Dict, Any, Tuple
from flask import Flask, request, jsonify

# ---- Mock store catalog (replace with DB/API later) ----
STORE_INDEX = [
  {"name":"Publix #123","brand":"Publix","lat":25.772,"lon":-80.196,"cats":["water","food","batteries"]},
  {"name":"Walmart #45","brand":"Walmart","lat":25.781,"lon":-80.209,"cats":["water","food","batteries","tarps"]},
  {"name":"Home Depot Midtown","brand":"Home Depot","lat":25.804,"lon":-80.195,"cats":["tarps","plywood","sandbags","batteries"]},
  {"name":"Lowe's","brand":"Lowe's","lat":25.738,"lon":-80.250,"cats":["tarps","plywood","sandbags","batteries"]},
  {"name":"CVS #100","brand":"CVS","lat":25.763,"lon":-80.190,"cats":["meds","batteries","ice"]}
]

ITEM_TO_CATS = {
  "Water (liters)":"water",
  "Shelf-stable food (calories)":"food",
  "Batteries AA/AAA (count)":"batteries",
  "Power bank (Wh)":"batteries",
  "First aid kit":"meds",
  "Prescription meds (days)":"meds",
  "Tarps / duct tape":"tarps",
  "Plywood panels":"plywood",
  "Sandbags":"sandbags",
  "Ice / cooler":"ice",
  "Infant formula (servings)":"food",
  "Diapers (count)":"food",   # or "baby"
  "Wipes (packs)":"food"
}

def haversine_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    R=6371.0
    import math
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2-lat1; dlon = lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

def estimate_eta_minutes(distance_km: float, penalty: float=0.0) -> int:
    # 35 km/h urban avg; add penalty minutes for closures
    base = distance_km / 35.0 * 60.0
    return int(round(base + penalty))

def score_store(store: Dict[str,Any], need_cats: List[str]) -> int:
    # simple score: matches - distance bucket
    matches = sum(1 for c in need_cats if c in store["cats"])
    return matches

def plan_routes(user_loc: Dict[str,float],
                items: List[Dict[str,Any]],
                max_radius_km: float,
                closures: List[str] | None = None) -> Dict[str, Any]:
    need_cats = sorted({ITEM_TO_CATS.get(i["name"], "").strip() for i in items if i.get("qty",0)>0})
    need_cats = [c for c in need_cats if c]

    # filter stores by radius and category match
    shops = []
    for s in STORE_INDEX:
        dist = haversine_km((user_loc["lat"], user_loc["lon"]), (s["lat"], s["lon"]))
        if dist > max_radius_km:
            continue
        sc = score_store(s, need_cats)
        if sc == 0:
            continue
        # naive closure penalty: if any closure string matches brand/name, add 5 min
        penalty = 0.0
        notes = []
        if closures:
            for c in closures:
                if any(tok.lower() in c.lower() for tok in [s["name"], s["brand"]]):
                    penalty += 5.0; notes.append(f"closure: {c}")
        eta = estimate_eta_minutes(dist, penalty)
        # which items can this store fulfill?
        supplied = [{"name": i["name"], "can_fulfill": i["qty"] if ITEM_TO_CATS.get(i["name"]) in s["cats"] else 0}
                    for i in items]
        shops.append({
            "name": s["name"], "brand": s["brand"], "lat": s["lat"], "lon": s["lon"],
            "eta_minutes": eta, "stock_confidence": 0.6 + 0.1*min(sc,3),
            "items": supplied, "notes": notes, "distance_km": round(dist,1)
        })

    # pick top 3 by (score desc, eta asc)
    shops.sort(key=lambda x: (-sum(1 for it in x["items"] if it["can_fulfill"]), x["eta_minutes"]))
    shops = shops[:3]

    # routes (straight-line placeholder)
    routes = [{"from":"user_loc","to":s["name"],"distance_km": s["distance_km"],"eta_minutes": s["eta_minutes"],"avoid": s["notes"]}
              for s in shops]

    return {"shops": shops, "routes": routes, "sources": ["store-index:v0", "fl511:v0"]}

# ---- HTTP exposure (A2A-like) ----
app = Flask(__name__)

@app.get("/.well-known/agent-card.json")
def card():
    return jsonify({
        "id":"store-routing",
        "name":"Store & Routing Agent",
        "version":"0.1.0",
        "skills":[{
          "name":"source_supplies",
          "description":"Find nearby shops and routes for a supply checklist.",
          "args":{"type":"object"}
        }]
    })

@app.post("/skills/source_supplies")
def skill_source():
    b = request.get_json(force=True) or {}
    items = b.get("items", [])
    user_loc = b.get("user_loc", {})
    max_r = float(b.get("max_radius_km", 25))
    closures = b.get("closures", [])  # wire FL511 later
    return jsonify(plan_routes(user_loc, items, max_r, closures))

if __name__ == "__main__":
    port = int(os.getenv("STORE_ROUTING_PORT","8083"))
    print(f"🛒 Store & Routing on http://127.0.0.1:{port}")
    app.run("0.0.0.0", port, debug=False)
