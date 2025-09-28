# Cvclone/src_back_end/mvp_server.py
from __future__ import annotations
import os, math
from typing import Dict, Any
from flask import Flask, request, render_template_string, redirect, url_for
import requests

# ---- Config: where the agents live ----
SUPPLY_PLANNER_URL = os.getenv("SUPPLY_PLANNER_URL", "http://localhost:8082/skills/build_checklist")
STORE_ROUTING_URL  = os.getenv("STORE_ROUTING_URL",  "http://localhost:8083/skills/source_supplies")

# ---- Fixed MVP presets (no frontend inputs) ----
DEFAULT_HOUSEHOLD = {
    "adults": 2, "children": 1, "infants": 0, "pets": 1,
    "special_needs": [], "diet": []
}
DEFAULT_HOURS = 72
# Demo locations: Miami & Orlando (choose by button)
LOCATIONS = {
    "miami":   {"label":"Miami, FL",   "lat": 25.7617, "lon": -80.1918},
    "orlando": {"label":"Orlando, FL", "lat": 28.5383, "lon": -81.3792},
}

# ---- Minimal HTML (server-rendered; no external assets needed) ----
PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Hurricane MVP</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; line-height: 1.4; }
    .btn { background:#0d6efd; border:none; padding:10px 14px; color:white; border-radius:8px; cursor:pointer; }
    .btn.secondary { background:#6c757d; }
    .card { border:1px solid #e5e7eb; border-radius:12px; padding:16px; margin:12px 0; }
    .row { display:flex; gap:12px; flex-wrap:wrap; }
    .muted { color:#6b7280; }
    table { border-collapse: collapse; width:100%; }
    th, td { border-bottom:1px solid #e5e7eb; padding:8px; text-align:left; }
    h1, h2 { margin: 8px 0; }
    small { color:#6b7280; }
  </style>
</head>
<body>
  <h1>Hurricane Prep — MVP</h1>
  <p class="muted">No inputs required. We assume a typical household and a preset city for demo.</p>

  <div class="card">
    <h2>Typical household needs ({{ hours }} hours)</h2>
    <ul>
      {% for it in essentials %}
        <li><strong>{{ it.name }}</strong>: {{ it.qty }} <small class="muted">({{ it.why }})</small></li>
      {% endfor %}
    </ul>
    <p class="muted">Priorities: A = life-sustaining, B = safety/comfort, C = situational.</p>
    <div class="row">
      <form method="post" action="{{ url_for('find_stores') }}">
        <input type="hidden" name="city" value="miami" />
        <button class="btn" type="submit">Find nearby stores — Miami</button>
      </form>
      <form method="post" action="{{ url_for('find_stores') }}">
        <input type="hidden" name="city" value="orlando" />
        <button class="btn secondary" type="submit">Find nearby stores — Orlando</button>
      </form>
    </div>
  </div>

  {% if result %}
    <div class="card">
      <h2>Suggested shops — {{ city_label }}</h2>
      {% if result and result["shops"] %}
        <table>
          <thead>
            <tr><th>Shop</th><th>ETA</th><th>Distance</th><th>Fulfills</th></tr>
          </thead>
          <tbody>
            {% for s in result["shops"] %}
              <tr>
                <td><strong>{{ s["name"] }}</strong> <small class="muted">({{ s["brand"] }})</small></td>
                <td>{{ s["eta_minutes"] }} min</td>
                <td>{{ s["distance_km"] }} km</td>
                <td>
                  {% set fulfilled = [] %}
                  {% for it in s["items"] %}
                    {% if it["can_fulfill"] and it["can_fulfill"] > 0 %}
                      {% set _ = fulfilled.append(it["name"] ~ " ×" ~ it["can_fulfill"]) %}
                    {% endif %}
                  {% endfor %}
                  {{ fulfilled|join(", ") if fulfilled else "—" }}
                </td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
        <p class="muted">Routing notes:
          {% for r in result["routes"] %}
            <br>• {{ r["to"] }} — {{ r["distance_km"] }} km / {{ r["eta_minutes"] }} min
            {% if r["avoid"] and r["avoid"]|length > 0 %} (avoid: {{ r["avoid"]|join("; ") }}){% endif %}
          {% endfor %}
        </p>
      {% else %}
        <p>No shops found in radius.</p>
      {% endif %}
      <p class="muted">Sources: {{ (result["sources"] or [])|join(", ") }}</p>
    </div>
  {% endif %}
</body>
</html>
"""

from flask import Flask
app = Flask(__name__)

def fetch_checklist() -> Dict[str, Any]:
    payload = {"household": DEFAULT_HOUSEHOLD, "hours": DEFAULT_HOURS}
    try:
        # If your Supply Planner isn’t running, we compute inline fallback:
        r = requests.post(SUPPLY_PLANNER_URL, json=payload, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        # minimal inline fallback (water + food only)
        H = max(1, math.ceil(DEFAULT_HOURS/24))
        adults, children, infants, pets = DEFAULT_HOUSEHOLD["adults"], DEFAULT_HOUSEHOLD["children"], DEFAULT_HOUSEHOLD["infants"], DEFAULT_HOUSEHOLD["pets"]
        water_l = math.ceil(3.0*(adults + 0.75*children)*H + 0.75*pets*H)
        kcal = (2000*adults + 1500*children + 800*infants) * H
        return {
            "items": [
                {"name":"Water (liters)","qty": water_l,"priority":"A","why":"3L/person/day; 0.75L/pet/day"},
                {"name":"Shelf-stable food (calories)","qty": int(kcal),"priority":"A","why":"2k adult, 1.5k child, 0.8k infant per day"},
                {"name":"Batteries AA/AAA (count)","qty": 12*H,"priority":"B","why":"flashlights, radios"},
            ],
            "rationale": ["inline fallback"]
        }

def essentials_from(checklist: Dict[str,Any]):
    # Keep just top items for readability on the MVP page
    names = {"Water (liters)","Shelf-stable food (calories)","Prescription meds (days)","Batteries AA/AAA (count)","First aid kit"}
    items = [it for it in checklist.get("items",[]) if it.get("name") in names]
    # ensure meds/first aid presence
    if not any(it["name"]=="Prescription meds (days)" for it in items):
        H = max(1, math.ceil(DEFAULT_HOURS/24))
        items.append({"name":"Prescription meds (days)","qty": max(H,7),"priority":"A","why":"buffer if pharmacies closed"})
    if not any(it["name"]=="First aid kit" for it in items):
        items.append({"name":"First aid kit","qty":1,"priority":"A","why":"basic care"})
    return items

@app.get("/")
def home():
    checklist = fetch_checklist()
    essentials = essentials_from(checklist)
    return render_template_string(PAGE, essentials=essentials, hours=DEFAULT_HOURS, result=None)

@app.post("/find-stores")
def find_stores():
    city = request.form.get("city","miami")
    loc = LOCATIONS.get(city, LOCATIONS["miami"])
    checklist = fetch_checklist()
    essentials = essentials_from(checklist)

    # Build a compact items list for routing
    items = [{"name": it["name"], "qty": it["qty"]} for it in essentials if it.get("qty",0) > 0]

    # Call store-routing agent
    payload = {
        "items": items,
        "user_loc": {"lat": loc["lat"], "lon": loc["lon"]},
        "max_radius_km": 25,
        # "closures": []  # wire FL511 later
    }
    result = None
    try:
        rr = requests.post(STORE_ROUTING_URL, json=payload, timeout=8)
        rr.raise_for_status()
        result = rr.json()
    except Exception as e:
        result = {"shops": [], "routes": [], "sources": ["store-index:v0"], "error": str(e)}

    return render_template_string(PAGE, essentials=essentials, hours=DEFAULT_HOURS, result=result, city_label=loc["label"])

if __name__ == "__main__":
    port = int(os.getenv("MVP_SERVER_PORT","8090"))
    print(f"🏁 MVP server at http://127.0.0.1:{port}")
    app.run("0.0.0.0", port, debug=False)
