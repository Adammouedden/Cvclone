from __future__ import annotations
import math, os
from typing import Dict, Any, List
from flask import Flask, request, jsonify

def round_up(x: float) -> int:
    return int(math.ceil(x))

def build_checklist(household: Dict[str, Any], hours: int) -> Dict[str, Any]:
    adults   = int(household.get("adults", 0))
    children = int(household.get("children", 0))
    infants  = int(household.get("infants", 0))
    pets     = int(household.get("pets", 0))
    needs    = set(household.get("special_needs", []) or [])
    H = max(1, math.ceil(hours/24))

    # Water
    human_liters = 3.0 * (adults + 0.75*children) * H
    pet_liters   = 0.75 * pets * H
    water_total  = round_up(human_liters + pet_liters)

    # Food calories
    total_kcal = (2000*adults + 1500*children + 800*infants) * H

    items: List[Dict[str, Any]] = [
        {"name":"Water (liters)","qty": water_total,"priority":"A","why":"3L/person/day; 0.75L/pet/day"},
        {"name":"Shelf-stable food (calories)","qty": int(total_kcal),"priority":"A","why":"2k adult, 1.5k child, 0.8k infant per day"},
        {"name":"Prescription meds (days)","qty": max(H,7),"priority":"A","why":"buffer if pharmacies closed"},
        {"name":"First aid kit","qty": 1,"priority":"A","why":"basic care"},
        {"name":"Batteries AA/AAA (count)","qty": 12*H,"priority":"B","why":"flashlights, radios"},
        {"name":"Power bank (Wh)","qty": 10*(adults+children+infants)*H,"priority":"B","why":"phones"},
        {"name":"Pet food (days)","qty": H if pets>0 else 0,"priority":"B","why":"daily ration"},
        {"name":"Tarps / duct tape","qty": 2,"priority":"C","why":"minor damage"},
        {"name":"Ice / cooler","qty": 1 if ("insulin" in needs) else 0,"priority":"B","why":"cold chain for meds"},
        {"name":"Plywood panels","qty": 0,"priority":"C","why":"board windows if advised"},
        {"name":"Sandbags","qty": 0,"priority":"C","why":"if flood-prone"},
    ]
    if infants>0:
        items += [
            {"name":"Infant formula (servings)","qty": 6*H*infants,"priority":"A","why":"~6 feeds/day"},
            {"name":"Diapers (count)","qty": 8*H*infants,"priority":"B","why":"~8/day"},
            {"name":"Wipes (packs)","qty": H,"priority":"B","why":"hygiene"},
        ]

    rationale = [
        "Water: 3 L/person/day; pets ~0.75 L/day",
        "Calories: 2000 adult, 1500 child, 800 infant per day",
        "Meds: aim for ≥7 days supply",
        "Priorities: A (life-sustaining), B (safety/comfort), C (situational)"
    ]

    return {"items": items, "rationale": rationale}

# Optional tiny server to expose as an A2A-like skill
app = Flask(__name__)

@app.get("/.well-known/agent-card.json")
def card():
    return jsonify({
        "id":"supply-planner",
        "name":"Supply Planner",
        "version":"0.1.0",
        "skills":[{"name":"build_checklist","args":{"type":"object"}}]
    })

@app.post("/skills/build_checklist")
def skill_build():
    b = request.get_json(force=True) or {}
    return jsonify(build_checklist(b.get("household", {}), int(b.get("hours",72))))

if __name__ == "__main__":
    port = int(os.getenv("SUPPLY_PLANNER_PORT","8082"))
    print(f"🧰 Supply Planner on http://127.0.0.1:{port}")
    app.run("0.0.0.0", port, debug=False)
