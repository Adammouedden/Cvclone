# 🌪️ Hurricane Vision Project — Agentic AI Plan

## A) Core Data Plane (feeds for agents)

- **Tracks / Cones / Watchesa & Warnings** → National Hurricane Center (NHC) GIS layers (GeoJSON/tiles).  
- **Local Forecasts & Alerts** → NWS API (`api.weather.gov`, CAP alerts).  
- **Flood Risk Context** → FEMA National Flood Hazard Layer (NFHL).  
- **Historical Storms** → HURDAT2 / IBTrACS (training + ground truth).  
- **Disaster Costs** → OpenFEMA disaster + assistance datasets.  
- **Road Closures (FL MVP)** → FL511 live events.  
- **Context / Baseline** → Google DeepMind Weather Lab cyclone model.  

⚠️ Note: **NWS multilingual alerts are paused** → we’ll do translations locally in our chatbot.

---

## B) Civilian Side — Agents

### 1. Orchestrator (Local)
- **Role:** Entry point; fans out requests, aggregates results.  
- **Skill:** `run_scenario({lat,lon,horizon<=24}) -> {cone, alerts, plan, news, checklist}`

### 2. Track & Alert Fetcher (Local, MCP/OpenAPI)
- **Tools:** NHC GIS + NWS API.  
- **Skill:** `get_hazard_snapshot({lat,lon}) -> {cone_poly, alerts[], winds}`

### 3. Supply Planner (Local)
- **Skill:** `build_checklist({household, hours}) -> {items[], quantities, rationale[]}`

### 4. Store & Routing Agent (Remote via A2A)
- **Skill:** `source_supplies({items[], user_loc}) -> {shops[], routes[]}`  
- Uses FL511 closures for Florida MVP.

### 5. Sandbag Finder (Remote via A2A)
- **Skill:**  
  - `crawl_official_sources({county,state}) -> {sites[]}`  
  - `nearest_sites({lat,lon}) -> {sites_sorted}`

### 6. Home Prep Advisor (Remote via A2A + Vision)
- **Input:** exterior photos of home.  
- **Skill:** `prep_recs_from_images({images[], wind_prob, surge_zone}) -> {actions[]}`

### 7. Local News Curator (Remote via A2A)
- **Skill:** `get_local_updates({lat,lon}) -> {headlines[], brief}`

### 8. Chatbot / Explainer (Local)
- **Skill:** `answer_user_q({qa, snapshot}) -> {answer, sources}`  
- Handles translations (since NWS paused theirs).

---

## C) Business Side — Agents

### 1. Exposure & Cost Modeler (Remote via A2A)
- **Skill:** `estimate_event_costs({assets[], current_cone}) -> {loss_mean, p95, drivers[]}`  
- Data: historical tracks + OpenFEMA assistance.

### 2. HQ / Site Optimizer (Remote via A2A)
- **Skill:** `rank_sites({candidates[], constraints}) -> {ranked[], why}`  
- Data: NFHL flood zones + historical track density.

### 3. Continuity & Logistics Agent (Remote via A2A)
- **Skill:** `route_plans({depots[], stores[], closures_feed}) -> {alternates[], risk_notes}`

### 4. Executive Briefing Agent (Local)
- **Skill:** `make_brief({hazard_snapshot, costs, logistics}) -> {one_pager_md, charts}`

---

## D) Orchestration (Civilian Flow)

1. **Orchestrator** → calls **Track/Alert Fetcher** → gets `hazard_snapshot`.  
2. **Parallel fan-out**:  
   - Supply Planner → checklist.  
   - Sandbag Finder → sites.  
   - Store & Routing → shops + routes.  
   - Local News Curator → headlines.  
   - (Optional) Home Prep Advisor → photo-based actions.  
3. **Chatbot** summarizes and answers questions.

---

## E) Orchestration (Business Flow)

1. **Orchestrator** → Exposure & Cost Modeler → loss estimates.  
2. HQ/Site Optimizer → safer location rankings.  
3. Continuity & Logistics Agent → alternate routing.  
4. Executive Briefing Agent → one-pager summary with charts + sources.

---

## Demo Flow (2–3 minutes)

1. Pick Florida MVP + historical storm → UI shows ground truth vs our model’s predicted track.  
2. Live Mode → enter ZIP → cone + alerts (NHC/NWS).  
3. Checklist, sandbag map, stores/routes (FL511).  
4. Upload photos → Home Prep Advisor returns prioritized action list.  
5. Switch to Business tab → cost estimates, HQ optimizer, continuity plans.  
6. Executive Briefing Agent outputs a 1-pager with sources.

---
