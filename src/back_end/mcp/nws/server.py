# server.py — HTTP tool server compatible with your mcp_client + curl
import os, requests, time
from flask import Flask, request, jsonify

UA = os.getenv("NWS_UA", "HurricaneGuardian/0.1 (+site; email)")
BASE = "https://api.weather.gov"
S = requests.Session()
app = Flask(__name__)

def _get(url, accept="application/geo+json", timeout=6):
    headers = {"User-Agent": UA, "Accept": accept}
    r = S.get(url, headers=headers, timeout=timeout)
    if r.status_code == 429:
        time.sleep(5)
        r = S.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    return r.json() if "json" in ctype else r.text

# --- Tools exposed as HTTP endpoints ---

@app.post("/tools/getPoints")
def getPoints():
    b = request.get_json(force=True) or {}
    lat, lon = b.get("lat"), b.get("lon")
    if lat is None or lon is None:
        return jsonify({"status":"error","error":"lat/lon required"}), 400
    try:
        data = _get(f"{BASE}/points/{lat},{lon}")
        return jsonify({"status":"ok","data": data})
    except Exception as e:
        return jsonify({"status":"error","error": str(e)}), 502

@app.post("/tools/getAlerts")
def getAlerts():
    b = request.get_json(force=True) or {}
    lat, lon = b.get("lat"), b.get("lon")
    status = b.get("status", "actual")
    if lat is None or lon is None:
        return jsonify({"status":"error","error":"lat/lon required"}), 400
    try:
        url = f"{BASE}/alerts?status={status}&message_type=alert,update&point={lat},{lon}"
        data = _get(url)
        return jsonify({"status":"ok","data": data})
    except Exception as e:
        return jsonify({"status":"error","error": str(e)}), 502

@app.post("/tools/getCapAlert")
def getCapAlert():
    b = request.get_json(force=True) or {}
    id_or_url = b.get("id_or_url")
    if not id_or_url:
        return jsonify({"status":"error","error":"id_or_url required"}), 400
    try:
        url = id_or_url if str(id_or_url).startswith("http") else f"{BASE}/alerts/{id_or_url}"
        data = _get(url, accept="application/cap+xml")
        return jsonify({"status":"ok","data": data})
    except Exception as e:
        return jsonify({"status":"error","error": str(e)}), 502

@app.post("/tools/getForecast")
def getForecast():
    b = request.get_json(force=True) or {}
    lat, lon = b.get("lat"), b.get("lon")
    hourly = bool(b.get("hourly", False))  # default False

    # Basic validation
    if lat is None or lon is None:
        return jsonify({"status":"error","error":"lat/lon required"}), 400
    try:
        # Normalize to string to avoid repr issues like many decimals
        lat_s, lon_s = str(lat), str(lon)

        # 1) Resolve the canonical forecast URL(s) for this point
        points = _get(f"{BASE}/points/{lat_s},{lon_s}")  # GeoJSON
        props = (points or {}).get("properties", {})
        if not props:
            return jsonify({"status":"error","error":"NWS /points response missing 'properties'"}), 502

        url = props.get("forecastHourly") if hourly else props.get("forecast")
        if not url:
            which = "forecastHourly" if hourly else "forecast"
            return jsonify({"status":"error","error": f"NWS /points missing '{which}' URL"}), 502

        # 2) Fetch the forecast payload at that canonical URL
        data = _get(url)

        # attach the URL we actually fetched (non-destructive)
        if isinstance(data, dict):
            data.setdefault("_meta", {})["resolved_url"] = url
        
        # 3) Return raw NWS forecast (your Forecast agent will normalize)
        return jsonify({"status": "ok", "data": data})

    except requests.Timeout:
        return jsonify({"status":"error","error":"NWS request timed out"}), 504
    except requests.HTTPError as e:
        # Bubble up upstream status if possible
        return jsonify({"status":"error","error": f"NWS HTTP error: {e}"}), 502
    except Exception as e:
        return jsonify({"status":"error","error": str(e)}), 502

if __name__ == "__main__":
    print("🚀 NWS tool server on http://127.0.0.1:7010")
    app.run(host="0.0.0.0", port=7010)
