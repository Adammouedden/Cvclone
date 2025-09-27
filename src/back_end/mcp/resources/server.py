from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os, time, requests
from flask import Flask, request, jsonify

# back_end/mcp/resources/server.py
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os, time, requests
from flask import Flask, request, jsonify
from back_end.mcp.resources.adapters.fetch import fetch_once
import back_end.mcp.resources.adapters.parse as parse_adapters  # <-- change

app = Flask(__name__)

@app.post("/tools/searchResources")
def search_resources():
    b = request.get_json(force=True) or {}
    rtype = (b.get("type") or "").lower()
    seeds = b.get("seed_urls") or []
    # MVP: just echo seeds for now; wire SERP later
    return jsonify({"status": "ok", "data": {"urls": seeds, "type": rtype}})

@app.post("/tools/fetchUrl")
def fetch_url():
    b = request.get_json(force=True) or {}
    url = b.get("url")
    if not url:
        return jsonify({"status": "error", "error": "url required"}), 400
    try:
        data = fetch_once(url)
        return jsonify({"status": "ok", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 502

@app.post("/tools/parseResources")
def parse_endpoint():
    b = request.get_json(force=True) or {}
    url = b.get("url")
    content = b.get("content", "")
    ctype = b.get("content_type", "text/html")
    rtype = (b.get("type") or "").lower()
    if not url or not rtype:
        return jsonify({"status": "error", "error": "url and type required"}), 400
    try:
        items = parse_adapters.parse_resources(url, content, ctype, rtype)
        return jsonify({"status": "ok", "data": {"items": items}})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 502

if __name__ == "__main__":
    port = int(os.getenv("RESOURCES_PORT", "7020"))
    print(f"🧭 Resources tool server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
