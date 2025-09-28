from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
from flask import Flask, request, jsonify

from back_end.mcp.resources.adapters.fetch import fetch_once
import back_end.mcp.resources.adapters.parse as parse_adapters
import back_end.mcp.resources.adapters.search as search_adapter


app = Flask(__name__)

@app.post("/tools/searchResources")
def search_resources():
    b = request.get_json(force=True) or {}
    rtype = (b.get("type") or "").lower().strip()
    seeds = b.get("seed_urls") or []
    q = (b.get("q") or "").strip() or None
    near = b.get("near")
    open_mode = bool(b.get("open"))
    max_results = int(b.get("max_results") or os.getenv("RESOURCE_SEARCH_MAX", "30"))
    allow_hosts = b.get("allow_hosts") or []
    deny_hosts = b.get("deny_hosts") or []

    result = search_adapter.search_urls(
        rtype=rtype,
        near=near,
        seeds=seeds,
        q=q,
        open_mode=open_mode,
        max_results=max_results,
        allow_hosts=allow_hosts,
        deny_hosts=deny_hosts
    )

    # ----------------------------------------------------------------------
    # FIX: The previous code was failing with KeyError: 'urls' if 
    # search_adapter.search_urls() returned a dictionary without a 'urls' key.
    # We now use .get("urls", []) to access it safely and prevent the 500 error.
    # ----------------------------------------------------------------------
    urls = result.get("urls", []) 

    return jsonify({
        "status": "ok",
        "data": {
            "urls": urls,
            "type": rtype,
            **{k: v for k, v in result.items() if k != "urls"} # THIS LINE WAS CHANGED
        }
    })

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
