# back_end/mcp/client/mcp_client.py
import os, requests

MCP_BASES = {
    "nws": os.getenv("MCP_BASE_NWS", "http://localhost:7010"),
    "resources": os.getenv("MCP_BASE_RESOURCES", "http://localhost:7020"),
}

def call_tool(agent: str, tool: str, payload: dict):
    base = MCP_BASES.get(agent)
    url = f"{base}/tools/{tool}" if base else None

    # Try MCP first (if configured)
    if url:
        try:
            r = requests.post(url, json=payload, timeout=5)
            r.raise_for_status()
            res = r.json()
            if isinstance(res, dict) and res.get("status") == "ok":
                return res["data"]
            # accept raw JSON too
            if isinstance(res, dict):
                if "properties" in res or "features" in res or "@context" in res:
                    return res
            raise RuntimeError(res.get("error", "tool error") if isinstance(res, dict) else "tool error")
        except Exception:
            # fall through to direct mode
            pass

    # DIRECT MODE (no MCP): call NWS yourself
    if agent == "nws" and tool == "getForecast":
        lat = payload["lat"]; lon = payload["lon"]; hourly = bool(payload.get("hourly", False))
        meta = requests.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=10).json()
        props = meta["properties"]
        url = props["forecastHourly"] if hourly else props["forecast"]
        return requests.get(url, timeout=10, headers={"User-Agent":"cvclone/forecast (dev)"}).json()

    if agent == "nws" and tool == "getAlerts":
        lat = payload["lat"]; lon = payload["lon"]
        url = f"https://api.weather.gov/alerts?point={lat},{lon}&status=actual&message_type=alert,update"
        return requests.get(url, timeout=10, headers={"User-Agent":"cvclone/forecast (dev)"}).json()

    raise RuntimeError(f"Unknown MCP agent '{agent}' or tool '{tool}'")
