# purpose: allow different agents (planner, resource finder, safety coach) call the NWS MCP server 
import requests

MCP_BASE = "http://localhost:7010"   # or wherever your MCP server runs

def call_tool(agent: str, tool: str, payload: dict):
    r = requests.post(f"{MCP_BASE}/tools/{tool}", json=payload, timeout=6)
    r.raise_for_status()
    res = r.json()
    if res.get("status") == "ok":
        return res["data"]
    raise RuntimeError(res.get("error", "tool error"))
