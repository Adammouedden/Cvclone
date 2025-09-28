# purpose: allow different agents (planner, resource finder, safety coach) call the NWS MCP server 
import os, requests

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


MCP_BASES = {
    "nws": os.getenv("MCP_BASE_NWS", "http://localhost:7010"),
    "resources": os.getenv("MCP_BASE_RESOURCES", "http://localhost:7020"),
}

def call_tool(agent: str, tool: str, payload: dict):
    base = MCP_BASES.get(agent)
    if not base:
        raise RuntimeError(f"Unknown MCP agent '{agent}'")
    r = requests.post(f"{base}/tools/{tool}", json=payload, timeout=10)
    r.raise_for_status()
    res = r.json()
    if res.get("status") == "ok":
        return res["data"]
    raise RuntimeError(res.get("error", "tool error"))