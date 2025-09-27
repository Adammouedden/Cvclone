import os, requests, time
from dotenv import load_dotenv
from mcp.server import MCPServer, tool   # example, depends on mcp lib you use

load_dotenv()
UA = os.getenv("NWS_UA", "HurricaneGuardian/0.1 (+site; email)")
BASE = "https://api.weather.gov"
S = requests.Session()

def _get(url, accept="application/geo+json", timeout=6):
    headers = {"User-Agent": UA, "Accept": accept}
    r = S.get(url, headers=headers, timeout=timeout)
    if r.status_code == 429:
        time.sleep(5)
        r = S.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json() if "json" in r.headers.get("Content-Type","") else r.text

@tool
def getPoints(lat: float, lon: float) -> dict:
    """Get forecast office/grid info for a location"""
    return _get(f"{BASE}/points/{lat},{lon}")

@tool
def getAlerts(lat: float, lon: float, status: str = "actual") -> dict:
    """Get active alerts for a location"""
    url = f"{BASE}/alerts?status={status}&message_type=alert,update&point={lat},{lon}"
    return _get(url)

@tool
def getCapAlert(id_or_url: str) -> str:
    """Fetch alert as CAP/XML"""
    url = id_or_url if id_or_url.startswith("http") else f"{BASE}/alerts/{id_or_url}"
    return _get(url, accept="application/cap+xml")

if __name__ == "__main__":
    server = MCPServer("nws")
    server.register_tool(getPoints)
    server.register_tool(getAlerts)
    server.register_tool(getCapAlert)
    server.run("0.0.0.0", 7010)
