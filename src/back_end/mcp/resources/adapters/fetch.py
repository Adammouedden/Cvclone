# back_end/mcp/resources/adapters/fetch.py
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os, time, requests

S = requests.Session()
UA = os.getenv("RESOURCES_UA", "ResourceFinder/0.1")

def fetch_once(url: str, timeout: int = 10) -> dict:
    """One polite GET. Returns url, status, content_type, content, fetched_at."""
    r = S.get(url, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "text/html").split(";")[0]
    text = r.text if ("text" in ctype or "json" in ctype) else r.content.decode("utf-8", "ignore")
    return {
        "url": url,
        "status": r.status_code,
        "content_type": ctype,
        "content": text,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
