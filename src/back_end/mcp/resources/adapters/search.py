# back_end/mcp/resources/adapters/search.py
from __future__ import annotations
import os, requests
from typing import List, Dict, Optional, Any

# Defaults / knobs

print("[search adapter] loading…")

_BAD_HOSTS_DEFAULT = {
    "facebook.com", "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "youtube.com"
}
_TLD_WHITELIST = (".gov", ".us", ".fl.us", ".org")

def _host(url: str) -> str:
    try:
        return url.split("//", 1)[1].split("/", 1)[0].lower()
    except Exception:
        return ""

def _dedupe(urls: List[str]) -> List[str]:
    seen, out = set(), []
    for u in urls:
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out

def _google_search_paginated(query: str, total: int = 30) -> List[str]:
    key = os.getenv("GOOGLE_CSE_KEY")
    cx  = os.getenv("GOOGLE_CSE_CX")
    endpoint = os.getenv("GOOGLE_CSE_ENDPOINT", "https://www.googleapis.com/customsearch/v1")
    if not key or not cx or not query:
        return []
    total = max(1, min(int(total), 100))
    out: List[str] = []
    start = 1  # Google starts at 1
    while len(out) < total:
        num = min(10, total - len(out))
        try:
            r = requests.get(
                endpoint,
                params={
                    "key": key,
                    "cx": cx,
                    "q": query,
                    "num": num,
                    "start": start,
                    "safe": "active",
                    "hl": "en",
                },
                timeout=8,
            )
            r.raise_for_status()
            data = r.json() or {}
            items = data.get("items") or []
            links = [it.get("link") for it in items if it.get("link")]
            out.extend(links)
            if not items or len(items) < num:
                break
            start += num
        except Exception:
            break
    return out

def build_query(rtype: str, near: Optional[Dict] | Optional[str], open_mode: bool) -> str:
    rtype = (rtype or "").lower().strip()
    terms_map = {
        "shelter":  ['(shelter OR "evacuation center" OR "emergency shelter")'],
        "sandbags": ['(sandbags OR "sandbag site" OR "sandbag locations")'],
        "food":     ['("food distribution" OR "relief food" OR "meal sites")'],
    }
    base_terms = terms_map.get(rtype, [rtype or ""])

    near_str = ""
    if isinstance(near, str) and near.strip():
        near_str = f' "{near.strip()}"'
    elif isinstance(near, dict):
        city = (near.get("city") or "").strip()
        state = (near.get("state") or "").strip()
        if city and state:
            near_str = f' "{city}, {state}"'
        elif city:
            near_str = f' "{city}"'
        elif state:
            near_str = f' "{state}"'

    site_bias = "" if open_mode else " (site:.gov OR site:.us OR site:.fl.us OR site:.org)"
    return f'{" ".join(base_terms)}{site_bias}{near_str}'.strip()

def filter_urls(
    urls: List[str],
    open_mode: bool,
    allow_hosts: Optional[List[str]] = None,
    deny_hosts: Optional[List[str]] = None,
) -> List[str]:
    allow_hosts = [h.lower() for h in (allow_hosts or [])]
    bad_hosts = set(_BAD_HOSTS_DEFAULT)
    if deny_hosts:
        bad_hosts |= set(h.lower() for h in deny_hosts)

    def allowed(u: str) -> bool:
        h = _host(u)
        if not h:
            return False
        if h in bad_hosts:
            return False
        if allow_hosts:
            return any(h == ah or h.endswith(ah) for ah in allow_hosts)
        if open_mode:
            return True
        return (h.endswith(_TLD_WHITELIST) or ".gov/" in u)

    return [u for u in urls if allowed(u)]

def search_urls(
    rtype: str,
    near: Optional[Dict] | Optional[str],
    seeds: Optional[List[str]] = None,
    q: Optional[str] = None,
    open_mode: bool = False,
    max_results: int = 30,
    allow_hosts: Optional[List[str]] = None,
    deny_hosts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return dict with urls + metadata; keeps working even if API keys missing (seeds only)."""
    query = q or build_query(rtype, near, open_mode)
    hits = _google_search_paginated(query, total=max_results)
    urls = _dedupe([*(seeds or []), *hits])
    urls = filter_urls(urls, open_mode=open_mode, allow_hosts=allow_hosts, deny_hosts=deny_hosts)
    return {
        "urls": urls,
        "query": query,
        "open": open_mode,
        "requested": int(max_results),
        "returned": len(urls),
        "engine": "google_cse",
    }

print("[search adapter] loaded. has search_urls:", "search_urls" in globals())