# back_end/mcp/resources/adapters/parse/__init__.py
from urllib.parse import urlparse
from typing import Dict, List, Callable, Tuple

ParserFn = Callable[[str, str, str], List[Dict]]
_registry: dict[Tuple[str, str], ParserFn] = {}

def register(domain: str, rtype: str, fn: ParserFn):
    _registry[(domain.lower(), rtype)] = fn

def parse_resources(url: str, content: str, content_type: str, desired_type: str) -> List[Dict]:
    netloc = urlparse(url).netloc.lower()
    parts = netloc.split(".")
    # exact and base-domain match
    keys = [(netloc, desired_type)] + [(".".join(parts[i:]), desired_type) for i in range(1, len(parts)-1)]
    for k in keys:
        fn = _registry.get(k)
        if fn:
            return fn(url, content, content_type)
    return []

# --- demo parser so pipeline returns something ---
def _demo_parse(url, content, content_type):
    return [{
        "type": "shelter",
        "name": "Demo High School Shelter",
        "address": "123 Demo Ave",
        "city": "Orlando",
        "state": "FL",
        "zip": "32801",
        "coordinates": {"lat": 28.5384, "lon": -81.3789},
        "hours": "Open 24h",
        "requirements": "ID recommended",
        "contact": "(407) 555-1234",
        "link": url,
        "publisher": "ocfl.net",
        "updated_at": None,
    }]

register("ocfl.net", "shelter", _demo_parse)
