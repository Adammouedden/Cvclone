# back_end/mcp/resources/adapters/parse/__init__.py
from typing import Dict, List, Callable, Tuple
from urllib.parse import urlparse

# --- registry ---
ParserFn = Callable[[str, str, str], List[Dict]]
_registry: dict[Tuple[str, str], ParserFn] = {}

def register(domain: str, rtype: str, fn: ParserFn):
    """
    Parsers call this on import to register themselves:
      register("ocfl.net", "shelter", parse)
    """
    _registry[(domain.lower(), (rtype or "").lower())] = fn

def parse_resources(url: str, content: str, content_type: str, desired_type: str) -> List[Dict]:
    """
    Route an (url, content) to the right per-domain parser for desired_type.
    """
    netloc = urlparse(url).netloc.lower()
    desired_type = (desired_type or "").lower()

    # exact domain hit
    fn = _registry.get((netloc, desired_type))
    if fn:
        return fn(url, content, content_type)

    # try base domain (e.g., news.ocfl.net -> ocfl.net)
    parts = netloc.split(".")
    for i in range(1, len(parts)-1):
        base = ".".join(parts[i:])
        fn = _registry.get((base, desired_type))
        if fn:
            return fn(url, content, content_type)

    # nothing matched
    return []

# --- auto-discover parser modules in this package ---
# Any file named shelters_*.py, sandbags_*.py, food_*.py will be imported,
# which should call register(...) at import time.
import pkgutil, importlib

for mod in pkgutil.iter_modules(__path__):
    if mod.name.startswith(("shelters_", "sandbags_", "food_")):
        importlib.import_module(f"{__name__}.{mod.name}")
