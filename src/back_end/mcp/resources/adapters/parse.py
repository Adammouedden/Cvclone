from __future__ import annotations
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

KEYWORDS = {
    "sandbags":  [r"sand\s*bag", r"sandbag", r"bag\s*distribution"],
    "shelter":   [r"shelter", r"evacuat(?:e|ion)", r"evac centers?"],
    "food":      [r"food\s+pantr(y|ies)", r"food\s+distribution", r"meals?", r"drive[-\s]?thru\s+food"],
}

def _match_keywords(text: str, rtype: str) -> bool:
    if not text:
        return False
    text = text.lower()
    for pat in KEYWORDS.get(rtype, []):
        if re.search(pat, text, re.I):
            return True
    return False

def _extract_items_from_lists_tables(soup: BeautifulSoup, rtype: str) -> list[dict]:
    items = []

    # 1) Look for headings that mention our keywords, then capture following ul/ol/table
    headings = soup.find_all(re.compile(r"^h[1-4]$"))
    for h in headings:
        if _match_keywords(h.get_text(" ", strip=True), rtype):
            # first list after this heading
            nxt = h.find_next_sibling(lambda tag: tag.name in ("ul","ol","table"))
            if not nxt:
                continue
            if nxt.name in ("ul","ol"):
                for li in nxt.find_all("li", recursive=False):
                    txt = li.get_text(" ", strip=True)
                    if txt:
                        items.append({"name": txt})
            elif nxt.name == "table":
                for tr in nxt.find_all("tr"):
                    cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td","th"])]
                    line = " | ".join([c for c in cells if c])
                    if line and not line.lower().startswith(("name", "location", "site", "address")):
                        items.append({"name": line})
    # 2) Fallback: any lists anywhere with obvious addresses
    if not items:
        for ul in soup.find_all(["ul","ol"]):
            for li in ul.find_all("li", recursive=False):
                txt = li.get_text(" ", strip=True)
                if _match_keywords(txt, rtype):
                    items.append({"name": txt})

    # normalize: ensure minimal fields
    for it in items:
        it.setdefault("address", None)
        it.setdefault("notes", None)
    return items

def _find_next_urls(soup: BeautifulSoup, base_url: str, rtype: str, same_host_only: bool = True) -> list[str]:
    base_host = urlparse(base_url).netloc
    out = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = a["href"].strip()
        if not _match_keywords(text, rtype) and not _match_keywords(href, rtype):
            continue
        absu = urljoin(base_url, href)
        if same_host_only and urlparse(absu).netloc != base_host:
            continue
        out.append(absu)
    # de-dupe, preserve order
    seen, deduped = set(), []
    for u in out:
        if u not in seen:
            seen.add(u); deduped.append(u)
    return deduped[:15]  # bound it
        

def parse_resources(url: str, content: str, content_type: str, rtype: str) -> dict:
    # Only attempt HTML here
    if not content_type.startswith("text/html"):
        return {"items": [], "next_urls": []}

    soup = BeautifulSoup(content, "html.parser")

    # Try to extract concrete items
    items = _extract_items_from_lists_tables(soup, rtype)

    # If nothing concrete, propose next_urls to follow (portal pages)
    next_urls = []
    if not items:
        next_urls = _find_next_urls(soup, url, rtype, same_host_only=True)

    return {"items": items, "next_urls": next_urls}
