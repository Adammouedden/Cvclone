from typing import List, Dict
from bs4 import BeautifulSoup
from back_end.mcp.resources.adapters.parse import register

DOMAIN = "ocfl.net"
RTYPE  = "shelter"

def _txt(el): 
    return (el.get_text(" ", strip=True) if el else "").strip() or None

def parse(url: str, content: str, content_type: str) -> List[Dict]:
    if "html" not in (content_type or "").lower():
        return []

    soup = BeautifulSoup(content, "html.parser")

    # Try a known table first (adjust selectors if needed)
    table = soup.select_one("table.shelter-list, table#shelters, table:has(th:matches(Shelter|Name))")
    items: List[Dict] = []

    if table:
        for tr in table.select("tbody tr"):
            tds = tr.find_all(["td", "th"])
            if not tds: 
                continue
            name   = _txt(tds[0]) if len(tds) > 0 else None
            addr   = _txt(tds[1]) if len(tds) > 1 else None
            city   = _txt(tds[2]) if len(tds) > 2 else None
            hours  = _txt(tds[3]) if len(tds) > 3 else None
            items.append({
                "type": RTYPE,
                "name": name,
                "address": addr,
                "city": city,
                "state": "FL",
                "zip": None,
                "coordinates": None,     # Agent can geocode later
                "hours": hours,
                "requirements": None,
                "contact": None,
                "link": url,
                "publisher": DOMAIN,
                "updated_at": None,
            })
        return items

    # Fallback: list items with addresses (very conservative)
    for li in soup.select("ul li, ol li"):
        txt = _txt(li)
        if not txt:
            continue
        # crude address hint
        if any(k in txt.lower() for k in ["ave", "st ", "road", "blvd", "dr ", "hwy", "fl "]):
            items.append({
                "type": RTYPE,
                "name": txt.split(",")[0][:120],
                "address": txt,
                "city": None, "state": "FL", "zip": None,
                "coordinates": None,
                "hours": None, "requirements": None, "contact": None,
                "link": url, "publisher": DOMAIN, "updated_at": None,
            })
    return items

register(DOMAIN, RTYPE, parse)
