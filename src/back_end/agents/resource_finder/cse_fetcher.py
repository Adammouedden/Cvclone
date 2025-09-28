# back_end/agents/resource_finder/cse_fetcher.py
from __future__ import annotations
import os, time, urllib.parse, urllib.request, json
from typing import List, Dict, Any

CSE_KEY = os.getenv("GOOGLE_CSE_KEY")
CSE_CX  = os.getenv("GOOGLE_CSE_CX")
CSE_NUM = int(os.getenv("GOOGLE_CSE_NUM", "8"))   # results per page (<=10)
CSE_PAGES = int(os.getenv("GOOGLE_CSE_PAGES", "2"))  # paginate pages
CSE_DATE_RESTRICT = os.getenv("GOOGLE_CSE_DATE_RESTRICT", "d30")  # d7/d30, etc.

def _cse_request(params: Dict[str, str]) -> Dict[str, Any]:
    base = "https://www.googleapis.com/customsearch/v1?"
    qs = urllib.parse.urlencode(params)
    url = base + qs
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _mk_queries(county: Dict[str, str], rtype: str) -> List[str]:
    name, state = county["name"], county["state"]
    Q = []
    if rtype == "sandbags":
        Q = [
            f'{name} {state} sandbag locations site:.gov',
            f'{name} {state} emergency management sandbags',
            f'{name} {state} storm sandbag distribution site:.gov OR site:.org'
        ]
    elif rtype == "food":
        Q = [
            f'{name} {state} food pantry site:.gov',
            f'{name} {state} food distribution site:.gov OR site:feedingamerica.org OR site:foodpantries.org'
        ]
    else:  # shelters
        Q = [
            f'{name} {state} emergency shelter site:.gov',
            f'{name} {state} evacuation shelter site:.gov OR site:redcross.org'
        ]
    return Q

def cse_discover_urls(county: Dict[str, str], rtype: str) -> List[str]:
    urls: List[str] = []
    for q in _mk_queries(county, rtype):
        start = 1
        for _ in range(CSE_PAGES):
            params = {
                "key": CSE_KEY, "cx": CSE_CX,
                "q": q, "num": str(CSE_NUM),
                "dateRestrict": CSE_DATE_RESTRICT,
                "safe": "off", "start": str(start)
            }
            data = _cse_request(params)
            for item in data.get("items", []):
                link = item.get("link")
                if link:
                    urls.append(link)
            # next page
            start += CSE_NUM
            # polite sleep to avoid quota bursts
            time.sleep(0.25)
    # simple de-dupe
    return list(dict.fromkeys(urls))
