from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os, time, math

# --- Debug: print env vars when run directly ---
if os.getenv("DEBUG_ENV") == "1":
    print("Loaded env values:")
    print("  MCP_BASE_NWS        =", os.getenv("MCP_BASE_NWS"))
    print("  MCP_BASE_RESOURCES  =", os.getenv("MCP_BASE_RESOURCES"))
    print("  RESOURCE_CACHE_TTL  =", os.getenv("RESOURCE_CACHE_TTL"))
    print("  RESOURCE_RADIUS_KM  =", os.getenv("RESOURCE_RADIUS_KM"))
    print("  RESOURCES_UA        =", os.getenv("RESOURCES_UA"))
    print("  NWS_UA              =", os.getenv("NWS_UA"))
    print("---")
