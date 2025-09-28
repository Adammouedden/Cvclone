#!/usr/bin/env python3
"""
Home Prep Advisor — single-file MVP

Inputs:
  - local image path
  - wind_prob (0..1 float)
  - surge_zone (e.g., "AE-9", "VE", "X")

Output:
  - JSON to stdout (and optionally --out to a file) with actions[] etc.
"""

import os, sys, json, argparse, time
from typing import Any, Dict, List

# 1) pip install google-generativeai
import google.generativeai as genai


# ---------- JSON schema (strict, concise) ----------
# small for MVP; can add fields later without breaking anything
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "fetched_at": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "where", "why", "priority"],
                "properties": {
                    "id":        {"type": "string"},     # e.g., "board_windows", "sandbag_entry"
                    "where":     {"type": "string"},     # short location ref (e.g., "front windows (img 0)")
                    "why":       {"type": "string"},     # concise rationale
                    "materials": {"type": "array", "items": {"type": "string"}},
                    "effort":    {"type": "string"},     # e.g., "45m", "2–3h"
                    "priority":  {"type": "string"},     # "A" | "B" | "C"
                    "confidence":{"type": "number"}      # 0..1
                }
            }
        },
        "gaps": {
            "type": "array",
            "items": {"type": "string"}
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["fetched_at", "actions"]
}


SYSTEM_PROMPT = (
    "System: You are an expert home safety advisor specializing in hurricane preparedness."
    "Your task is to analyze user-provided images of a home's exterior along with weather risk data."
    "Your ONLY output must be a JSON object that conforms to the provided tool schema."
    "Do not add any conversational text or markdown. Keep recommendations short and actionable."
    "Prefer life/property-critical items as priority 'A'. "
    "For the 'id' field, use a short, snake_case identifier like 'board_windows' or 'secure_patio_furniture'."
    "For the 'effort' field, provide a time estimate like '30m' or '1-2h'."
    "In the 'where' field, reference which image shows the issue, for example: 'front windows (image 0)' or 'back patio (image 1)'."
)

USER_INSTRUCTIONS = (
    "Given the image and the risk context, recommend actions to protect the home. "
    "Use short phrases and keep materials lists practical for a weekend prep. "
    "If information is missing, add a short entry in 'gaps'."
)


def load_image_for_gemini(path: str) -> Dict[str, Any]:
    # The SDK accepts raw bytes dictionaries for local files.
    # Infer mime type from extension (simple).
    ext = os.path.splitext(path.lower())[1]
    mt = "image/jpeg"
    if ext in (".png",):
        mt = "image/png"
    elif ext in (".webp",):
        mt = "image/webp"
    with open(path, "rb") as f:
        data = f.read()
    return {"mime_type": mt, "data": data}


def main():
    ap = argparse.ArgumentParser(description="Home Prep Advisor — single-file MVP")
    ap.add_argument("--image", required=True, help="Path to local image (jpg/png/webp)")
    ap.add_argument("--wind-prob", type=float, required=True, help="0..1 probability of TS-force winds")
    ap.add_argument("--surge-zone", required=True, help='e.g., "AE-9", "VE", "X"')
    ap.add_argument("--model", default="models/gemini-2.5-flash", help="Gemini model name")
    ap.add_argument("--out", default="", help="Optional path to save JSON result")
    args = ap.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY in your environment.", file=sys.stderr)
        sys.exit(2)

    if not (0.0 <= args.wind_prob <= 1.0):
        print("ERROR: --wind-prob must be between 0 and 1.", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        sys.exit(2)

    # Configure SDK
    genai.configure(api_key=api_key)

    # Model with strict JSON output
    model = genai.GenerativeModel(
        model_name=args.model,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "response_schema": SCHEMA,   # enforce the structure
        },
        system_instruction=SYSTEM_PROMPT,
    )

    # Build the prompt parts
    img_part = load_image_for_gemini(args.image)
    context = {
        "wind_prob": round(float(args.wind_prob), 2),
        "surge_zone": args.surge_zone,
    }

    # Call the model
    try:
        resp = model.generate_content(
            [
                {"text": USER_INSTRUCTIONS},
                img_part,
                {"text": f"Context:\n{json.dumps(context, ensure_ascii=False)}"}
            ]
        )
    except Exception as e:
        print(f"ERROR: Gemini call failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse JSON (the SDK returns text even with JSON MIME)
    try:
        # Some SDK versions expose resp.text; others use candidates[0].content.parts
        text = getattr(resp, "text", None)
        if not text:
            # Fallback extraction
            if resp.candidates and resp.candidates[0].content.parts:
                text = "".join([getattr(p, "text", "") for p in resp.candidates[0].content.parts])
        data = json.loads(text)
    except Exception as e:
        # Dump raw for debugging
        print("ERROR: Failed to parse JSON response.", file=sys.stderr)
        print("----- RAW RESPONSE -----", file=sys.stderr)
        print(getattr(resp, "text", str(resp)), file=sys.stderr)
        sys.exit(1)

    # Stamp fetched_at if model forgot
    if "fetched_at" not in data:
        data["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Minimal validation
    if "actions" not in data or not isinstance(data["actions"], list):
        print("ERROR: Model did not return 'actions' list.", file=sys.stderr)
        print(json.dumps(data, indent=2))
        sys.exit(1)

    # Output
    out_json = json.dumps(data, ensure_ascii=False)
    print(out_json)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_json)


if __name__ == "__main__":
    main()
