# base_agent.py
# ---------------------------------------
# A simple Agent wrapper around Google Gemini (genai) that:
# - Initializes with one of two system prompts (civilian vs enterprise)
# - Maintains lightweight turn history
# - Supports optional `images` (list of URLs/paths) by appending them to the text prompt
#   so text-only flows still work today
# - Can be upgraded later to true multimodal parts if desired

from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

try:
    from google import genai
except Exception as e:
    raise RuntimeError(
        "google-genai SDK not found. Install it with `pip install google-genai`."
    ) from e

# If you decide to use structured parts later, you'll also need:
# from google.genai import types


# ----------------------------
# Config
# ----------------------------
API_KEY: Optional[str] = os.getenv("GEMINI_API")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not API_KEY:
    # Fail fast with a clear error to avoid silent 401s later.
    raise EnvironmentError("GEMINI_API is not set. Export GEMINI_API in your environment.")


# ----------------------------
# System Prompts
# ----------------------------
CIVILIAN_SYSTEM_PROMPT = (
    "You are an expert assistant specializing in risk analysis and preparedness for natural disasters, "
    "with deep focus on tropical storms and hurricanes.\n\n"
    "Goals:\n"
    "- Provide reliable, practical, and empathetic guidance to people preparing for hurricanes.\n"
    "- Focus on resources near the user such as shelters, sandbag distribution sites, food and water distribution, "
    "medical stations, and official emergency contacts.\n"
    "- Summarize information in clear, plain English, avoiding technical jargon or code.\n"
    "- If information is uncertain or unavailable, say so directly and suggest where the person can verify "
    "(e.g., county emergency management, Red Cross).\n"
    "- Never fabricate locations or resources. Only mention sites explicitly provided by trusted sources "
    "(.gov, .org, Red Cross, local emergency management).\n"
    "- Keep responses concise but helpful: 2–4 sentences unless the user asks for more detail.\n"
    "- Maintain a calm, reassuring, and professional tone.\n\n"
    "Output: plain English text only (no JSON or code)."
)

ENTERPRISE_SYSTEM_PROMPT = (
    "You are a trusted advisor helping businesses stay safe, minimize losses, and remain profitable before, "
    "during, and after hurricanes.\n\n"
    "Goals:\n"
    "- Give reliable, practical guidance for business owners, employees, and managers on protecting people, "
    "property, operations, and finances during hurricanes.\n"
    "- Offer clear steps to reduce costs, safeguard revenue streams, and plan for quick recovery.\n"
    "- Write in plain, straightforward English—no technical jargon or code.\n"
    "- If any detail is uncertain or unavailable, say so honestly and point to official sources (FEMA.gov, "
    "RedCross.org, local emergency management).\n"
    "- Mention only well-known, trustworthy resources (.gov, .org, Red Cross, local emergency agencies).\n\n"
    "Style:\n"
    "- Keep advice concise but helpful: 2–4 sentences unless more detail is requested.\n"
    "- Maintain a calm, reassuring, professional tone suited for people under stress.\n\n"
    "Output: plain English text only (no JSON or code)."
)


# ----------------------------
# Agent
# ----------------------------
class Agent:
    """
    A minimal stateful agent:
      - Initializes with a scenario-specific system prompt (civilian/enterprise).
      - Keeps a small in-memory history (plain text turns).
      - generate_reply(message, images=None) returns model text.
    """

    def __init__(
        self,
        civilian: int = 0,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = API_KEY,
        max_history_turns: int = 10,  # keep the last N user+assistant lines (excl. system)
    ):
        if not api_key:
            raise EnvironmentError("Missing API key for Gemini. Set GEMINI_API.")

        self.client = genai.Client(api_key=api_key)
        self.model = model

        # Persist the chosen system prompt; we prepend it each call for clarity/reliability.
        self.system_prompt = CIVILIAN_SYSTEM_PROMPT if civilian else ENTERPRISE_SYSTEM_PROMPT

        # Keep a lightweight flat history of text turns (not roles/parts).
        # We'll join this for the request. This is simple and robust with your current server.
        self.history: List[str] = []

        # How many turns to keep in memory (keeps prompts small)
        self.max_history_turns = max(0, int(max_history_turns))

    # ----------------------------
    # Public API
    # ----------------------------
    def generate_reply(self, message: str, images: Optional[List[str]] = None) -> str:
        """
        Generate a reply to the user's message.
        - `images`: optional list of image URLs/paths. For now, we reference them in-text so this
          works even if you haven't enabled true multimodal parts yet.

        Returns: model text string
        """

        # Build the user prompt, embedding image URLs (if any).
        user_prompt = message.strip() if message else ""
        if images:
            # Keep it simple & reliable: reference URLs in-text
            # (This works with text-only request bodies and your existing server routes.)
            urls_block = "\n\nAttached images:\n" + "\n".join(images)
            user_prompt = (user_prompt or "See attached images.") + urls_block

        # Add the new user turn to local memory
        if user_prompt:
            self._push_turn(f"USER: {user_prompt}")

        # Build the final input text: system + recent history joined.
        # Keeping it as raw text is compatible with your current usage and SDK call.
        contents_text = self._compose_contents_text()

        # --- Call the model (text-only request body, robust across SDK versions) ---
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents_text,
        )

        # Read model text & store in history
        reply_text = getattr(resp, "text", "") or ""
        if reply_text:
            self._push_turn(f"ASSISTANT: {reply_text}")

        return reply_text

    # ----------------------------
    # Optional upgrades (kept simple for now)
    # ----------------------------
    # If/when you want *true multimodal* (sending actual image data/parts rather than URLs),
    # replace the call above with a structured `types.Content` request using text + image parts.
    # Keep a fallback to the current text path to avoid breaking if the SDK behavior changes.

    # def _generate_with_parts(self, message: str, image_refs: List[str]) -> str:
    #     from google.genai import types
    #     parts: list[types.Part] = [types.Part(text=message or "")]
    #     # For URLs, some SDK versions support FileData(file_uri=...).
    #     # For local paths, read bytes -> Blob(inline_data=...).
    #     # After building parts, call:
    #     # resp = self.client.models.generate_content(model=self.model, contents=[types.Content(role="user", parts=parts)])
    #     # return resp.text
    #     pass

    # ----------------------------
    # Internals
    # ----------------------------
    def _push_turn(self, text: str) -> None:
        """Append a single text turn and clip to the last N turns."""
        self.history.append(text)
        if self.max_history_turns > 0 and len(self.history) > self.max_history_turns:
            # Keep the most recent N turns
            self.history = self.history[-self.max_history_turns :]

    def _compose_contents_text(self) -> str:
        """
        Compose the final text we send to the model:
          - Prepend the system prompt on every request (keeps behavior consistent).
          - Then join the recent history (USER/ASSISTANT lines).
        """
        if self.history:
            return f"{self.system_prompt}\n\n" + "\n".join(self.history)
        else:
            # If user calls without message (rare), still send system prompt.
            return self.system_prompt

    # ----------------------------
    # Maintenance helpers
    # ----------------------------
    def reset(self) -> None:
        """Clear conversation history (keeps the same system prompt)."""
        self.history.clear()

    def set_model(self, model: str) -> None:
        """Swap models at runtime if desired."""
        self.model = model or self.model
