# run_adk.py
import os
import asyncio
from typing import Optional

from dotenv import load_dotenv

# ADK core pieces
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Multi-model client for LLMs (Gemini here)
from google.adk.models.lite_llm import LiteLlm

# Types to inspect event contents
from google.genai import types as gt  # Content / Part / function_call, etc.

# === Your tool(s) ===
# Assumes you already have: from news import news_tool
# If not, comment this out or replace with an empty list
try:
    from news import news_tool
    TOOLS = [news_tool]
except Exception:
    TOOLS = []  # safe fallback if you haven't defined the tool yet

# ----------------------------
# Setup
# ----------------------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_KEY")  # put your API key in .env as GEMINI_KEY

# Model client (Gemini via LiteLlm)
# provider can be "google" (Gemini), "openai", "anthropic", etc., depending on your setup.
model_client = LiteLlm(provider="google", api_key=GEMINI_KEY)

# Define your agent (root agent)
root_agent = LlmAgent(
    model="gemini-2.0-flash",
    model_client=model_client,
    name="capital_agent",
    description="Answers user questions about the capital city of a given country.",
    instruction="Respond as a helpful chatbot.",
    tools=TOOLS,  # [] is fine if you don't have one yet
)

# Runner + Session storage
runner = Runner(session_service=InMemorySessionService())

# ----------------------------
# Helpers to pretty-print events
# ----------------------------
def _extract_text(event) -> Optional[str]:
    """Safely extract any text parts from an event."""
    if not getattr(event, "content", None):
        return None
    parts = getattr(event.content, "parts", None)
    if not parts:
        return None
    texts = []
    for p in parts:
        # p could be text, function_call, function_response, tool_result, etc.
        if getattr(p, "text", None):
            texts.append(p.text)
    return "\n".join(texts) if texts else None

def _maybe_print_function_call(event) -> None:
    """Show tool/function calls emitted by the LLM."""
    if not getattr(event, "content", None):
        return
    for p in event.content.parts or []:
        fn_call = getattr(p, "function_call", None)
        if fn_call:
            print(f"\n[tool-call] {fn_call.name}({fn_call.args})")

def _maybe_print_function_response(event) -> None:
    """Show tool/function results returned to the agent."""
    if not getattr(event, "content", None):
        return
    for p in event.content.parts or []:
        fn_resp = getattr(p, "function_response", None)
        if fn_resp:
            print(f"[tool-result] {fn_resp.name} -> {fn_resp.response}")

# ----------------------------
# One-off invocation (streaming)
# ----------------------------
async def ask_once(prompt: str) -> str:
    """
    Sends a single user message into the ADK runtime,
    streams events, and returns the final assembled text.
    """
    final_text_chunks = []

    # The Runner drives the event loop. We stream every event it yields.
    async for event in runner.run_async(root_agent, new_message=prompt):
        # Print function/tool activity if any
        _maybe_print_function_call(event)
        _maybe_print_function_response(event)

        # Print streamed text (partial or final)
        txt = _extract_text(event)
        if txt:
            # Many LLM flows emit partial chunks before a final message.
            # You can print live and also accumulate them if you want a final string.
            print(txt, end="", flush=True)
            final_text_chunks.append(txt)

        # You could also inspect:
        # - event.partial (bool) for streaming vs final
        # - event.actions.state_delta / artifact_delta
        # - event.author / event.role
        # - event.turn_complete (in some flows) to detect final message

    # Join everything we saw; depending on your model flow,
    # you may prefer to only keep the last non-partial chunk.
    return "".join(final_text_chunks).strip()

# ----------------------------
# CLI entry
# ----------------------------
def main():
    prompt = "What is the capital of Morocco? Also summarize one recent news headline about it."
    print(f"\n[USER] {prompt}\n")
    result = asyncio.run(ask_once(prompt))
    print("\n\n[FINAL TEXT]\n" + result)

if __name__ == "__main__":
    if not GEMINI_KEY:
        raise RuntimeError("Missing GEMINI_KEY in environment. Put it in a .env file or export it.")
    main()
