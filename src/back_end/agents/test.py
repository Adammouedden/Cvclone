import os
import asyncio
from google.adk.agents import Agent, LlmAgent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts
from dotenv import load_dotenv
from news import news_tool

load_dotenv()
key = os.getenv("GEMINI_KEY")

test_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country.",
    tools=[news_tool]
)