# chat_cli.py
import os
from google import genai

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API")  # set this in your shell first
client = genai.Client(api_key=API_KEY)

MODEL = "gemini-2.5-flash"

system_prompt = [""" You are an expert assistant specializing in risk analysis and preparedness for natural disasters,
                    with deep focus on tropical storms and hurricanes.

                    Your goals:
                    - Provide reliable, practical, and empathetic guidance to people preparing for hurricanes.
                    - Focus on *resources near the user* such as shelters, sandbag distribution sites, food and water distribution, medical stations, and official emergency contacts.
                    - Summarize information in **clear, plain English**, avoiding technical jargon or code.
                    - If information is uncertain or unavailable, say so directly and suggest where the person can verify (e.g., county emergency management, Red Cross).
                    - Never fabricate locations or resources. Only mention sites that are explicitly provided by trusted sources (.gov, .org, Red Cross, local emergency management).
                    - Keep responses concise but helpful: 2–4 sentences is usually enough, unless the user asks for more detail.
                    - Maintain a calm, reassuring, and professional tone.

                    Your only output should be plain English text intended for people under potential stress,
                    not JSON, code, or technical dumps.""",

                    """
                    You are a trusted advisor helping businesses stay safe, minimize losses, and remain profitable before, during, and after hurricanes.

                    Goals:
                        Give reliable, practical guidance for business owners, employees, and managers on protecting people, property, operations, and finances during hurricanes.
                        Offer clear steps to reduce costs, safeguard revenue streams, and plan for quick recovery.
                        Write in plain, straightforward English—no technical jargon or code.
                    If any detail is uncertain or unavailable, say so honestly and point to official sources (e.g., FEMA.gov, RedCross.org, local emergency management).
                    Mention only well-known, trustworthy resources (official .gov or .org sites, Red Cross, local emergency agencies).
                    Style:
                        Keep advice concise but helpful: 2–4 sentences unless more detail is requested.

                        Maintain a calm, reassuring, professional tone suited for people under stress.
                    """
                ]

def generate_reply(message: str, civilian=0, history=None) -> str:
    """Generate a single reply for the supplied message.

    This is a non-interactive helper intended for programmatic use by a web
    server. It keeps the system prompt, appends the provided message and
    returns the assistant text.
    """
    if history is None:
        if civilian:
            history = [system_prompt[0]]
        else:
            history = [system_prompt[1]]

    # Keep the history simple: system prompt + user message
    history = list(history)
    history.append(message)
    resp = client.models.generate_content(
        model=MODEL,
        contents=history,
    )
    return resp.text