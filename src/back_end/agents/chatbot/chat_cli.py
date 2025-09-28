# chat_cli.py
# pip install google-genai
import os
from google import genai

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API")  # set this in your shell first
client = genai.Client(api_key=API_KEY)

MODEL = "gemini-2.5-flash"

system_prompt = """ You are an expert assistant specializing in risk analysis and preparedness for natural disasters,
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
                    not JSON, code, or technical dumps."""


def chat(text: str)->str:
    history = []
    history.append(system_prompt)
    print("Chatbot ready. Type 'exit' to quit.\n")
    while True:
        history.append(text)
        resp = client.models.generate_content(
            model=MODEL,
            contents=history,
        )
        answer = resp.text
        print(f"Bot: {answer}\n")
        history.append(answer)


def generate_reply(message: str, history=None) -> str:
    """Generate a single reply for the supplied message.

    This is a non-interactive helper intended for programmatic use by a web
    server. It keeps the system prompt, appends the provided message and
    returns the assistant text.
    """
    if history is None:
        history = [system_prompt]
    # Keep the history simple: system prompt + user message
    history = list(history)
    history.append(message)
    resp = client.models.generate_content(
        model=MODEL,
        contents=history,
    )
    return resp.text

if __name__ == "__main__":
    chat()
