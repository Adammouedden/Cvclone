#Imports
import os
from google import genai
from google.genai import types
#Agent tools

#from .forecast.agent import get_snapshot
#from .store_routing.agent import plan_routes

from dotenv import load_dotenv
load_dotenv()

sample_latitude = 25.758925
sample_longitude = -80.376627
#forecast = get_snapshot(sample_latitude, sample_longitude)

sample_user_location = {"lat": 25.772, "lon": -80.196}  # Miami, FL 
items = [
  {"name": "Water (liters)", "qty": 24},
  {"name": "Shelf-stable food (calories)", "qty": 6000},
  {"name": "Batteries AA/AAA (count)", "qty": 16},
  {"name": "Tarps / duct tape", "qty": 2},
  {"name": "Plywood panels", "qty": 8},
  {"name": "Sandbags", "qty": 20},
  {"name": "Prescription meds (days)", "qty": 7},
  {"name": "Ice / cooler", "qty": 1},
  {"name": "Infant formula (servings)", "qty": 20},
  {"name": "Diapers (count)", "qty": 40},
  {"name": "Wipes (packs)", "qty": 3}
]
max_radius_km = 12.0
closures = [
  "Home Depot Midtown reports limited access",
  "Lowe's experiencing partial power outage"
]

#planned_routes = plan_routes(sample_user_location, items, max_radius_km, closures)



API_KEY = os.getenv("GEMINI_API")  # set this in your shell first
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
                    not JSON, code, or technical dumps.

                    If you ever recieve an image as input, you must inform users how they can specifically protect their house. 
                    Identify structural weaknesses, suggest reinforcements, and recommend safety measures based on the image provided. 
                    But respond to the image in 2-4 sentences only, in HTML without any other text.

                    """,

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
                    If you ever recieve an image as input, you must inform users how they can specifically protect their house. 
                    Identify structural weaknesses, suggest reinforcements, and recommend safety measures based on the image provided. 
                    But respond to the image in 2-4 sentences only, in MARKDOWN without any other text.
                 
                    
                      """
                ]

class Agent():
    def __init__(self, civilian=0):
        self.client = genai.Client(api_key=API_KEY)
        self.history = []

        if civilian:
            self.history.append(system_prompt[0])
        else:
            self.history.append(system_prompt[1])

        #self.history.append(f"The current weather forecast for your area is: {forecast}.")
        #self.history.append(f"Here are some nearby stores and routes that can help you get supplies: {planned_routes}.")        

    def generate_reply(self, message: str) -> str:
        """Generate a single reply for the supplied message.

        This is a non-interactive helper intended for programmatic use by a web
        server. It keeps the system prompt, appends the provided message and
        returns the assistant text.
        """

        # Keep the history simple: system prompt + user message
        self.history.append(message)
        resp = self.client.models.generate_content(
            model=MODEL,
            contents=str(self.history),
        )

        self.history.append(resp.text)
        return resp.text
    
    
    
    def respond_to_image(self, image_bytes):
        response = self.client.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                      "Tell the user how to protect their home from hurricanes based on the image."], 
        )
        return response.text
    
    def cache_augmented_generation(self, content: list):
        self.history.append(content)
        response = self.client.models.generate_content(
            model=MODEL,
            contents=str(content)
        )
        self.history.append(response.text)
        return response.text

        