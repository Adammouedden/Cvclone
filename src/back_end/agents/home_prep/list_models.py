import os, google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
for m in genai.list_models():
    if "generateContent" in getattr(m, "supported_generation_methods", []):
        print(m.name)
