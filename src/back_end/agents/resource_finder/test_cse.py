# test_cse.py
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("GOOGLE_CSE_KEY")
CSE_CX = os.getenv("GOOGLE_CSE_CX")

def test_cse_search(query: str):
    if not API_KEY or not CSE_CX:
        print("ERROR: GOOGLE_CSE_KEY and/or GOOGLE_CSE_CX not found in environment.")
        print("Please ensure your .env file is in the same directory and is populated.")
        return

    print(f"--- Testing CSE with query: '{query}' ---")
    print(f"API Key found: {'Yes' if API_KEY else 'No'}")
    print(f"CSE CX ID found: {'Yes' if CSE_CX else 'No'}")

    try:
        # Build the service object
        service = build("customsearch", "v1", developerKey=API_KEY)

        # Execute the search
        result = service.cse().list(
            q=query,
            cx=CSE_CX,
            num=5 # ask for 5 results
        ).execute()

        # Check for items
        items = result.get("items", [])
        if not items:
            print("\n>>> RESULT: SUCCESS, but 0 URLs found.")
            print(">>> This means your keys are working, but your CSE configuration is not finding results for this query.")
            print(">>> Check your CSE setup in the control panel (see checklist).")
        else:
            print(f"\n>>> RESULT: SUCCESS! Found {len(items)} URLs.")
            for i, item in enumerate(items):
                print(f"  {i+1}. {item['link']}")

    except HttpError as e:
        print(f"\n>>> RESULT: FAILED! An API error occurred.")
        print(f">>> Status Code: {e.resp.status}")
        print(f">>> Error Details: {e.content.decode('utf-8')}")
        print("\n>>> This means there is a problem with your API key or Cloud project setup.")

if __name__ == "__main__":
    # A realistic query for your project
    test_query = "sandbag distribution miami-dade county emergency"
    test_cse_search(test_query)