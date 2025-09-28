# test_fetch.py
import requests

# Use one of the URLs that failed in your log
url_to_test = 'https://www.ready.gov/shelter' 

print(f"--- Testing basic internet connectivity with requests ---")
print(f"Attempting to fetch: {url_to_test}")

try:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    # Use a generous timeout
    response = requests.get(url_to_test, headers=headers, timeout=20, allow_redirects=True)
    
    # This will raise an error if the status is 4xx or 5xx
    response.raise_for_status() 
    
    print(f"\n>>> RESULT: SUCCESS!")
    print(f">>> Status Code: {response.status_code}")
    print(f">>> Content Length: {len(response.text)}")

except Exception as e:
    print(f"\n>>> RESULT: FAILED!")
    print(f">>> An error occurred: {e}")