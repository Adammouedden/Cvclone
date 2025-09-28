import os
from dotenv import load_dotenv
from newsapi import NewsApiClient
import time

load_dotenv()
API_KEY = os.getenv("NEWS_API")
newsapi = NewsApiClient(api_key=API_KEY)

location = "Miami, Florida"
QUERY = f"Tropical storm OR hurricane {location}"

DOMAINS = (
        "weather.com,cnn.com,reuters.com,apnews.com,abcnews.go.com,nbcnews.com,"
        "cbsnews.com,usatoday.com,miamiherald.com,sun-sentinel.com,tampabay.com,"
        "orlandosentinel.com,noaa.gov"
    )

def fetch_news():
    
    all_articles = newsapi.get_everything(
        q=QUERY,
        domains=DOMAINS,
        from_param="2025-09-01",
        to=time.time(),
        language="en",
        sort_by="relevancy",
        page=1,
        page_size=50
    )
    return all_articles

if __name__ == "__main__":
    news = fetch_news()
    print(news)
