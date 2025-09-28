import os
from dotenv import load_dotenv
from google.adk.agents import Agent, ParallelAgent
from newsapi import NewsApiClient
import time

load_dotenv()
API_KEY = os.getenv("NEWS_API")
newsapi = NewsApiClient(api_key=API_KEY)
location = "Miami, Florida"
QUERY = f"""Tropical Storms in {location}"""

def fetch_headlines():
    top_headlines = newsapi.get_top_headlines(q=QUERY, 
                                              sources='bbc-news,the-verge,the-weather-channel,cnn', 
                                              category='weather', 
                                              language='en', 
                                              country='us')
    

    return top_headlines

def fetch_news():
    all_articles = newsapi.get_everything(q=QUERY,
                                      sources='bbc-news,the-verge,the-weather-channel,cnn',
                                      domains='weather',
                                      from_param=time.localtime(),
                                      to=time.localtime(),
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)
    
    return all_articles

    

