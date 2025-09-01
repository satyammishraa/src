import requests
from bs4 import BeautifulSoup
import random
import time
import pandas as pd

# Example proxy pool
PROXIES = [
    "http://proxy1:port",
    "http://proxy2:port",
    "http://proxy3:port"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def get_html(url):
    proxy = {"http": random.choice(PROXIES)}
    try:
        response = requests.get(url, headers=HEADERS, proxies=proxy, timeout=5)
        if response.status_code == 200:
            return response.text
    except:
        return None

def scrape_reviews(base_url, pages=3):
    reviews = []
    for page in range(1, pages+1):
        url = f"{base_url}?page={page}"
        html = get_html(url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for review in soup.find_all("p", class_="review-text"):
                reviews.append(review.get_text(strip=True))
        time.sleep(random.uniform(1,3))  # polite crawling
    return pd.DataFrame({"review": reviews})
