import requests
from bs4 import BeautifulSoup

def fetch_cdc_guidance(topic="covid"):
    url = f"https://www.cdc.gov/{topic}/"
    response = requests.get(url)
    if not response.ok:
        return "CDC page not found."
    soup = BeautifulSoup(response.text, "html.parser")
    # Very basic extraction: get visible text only
    text = soup.get_text(separator=' ', strip=True)
    return text[:2000]  # Limit for demo; adjust as needed
