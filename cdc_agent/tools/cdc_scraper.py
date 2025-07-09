import requests
from bs4 import BeautifulSoup

def fetch_cdc_guidance(topic="coronavirus"):
    # Basic topic-to-path mapping for demo
    topic_paths = {
        "covid": "coronavirus/2019-ncov/index.html",
        "flu": "flu/index.html",
        "monkeypox": "mpox/"
    }
    path = topic_paths.get(topic.lower(), "coronavirus/2019-ncov/index.html")
    url = f"https://www.cdc.gov/{path}"
    response = requests.get(url)
    if not response.ok:
        return "CDC page not found."
    soup = BeautifulSoup(response.text, "html.parser")
    # Extract all visible text, limit length for demo
    text = soup.get_text(separator=' ', strip=True)
    return text[:2000]

