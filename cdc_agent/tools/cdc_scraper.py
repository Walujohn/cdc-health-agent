import requests
import requests_cache
from bs4 import BeautifulSoup

# --- Enable disk-based HTTP caching for performance ---
# This saves each CDC.gov response to a local file (http_cache.sqlite).
# If you request the same page again (even on a different run), it’s loaded instantly from disk.
requests_cache.install_cache("http_cache", expire_after=3600)  # 1 hour cache

def fetch_cdc_guidance(topic, max_depth=1):
    """
    # --- NOTE: Production API Stub Example ---

    In a production environment, I would use a structured CDC API if available, like this:

        import requests
        resp = requests.get(f"https://api.cdc.gov/guidance/{topic}")
        if resp.ok:
            return resp.json()["guidance"]

    For this proof-of-concept, I’m scraping with BeautifulSoup and using HTTP caching.
    """

    # Map topic to CDC URLs
    topic_urls = {
        "covid": "https://www.cdc.gov/coronavirus/2019-ncov/index.html",
        "flu": "https://www.cdc.gov/flu/index.htm",
        "monkeypox": "https://www.cdc.gov/mpox/index.html"
    }
    url = topic_urls.get(topic, "https://www.cdc.gov/")
    return fetch_links_and_content(url, max_depth=max_depth)

def fetch_links_and_content(start_url, max_depth=1):
    visited = set()
    queue = [(start_url, 0)]
    all_text = []
    session = requests.Session()  # Use a session for HTTP keep-alive

    while queue:
        url, depth = queue.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)
        try:
            resp = session.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            all_text.append(soup.get_text(separator=" ", strip=True))
            # Follow CDC.gov links if within depth
            if depth < max_depth:
                for a in soup.find_all("a", href=True):
                    link = a["href"]
                    if link.startswith("https://www.cdc.gov") and link not in visited:
                        queue.append((link, depth + 1))
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
    return "\n".join(all_text)


