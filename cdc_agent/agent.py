from cdc_agent.tools.cdc_scraper import fetch_cdc_guidance

def main():
    print("CDC Health Agent")
    question = input("Ask a CDC-related question: ")
    print("Fetching CDC.gov info...")
    cdc_info = fetch_cdc_guidance("coronavirus")  # Demo; later, parse topic from question
    print("\nCDC Guidance (preview):\n", cdc_info[:500], "\n...")

if __name__ == "__main__":
    main()
