import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

CONFIG = {
    "chunk_size": 500,
    "overlap": 100,
    "max_chunks": 100,
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
    "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
    "llm_model": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
    "cdc_urls": {
        "covid": "https://www.cdc.gov/coronavirus/2019-ncov/index.html",
        "flu": "https://www.cdc.gov/flu/index.htm",
        "monkeypox": "https://www.cdc.gov/mpox/index.html",
    },
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
}

