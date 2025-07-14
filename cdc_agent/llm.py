import os

def get_llm_provider(model=None, provider=None):
    """
    Returns an LLM object from OpenAI, Vertex AI, or Anthropic based on config/env.
    Usage:
        get_llm_provider()                   # Uses env/config
        get_llm_provider(model="gpt-4")      # Override model
        get_llm_provider(provider="vertex")  # Override provider
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()
    model = model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=model)

    elif provider == "vertex":
        # Example stub — update for Vertex in your env!
        from vertexai.language_models import TextGenerationModel
        return TextGenerationModel.from_pretrained(model)

    elif provider == "anthropic":
        from langchain_anthropic import Anthropic
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), model=model)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

