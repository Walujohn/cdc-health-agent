import os

def get_llm_provider(model=None):
    """
    Returns an LLM object from OpenAI, Vertex AI, or Anthropic based on env variables.

    Set LLM_PROVIDER to "openai", "vertex", or "anthropic" to control which one is used.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()  # Always lower for consistency

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=model or "gpt-3.5-turbo")

    elif provider == "vertex":
        # This block is a stub -- adjust for actual Vertex AI integration!
        from vertexai.language_models import TextGenerationModel
        # For Google Vertex AI, authentication will need extra setup!
        return TextGenerationModel.from_pretrained(model or "text-bison@001")
    
    elif provider == "anthropic":
        from langchain_anthropic import Anthropic
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), model=model or "claude-3-haiku-20240307")
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
