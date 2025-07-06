import logging
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import get_api_key

logger = logging.getLogger(__name__)

# A simple cache to avoid re-initializing the same model repeatedly
embedding_model_cache = {}


def get_embedding_model(model_name: str):
    """
    Initializes and returns an embedding model by name.
    Caches the model to avoid re-initialization.
    """
    if model_name in embedding_model_cache:
        return embedding_model_cache[model_name]

    logger.info(f"Initializing embedding model: {model_name}")
    try:
        if model_name.startswith("text-embedding"):
            provider = "openai"
            api_key = get_api_key(provider)
            embeddings = OpenAIEmbeddings(model=model_name, api_key=api_key)

        elif "gemini" in model_name or model_name.startswith("models/embedding"):
            provider = "google"
            api_key = get_api_key(provider)
            embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

        else:
            # Assume it's a local Hugging Face model
            provider = "huggingface"
            logger.info(f"Using local Hugging Face model '{model_name}'. No API key needed.")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)

        logger.info(f"Successfully initialized embedding model '{model_name}' from provider '{provider}'.")
        embedding_model_cache[model_name] = embeddings
        return embeddings

    except Exception as e:
        logger.error(f"Could not initialize embedding model '{model_name}': {e}", exc_info=True)
        # Re-raise the exception to be caught by the calling function
        raise e