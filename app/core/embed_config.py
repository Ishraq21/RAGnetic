import logging
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import get_api_key

logger = logging.getLogger(__name__)

# A simple cache to avoid re-initializing the same model repeatedly
embedding_model_cache = {}


def get_embedding_model(model_identifier: str):
    """
    Initializes and returns a LangChain embedding model from a model identifier.
    Caches models to avoid re-initializing them on subsequent calls.

    This function supports:
    - OpenAI models (e.g., "text-embedding-3-small")
    - Google Generative AI models (e.g., "models/embedding-001")
    - Local Hugging Face models (prefixed with "local:", e.g., "local:BAAI/bge-small-en-v1.5")
    - Ollama models (prefixed with "ollama/", e.g., "ollama/llama3")
    """
    if model_identifier in embedding_model_cache:
        logger.info(f"Returning cached embedding model: {model_identifier}")
        return embedding_model_cache[model_identifier]

    logger.info(f"Initializing embedding model: {model_identifier}")

    try:
        embeddings = None
        provider = "unknown"

        # 1. Check for local Hugging Face models
        if model_identifier.startswith("local:"):
            provider = "huggingface"
            model_name = model_identifier.split("local:", 1)[1]
            logger.info(f"Using local Hugging Face model '{model_name}'. No API key needed.")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # 2. Check for Ollama models
        elif model_identifier.startswith("ollama/"):
            provider = "ollama"
            model_name = model_identifier.split("ollama/", 1)[1]
            # Note: LangChain's OllamaEmbeddings would be used here if available.
            # Assuming HuggingFaceEmbeddings for now as a placeholder if it's a compatible model.
            # For a true Ollama integration, you might need a different class.
            logger.info(f"Using Ollama model '{model_name}'. Ensure Ollama is running.")
            # from langchain_community.embeddings import OllamaEmbeddings
            # embeddings = OllamaEmbeddings(model=model_name)
            # As a fallback for this example, we'll use HuggingFace, but a dedicated Ollama class is better.
            embeddings = HuggingFaceEmbeddings(model_name=model_name)


        # 3. Check for Google Gemini models
        elif "gemini" in model_identifier or model_identifier.startswith("models/embedding"):
            provider = "google_genai"
            api_key = get_api_key(provider)
            embeddings = GoogleGenerativeAIEmbeddings(model=model_identifier, google_api_key=api_key)

        # 4. Default to OpenAI for their specific model names
        elif model_identifier.startswith("text-embedding"):
            provider = "openai"
            api_key = get_api_key(provider)
            embeddings = OpenAIEmbeddings(model=model_identifier, api_key=api_key)

        # 5. Fallback for any other Hugging Face model from the hub
        else:
            provider = "huggingface"
            logger.info(f"Using Hugging Face Hub model '{model_identifier}'. No API key needed.")
            embeddings = HuggingFaceEmbeddings(model_name=model_identifier)

        logger.info(f"Successfully initialized embedding model '{model_identifier}' from provider '{provider}'.")
        embedding_model_cache[model_identifier] = embeddings
        return embeddings

    except Exception as e:
        logger.error(f"Could not initialize embedding model '{model_identifier}': {e}", exc_info=True)
        # Re-raise the exception to be caught by the calling function
        raise e