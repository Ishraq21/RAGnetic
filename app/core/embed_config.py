import logging
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import get_api_key

logger = logging.getLogger(__name__)


def get_embedding_model(model_name: str):
    """
    Factory function to get the embedding model instance based on the model name.

    This function centralizes the logic for initializing different embedding models,
    making it easy to add new providers in the future. It automatically handles
    API key retrieval for providers that require it.

    Args:
        model_name: The name of the model to initialize (e.g., "text-embedding-3-small",
                    "models/embedding-001", or a HuggingFace model name).

    Returns:
        An instance of the appropriate LangChain embedding model.
    """
    logger.info(f"Initializing embedding model: {model_name}")

    # HuggingFace models are identified by a 'sentence-transformers' prefix.
    # These models are run locally and do not require an API key.
    if "sentence-transformers" in model_name:
        logger.info(f"Using HuggingFace embedding model: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)

    # For other models (OpenAI, Google), an API key is required.
    # The get_api_key function determines the correct key from the model name.
    try:
        api_key = get_api_key(model_name)
        logger.info(f"API key found for model: {model_name}")
    except ValueError as e:
        logger.error(f"Could not initialize embedding model '{model_name}': {e}")
        raise  # Re-raise the error to stop the process if no key is found.

    # Select the provider based on common model name patterns.
    if "text-embedding" in model_name:
        logger.info(f"Using OpenAI embedding model: {model_name}")
        return OpenAIEmbeddings(model=model_name, openai_api_key=api_key)

    elif "models/embedding" in model_name:
        logger.info(f"Using Google embedding model: {model_name}")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    else:
        # Fallback for unrecognized but potentially valid models that need a key.
        # This defaults to OpenAI, which is a safe bet for many custom model names.
        logger.warning(f"Unknown embedding model provider for '{model_name}'. Defaulting to OpenAI.")
        return OpenAIEmbeddings(model=model_name, openai_api_key=api_key)

