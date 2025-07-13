import os
import configparser
import logging
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.schemas.agent import ModelParams

# --- Constants ---
RAGNETIC_DIR = ".ragnetic"
CONFIG_FILE = os.path.join(RAGNETIC_DIR, "config.ini")

# --- Mappings ---
# Maps service names to their environment variable and config.ini key names
SERVICE_KEY_MAPPING = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "pinecone": "PINECONE_API_KEY",
    "mongodb": "MONGODB_CONN_STRING",
}

# Maps model prefixes to their respective classes and required service keys
MODEL_PROVIDER_MAPPING = {
    "gpt-": (ChatOpenAI, "openai"),
    "claude-": (ChatAnthropic, "anthropic"),
    "gemini-": (ChatGoogleGenerativeAI, "google"),
    "ollama/": (ChatOllama, None),  # Local models don't need a key
}

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Cache ---
_llm_cache: Dict[str, Any] = {}


def get_api_key(service_name: str) -> Optional[str]:
    """
    Retrieves an API key, prioritizing environment variables over the config file.
    This is a more secure approach for production environments.

    Args:
        service_name: The service to get the key for (e.g., 'openai', 'pinecone').

    Returns:
        The API key string, or None if not found.
    """
    key_name = SERVICE_KEY_MAPPING.get(service_name.lower())
    if not key_name:
        raise ValueError(f"Service '{service_name}' is not a valid service.")

    # 1. Prioritize Environment Variable (Production-ready)
    api_key = os.environ.get(key_name)
    if api_key:
        logger.info(f"Loaded API key for '{service_name}' from environment variable.")
        return api_key

    # 2. Fallback to config.ini (for local development)
    if os.path.exists(CONFIG_FILE):
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        api_key = config.get('API_KEYS', key_name, fallback=None)
        if api_key and api_key != "...":
            logger.info(f"Loaded API key for '{service_name}' from config file (fallback).")
            return api_key

    logger.warning(f"API key for '{service_name}' not found in environment variables or config file.")
    return None


def get_llm_model(
        model_name: str,
        model_params: Optional[ModelParams] = None,
        retries: int = 0,
        timeout: Optional[int] = 60,
        temperature: Optional[float] = None  # Allow direct override for judge
) -> Any:
    """
    Initializes and returns a cached LLM instance based on the model name.
    """
    # Use provided params or create a default if None
    params = model_params or ModelParams()

    # The direct temperature override is for the judge LLM, which needs to be deterministic
    if temperature is not None:
        params.temperature = temperature

    # Create a unique cache key based on model name and all its parameters
    cache_key = f"{model_name}-{params.json()}-{retries}-{timeout}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    logger.info(f"Initializing LLM model: {model_name} (retries={retries}, timeout={timeout})...")

    # Prepare model arguments
    model_kwargs = {
        "model": model_name,
        "temperature": params.temperature or 0.7,  # Default temp if not set
        "max_tokens": params.max_tokens,
        "top_p": params.top_p,
        "max_retries": retries,
        "timeout": timeout,
    }
    # Filter out None values so we don't pass them to the constructor
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    # Find the correct provider and initialize the model
    for prefix, (model_class, service_key) in MODEL_PROVIDER_MAPPING.items():
        if model_name.startswith(prefix):
            if service_key:
                api_key = get_api_key(service_key)
                if not api_key:
                    raise ValueError(f"API key for {service_key} not found, cannot initialize {model_name}.")

                # Specific handling for different providers if necessary
                if service_key == "google":
                    model_kwargs["google_api_key"] = api_key
                else:
                    model_kwargs["api_key"] = api_key

            # Special handling for local Ollama models
            if model_name.startswith("ollama/"):
                model_kwargs["model"] = model_name.replace("ollama/", "")
                logger.info(f"Using local Ollama model: '{model_kwargs['model']}'.")

            llm = model_class(**model_kwargs)
            _llm_cache[cache_key] = llm
            logger.info(f"Successfully initialized LLM model '{model_name}'.")
            return llm

    raise ValueError(f"Unsupported or unknown LLM model: {model_name}")