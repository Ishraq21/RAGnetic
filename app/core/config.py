import os
import configparser
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.ollama import ChatOllama
from app.schemas.agent import ModelParams

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Mappings ---
# Maps service names to their environment variable and config.ini key names
SERVICE_KEY_MAPPING: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "pinecone": "PINECONE_API_KEY",
    "mongodb": "MONGODB_CONN_STRING",
    "brave_search": "BRAVE_SEARCH_API_KEY",
}

# Maps model prefixes to their respective classes and required service keys
MODEL_PROVIDER_MAPPING = {
    "gpt-": (ChatOpenAI, "openai"),
    "claude-": (ChatAnthropic, "anthropic"),
    "gemini-": (ChatGoogleGenerativeAI, "google"),
    "ollama/": (ChatOllama, None),  # Local models don't need a key
}

# --- Caching ---
_llm_cache: Dict[str, Any] = {}


# --- Core Configuration Functions ---

@lru_cache(maxsize=1)
def _get_config_parser() -> configparser.ConfigParser:
    """
    Reads and caches the main config.ini file. Using a cached function
    prevents reading the file from disk multiple times per command.
    """
    # This assumes get_project_root() is defined or paths are handled consistently.
    # For simplicity, we'll calculate the config file path here directly.
    project_root = Path(os.environ.get("RAGNETIC_PROJECT_ROOT", Path.cwd()))
    config_file_path = project_root / ".ragnetic" / "config.ini"

    config = configparser.ConfigParser()
    if config_file_path.exists():
        config.read(config_file_path)
    else:
        logger.debug(f"Config file not found at {config_file_path}. Using defaults.")
    return config


@lru_cache(maxsize=1)
def get_path_settings() -> Dict[str, Path | List[Path]]:
    """
    Retrieves and resolves all critical application paths.
    It establishes sensible defaults and allows overrides from config.ini
    without printing warnings for default usage.
    """
    config = _get_config_parser()

    # 1. Determine Project Root
    # The default is the current working directory.
    project_root = Path(os.environ.get("RAGNETIC_PROJECT_ROOT", Path.cwd())).resolve()

    # Allow override from config file, relative to the project root.
    if config.has_option('PATH_SETTINGS', 'PROJECT_ROOT'):
        project_root = (project_root / config.get('PATH_SETTINGS', 'PROJECT_ROOT')).resolve()

    # 2. Define all other paths based on the final project root
    paths = {
        "PROJECT_ROOT": project_root,
        "RAGNETIC_DIR": project_root / ".ragnetic",
        "LOGS_DIR": project_root / ".ragnetic" / "logs",
        "CONFIG_FILE_PATH": project_root / ".ragnetic" / "config.ini",
        "DATA_DIR": project_root / "data",
        "AGENTS_DIR": project_root / "agents_data",
        "VECTORSTORE_DIR": project_root / "vectorstore",
        "MEMORY_DIR": project_root / "memory",
        "TEMP_CLONES_DIR": project_root / ".ragnetic" / ".ragnetic_temp_clones",
    }

    # 3. Define allowed data directories for security
    default_allowed_dirs = f"{paths['DATA_DIR']},{paths['AGENTS_DIR']},{paths['TEMP_CLONES_DIR']}"
    allowed_dirs_str = config.get('PATH_SETTINGS', 'ALLOWED_DATA_DIRS', fallback=default_allowed_dirs)
    paths["ALLOWED_DATA_DIRS"] = [Path(p.strip()).resolve() for p in allowed_dirs_str.split(',')]

    return paths


def get_api_key(service_name: str) -> Optional[str]:
    """
    Retrieves an API key, prioritizing environment variables over the config file.
    """
    key_name = SERVICE_KEY_MAPPING.get(service_name.lower())
    if not key_name:
        raise ValueError(f"Service '{service_name}' is not a valid service.")

    # 1. Prioritize Environment Variable (ideal for production)
    api_key = os.environ.get(key_name)
    if api_key:
        logger.debug(f"Loaded API key for '{service_name}' from environment variable.")
        return api_key

    # 2. Fallback to config.ini for local development
    config = _get_config_parser()
    if config.has_option('API_KEYS', key_name):
        api_key = config.get('API_KEYS', key_name)
        if api_key and api_key != "...":
            logger.debug(f"Loaded API key for '{service_name}' from config file (development fallback).")
            return api_key

    logger.warning(f"API key for '{service_name}' not found. Calls to this service will likely fail.")
    return None


def get_server_api_keys() -> List[str]:
    """
    Retrieves a list of valid RAGnetic server API keys to protect the API.
    Prioritizes environment variables, then falls back to config.ini.
    """
    # 1. Prioritize Environment Variable
    keys_str = os.environ.get("RAGNETIC_API_KEYS")
    if keys_str:
        logger.info("Loaded server API keys from RAGNETIC_API_KEYS environment variable.")
        return [key.strip() for key in keys_str.split(",") if key.strip()]

    # 2. Fallback to config.ini
    config = _get_config_parser()
    if config.has_option('AUTH', 'server_api_keys'):
        keys_str = config.get('AUTH', 'server_api_keys')
        if keys_str:
            logger.info("Loaded server API keys from config.ini.")
            return [key.strip() for key in keys_str.split(",") if key.strip()]

    # 3. If no keys are found anywhere, run in insecure mode.
    logger.warning("No server API keys found. API is running without authentication.")
    return []


def get_db_connection(name_or_string: str) -> str:
    """
    Resolves a database connection name into a full connection string from config.ini.
    If the input already looks like a connection string, it's returned directly.
    """
    if "://" in name_or_string:
        return name_or_string

    config = _get_config_parser()
    if config.has_option('DATABASE_CONNECTIONS', name_or_string):
        return config.get('DATABASE_CONNECTIONS', name_or_string)

    raise ValueError(
        f"Database connection name '{name_or_string}' not found in [DATABASE_CONNECTIONS] section of .ragnetic/config.ini")


def get_llm_model(
        model_name: str,
        model_params: Optional[ModelParams] = None,
        retries: int = 0,
        timeout: Optional[int] = 60,
        temperature: Optional[float] = None,
) -> Any:
    """
    Initializes (and caches) an LLM instance based on the model name and params.
    """
    params = model_params or ModelParams()
    if temperature is not None:
        params.temperature = temperature

    cache_key = f"{model_name}-{params.json()}-{retries}-{timeout}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    logger.info(f"Initializing LLM model: {model_name} (retries={retries}, timeout={timeout})...")

    # Prepare common model arguments
    model_kwargs = {
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "max_retries": retries,
        "timeout": timeout,
    }
    # Clean kwargs with None values
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    # Find the provider and initialize the model
    for prefix, (model_class, service_key) in MODEL_PROVIDER_MAPPING.items():
        if model_name.startswith(prefix):
            # Add API key if the service requires one
            if service_key:
                api_key = get_api_key(service_key)
                if not api_key:
                    raise ValueError(f"API key for {service_key} is required to use model {model_name}.")
                # Different providers have different argument names for the key
                if service_key == "google":
                    model_kwargs["google_api_key"] = api_key
                else:
                    model_kwargs["api_key"] = api_key

            # Specific handling for local Ollama models
            if model_name.startswith("ollama/"):
                model_name_only = model_name.split("/", 1)[1]
                model_kwargs["model"] = model_name_only
                logger.info(f"Using local Ollama model: '{model_name_only}'.")
            else:
                model_kwargs["model"] = model_name

            # Initialize the model class
            try:
                llm = model_class(**model_kwargs)
                _llm_cache[cache_key] = llm
                logger.info(f"Successfully initialized LLM model '{model_name}'.")
                return llm
            except Exception as e:
                logger.error(f"Failed to initialize model '{model_name}' with class '{model_class.__name__}': {e}",
                             exc_info=True)
                raise

    raise ValueError(f"Unsupported or unknown LLM model prefix for: {model_name}")