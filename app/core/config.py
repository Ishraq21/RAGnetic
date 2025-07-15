import os
import configparser
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.schemas.agent import ModelParams

# --- Mappings ---
# Maps service names to their environment variable and config.ini key names
SERVICE_KEY_MAPPING: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "pinecone": "PINECONE_API_KEY",
    "mongodb": "MONGODB_CONN_STRING",
    "brave_search": "BRAVE_SEARCH_API_KEY"
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
_path_settings_cache: Dict[str, Any] = {}


def get_path_settings() -> Dict[str, Path | List[Path]]:
    """
    Retrieves project root and various application directories (logs, data, vectorstore, etc.),
    prioritizing environment variables over config.ini. Caches the result.
    """
    if _path_settings_cache:
        return _path_settings_cache

    is_production = os.environ.get("ENVIRONMENT", "development").lower() == "production"
    settings: Dict[str, Path | List[Path]] = {}

    DEFAULT_RAGNETIC_DIR_NAME = ".ragnetic"
    DEFAULT_LOGS_DIR_NAME = "logs"
    DEFAULT_DATA_DIR_NAME = "data"
    DEFAULT_AGENTS_DIR_NAME = "agents_data"
    DEFAULT_VECTORSTORE_DIR_NAME = "vectorstore"
    DEFAULT_MEMORY_DIR_NAME = "memory"
    DEFAULT_TEMP_CLONES_DIR_NAME = ".ragnetic_temp_clones"

    env_project_root_str = os.environ.get("RAGNETIC_PROJECT_ROOT")
    if env_project_root_str:
        settings["PROJECT_ROOT"] = Path(env_project_root_str).resolve()
        logger.info("Loaded PROJECT_ROOT from environment variable.")
    else:
        config_file_path_relative_to_cwd = os.path.join(DEFAULT_RAGNETIC_DIR_NAME, "config.ini")
        if os.path.exists(config_file_path_relative_to_cwd):
            config = configparser.ConfigParser()
            config.read(config_file_path_relative_to_cwd)
            if 'PATH_SETTINGS' in config:
                config_project_root_str = config.get("PATH_SETTINGS", "PROJECT_ROOT", fallback=".")
                settings["PROJECT_ROOT"] = (Path(os.getcwd()) / config_project_root_str).resolve()
                logger.info("Loaded PROJECT_ROOT from config.ini (development fallback).")
            else:
                logger.warning(
                    f"No '[PATH_SETTINGS]' section found in {config_file_path_relative_to_cwd}. Using defaults for PROJECT_ROOT.")
        else:
            logger.warning(f"Config file not found at {config_file_path_relative_to_cwd}. Using default PROJECT_ROOT.")

        if "PROJECT_ROOT" not in settings:
            settings["PROJECT_ROOT"] = Path(os.getcwd()).resolve()
            logger.warning(f"Using default PROJECT_ROOT: {settings['PROJECT_ROOT']}")

    project_root = settings["PROJECT_ROOT"]

    settings["RAGNETIC_DIR"] = (project_root / DEFAULT_RAGNETIC_DIR_NAME).resolve()
    settings["CONFIG_FILE_PATH"] = (settings["RAGNETIC_DIR"] / "config.ini").resolve()
    settings["LOGS_DIR"] = (project_root / DEFAULT_LOGS_DIR_NAME).resolve()
    settings["DATA_DIR"] = (project_root / DEFAULT_DATA_DIR_NAME).resolve()
    settings["AGENTS_DIR"] = (project_root / DEFAULT_AGENTS_DIR_NAME).resolve()
    settings["VECTORSTORE_DIR"] = (project_root / DEFAULT_VECTORSTORE_DIR_NAME).resolve()
    settings["MEMORY_DIR"] = (project_root / DEFAULT_MEMORY_DIR_NAME).resolve()
    settings["TEMP_CLONES_DIR"] = (project_root / DEFAULT_TEMP_CLONES_DIR_NAME).resolve()

    env_allowed_dirs_str = os.environ.get("RAGNETIC_ALLOWED_DATA_DIRS")
    if env_allowed_dirs_str:
        settings["ALLOWED_DATA_DIRS"] = [(project_root / d).resolve() for d in env_allowed_dirs_str.split(",")]
        logger.info("Loaded ALLOWED_DATA_DIRS from environment variable.")
    else:
        config = configparser.ConfigParser()
        if os.path.exists(settings["CONFIG_FILE_PATH"]):
            config.read(settings["CONFIG_FILE_PATH"])
            if 'PATH_SETTINGS' in config:
                config_allowed_dirs_str = config.get("PATH_SETTINGS", "ALLOWED_DATA_DIRS", fallback="data,agents_data")
                settings["ALLOWED_DATA_DIRS"] = [(project_root / d).resolve() for d in config_allowed_dirs_str.split(",")]
                logger.info("Loaded ALLOWED_DATA_DIRS from config.ini (development fallback).")
            else:
                logger.warning(
                    f"No '[PATH_SETTINGS]' section found in {settings['CONFIG_FILE_PATH']}. Using defaults for ALLOWED_DATA_DIRS.")
        else:
            logger.warning(f"Config file not found at {settings['CONFIG_FILE_PATH']}. Using default ALLOWED_DATA_DIRS.")

        if "ALLOWED_DATA_DIRS" not in settings:
            settings["ALLOWED_DATA_DIRS"] = [
                settings["DATA_DIR"],
                settings["AGENTS_DIR"],
                settings["TEMP_CLONES_DIR"]
            ]
            logger.warning(f"Using default ALLOWED_DATA_DIRS: {[str(d) for d in settings['ALLOWED_DATA_DIRS']]}")

    settings["ALLOWED_DATA_DIRS"] = [d.resolve() for d in settings["ALLOWED_DATA_DIRS"]]

    _path_settings_cache.update(settings)
    return settings


def get_api_key(service_name: str) -> Optional[str]:
    """
    Retrieves an API key, prioritizing environment variables over the config file.
    In production mode, missing keys are fatal; in development, we log an error but continue.
    """
    is_production = os.environ.get("ENVIRONMENT", "development").lower() == "production"
    key_name = SERVICE_KEY_MAPPING.get(service_name.lower())
    if not key_name:
        raise ValueError(f"Service '{service_name}' is not a valid service.")

    api_key = os.environ.get(key_name)
    if api_key:
        logger.info(f"Loaded API key for '{service_name}' from environment variable.")
        return api_key

    if is_production:
        logger.critical(f"FATAL: Required API key '{key_name}' not found in environment for production mode.")
        raise ValueError(f"Missing required environment variable: {key_name}")

    path_settings = get_path_settings()
    config_file_path = path_settings["CONFIG_FILE_PATH"]

    if os.path.exists(config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        api_key = config.get("API_KEYS", key_name, fallback=None)
        if api_key and api_key != "...":
            logger.info(f"Loaded API key for '{service_name}' from config file (development fallback).")
            return api_key

    logger.error(f"No API key found for '{service_name}' in development mode—calls will likely fail.")
    return None

def get_server_api_keys() -> List[str]:
    """
    Retrieves a list of valid RAGnetic server API keys to protect the API.
    It prioritizes the RAGNETIC_API_KEYS environment variable. If not found,
    it falls back to reading the 'server_api_keys' from the [AUTH] section
    of the config.ini file. Keys should be a comma-separated string.
    """
    # 1. Prioritize Environment Variable (ideal for Docker/k8s)
    keys_str = os.environ.get("RAGNETIC_API_KEYS")
    if keys_str:
        logger.info("Loaded server API keys from RAGNETIC_API_KEYS environment variable.")
        return [key.strip() for key in keys_str.split(",") if key.strip()]

    # 2. Fallback to config.ini
    path_settings = get_path_settings()
    config_file_path = path_settings["CONFIG_FILE_PATH"]
    if os.path.exists(config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        if 'AUTH' in config and 'server_api_keys' in config['AUTH']:
            keys_str = config['AUTH']['server_api_keys']
            if keys_str:
                logger.info("Loaded server API keys from config.ini.")
                return [key.strip() for key in keys_str.split(",") if key.strip()]

    # 3. If no keys are found anywhere, run in insecure mode.
    logger.warning("No server API keys found in environment or config.ini. API is running without authentication.")
    return []


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
    model_kwargs = {
        "model": model_name,
        "temperature": params.temperature or 0.7,
        "max_tokens": params.max_tokens,
        "top_p": params.top_p,
        "max_retries": retries,
        "timeout": timeout,
    }
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    for prefix, (model_class, service_key) in MODEL_PROVIDER_MAPPING.items():
        if not model_name.startswith(prefix):
            continue

        if service_key:
            api_key = get_api_key(service_key)
            if not api_key:
                raise ValueError(f"API key for {service_key} not found; cannot initialize {model_name}.")
            if service_key == "google":
                model_kwargs["google_api_key"] = api_key
            else:
                model_kwargs["api_key"] = api_key

        if model_name.startswith("ollama/"):
            model_kwargs["model"] = model_name.split("/", 1)[1]
            logger.info(f"Using local Ollama model: '{model_kwargs['model']}'.")

        llm = model_class(**model_kwargs)
        _llm_cache[cache_key] = llm
        logger.info(f"Successfully initialized LLM model '{model_name}'.")
        return llm

    raise ValueError(f"Unsupported or unknown LLM model: {model_name}")