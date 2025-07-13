import os
import configparser
import logging
from typing import Dict, Any, Optional, List  # Added List for get_path_settings return type
from pathlib import Path  # NEW: Added import

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
SERVICE_KEY_MAPPING: Dict[str, str] = {
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

# --- Path Settings Cache ---
_path_settings_cache: Dict[str, Any] = {}  # NEW: Cache for path settings


def get_path_settings() -> Dict[str, Path | List[Path]]:  # MODIFIED: Return type hint to include List[Path]
    """
    Retrieves project root and allowed data directories, prioritizing environment variables
    over config.ini. Caches the result.
    """
    if _path_settings_cache:
        return _path_settings_cache

    is_production = os.environ.get("ENVIRONMENT", "development").lower() == "production"
    settings: Dict[str, Path | List[Path]] = {}  # Initialize with correct type hint

    # 1. Prioritize Environment Variables
    env_project_root = os.environ.get("RAGNETIC_PROJECT_ROOT")
    env_allowed_dirs = os.environ.get("RAGNETIC_ALLOWED_DATA_DIRS")

    if env_project_root:
        settings["PROJECT_ROOT"] = Path(env_project_root).resolve()
        logger.info("Loaded PROJECT_ROOT from environment variable.")
    if env_allowed_dirs:
        # Resolve each path in the comma-separated list
        settings["ALLOWED_DATA_DIRS"] = [Path(d).resolve() for d in env_allowed_dirs.split(",")]
        logger.info("Loaded ALLOWED_DATA_DIRS from environment variable.")

    # 2. Fallback to config.ini if env vars not set (for development or defaults)
    if not settings.get("PROJECT_ROOT") or not settings.get("ALLOWED_DATA_DIRS"):
        if os.path.exists(CONFIG_FILE):
            config = configparser.ConfigParser()
            config.read(CONFIG_FILE)

            if 'PATH_SETTINGS' in config:
                if "PROJECT_ROOT" not in settings:  # Check if not already set by env var
                    config_project_root = config.get("PATH_SETTINGS", "PROJECT_ROOT", fallback=".")
                    # Resolve relative to the directory where config.ini is assumed to be or current working directory
                    settings["PROJECT_ROOT"] = (Path(os.getcwd()) / config_project_root).resolve()
                    logger.info("Loaded PROJECT_ROOT from config.ini (development fallback).")

                if "ALLOWED_DATA_DIRS" not in settings:  # Check if not already set by env var
                    config_allowed_dirs_str = config.get("PATH_SETTINGS", "ALLOWED_DATA_DIRS",
                                                         fallback="data,agents_data")
                    # Resolve relative to PROJECT_ROOT or current working directory for flexibility
                    project_base = settings.get("PROJECT_ROOT",
                                                Path(os.getcwd()).resolve())  # Use loaded PROJECT_ROOT or CWD
                    settings["ALLOWED_DATA_DIRS"] = [(Path(project_base) / d).resolve() for d in
                                                     config_allowed_dirs_str.split(",")]
                    logger.info("Loaded ALLOWED_DATA_DIRS from config.ini (development fallback).")
            else:
                logger.warning(
                    f"No '[PATH_SETTINGS]' section found in {CONFIG_FILE}. Using defaults if not set by env vars.")
        else:
            logger.warning(
                f"Config file not found at {CONFIG_FILE}. Path settings will use defaults if not in env vars.")

    # 3. Apply sensible defaults if nothing loaded (for cases where config.ini or env vars are incomplete)
    if "PROJECT_ROOT" not in settings:
        settings["PROJECT_ROOT"] = Path(os.getcwd()).resolve()
        logger.warning(f"Using default PROJECT_ROOT: {settings['PROJECT_ROOT']}")
    if "ALLOWED_DATA_DIRS" not in settings:
        settings["ALLOWED_DATA_DIRS"] = [
            settings["PROJECT_ROOT"] / "data",
            settings["PROJECT_ROOT"] / "agents_data",
            settings["PROJECT_ROOT"] / ".ragnetic_temp_clones"
            # Include this default temp dir from code_repository_loader
        ]
        logger.warning(f"Using default ALLOWED_DATA_DIRS: {[str(d) for d in settings['ALLOWED_DATA_DIRS']]}")

    # Ensure all paths in ALLOWED_DATA_DIRS are resolved absolute paths
    # This step is already done during loading, but a final ensure for safety.
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

    # 1. Always prioritize environment variable
    api_key = os.environ.get(key_name)
    if api_key:
        logger.info(f"Loaded API key for '{service_name}' from environment variable.")
        return api_key

    # 2. In production, fail immediately if missing
    if is_production:
        logger.critical(f"FATAL: Required API key '{key_name}' not found in environment for production mode.")
        raise ValueError(f"Missing required environment variable: {key_name}")

    # 3. Fallback to config.ini in development
    if os.path.exists(CONFIG_FILE):
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        api_key = config.get("API_KEYS", key_name, fallback=None)
        if api_key and api_key != "...":
            logger.info(f"Loaded API key for '{service_name}' from config file (development fallback).")
            return api_key

    # 4. Development‐mode warning and error log
    logger.error(f"No API key found for '{service_name}' in development mode—calls will likely fail.")
    return None


def get_llm_model(
        model_name: str,
        model_params: Optional[ModelParams] = None,
        retries: int = 0,
        timeout: Optional[int] = 60,
        temperature: Optional[float] = None,  # Override for deterministic evaluation
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
    # Remove any None values
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    # Select provider by prefix
    for prefix, (model_class, service_key) in MODEL_PROVIDER_MAPPING.items():
        if not model_name.startswith(prefix):
            continue

        # If the provider needs an API key, fetch it
        if service_key:
            api_key = get_api_key(service_key)
            if not api_key:
                raise ValueError(f"API key for {service_key} not found; cannot initialize {model_name}.")
            # special case for Google
            if service_key == "google":
                model_kwargs["google_api_key"] = api_key
            else:
                model_kwargs["api_key"] = api_key

        # Adjust for Ollama local naming
        if model_name.startswith("ollama/"):
            model_kwargs["model"] = model_name.split("/", 1)[1]
            logger.info(f"Using local Ollama model: '{model_kwargs['model']}'.")

        llm = model_class(**model_kwargs)
        _llm_cache[cache_key] = llm
        logger.info(f"Successfully initialized LLM model '{model_name}'.")
        return llm

    raise ValueError(f"Unsupported or unknown LLM model: {model_name}")