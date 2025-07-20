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
import typer

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Mappings ---
SERVICE_KEY_MAPPING: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "pinecone": "PINECONE_API_KEY",
    "mongodb": "MONGODB_CONN_STRING",
    "brave_search": "BRAVE_SEARCH_API_KEY",
}

MODEL_PROVIDER_MAPPING = {
    "gpt-": (ChatOpenAI, "openai"),
    "claude-": (ChatAnthropic, "anthropic"),
    "gemini-": (ChatGoogleGenerativeAI, "google"),
    "ollama/": (ChatOllama, None),
}

# --- Caching ---
_llm_cache: Dict[str, Any] = {}


# --- Core Configuration Functions ---

@lru_cache(maxsize=1)
def _get_config_parser() -> configparser.ConfigParser:
    """Reads and caches the main config.ini file."""
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
    """Retrieves and resolves all critical application paths."""
    config = _get_config_parser()
    project_root = Path(os.environ.get("RAGNETIC_PROJECT_ROOT", Path.cwd())).resolve()
    if config.has_option('PATH_SETTINGS', 'PROJECT_ROOT'):
        project_root = (project_root / config.get('PATH_SETTINGS', 'PROJECT_ROOT')).resolve()

    paths = {
        "PROJECT_ROOT": project_root,
        "RAGNETIC_DIR": project_root / ".ragnetic",
        "LOGS_DIR": project_root / "logs",
        "CONFIG_FILE_PATH": project_root / ".ragnetic" / "config.ini",
        "DATA_DIR": project_root / "data",
        "AGENTS_DIR": project_root / "agents",
        "VECTORSTORE_DIR": project_root / "vectorstore",
        "MEMORY_DIR": project_root / "memory",
        "TEMP_CLONES_DIR": project_root / ".ragnetic" / ".ragnetic_temp_clones",
        "BENCHMARK_DIR": project_root / "benchmark",
        "WORKFLOWS_DIR": project_root / "workflows",
    }

    default_allowed_dirs = f"{paths['DATA_DIR']},{paths['AGENTS_DIR']},{paths['TEMP_CLONES_DIR']}"
    allowed_dirs_str = config.get('PATH_SETTINGS', 'ALLOWED_DATA_DIRS', fallback=default_allowed_dirs)
    paths["ALLOWED_DATA_DIRS"] = [Path(p.strip()).resolve() for p in allowed_dirs_str.split(',')]

    return paths


def get_api_key(service_name: str) -> Optional[str]:
    """Retrieves an API key, prioritizing environment variables."""
    key_name = SERVICE_KEY_MAPPING.get(service_name.lower())
    if not key_name:
        raise ValueError(f"Service '{service_name}' is not a valid service.")
    api_key = os.environ.get(key_name)
    if api_key:
        return api_key
    config = _get_config_parser()
    if config.has_option('API_KEYS', key_name):
        api_key = config.get('API_KEYS', key_name)
        if api_key and api_key != "...":
            return api_key
    return None


def get_server_api_keys() -> List[str]:
    """Retrieves server API keys, prioritizing environment variables."""
    keys_str = os.environ.get("RAGNETIC_API_KEYS")
    if keys_str:
        return [key.strip() for key in keys_str.split(",") if key.strip()]
    config = _get_config_parser()
    if config.has_option('AUTH', 'server_api_keys'):
        keys_str = config.get('AUTH', 'server_api_keys')
        if keys_str:
            return [key.strip() for key in keys_str.split(",") if key.strip()]
    return []



def get_db_connection(name: str) -> str:
    """
    Constructs a database connection string from a structured configuration
    section in config.ini. It prioritizes reading the password from an
    environment variable and falls back to a secure interactive prompt.
    """
    config = _get_config_parser()
    section_name = f"DATABASE_{name}"

    if not config.has_section(section_name):
        raise ValueError(f"Configuration section '[{section_name}]' not found in .ragnetic/config.ini.")

    dialect = config.get(section_name, 'dialect')

    if 'sqlite' in dialect:
        project_root = get_path_settings()["PROJECT_ROOT"]
        db_path_str = config.get(section_name, 'database_path')
        db_path = project_root / db_path_str
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"{dialect}:///{db_path.resolve()}"

    username = config.get(section_name, 'username')
    host = config.get(section_name, 'host')
    port = config.get(section_name, 'port')
    database = config.get(section_name, 'database')

    password_env_var = f"{name.upper()}_PASSWORD"
    password = os.environ.get(password_env_var)

    if not password:
        logger.warning(f"Environment variable '{password_env_var}' not set for db connection '{name}'.")
        if "CI" not in os.environ:
            password = typer.prompt(f"Enter password for database user '{username}' on '{host}'", hide_input=True)

    if not password:
        raise ValueError(f"Password for database connection '{name}' is required but was not provided via the '{password_env_var}' environment variable or interactive prompt.")

    return f"{dialect}://{username}:{password}@{host}:{port}/{database}"


def get_db_connection_config() -> Optional[Dict[str, str]]:
    """
    Retrieves the raw configuration dictionary for the currently active
    system database connection (memory or logs).
    """
    config = _get_config_parser()
    mem_config = get_memory_storage_config()
    log_config = get_log_storage_config()

    conn_name = None
    if mem_config.get("type") in ["db", "sqlite"]:
        conn_name = mem_config.get("connection_name")
    elif log_config.get("type") == "db":
        conn_name = log_config.get("connection_name")

    if not conn_name:
        return None

    section_name = f"DATABASE_{conn_name}"
    if config.has_section(section_name):
        return dict(config.items(section_name))

    return None

def get_memory_storage_config() -> Dict[str, str]:
    """Reads the [MEMORY_STORAGE] section from the config file."""
    config = _get_config_parser()
    if config.has_section('MEMORY_STORAGE'):
        return dict(config.items('MEMORY_STORAGE'))
    return {"type": "sqlite"}  # Default


def get_log_storage_config() -> Dict[str, str]:
    """Reads the [LOG_STORAGE] section from the config file."""
    config = _get_config_parser()
    if config.has_section('LOG_STORAGE'):
        return dict(config.items('LOG_STORAGE'))
    return {"type": "file"}  # Default


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


def get_cors_settings() -> List[str]:
    """Retrieves CORS origins, prioritizing environment variables."""
    origins_str = os.environ.get("CORS_ALLOWED_ORIGINS")
    if origins_str:
        return [origin.strip() for origin in origins_str.split(',') if origin.strip()]

    config = _get_config_parser()
    if config.has_option('SERVER', 'cors_allowed_origins'):
        origins_str = config.get('SERVER', 'cors_allowed_origins')
        if origins_str:
            return [origin.strip() for origin in origins_str.split(',') if origin.strip()]

    return ["*"]  # Default to all origins if not set