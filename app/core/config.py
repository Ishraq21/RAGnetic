import os
import configparser

# Define constants for the config file path, assuming the script runs from the project root
RAGNETIC_DIR = ".ragnetic"
CONFIG_FILE = os.path.join(RAGNETIC_DIR, "config.ini")

# A mapping from the simple provider name to the key name in the config file
PROVIDER_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "pinecone": "PINECONE_API_KEY",
    "mongodb": "MONGODB_CONN_STRING", # For MongoDB connection string
}

def get_api_key(provider: str) -> str:
    """
    Retrieves a specific provider's API key from the .ragnetic/config.ini file.

    Args:
        provider: The name of the API provider (e.g., "openai").

    Returns:
        The API key as a string.

    Raises:
        ValueError: If the provider is not supported or the key is not found.
    """
    provider_key_name = PROVIDER_MAP.get(provider.lower())
    if not provider_key_name:
        raise ValueError(f"API provider '{provider}' is not supported.")

    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"Configuration file not found at '{CONFIG_FILE}'. "
            "Please run 'ragnetic init' to create it."
        )

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    # Safely get the key from the [API_KEYS] section
    api_key = config.get('API_KEYS', provider_key_name, fallback=None)

    if not api_key or api_key == "...":
        raise ValueError(
            f"API key for '{provider}' not found or not set in {CONFIG_FILE}. "
            "Please use the 'ragnetic set-api' command."
        )

    return api_key