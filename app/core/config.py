import os
from dotenv import load_dotenv

def get_api_key(service_name: str) -> str:
    """
    Retrieves an API key for a given service and returns it.
    It checks environment variables first, then a .env file.
    """
    load_dotenv()

    # Standardize the service name for comparison
    service_name_lower = service_name.lower()
    env_var_name = ""

    # Determine the environment variable name based on the service
    # ADDED 'text-embedding' to the check for OpenAI models.
    if "openai" in service_name_lower or "gpt" in service_name_lower or "text-embedding" in service_name_lower:
        env_var_name = "OPENAI_API_KEY"
    elif "anthropic" in service_name_lower or "claude" in service_name_lower:
        env_var_name = "ANTHROPIC_API_KEY"
    elif "google" in service_name_lower or "gemini" in service_name_lower or "models/embedding" in service_name_lower:
        env_var_name = "GOOGLE_API_KEY"
    elif "xai" in service_name_lower:
        env_var_name = "XAI_API_KEY"
    else:
        # Fallback for other potential service names (like HuggingFace which doesn't need a key)
        # We can return an empty string if no specific key pattern is matched.
        # The embedding factory for HuggingFace doesn't require a key, so this is safe.
        if "sentence-transformers" in service_name_lower:
            return ""
        env_var_name = f"{service_name.upper().replace('-', '_')}_API_KEY"

    api_key = os.getenv(env_var_name)

    if not api_key:
        error_message = (
            f"Error: API key for '{service_name}' (expected as {env_var_name}) not found.\n"
            f"Please set it using the command: ragnetic set-api"
        )
        raise ValueError(error_message)

    return api_key
