import os
from dotenv import load_dotenv

def get_api_key(service_name: str) -> str:
    """
    Retrieves an API key for a given service and returns it.
    It checks environment variables first, then a .env file.
    """
    load_dotenv()

    # Determine the environment variable name based on the service
    env_var_name = ""
    if "openai" in service_name.lower() or "gpt" in service_name.lower():
        env_var_name = "OPENAI_API_KEY"
    elif "anthropic" in service_name.lower() or "claude" in service_name.lower():
        env_var_name = "ANTHROPIC_API_KEY"
    elif "google" in service_name.lower() or "gemini" in service_name.lower():
        env_var_name = "GOOGLE_API_KEY"
    elif "xai" in service_name.lower():
        env_var_name = "XAI_API_KEY"
    else:
        env_var_name = f"{service_name.upper()}_API_KEY"

    api_key = os.getenv(env_var_name)

    if not api_key:
        error_message = (
            f"Error: API key for '{service_name}' ({env_var_name}) not found.\n"
            f"Please set it using the command: ragnetic set-api"
        )
        raise ValueError(error_message)

    return api_key