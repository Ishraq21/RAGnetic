import os
import getpass
from dotenv import load_dotenv, set_key


def get_api_key(service_name: str) -> str:
    """
    Retrieves an API key for a given service.
    It checks environment variables first, then a .env file.
    If not found, it interactively prompts the user and saves the key to .env.
    """
    # Determine the environment variable name based on the service
    env_var_name = ""
    if "openai" in service_name.lower() or "gpt" in service_name.lower():
        env_var_name = "OPENAI_API_KEY"
    elif "anthropic" in service_name.lower() or "claude" in service_name.lower():
        env_var_name = "ANTHROPIC_API_KEY"
    elif "google" in service_name.lower() or "gemini" in service_name.lower():
        env_var_name = "GOOGLE_API_KEY"
    else:
        # Fallback for other potential services
        env_var_name = f"{service_name.upper()}_API_KEY"

    # 1. Try to get the key from the environment
    api_key = os.getenv(env_var_name)
    if api_key:
        return api_key

    # 2. If not found, try to load from .env file
    dotenv_path = os.path.join(os.getcwd(), '.env')
    load_dotenv(dotenv_path)
    api_key = os.getenv(env_var_name)
    if api_key:
        return api_key

    # 3. If still not found, prompt the user interactively
    print(f"--- API Key Configuration ---")
    print(f"API key for '{service_name}' ({env_var_name}) not found.")

    try:
        api_key = getpass.getpass(f"Please enter your {service_name} API key: ")
    except (IOError, EOFError):
        print("\nCould not read API key from input.")
        raise ValueError(f"API key for {service_name} is required.")

    if not api_key:
        raise ValueError(f"API key for {service_name} cannot be empty.")

    # 4. Save the key to the .env file for future use
    set_key(dotenv_path, env_var_name, api_key)
    print(f"API key for {service_name} saved to .env file for future use.")

    return api_key
