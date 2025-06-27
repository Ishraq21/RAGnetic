import typer
import uvicorn
import os
import shutil
import yaml
import glob

from app.agents.config_manager import load_agent_config, load_agent_from_yaml_file, AGENTS_DIR
from app.pipelines.embed import embed_agent_data
from app.core.config import get_api_key
from dotenv import set_key

cli_help = """
RAGnetic: Your on-premise, plug-and-play AI agent framework.

Provides a CLI for initializing projects, managing agents, and running the server.

Examples:
\n- Initialize a new project:
  $ ragnetic init
\n- Set an API key:
  $ ragnetic set-api
\n- Deploy an agent from a config file:
  $ ragnetic deploy-agent --config ./agents_data/my_agent.yaml
"""

app = typer.Typer(
    name="ragnetic",
    help=cli_help,
    add_completion=False,
    no_args_is_help=True,
)

# --- Define model providers and models ---
MODEL_PROVIDERS = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google": "GOOGLE_API_KEY",
}


@app.command(help="Initialize a new RAGnetic project in the current directory.")
def init():
    """
    Creates the necessary folders (agents_data, data) and a default .env.example file.
    """
    print("Initializing new RAGnetic project...")
    folders_to_create = ["agents_data", "data", "vectorstore", "memory"]
    for folder in folders_to_create:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"  - Created directory: ./{folder}")

    env_example_path = ".env.example"
    if not os.path.exists(env_example_path):
        with open(env_example_path, "w") as f:
            f.write("# This file is an example. Use 'ragnetic set-api' to create your .env file.\n")
            f.write("OPENAI_API_KEY=\"...\"\n")
            f.write("ANTHROPIC_API_KEY=\"...\"\n")
            f.write("GOOGLE_API_KEY=\"...\"\n")
        print(f"  - Created .env.example file.")

    print("\nProject initialized successfully.")
    print("Next step: Use the 'ragnetic set-api' command to configure your API keys.")


@app.command(help="Set and save an API key to the .env file.")
def set_api():
    """
    Prompts the user to select a provider and enter an API key,
    then saves it to the .env file.
    """
    typer.echo("Select the provider for the API key you want to set:")

    provider_menu = list(MODEL_PROVIDERS.keys())
    for i, provider in enumerate(provider_menu, 1):
        typer.echo(f"  [{i}] {provider}")

    try:
        choice_str = typer.prompt("Enter the number of your choice")
        choice_index = int(choice_str) - 1
        if not 0 <= choice_index < len(provider_menu):
            raise ValueError
        selected_provider = provider_menu[choice_index]
        env_var_name = MODEL_PROVIDERS[selected_provider]
    except (ValueError, IndexError):
        typer.secho("Error: Invalid selection.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    api_key = typer.prompt(f"Enter your {selected_provider} API Key", hide_input=True)

    if not api_key:
        typer.secho("Error: API Key cannot be empty.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    dotenv_path = ".env"
    try:
        set_key(dotenv_path, env_var_name, api_key)
        typer.secho(f"Successfully saved {selected_provider} API key to the .env file.", fg=typer.colors.GREEN)
    except IOError as e:
        typer.secho(f"Error: Could not write to .env file: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(help="Lists all configured agents.")
def list_agents():
    """Scans the agents_data directory and lists all configured agents."""
    if not os.path.exists(AGENTS_DIR):
        print("Error: Directory 'agents_data' not found. Have you run 'ragnetic init'?")
        raise typer.Exit(code=1)

    agents = [f.split(".")[0] for f in os.listdir(AGENTS_DIR) if f.endswith((".yaml", ".yml"))]

    if not agents:
        print("No agents found in the 'agents_data' directory.")
        return

    print("Available Agents:")
    for agent_name in agents:
        print(f"  - {agent_name}")


@app.command(help="Displays the configuration of a specific agent.")
def inspect_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to inspect.")
):
    """Loads and prints the specified agent's configuration."""
    try:
        print(f"Inspecting configuration for agent: '{agent_name}'")
        config = load_agent_config(agent_name)
        print(yaml.dump(config.model_dump(), indent=2, sort_keys=False))
    except FileNotFoundError:
        print(f"Error: Agent '{agent_name}' not found.")
        raise typer.Exit(code=1)


@app.command(help="Deploys an agent by processing and embedding its data sources.")
def deploy_agent(
        config_path: str = typer.Option(..., "--config", "-c", help="Path to the agent's YAML configuration file."),
):
    """Loads an agent config from YAML and creates a vector store."""
    try:
        print(f"Loading agent configuration from: {config_path}")
        agent_config = load_agent_from_yaml_file(config_path)

        print(f"\nDeploying agent '{agent_config.name}' using embedding model '{agent_config.embedding_model}'...")
        # embed_agent_data uses the factory to get the right model and API key.
        embed_agent_data(agent_config)

        print("\nAgent deployment successful!")
        print(f"  - Vector store created at: vectorstore/{agent_config.name}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"An unexpected error occurred during deployment: {e}")
        raise typer.Exit(code=1)


@app.command(help="Validates an agent's configuration and associated files.")
def validate_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to validate.")
):
    """Performs a health check on an agent's setup."""
    print(f"Validating agent: '{agent_name}'...")
    errors = 0

    try:
        load_agent_config(agent_name)
        print("  - [PASS] YAML configuration is valid.")
    except Exception as e:
        print(f"  - [FAIL] Could not load or parse YAML config: {e}")
        errors += 1

    vectorstore_path = f"vectorstore/{agent_name}"
    if os.path.exists(vectorstore_path) and os.path.isdir(vectorstore_path):
        print(f"  - [PASS] Vector store directory exists at: {vectorstore_path}")
    else:
        print(f"  - [WARN] Vector store not found. Agent may need to be deployed with 'ragnetic deploy-agent'.")

    memory_files = glob.glob(f"memory/{agent_name}_*.db")
    if memory_files:
        print(f"  - [INFO] Found {len(memory_files)} conversation memory file(s).")
    else:
        print("  - [INFO] No conversation memory files found (this is normal for a new agent).")

    print("-" * 20)
    if errors == 0:
        print("Validation successful. Agent appears to be configured correctly.")
    else:
        print(f"Validation failed with {errors} critical error(s). Please resolve the issues above.")
        raise typer.Exit(code=1)


@app.command(help="Starts the RAGnetic FastAPI server.")
def start_server(
        host: str = typer.Option("127.0.0.1", help="The host to bind the server to."),
        port: int = typer.Option(8000, help="The port to run the server on."),
        reload: bool = typer.Option(False, "--reload", help="Enable auto-reloading for development."),
):
    """Starts the Uvicorn server."""
    print(f"Starting RAGnetic server on http://{host}:{port}")
    try:
        get_api_key("openai")
    except ValueError as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


@app.command(help="Deletes all data associated with an agent (vector store and memory).")
def delete_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent whose data will be deleted."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    """
    Deletes an agent's vector store and conversation memory.
    Does not delete the agent's YAML config file.
    """
    vectorstore_path = f"vectorstore/{agent_name}"
    memory_pattern = f"memory/{agent_name}_*.db"

    print(f"Warning: This will permanently delete the following for agent '{agent_name}':")
    print(f"  - Vector store directory: {vectorstore_path}")
    print(f"  - All memory files matching: {memory_pattern}")

    if not force:
        typer.confirm("Are you sure you want to proceed?", abort=True)

    try:
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
            print(f"  - Deleted vector store: {vectorstore_path}")
        else:
            print(f"  - No vector store found to delete at {vectorstore_path}")

        memory_files = glob.glob(memory_pattern)
        if memory_files:
            for f in memory_files:
                os.remove(f)
            print(f"  - Deleted {len(memory_files)} memory file(s).")
        else:
            print("  - No memory files found to delete.")

        print(f"\nAgent '{agent_name}' data has been deleted.")
    except Exception as e:
        print(f"An error occurred during deletion: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
