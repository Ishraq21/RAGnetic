import typer
import uvicorn
import os
import shutil
import yaml
import glob
import configparser
import logging
from multiprocessing import Process

# Local application imports
from app.agents.config_manager import load_agent_config, load_agent_from_yaml_file, AGENTS_DIR
from app.pipelines.embed import embed_agent_data
from app.core.config import get_api_key
from app.watcher import start_watcher

# Set up logging for cleaner output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

cli_help = """
RAGnetic: Your on-premise, plug-and-play AI agent framework.

Provides a CLI for initializing projects, managing agents, and running the server.

Examples:
\n- Initialize a new project:
  $ ragnetic init
\n- Deploy an agent by its name:
  $ ragnetic deploy your-agent-name
\n- Start the server and automated file watcher:
  $ ragnetic start-server
"""

app = typer.Typer(
    name="ragnetic",
    help=cli_help,
    add_completion=False,
    no_args_is_help=True,
)

# Define constants for the config directory and file
RAGNETIC_DIR = ".ragnetic"
CONFIG_FILE = os.path.join(RAGNETIC_DIR, "config.ini")

MODEL_PROVIDERS = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google": "GOOGLE_API_KEY",
}


@app.command(help="Initialize a new RAGnetic project in the current directory.")
def init():
    """
    Creates the necessary folders (agents_data, data, .ragnetic) and a default config.ini file.
    """
    logger.info("Initializing new RAGnetic project...")
    folders_to_create = ["agents_data", "data", "vectorstore", "memory", RAGNETIC_DIR]
    for folder in folders_to_create:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logger.info(f"  - Created directory: ./{folder}")

    if not os.path.exists(CONFIG_FILE):
        config = configparser.ConfigParser()
        config['API_KEYS'] = {
            '# Please use the "ragnetic set-api" command to add your keys securely': '',
            'OPENAI_API_KEY': '...',
            'ANTHROPIC_API_KEY': '...',
            'GOOGLE_API_KEY': '...',
        }
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        logger.info(f"  - Created config file at: {CONFIG_FILE}")

    logger.info("\nProject initialized successfully.")
    logger.info("Next step: Use the 'ragnetic set-api' command to configure your API keys.")


@app.command(help="Set and save an API key to the config.ini file.")
def set_api():
    """
    Prompts the user to select a provider and enter an API key.
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
        config_key_name = MODEL_PROVIDERS[selected_provider]
    except (ValueError, IndexError):
        typer.secho("Error: Invalid selection.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    api_key = typer.prompt(f"Enter your {selected_provider} API Key", hide_input=True)
    if not api_key:
        typer.secho("Error: API Key cannot be empty.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        if 'API_KEYS' not in config:
            config['API_KEYS'] = {}
        config['API_KEYS'][config_key_name] = api_key
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        typer.secho(f"Successfully saved {selected_provider} API key to {CONFIG_FILE}.", fg=typer.colors.GREEN)
    except IOError as e:
        typer.secho(f"Error: Could not write to config file: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


# ** REFINED START-SERVER COMMAND **
@app.command(help="Starts the RAGnetic server and the file watcher.")
def start_server(
        host: str = typer.Option("127.0.0.1", help="The host to bind the server to."),
        port: int = typer.Option(8000, help="The port to run the server on."),
        reload: bool = typer.Option(False, "--reload",
                                    help="Enable auto-reloading for development (disables watcher)."),
        no_watcher: bool = typer.Option(False, "--no-watcher", help="Do not start the file watcher process."),
):
    """
    Starts the Uvicorn server and, by default, a background process to watch for
    data file changes.
    """
    # The --reload flag is for development and is incompatible with also running the watcher process.
    if reload:
        logger.warning("Running in --reload mode. The file watcher will be disabled.")
        uvicorn.run("app.main:app", host=host, port=port, reload=True)
        return

    # Start the watcher process in the background unless disabled.
    watcher_process = None
    if not no_watcher:
        data_directory = "data"
        if not os.path.exists(data_directory):
            logger.error(f"Error: The '{data_directory}' directory does not exist. Please run 'ragnetic init' first.")
            raise typer.Exit(code=1)

        # We run the watcher in a separate process so it doesn't block the web server.
        watcher_process = Process(target=start_watcher, args=(data_directory,), daemon=True)
        watcher_process.start()
        logger.info("Automated file watcher started in the background.")

    # Start the main FastAPI server.
    logger.info(f"Starting RAGnetic server on http://{host}:{port}")
    try:
        get_api_key("openai")
    except ValueError as e:
        logger.warning(f"{e} - some models may not work.")

    uvicorn.run("app.main:app", host=host, port=port)

    # Clean up the watcher process if it was started.
    if watcher_process and watcher_process.is_alive():
        watcher_process.terminate()
        watcher_process.join()
        logger.info("File watcher process stopped.")


# --- Agent Management Commands ---

@app.command(help="Lists all configured agents.")
def list_agents():
    if not os.path.exists(AGENTS_DIR):
        logger.error("Error: Directory 'agents_data' not found. Have you run 'ragnetic init'?")
        raise typer.Exit(code=1)
    agents = [f.split(".")[0] for f in os.listdir(AGENTS_DIR) if f.endswith((".yaml", ".yml"))]
    if not agents:
        logger.info("No agents found in the 'agents_data' directory.")
        return
    typer.echo("Available Agents:")
    for agent_name in agents:
        typer.echo(f"  - {agent_name}")


@app.command(name="deploy", help="Deploys an agent by its name, processing its data sources.")
def deploy_agent_by_name(
        agent_name: str = typer.Argument(..., help="The name of the agent to deploy (must match the YAML filename).")
):
    try:
        config_path = os.path.join(AGENTS_DIR, f"{agent_name}.yaml")
        logger.info(f"Loading agent configuration from: {config_path}")
        if not os.path.exists(config_path):
            typer.secho(f"Error: Configuration file not found at {config_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        agent_config = load_agent_from_yaml_file(config_path)
        typer.echo(f"\nDeploying agent '{agent_config.name}' using embedding model '{agent_config.embedding_model}'...")
        embed_agent_data(agent_config)
        typer.secho("\nAgent deployment successful!", fg=typer.colors.GREEN)
        typer.echo(f"  - Vector store created at: vectorstore/{agent_config.name}")
    except Exception as e:
        typer.secho(f"An unexpected error occurred during deployment: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(help="Displays the configuration of a specific agent.")
def inspect_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to inspect.")
):
    try:
        typer.echo(f"Inspecting configuration for agent: '{agent_name}'")
        config = load_agent_config(agent_name)
        typer.echo(yaml.dump(config.model_dump(), indent=2, sort_keys=False))
    except FileNotFoundError:
        typer.secho(f"Error: Agent '{agent_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(help="Validates an agent's configuration and associated files.")
def validate_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to validate.")
):
    typer.echo(f"Validating agent: '{agent_name}'...")
    errors = 0
    try:
        load_agent_config(agent_name)
        typer.secho("  - [PASS] YAML configuration is valid.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"  - [FAIL] Could not load or parse YAML config: {e}", fg=typer.colors.RED)
        errors += 1
    vectorstore_path = f"vectorstore/{agent_name}"
    if os.path.exists(vectorstore_path) and os.path.isdir(vectorstore_path):
        typer.secho(f"  - [PASS] Vector store directory exists at: {vectorstore_path}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"  - [WARN] Vector store not found. Agent may need to be deployed.", fg=typer.colors.YELLOW)
    memory_files = glob.glob(f"memory/{agent_name}_*.db")
    if memory_files:
        typer.echo(f"  - [INFO] Found {len(memory_files)} conversation memory file(s).")
    else:
        typer.echo("  - [INFO] No conversation memory files found (this is normal for a new agent).")
    typer.echo("-" * 20)
    if errors == 0:
        typer.secho("Validation successful. Agent appears to be configured correctly.", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Validation failed with {errors} critical error(s).", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="reset-agent", help="Resets an agent by deleting its learned data (vector store and memory).")
def reset_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to reset."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    vectorstore_path = f"vectorstore/{agent_name}"
    memory_pattern = f"memory/{agent_name}_*.db"
    typer.secho(f"Warning: This will reset agent '{agent_name}' by deleting its generated data:",
                fg=typer.colors.YELLOW)
    typer.echo(f"  - Vector store directory: {vectorstore_path}")
    typer.echo(f"  - All memory files matching: {memory_pattern}")
    if not force:
        typer.confirm("Are you sure you want to proceed?", abort=True)
    try:
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
            typer.echo(f"  - Deleted vector store: {vectorstore_path}")
        memory_files = glob.glob(memory_pattern)
        if memory_files:
            for f in memory_files: os.remove(f)
            typer.echo(f"  - Deleted {len(memory_files)} memory file(s).")
        typer.secho(f"\nAgent '{agent_name}' has been reset.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"An error occurred during reset: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="delete-agent", help="Permanently deletes an agent, including its configuration file.")
def delete_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to permanently delete."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    vectorstore_path = f"vectorstore/{agent_name}"
    memory_pattern = f"memory/{agent_name}_*.db"
    config_path = os.path.join(AGENTS_DIR, f"{agent_name}.yaml")
    typer.secho(f"DANGER: This will permanently delete agent '{agent_name}' and all its data:", fg=typer.colors.RED)
    typer.echo(f"  - Vector store directory: {vectorstore_path}")
    typer.echo(f"  - All memory files matching: {memory_pattern}")
    typer.echo(f"  - Agent configuration file: {config_path}")
    if not force:
        typer.confirm("This action is irreversible. Are you absolutely sure?", abort=True)
    try:
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
            typer.echo(f"  - Deleted vector store: {vectorstore_path}")
        memory_files = glob.glob(memory_pattern)
        if memory_files:
            for f in memory_files: os.remove(f)
            typer.echo(f"  - Deleted {len(memory_files)} memory file(s).")
        if os.path.exists(config_path):
            os.remove(config_path)
            typer.echo(f"  - Deleted agent configuration: {config_path}")
        typer.secho(f"\nAgent '{agent_name}' has been permanently deleted.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"An error occurred during deletion: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()