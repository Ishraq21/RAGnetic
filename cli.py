import typer
import uvicorn
import os
import shutil
import yaml
import glob
import configparser
import logging
from multiprocessing import Process
import json
import pandas as pd
from app.evaluation.dataset_generator import generate_test_set
from app.evaluation.benchmark import run_benchmark
# Local application imports
from app.agents.config_manager import load_agent_config, load_agent_from_yaml_file, AGENTS_DIR
from app.pipelines.embed import embed_agent_data
from app.core.config import get_api_key
from app.watcher import start_watcher
import pytest
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

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

auth_app = typer.Typer(name="auth", help="Manage authentication for external services.")
app.add_typer(auth_app)

# Define constants for the config directory and file
RAGNETIC_DIR = ".ragnetic"
CONFIG_FILE = os.path.join(RAGNETIC_DIR, "config.ini")

MODEL_PROVIDERS = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google (Gemini)": "GOOGLE_API_KEY",
    "Pinecone": "PINECONE_API_KEY",
    "MongoDB Atlas": "MONGODB_CONN_STRING",
    "Hugging Face": None,
    "Ollama (Local LLMs)": None
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


@app.command(help="Set and save API keys for cloud providers.")
def set_api():
    """
    Guides the user through setting one or more API keys and saves them
    to the .ragnetic/config.ini file.
    """
    typer.echo("API Key Configuration Wizard")
    typer.echo(
        "You can set keys for multiple providers if your agents mix services (e.g., OpenAI chat + Anthropic embeddings).")

    while True:
        typer.echo("\nPlease select a provider to configure:")
        provider_menu = list(MODEL_PROVIDERS.keys())
        for i, provider in enumerate(provider_menu, 1):
            typer.echo(f"  [{i}] {provider}")

        try:
            choice_str = typer.prompt("Enter the number of your choice")
            choice_index = int(choice_str) - 1
            if not 0 <= choice_index < len(provider_menu):
                raise ValueError
            selected_provider = provider_menu[choice_index]

            if selected_provider == "Hugging Face":
                typer.secho("\nHugging Face models run locally and do not require an API key.", fg=typer.colors.GREEN)
                typer.echo("You only need to set an API key for the CHAT model you plan to use (e.g., OpenAI).")

            elif selected_provider == "Ollama (Local LLMs)":
                typer.secho("\nOllama models run locally on your machine and do not require an API key.",
                            fg=typer.colors.GREEN)
                typer.echo("   Please ensure the Ollama application is running before you start RAGnetic.")
                typer.echo("   You can specify a model in your agent's YAML file like: llm_model: 'ollama/llama3'")

            else:
                config_key_name = MODEL_PROVIDERS[selected_provider]
                api_key = typer.prompt(f"Enter your {selected_provider} API Key", hide_input=True)
                if not api_key:
                    typer.secho("Error: API Key cannot be empty.", fg=typer.colors.RED)
                    continue

                config = configparser.ConfigParser()
                config.read(CONFIG_FILE)
                if 'API_KEYS' not in config:
                    config['API_KEYS'] = {}
                config['API_KEYS'][config_key_name] = api_key
                with open(CONFIG_FILE, 'w') as configfile:
                    config.write(configfile)
                typer.secho(f"Successfully saved {selected_provider} API key.", fg=typer.colors.GREEN)

        except (ValueError, IndexError):
            typer.secho("Error: Invalid selection.", fg=typer.colors.RED)

        # Ask the user if they want to set another key.
        if not typer.confirm("Do you want to set another API key?", default=False):
            break

    typer.echo("\nAPI key configuration complete.")

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
    if reload:
        logger.warning("Running in --reload mode. The file watcher will be disabled.")
        uvicorn.run("app.main:app", host=host, port=port, reload=True)
        return

    watcher_process = None
    if not no_watcher:
        data_directory = "data"
        if not os.path.exists(data_directory):
            logger.error(f"Error: The '{data_directory}' directory does not exist. Please run 'ragnetic init' first.")
            raise typer.Exit(code=1)

        watcher_process = Process(target=start_watcher, args=(data_directory,), daemon=True)
        watcher_process.start()
        logger.info("Automated file watcher started in the background.")

    logger.info(f"Starting RAGnetic server on http://{host}:{port}")
    try:
        get_api_key("openai")
    except ValueError as e:
        logger.warning(f"{e} - some models may not work.")

    uvicorn.run("app.main:app", host=host, port=port)

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
        agent_name: str = typer.Argument(..., help="The name of the agent to deploy (must match the YAML filename)."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation and overwrite existing data."),
):
    """Loads an agent config from YAML and creates a vector store."""
    try:
        config_path = os.path.join(AGENTS_DIR, f"{agent_name}.yaml")
        logger.info(f"Loading agent configuration from: {config_path}")
        if not os.path.exists(config_path):
            typer.secho(f"Error: Configuration file not found at {config_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        vectorstore_path = f"vectorstore/{agent_name}"
        if os.path.exists(vectorstore_path) and not force:
            typer.secho(f"Warning: A vector store for agent '{agent_name}' already exists.", fg=typer.colors.YELLOW)
            if not typer.confirm("Do you want to overwrite it and re-deploy the agent?"):
                typer.echo("Deployment cancelled.")
                raise typer.Exit()

            # Clean up the old directory before deploying again
            shutil.rmtree(vectorstore_path)
            logger.info(f"Removed existing vector store at: {vectorstore_path}")

        agent_config = load_agent_from_yaml_file(config_path)
        typer.echo(f"\nDeploying agent '{agent_config.name}' using embedding model '{agent_config.embedding_model}'...")
        embed_agent_data(agent_config)

        typer.secho("\nAgent deployment successful!", fg=typer.colors.GREEN)
        typer.echo(f"  - Vector store created at: {vectorstore_path}")

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


@auth_app.command("gdrive", help="Authenticate with Google Drive using a service account key.")
def auth_gdrive():
    """
    Guides the user to provide the path to their Google service account JSON file
    and saves the content to the main config file.
    """
    typer.echo("Google Drive Authentication Setup")
    typer.echo("Please provide the path to your service account JSON key file.")
    json_path = typer.prompt("Path to service account .json file")

    if not os.path.exists(json_path) or not json_path.endswith('.json'):
        typer.secho(f"Error: File not found or not a .json file at '{json_path}'", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        with open(json_path, 'r') as f:
            credentials_json_str = f.read()

        json.loads(credentials_json_str)

        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)

        if 'GOOGLE_CREDENTIALS' not in config:
            config.add_section('GOOGLE_CREDENTIALS')

        # Escape the '%' character for configparser by replacing it with '%%'
        safe_credentials_str = credentials_json_str.replace('%', '%%')
        config.set('GOOGLE_CREDENTIALS', 'json_key', safe_credentials_str)

        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)

        typer.secho("Google Drive credentials saved successfully!", fg=typer.colors.GREEN)
        typer.echo("You can now use the 'gdoc' source type in your agents.")

    except json.JSONDecodeError:
        typer.secho("Error: The provided file is not a valid JSON file.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(help="Runs the entire test suite using pytest.")
def test():
    """
    Discovers and runs all automated tests in the 'tests/' directory.
    """
    typer.echo("Running the RAGnetic test suite...")

    # Define the arguments for pytest.
    # We'll add '-v' for verbose output.
    pytest_args = ["-v", "tests/"]

    # Run pytest programmatically
    result_code = pytest.main(pytest_args)

    if result_code == 0:
        typer.secho("\nAll tests passed!", fg=typer.colors.GREEN)
    else:
        typer.secho(f"\n{result_code} test(s) failed.", fg=typer.colors.RED)


@app.command(help="Validates an agent's configuration and associated files.")
def validate_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to validate."),
        check_connections: bool = typer.Option(
            False,
            "--check-connections",
            "-c",
            help="Test data source connections."
        )
):
    """
    Validates an agent's configuration, checking YAML syntax and the existence
    of its vector store. If --dry-run is specified, it will also test live
    connections to databases and URLs.
    """
    typer.echo(f"Validating agent: '{agent_name}'...")
    errors = 0

    # --- Basic Validation (existing logic) ---
    try:
        agent_config = load_agent_config(agent_name)
        typer.secho("  - [PASS] YAML configuration is valid.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"  - [FAIL] Could not load or parse YAML config: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

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

    # --- Check Connections ---
    if check_connections:
        typer.echo("\n--- Performing Connection Check ---")

        for i, source in enumerate(agent_config.sources):
            source_info = f"Source #{i + 1} (type: {source.type})"

            if source.type == 'db' and source.db_connection:
                typer.echo(f"  - Testing {source_info}...")
                try:
                    engine = create_engine(source.db_connection)
                    with engine.connect() as connection:
                        # Execute a simple query to confirm connectivity
                        connection.execute(text("SELECT 1"))
                    typer.secho("    - [PASS] Database connection successful.", fg=typer.colors.GREEN)
                except SQLAlchemyError as e:
                    typer.secho(f"    - [FAIL] Database connection failed: {e}", fg=typer.colors.RED)
                    errors += 1
                except Exception as e:
                    typer.secho(f"    - [FAIL] An unexpected error occurred with the database connection: {e}",
                                fg=typer.colors.RED)
                    errors += 1

            elif source.type == 'url' and source.url:
                typer.echo(f"  - Testing {source_info}...")
                try:
                    # Use a HEAD request to check the URL without downloading the content
                    response = requests.head(source.url, timeout=10)
                    response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
                    typer.secho(f"    - [PASS] URL is reachable (Status: {response.status_code}).",
                                fg=typer.colors.GREEN)
                except requests.exceptions.RequestException as e:
                    typer.secho(f"    - [FAIL] URL is not reachable: {e}", fg=typer.colors.RED)
                    errors += 1

    typer.echo("-" * 20)
    if errors == 0:
        typer.secho("Validation successful. Agent appears to be configured correctly.", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Validation failed with {errors} critical error(s).", fg=typer.colors.RED)
        raise typer.Exit(code=1)

eval_app = typer.Typer(name="evaluate", help="Commands for evaluating agent performance.")
app.add_typer(eval_app)
@eval_app.command("generate-test", help="Generates a test set of Q&A pairs from an agent's sources.")
def generate_test_command(
        agent_name: str = typer.Argument(..., help="The name of the source agent to build the test set from."),
        output_file: str = typer.Option(
            "test_set.json",
            "--output",
            "-o",
            help="The path to save the generated JSON test set file."
        ),
        num_questions: int = typer.Option(50, "--num-questions", "-n", help="The number of questions to generate.")
):
    """
    Uses a local LLM to automatically create a test set for evaluating RAG quality.
    """
    typer.echo(f"--- Generating Test Set for Agent: '{agent_name}' ---")
    try:
        agent_config = load_agent_config(agent_name)

        qa_pairs = generate_test_set(agent_config, num_questions)

        if qa_pairs:
            with open(output_file, 'w') as f:
                json.dump(qa_pairs, f, indent=2)
            typer.secho(f"\nSuccessfully generated {len(qa_pairs)} Q&A pairs and saved to '{output_file}'",
                        fg=typer.colors.GREEN)
        else:
            typer.secho("Could not generate any Q&A pairs. Please check the agent's data sources.", fg=typer.colors.RED)

    except Exception as e:
        typer.secho(f"An unexpected error occurred during test set generation: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@eval_app.command("benchmark", help="Runs a retrieval quality benchmark on an agent.")
def benchmark_command(
        agent_name: str = typer.Argument(..., help="The name of the agent to benchmark."),
        test_set_file: str = typer.Option(
            ...,
            "--test-set",
            "-t",
            help="Path to a JSON file containing the ground truth test set."
        )
):
    """
    Uses a ground truth test set to calculate objective retrieval metrics like
    Key Fact Recall and Contextual Noise.
    """
    typer.echo(f"--- Running Benchmark for Agent: '{agent_name}' ---")

    try:
        config_path = os.path.join(AGENTS_DIR, f"{agent_name}.yaml")
        if not os.path.exists(config_path):
            typer.secho(f"Error: Configuration file not found at {config_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        agent_config = load_agent_from_yaml_file(config_path)

        if not os.path.exists(test_set_file):
            typer.secho(f"Error: Test set file not found at '{test_set_file}'", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        with open(test_set_file, 'r') as f:
            test_set = json.load(f)

        results_df = run_benchmark(agent_config, test_set)

        if not results_df.empty:
            recall_score = results_df["key_fact_recalled"].mean()
            avg_noise = results_df[results_df["contextual_noise"] != -1]["contextual_noise"].mean()

            typer.secho("\n--- Benchmark Complete ---", bold=True)
            typer.echo("\nOverall Scores:")
            typer.secho(f"  - Key Fact Recall: {recall_score:.2%}", fg=typer.colors.GREEN)
            typer.secho(f"  - Average Context Size (Noise): {avg_noise:.2f} docs", fg=typer.colors.YELLOW)

            typer.echo("\n(Key Fact Recall measures if the correct answer was in the retrieved documents.)")
            typer.echo("(Lower Context Size is better, as it indicates less noise.)")

            if typer.confirm("\nShow detailed results for each question?", default=False):
                typer.echo(results_df.to_string())
        else:
            typer.secho("Benchmark failed to produce results.", fg=typer.colors.RED)

    except Exception as e:
        typer.secho(f"An unexpected error occurred during benchmark: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()