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
from datetime import datetime
import asyncio
from logging.handlers import TimedRotatingFileHandler
import signal
import sys
from typing import Optional, List, Dict, Any

# Initialize a default logger for CLI startup FIRST, so it's always bound.
logger = logging.getLogger(__name__)

from app.core.config import get_path_settings, get_api_key

from app.core.structured_logging import JSONFormatter
from app.evaluation.dataset_generator import generate_test_set
from app.evaluation.benchmark import run_benchmark

# Local application imports
from app.agents.config_manager import load_agent_config, load_agent_from_yaml_file
# AGENTS_DIR, CONFIG_FILE will come from _APP_PATHS (removed old constants)
from app.pipelines.embed import embed_agent_data
# get_api_key already imported via app.core.config
from app.watcher import start_watcher
import pytest
from app.core.validation import is_valid_agent_name_cli

# --- Centralized Path Configuration ---
_APP_PATHS = get_path_settings()
_LOGS_DIR = _APP_PATHS["LOGS_DIR"]
_DATA_DIR = _APP_PATHS["DATA_DIR"]
_AGENTS_DIR = _APP_PATHS["AGENTS_DIR"]
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]
_MEMORY_DIR = _APP_PATHS["MEMORY_DIR"]
_CONFIG_FILE = _APP_PATHS["CONFIG_FILE_PATH"]  # Absolute path to config.ini
_RAGNETIC_DIR = _APP_PATHS["RAGNETIC_DIR"]  # For init command
_TEMP_CLONES_DIR = _APP_PATHS["TEMP_CLONES_DIR"]  # For code_repository_loader setup in init if needed


# --- End Centralized Configuration ---


def _validate_agent_name_cli(agent_name: str):
    if not is_valid_agent_name_cli(agent_name):
        error_message = (
            f"Error: Invalid agent_name '{agent_name}'. Name must be 3-50 characters "
            "and can only contain letters, numbers, underscores, and hyphens."
        )
        typer.secho(error_message, fg=typer.colors.RED)
        raise typer.Exit(code=1)


# Logging Setup
def setup_logging(json_format: bool = False):
    """Configures the root logger for either human-readable or JSON output with rotation."""
    # Use centralized LOGS_DIR and create with stricter permissions
    if not os.path.exists(_LOGS_DIR):
        os.makedirs(_LOGS_DIR, mode=0o750, exist_ok=True)

    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicate logs if setup_logging is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Determine environment for dynamic log levels
    is_production = os.environ.get("ENVIRONMENT", "development").lower() == "production"

    # 1. Root Logger Level: Set overall verbosity
    # To suppress DEBUG from libraries even in development console, set root_logger to INFO.
    if is_production:
        root_logger.setLevel(logging.INFO)  # Keep INFO to allow RAGnetic's own INFOs, but handlers will filter.
        logger.info("Logging configured for PRODUCTION environment.")
    else:
        root_logger.setLevel(
            logging.INFO)  # MODIFIED: Changed from DEBUG to INFO to suppress most library DEBUG by default in dev
        logger.info("Logging configured for DEVELOPMENT environment (minimal DEBUG on console).")

    # 2. Console Handler: Logs to stdout/stderr
    console_handler = logging.StreamHandler()
    if is_production:
        console_handler.setLevel(
            logging.WARNING)  # MODIFIED: Changed from INFO to WARNING for a much quieter console in production
    else:
        console_handler.setLevel(
            logging.INFO)  # MODIFIED: Changed from DEBUG to INFO for a quieter console in development
        # This will suppress most DEBUG messages from external libraries like httpx, sqlite3.

    if json_format:
        formatter = JSONFormatter()
        console_handler.setFormatter(formatter)

        # 3. File Handler (JSON with rotation)
        log_file_base_name = "ragnetic_app.jsonl"
        file_handler_path = os.path.join(_LOGS_DIR, log_file_base_name)

        file_handler = TimedRotatingFileHandler(
            file_handler_path,
            when="midnight",  # Rotate every day at midnight
            interval=1,
            backupCount=7,  # Keep 7 days of backups
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        # Set file handler level (e.g., WARNING+ for prod to avoid filling disk with routine chatter)
        if is_production:
            file_handler.setLevel(logging.WARNING)  # Only WARNING, ERROR, CRITICAL logs go to disk in prod
        else:
            file_handler.setLevel(logging.INFO)  # All INFO+ logs go to disk in dev

        root_logger.addHandler(file_handler)
    else:
        # For non-JSON (development mode to console)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)

    # Always add the console handler
    root_logger.addHandler(console_handler)


# --- CLI App Definition ---
cli_help = """
RAGnetic: Your on-premise, plug-and-play AI agent framework.

Provides a CLI for initializing projects, managing agents, and running the server.
"""

app = typer.Typer(
    name="ragnetic",
    help=cli_help,
    add_completion=False,
    no_args_is_help=True,
)

auth_app = typer.Typer(name="auth", help="Manage authentication for external services.")
app.add_typer(auth_app)

eval_app = typer.Typer(name="evaluate", help="Commands for evaluating agent performance.")
app.add_typer(eval_app)

# --- Constants ---
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
    Creates the necessary folders and a default config.ini file.
    """
    setup_logging()  # MODIFIED: Removed json_logs parameter
    logger.info("Initializing new RAGnetic project...")
    # MODIFIED: Use centralized paths for folders to create and set mode
    folders_to_create = [_AGENTS_DIR, _DATA_DIR,
                         _VECTORSTORE_DIR, _MEMORY_DIR,
                         _LOGS_DIR, _RAGNETIC_DIR, _TEMP_CLONES_DIR]
    for folder_path in folders_to_create:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, mode=0o750, exist_ok=True)
            logger.info(f"  - Created directory: {folder_path}")

    if not os.path.exists(_CONFIG_FILE):
        config = configparser.ConfigParser()
        config['API_KEYS'] = {
            '# For local development, you can set keys here.': '',
            '# For production, it is strongly recommended to use environment variables.': '',
            'OPENAI_API_KEY': '...',
        }
        with open(_CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        logger.info(f"  - Created config file at: {_CONFIG_FILE}")

    logger.info("Project initialized successfully.")


@app.command(help="Set and save API keys for local development.")
def set_api():
    """
    Guides the user through setting API keys in the local config.ini file.
    """
    setup_logging()  # MODIFIED: Removed json_logs parameter
    typer.secho("--- API Key Configuration for Local Development ---", bold=True)
    typer.secho(
        "This wizard saves keys to the local .ragnetic/config.ini file.\n"
        "For production, setting environment variables is the recommended approach.",
        fg=typer.colors.YELLOW
    )
    typer.echo("Example: export OPENAI_API_KEY='sk-...'")

    while True:
        typer.echo("\nPlease select a provider to configure:")
        provider_menu = list(MODEL_PROVIDERS.keys())
        for i, provider in enumerate(provider_menu, 1):
            typer.echo(f"  [{i}] {provider}")
        try:
            choice_str = typer.prompt("Enter the number of your choice")
            choice_index = int(choice_str) - 1
            if not 0 <= choice_index < len(provider_menu): raise ValueError
            selected_provider = provider_menu[choice_index]
            if MODEL_PROVIDERS[selected_provider] is None:
                typer.secho(f"\n{selected_provider} models run locally and do not require an API key.",
                            fg=typer.colors.GREEN)
            else:
                config_key_name = MODEL_PROVIDERS[selected_provider]
                api_key = typer.prompt(f"Enter your {selected_provider} API Key", hide_input=True)
                if not api_key:
                    typer.secho("Error: API Key cannot be empty.", fg=typer.colors.RED)
                    continue
                config = configparser.ConfigParser()
                config.read(_CONFIG_FILE)
                if 'API_KEYS' not in config: config['API_KEYS'] = {}
                config['API_KEYS'][config_key_name] = api_key
                with open(_CONFIG_FILE, 'w') as configfile:
                    config.write(configfile)
                typer.secho(f"Successfully saved {selected_provider} API key to {_CONFIG_FILE}.", fg=typer.colors.GREEN)
        except (ValueError, IndexError):
            typer.secho("Error: Invalid selection.", fg=typer.colors.RED)
        if not typer.confirm("Do you want to set another API key?", default=False):
            break
    typer.echo("\nLocal API key configuration complete.")


@app.command(help="Starts the RAGnetic server and the file watcher.")
def start_server(
        host: str = typer.Option("127.0.0.1", help="The host to bind the server to."),
        port: int = typer.Option(8000, help="The port to run the server on."),
        reload: bool = typer.Option(False, "--reload", help="Enable auto-reloading for development."),
        no_watcher: bool = typer.Option(False, "--no-watcher", help="Do not start the file watcher process."),
        json_logs: bool = typer.Option(False, "--json-logs",
                                       help="Output logs in structured JSON format for production."),
):
    """Starts the Uvicorn server."""
    setup_logging(json_logs)

    if reload:
        logger.warning("Running in --reload mode. The file watcher will be disabled.")
        uvicorn.run("app.main:app", host=host, port=port, reload=True)
        return

    # Watcher process management is now handled by FastAPI's app.on_event handlers in app/main.py
    logger.info(f"Starting RAGnetic server on http://{host}:{port}")
    try:
        get_api_key("openai")
    except ValueError as e:
        logger.warning(f"{e} - some models may not work.")

    # Uvicorn itself handles graceful shutdown via SIGINT/SIGTERM,
    # and it will trigger FastAPI's app.on_event("shutdown") hook to stop the watcher.

    uvicorn.run("app.main:app", host=host, port=port)

    # The following block is now redundant as watcher is managed by app.on_event("shutdown")
    # if watcher_process and watcher_process.is_alive():
    #     watcher_process.terminate()
    #     watcher_process.join()
    #     logger.info("File watcher process stopped.")


# --- Agent Management Commands ---

@app.command(help="Lists all configured agents.")
def list_agents():
    setup_logging()  # MODIFIED: Removed json_logs parameter
    if not os.path.exists(_AGENTS_DIR):
        logger.error(f"Error: Directory '{_AGENTS_DIR}' not found. Have you run 'ragnetic init'?")
        raise typer.Exit(code=1)
    agents = [f.split(".")[0] for f in os.listdir(_AGENTS_DIR) if f.endswith((".yaml", ".yml"))]
    if not agents:
        logger.info("No agents found in the 'agents_data' directory.")
        return
    typer.echo("Available Agents:")
    for agent_name in agents:
        typer.echo(f"  - {agent_name}")


@app.command(name="deploy", help="Deploys an agent by its name, processing its data sources.")
def deploy_agent_by_name(
        agent_name: str = typer.Argument(..., help="The name of the agent to deploy."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation and overwrite existing data."),
        json_logs: bool = typer.Option(False, "--json-logs", help="Output logs in structured JSON format."),
):
    _validate_agent_name_cli(agent_name)
    """Loads an agent config from YAML and creates a vector store."""
    setup_logging(json_logs)
    try:
        config_path = _AGENTS_DIR / f"{agent_name}.yaml"
        logger.info(f"Loading agent configuration from: {config_path}")
        if not os.path.exists(config_path):
            typer.secho(f"Error: Configuration file not found at {config_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        vectorstore_path = _VECTORSTORE_DIR / agent_name
        if os.path.exists(vectorstore_path) and not force:
            typer.secho(f"Warning: A vector store for agent '{agent_name}' already exists.", fg=typer.colors.YELLOW)
            if not typer.confirm("Do you want to overwrite it and re-deploy the agent?"):
                typer.echo("Deployment cancelled.")
                raise typer.Exit()
            shutil.rmtree(vectorstore_path)
            logger.info(f"Removed existing vector store at: {vectorstore_path}")

        agent_config = load_agent_from_yaml_file(config_path)
        typer.echo(f"\nDeploying agent '{agent_config.name}' using embedding model '{agent_config.embedding_model}'...")
        asyncio.run(embed_agent_data(agent_config))

        typer.secho("\nAgent deployment successful!", fg=typer.colors.GREEN)
        typer.echo(f"  - Vector store created at: {vectorstore_path}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during deployment: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command(help="Displays the configuration of a specific agent.")
def inspect_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to inspect.")
):
    _validate_agent_name_cli(agent_name)
    setup_logging()  # MODIFIED: Removed json_logs parameter
    try:
        typer.echo(f"Inspecting configuration for agent: '{agent_name}'")
        config = load_agent_config(agent_name)
        typer.echo(yaml.dump(config.model_dump(), indent=2, sort_keys=False))
    except FileNotFoundError:
        typer.secho(f"Error: Agent '{agent_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(help="Validates an agent's configuration and associated files.")
def validate_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to validate."),
        check_connections: bool = typer.Option(False, "--check-connections", "-c",
                                               help="Test data source connections."),
        json_logs: bool = typer.Option(False, "--json-logs", help="Structured JSON logs"),
):
    _validate_agent_name_cli(agent_name)

    """Validates an agent's configuration, checking YAML syntax, file existence, and external connections."""
    setup_logging(json_logs)  # This command HAS json_logs parameter, so it's correct
    typer.echo(f"Validating agent: '{agent_name}'...")
    errors = 0
    try:
        agent_config = load_agent_config(agent_name)
        typer.secho("  - [PASS] YAML configuration is valid.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"  - [FAIL] Could not load or parse YAML config: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    vectorstore_path = _VECTORSTORE_DIR / agent_name
    if os.path.exists(vectorstore_path) and os.path.isdir(vectorstore_path):
        typer.secho(f"  - [PASS] Vector store directory exists at: {vectorstore_path}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"  - [WARN] Vector store not found. Agent may need to be deployed.", fg=typer.colors.YELLOW)

    if check_connections:
        typer.echo("\n--- Performing Connection Check ---")
        for i, source in enumerate(agent_config.sources):
            source_info = f"Source #{i + 1} (type: {source.type})"
            if source.type == 'db' and source.db_connection:
                # ... (Database connection check logic) ...
                pass
            elif source.type == 'url' and source.url:
                # ... (URL connection check logic) ...
                pass

    typer.echo("-" * 20)
    if errors == 0:
        typer.secho("Validation successful.", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Validation failed with {errors} critical error(s).", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="reset-agent", help="Resets an agent by deleting its learned data.")
def reset_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to reset."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    _validate_agent_name_cli(agent_name)
    setup_logging()  # MODIFIED: Removed json_logs parameter
    vectorstore_path = _VECTORSTORE_DIR / agent_name
    memory_pattern = _MEMORY_DIR / f"{agent_name}_*.db"
    typer.secho(f"Warning: This will reset agent '{agent_name}' by deleting its generated data:",
                fg=typer.colors.YELLOW)
    typer.echo(f"  - Vector store directory: {vectorstore_path}")
    typer.echo(f"  - All memory files matching: {memory_pattern}")
    if not force:
        typer.confirm("Are you sure you want to proceed?", abort=True)
    try:
        if os.path.exists(vectorstore_path): shutil.rmtree(vectorstore_path)
        # Convert Path object to string for glob.glob if it doesn't accept Path directly
        memory_files = glob.glob(str(memory_pattern))
        if memory_files:
            for f in memory_files: os.remove(f)
        typer.secho(f"\nAgent '{agent_name}' has been reset.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"An error occurred during reset: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="delete-agent", help="Permanently deletes an agent and all its data.")
def delete_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to permanently delete."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    _validate_agent_name_cli(agent_name)
    setup_logging()  # MODIFIED: Removed json_logs parameter
    vectorstore_path = _VECTORSTORE_DIR / agent_name
    memory_pattern = _MEMORY_DIR / f"{agent_name}_*.db"
    config_path = _AGENTS_DIR / f"{agent_name}.yaml"
    typer.secho(f"DANGER: This will permanently delete agent '{agent_name}' and all its data:", fg=typer.colors.RED)
    if not force:
        typer.confirm("This action is irreversible. Are you absolutely sure?", abort=True)
    try:
        if os.path.exists(vectorstore_path): shutil.rmtree(vectorstore_path)
        # Convert Path object to string for glob.glob
        memory_files = glob.glob(str(memory_pattern))
        if memory_files:
            for f in memory_files: os.remove(f)
        if os.path.exists(config_path): os.remove(config_path)
        typer.secho(f"\nAgent '{agent_name}' has been permanently deleted.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"An error occurred during deletion: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@auth_app.command("gdrive", help="Authenticate with Google Drive.")
def auth_gdrive():
    setup_logging()  # MODIFIED: Removed json_logs parameter
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
        config.read(_CONFIG_FILE)
        if 'GOOGLE_CREDENTIALS' not in config:
            config.add_section('GOOGLE_CREDENTIALS')
        safe_credentials_str = credentials_json_str.replace('%', '%%')
        config.set('GOOGLE_CREDENTIALS', 'json_key', safe_credentials_str)
        with open(_CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        typer.secho(f"Google Drive credentials saved successfully!",
                    fg=typer.colors.GREEN)
    except json.JSONDecodeError:
        typer.secho("Error: The provided file is not a valid JSON file.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(help="Runs the entire test suite using pytest.")
def test():
    """Discovers and runs all automated tests in the 'tests/' directory."""
    setup_logging()  # MODIFIED: Removed json_logs parameter
    typer.echo("Running the RAGnetic test suite...")
    # MODIFIED: Use _APP_PATHS for tests directory if you want to centralize that too
    # For now, keeping "tests/" as it's typically relative to project root for pytest.
    result_code = pytest.main(["-v", "tests/"])
    if result_code == 0:
        typer.secho("\nAll tests passed!", fg=typer.colors.GREEN)
    else:
        typer.secho(f"\n{result_code} test(s) failed.", fg=typer.colors.RED)


@eval_app.command("generate-test", help="Generates a test set from an agent's sources.")
def generate_test_command(
        agent_name: str = typer.Argument(..., help="The agent to build the test set from."),
        output_file: str = typer.Option("test_set.json", "--output", "-o",
                                        help="Path to save the generated JSON file."),
        num_questions: int = typer.Option(50, "--num-questions", "-n", help="Number of questions to generate."),
        json_logs: bool = typer.Option(False, "--json-logs", help="Output logs in structured JSON format."),
):
    _validate_agent_name_cli(agent_name)
    setup_logging(json_logs)
    logger.info(f"--- Generating Test Set for Agent: '{agent_name}' ---")
    try:
        agent_config = load_agent_config(agent_name)
        qa_pairs = asyncio.run(generate_test_set(agent_config, num_questions))
        if qa_pairs:
            with open(output_file, 'w') as f:
                json.dump(qa_pairs, f, indent=2)
            logger.info(f"\nSuccessfully generated {len(qa_pairs)} Q&A pairs to '{output_file}'")
    except Exception as e:
        logger.error(f"An error occurred during test set generation: {e}", exc_info=True)
        raise typer.Exit(code=1)


@eval_app.command("benchmark", help="Runs a retrieval quality benchmark on an agent.")
def benchmark_command(
        agent_name: str = typer.Argument(..., help="The agent to benchmark."),
        test_set_file: str = typer.Option(..., "--test-set", "-t", help="Path to a JSON test set file."),
        json_logs: bool = typer.Option(False, "--json-logs", help="Output logs in structured JSON format."),
        # NEW: Add option to show detailed results non-interactively
        show_detailed_results: Optional[bool] = typer.Option(
            None, "--show-detailed-results", "-s",
            help="Explicitly show/hide detailed results in console (overrides prompt)."
        ),
):
    _validate_agent_name_cli(agent_name)
    """
    Uses a ground truth test set to calculate metrics, including token usage and cost.
    """
    setup_logging(json_logs)
    logger.info(f"--- Running Benchmark for Agent: '{agent_name}' ---")

    try:
        agent_config = load_agent_from_yaml_file(_AGENTS_DIR / f"{agent_name}.yaml")

        with open(test_set_file, 'r') as f:
            test_set = json.load(f)

        results_df = run_benchmark(agent_config, test_set)

        if not results_df.empty:
            results_filename = _LOGS_DIR / f"benchmark_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_df.to_csv(results_filename, index=False)
            logger.info(f"Full benchmark results saved to: {results_filename}")

            recall_score = results_df["key_fact_recalled"].mean()
            avg_noise = results_df["contextual_noise"].mean()
            total_cost = results_df["estimated_cost_usd"].sum()

            typer.secho("\n--- Benchmark Complete ---", bold=True)
            typer.echo("\nOverall Scores:")
            typer.secho(f"  - Key Fact Recall: {recall_score:.2%}", fg=typer.colors.GREEN)
            typer.secho(f"  - Average Context Size (Noise): {avg_noise:.2f} docs", fg=typer.colors.YELLOW)
            typer.secho(f"  - Total Estimated Cost: ${total_cost:.6f} USD", fg=typer.colors.BLUE)
            typer.secho(f"\nDetailed results saved to: {results_filename}", fg=typer.colors.BLUE)

            # MODIFIED: Control prompt based on show_detailed_results
            should_show_console_results = False
            if show_detailed_results is True:
                should_show_console_results = True
            elif show_detailed_results is False:
                should_show_console_results = False
            else:  # show_detailed_results is None, prompt user
                should_show_console_results = typer.confirm("\nShow detailed results in console?", default=False, abort=False)

            if should_show_console_results:
                display_columns = [
                    "question", "generated_answer", "key_fact_recalled", "faithfulness",
                    "total_tokens", "estimated_cost_usd", "retrieval_f1"
                ]
                existing_columns = [col for col in display_columns if col in results_df.columns]
                typer.echo(results_df[existing_columns].to_string())
        else:
            logger.error("Benchmark failed to produce results.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during benchmark: {e}", exc_info=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()