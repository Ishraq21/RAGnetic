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
import secrets # CRITICAL FIX: ADDED MISSING IMPORT FOR SECRETS

from langchain_community.vectorstores import FAISS, Chroma # For local vector stores
from langchain_qdrant import Qdrant # For Qdrant
from langchain_pinecone import Pinecone as PineconeLangChain # For Pinecone
from langchain_mongodb import MongoDBAtlasVectorSearch # For MongoDB Atlas
from pinecone import Pinecone as PineconeClient # Pinecone client for initialization

from app.core.embed_config import get_embedding_model # For dynamically loading the correct embedding model
from langchain_core.documents import Document as LangChainDocument # For type hinting if needed for clarity
from app.agents.config_manager import load_agent_config, load_agent_from_yaml_file # Ensure these are correctly imported
from pathlib import Path # For path manipulation

from sqlalchemy import create_engine, text # For DB connection checks
from sqlalchemy.exc import SQLAlchemyError # For DB exception handling
import requests # For URL/API connection checks
from urllib.parse import urlparse # For URL parsing
from app.schemas.agent import DataSource # For DataSource type hinting in check_connections
from google.oauth2 import service_account # For GDrive credentials check


# Initialize a default logger for CLI startup FIRST, so it's always bound.
logger = logging.getLogger(__name__)

from app.core.config import get_path_settings, get_api_key, get_server_api_keys
from app.core.structured_logging import JSONFormatter
from app.evaluation.dataset_generator import generate_test_set
from app.evaluation.benchmark import run_benchmark
# from app.agents.config_manager import load_agent_config, load_agent_from_yaml_file # Already imported above
from app.pipelines.embed import embed_agent_data
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
_CONFIG_FILE = _APP_PATHS["CONFIG_FILE_PATH"]
_RAGNETIC_DIR = _APP_PATHS["RAGNETIC_DIR"]
_TEMP_CLONES_DIR = _APP_PATHS["TEMP_CLONES_DIR"]


def _validate_agent_name_cli(agent_name: str):
    if not is_valid_agent_name_cli(agent_name):
        error_message = (
            f"Error: Invalid agent_name '{agent_name}'. Name must be 3-50 characters "
            "and can only contain letters, numbers, underscores, and hyphens."
        )
        typer.secho(error_message, fg=typer.colors.RED)
        raise typer.Exit(code=1)


def setup_logging(json_format: bool = False):
    """Configures the root logger for either human-readable or JSON output with rotation."""
    if not os.path.exists(_LOGS_DIR):
        os.makedirs(_LOGS_DIR, mode=0o750, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    is_production = os.environ.get("ENVIRONMENT", "development").lower() == "production"

    if is_production:
        root_logger.setLevel(logging.INFO)
        logger.info("Logging configured for PRODUCTION environment.")
    else:
        root_logger.setLevel(logging.INFO) # Keep INFO for general CLI output
        logger.info("Logging configured for DEVELOPMENT environment (minimal DEBUG on console).")

    console_handler = logging.StreamHandler()
    if is_production:
        console_handler.setLevel(logging.WARNING)
    else:
        console_handler.setLevel(logging.INFO) # Keep INFO for general CLI output

    if json_format:
        formatter = JSONFormatter()
        console_handler.setFormatter(formatter)
        log_file_base_name = "ragnetic_app.jsonl"
        file_handler_path = os.path.join(_LOGS_DIR, log_file_base_name)
        file_handler = TimedRotatingFileHandler(
            file_handler_path, when="midnight", interval=1, backupCount=7, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        if is_production:
            file_handler.setLevel(logging.WARNING)
        else:
            file_handler.setLevel(logging.INFO) # Keep INFO for general CLI output
        root_logger.addHandler(file_handler)
    else:
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)

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

auth_app = typer.Typer(name="auth", help="Manage authentication for external services like Google Drive.")
app.add_typer(auth_app)

eval_app = typer.Typer(name="evaluate", help="Commands for evaluating agent performance.")
app.add_typer(eval_app)

MODEL_PROVIDERS = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google (Gemini)": "GOOGLE_API_KEY",
    "Pinecone": "PINECONE_API_KEY",
    "MongoDB Atlas": "MONGODB_CONN_STRING",
    "Hugging Face": None,
    "Ollama (Local LLMs)": None,
    "Brave Search": "BRAVE_SEARCH_API_KEY"
}


@app.command(help="Initialize a new RAGnetic project in the current directory.")
def init():
    setup_logging()
    logger.info("Initializing new RAGnetic project...")
    folders_to_create = [
        _AGENTS_DIR, _DATA_DIR, _VECTORSTORE_DIR, _MEMORY_DIR,
        _LOGS_DIR, _RAGNETIC_DIR, _TEMP_CLONES_DIR
    ]
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
        config['AUTH'] = {
            '# A comma-separated list of secret keys to protect the RAGnetic server API.': '',
            '# Use the `ragnetic set-server-key` command to generate a secure key.': '',
            'server_api_keys': ''
        }
        with open(_CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        logger.info(f"  - Created config file at: {_CONFIG_FILE}")
        typer.secho("\nProject initialized. To secure your server, run:", fg=typer.colors.CYAN)
        typer.secho("  ragnetic set-server-key", bold=True)
    else:
        logger.info("Project already initialized.")


@app.command(name="set-server-key", help="Generate and set a new secret key to protect the server API.")
def set_server_key():
    """
    Generates a new secure key and saves it to config.ini.
    """
    setup_logging()
    config = configparser.ConfigParser()
    config.read(_CONFIG_FILE)
    if 'AUTH' not in config:
        config['AUTH'] = {}

    new_key = secrets.token_hex(32)
    config['AUTH']['server_api_keys'] = new_key

    with open(_CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

    typer.secho("Successfully set a new server API key.", fg=typer.colors.GREEN)
    typer.echo("Your new key is:")
    typer.secho(f"  {new_key}", bold=True)
    typer.echo("\nUse this key in the 'X-API-Key' header for all API requests.")


@app.command(name="set-api-key", help="Set and save API keys for external services (e.g., OpenAI).")
def set_api():
    setup_logging()
    typer.secho("--- External Service API Key Configuration ---", bold=True)
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
        json_logs: bool = typer.Option(False, "--json-logs", help="Output logs in structured JSON format."),
):
    setup_logging(json_logs)
    if not get_server_api_keys():
        typer.secho("--- SECURITY WARNING ---", fg=typer.colors.YELLOW, bold=True)
        typer.secho("The server is starting without an API key. The API will be open to anyone who can access it.",
                    fg=typer.colors.YELLOW)
        typer.secho("For production or shared environments, please secure your server by running:",
                    fg=typer.colors.YELLOW)
        # Updated command prompt
        typer.secho("  ragnetic set-server-key\n", bold=True)

    if reload:
        logger.warning("Running in --reload mode. The file watcher will be disabled.")
        uvicorn.run("app.main:app", host=host, port=port, reload=True)
        return

    logger.info(f"Starting RAGnetic server on http://{host}:{port}")
    try:
        get_api_key("openai")
    except ValueError as e:
        logger.warning(f"{e} - some models may not work.")

    uvicorn.run("app.main:app", host=host, port=port)


@app.command(help="Lists all configured agents.")
def list_agents():
    setup_logging()
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

        vector_store_created = asyncio.run(embed_agent_data(agent_config))

        typer.secho("\nAgent deployment successful!", fg=typer.colors.GREEN)
        if vector_store_created:
            typer.echo(f"  - Vector store created at: {vectorstore_path}")
        else:
            typer.echo("  - No vector store created (no sources provided or no valid documents found).")

    except Exception as e:
        logger.error(f"An unexpected error occurred during deployment: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command(name='inspect-agent', help="Displays the configuration of a specific agent.")
def inspect_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to inspect."),
        show_documents_metadata: bool = typer.Option(
            False, "--metadata", "-m",
            help="Display detailed metadata for ingested documents."
        ),
        check_connections: bool = typer.Option(
            False, "--check-connections", "-c",
            help="Verify connectivity for each configured external source."
        ),

        num_docs: int = typer.Option(
            5, "--num-docs",
            help="Number of sample documents to retrieve and display from the vector store."
        ),
):
    _validate_agent_name_cli(agent_name)
    setup_logging()

    errors = 0

    typer.echo(f"Inspecting configuration for agent: '{agent_name}'")

    # 1) Validate YAML config
    try:
        agent_config = load_agent_config(agent_name)
        typer.echo(yaml.dump(agent_config.model_dump(), indent=2, sort_keys=False))
        typer.secho("  - [PASS] YAML configuration is valid.", fg=typer.colors.GREEN)
    except FileNotFoundError:
        typer.secho(f"Error: Agent '{agent_name}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"  - [FAIL] Could not load or parse YAML config: {e}", fg=typer.colors.RED, err=True)
        errors += 1

    # 2) Optionally inspect vector-store documents
    if show_documents_metadata:
        typer.secho(f"\n--- Inspecting Document Metadata for '{agent_name}' ---", bold=True)
        vectorstore_path = _VECTORSTORE_DIR / agent_name

        if not os.path.exists(vectorstore_path):
            typer.secho(
                f"Error: Vector store not found at {vectorstore_path}. Please deploy the agent first.",
                fg=typer.colors.RED
            )
            errors += 1
        else:
            async def _inspect_vector_store_async(k_value: int):
                # load embeddings
                embeddings = get_embedding_model(agent_config.embedding_model)
                typer.echo(f"Loading vector store: {agent_config.vector_store.type}")

                # choose the right loader
                cls_map = {
                    'faiss': FAISS,
                    'chroma': Chroma,
                    'qdrant': Qdrant,
                    'pinecone': PineconeLangChain,
                    'mongodb_atlas': MongoDBAtlasVectorSearch,
                }
                db_type = agent_config.vector_store.type
                if db_type not in cls_map:
                    typer.secho(f"Unsupported vector store type '{db_type}'", fg=typer.colors.RED)
                    return False

                # handle external API keys
                if db_type in ('pinecone', 'mongodb_atlas', 'qdrant'):
                    key_map = {'pinecone': 'pinecone', 'mongodb_atlas': 'mongodb', 'qdrant': 'qdrant'}
                    key = get_api_key(key_map[db_type])
                    if not key:
                        typer.secho(f"Missing API key for {db_type}", fg=typer.colors.YELLOW)
                        return False
                    if db_type == 'pinecone':
                        PineconeClient(api_key=key)

                # load the store
                if db_type == 'faiss':
                    db = await asyncio.to_thread(
                        FAISS.load_local, str(vectorstore_path), embeddings, allow_dangerous_deserialization=True
                    )
                elif db_type == 'chroma':
                    db = await asyncio.to_thread(Chroma, persist_directory=str(vectorstore_path),
                                                 embedding_function=embeddings)
                elif db_type == 'qdrant':
                    cfg = agent_config.vector_store
                    db = await asyncio.to_thread(
                        Qdrant, client=None, collection_name=agent_name,
                        embeddings=embeddings, host=cfg.qdrant_host, port=cfg.qdrant_port, prefer_grpc=True
                    )
                elif db_type == 'pinecone':
                    idx = agent_config.vector_store.pinecone_index_name
                    db = await asyncio.to_thread(PineconeLangChain.from_existing_index, index_name=idx,
                                                 embedding=embeddings)
                else:  # mongodb_atlas
                    vs = agent_config.vector_store
                    db = await asyncio.to_thread(
                        MongoDBAtlasVectorSearch.from_connection_string,
                        get_api_key("mongodb"),
                        vs.mongodb_db_name,
                        vs.mongodb_collection_name,
                        embeddings,
                        vs.mongodb_index_name
                    )

                if not db:
                    typer.secho("Failed to initialize vector store.", fg=typer.colors.RED)
                    return False

                typer.echo("Vector store loaded successfully.")
                typer.echo("\nRetrieving sample entries…")
                docs = await asyncio.to_thread(db.similarity_search_with_score, "document", k=k_value)
                if docs:
                    for i, (doc, score) in enumerate(docs, 1):
                        typer.secho(f"\n--- Entry {i} (Distance Score: {score:.4f} | Lower is More Relevant) ---",
                                    fg=typer.colors.CYAN, bold=True)
                        typer.echo(f"{doc.page_content[:400]}…")
                        typer.secho("Metadata:", fg=typer.colors.BLUE)
                        typer.echo(json.dumps(doc.metadata, indent=2))
                        typer.secho("-" * 60, fg=typer.colors.MAGENTA)
                else:
                    typer.secho("No entries retrieved; store may be empty.", fg=typer.colors.YELLOW)
                return True

            try:
                success = asyncio.run(_inspect_vector_store_async(k_value=num_docs))

                if not success:
                    errors += 1
            except Exception as e:
                typer.secho(f"Error during metadata inspection: {e}", fg=typer.colors.RED)
                errors += 1

    # 3) Optionally check external connections
    if check_connections:
        typer.secho(f"\n--- Performing Connection Checks for '{agent_name}' ---", bold=True)
        if not agent_config.sources:
            typer.secho("No data sources configured.", fg=typer.colors.YELLOW)
        else:
            for idx, source in enumerate(agent_config.sources, 1):
                info = f"Source {idx} ({source.type})"
                status = "[UNKNOWN]"
                try:
                    if source.type == "db" and source.db_connection:
                        connect_args = {}
                        conn_str = source.db_connection

                        if conn_str.startswith("postgresql"):
                            connect_args['connect_timeout'] = 5
                        elif conn_str.startswith("mysql"):
                            # The mysql-connector-python driver uses 'connect_timeout'
                            # Other MySQL drivers might use 'connection_timeout'
                            connect_args['connect_timeout'] = 5
                        elif not conn_str.startswith("sqlite"):
                            # Generic fallback for other network-based DBs
                            connect_args['timeout'] = 5
                        # SQLite does not need a network timeout

                        eng = create_engine(conn_str, connect_args=connect_args)
                        with eng.connect() as conn:
                            conn.execute(text("SELECT 1"))
                        status = "[PASS]"
                        eng.dispose()
                    elif source.type in ("url", "api") and source.url:
                        r = requests.head(source.url, timeout=5)
                        r.raise_for_status()
                        status = "[PASS]"
                    elif source.type == "gdoc":
                        cfg = configparser.ConfigParser()
                        cfg.read(_CONFIG_FILE)
                        if cfg.has_section("GOOGLE_CREDENTIALS") and cfg.get("GOOGLE_CREDENTIALS", "json_key",
                                                                             fallback=None):
                            status = "[PASS] GDrive creds"
                        else:
                            status = "[FAIL] Missing GDrive creds"
                            errors += 1
                    else:
                        status = "[SKIP]"
                except Exception as e:
                    status = f"[FAIL] {e}"
                    errors += 1

                typer.echo(f"  - {info}: {status}")

    typer.secho("\n" + "-" * 20, fg=typer.colors.WHITE)
    if errors == 0:
        typer.secho("Validation successful.", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Validation failed with {errors} error(s).", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="reset-agent", help="Resets an agent by deleting its learned data.")
def reset_agent(
        agent_name: str = typer.Argument(..., help="The name of the agent to reset."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    _validate_agent_name_cli(agent_name)
    setup_logging()
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
    setup_logging()
    vectorstore_path = _VECTORSTORE_DIR / agent_name
    memory_pattern = _MEMORY_DIR / f"{agent_name}_*.db"
    config_path = _AGENTS_DIR / f"{agent_name}.yaml"
    typer.secho(f"DANGER: This will permanently delete agent '{agent_name}' and all its data:", fg=typer.colors.RED)
    if not force:
        typer.confirm("This action is irreversible. Are you absolutely sure?", abort=True)
    try:
        if os.path.exists(vectorstore_path): shutil.rmtree(vectorstore_path)
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
    setup_logging()
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
    setup_logging()
    typer.echo("Running the RAGnetic test suite...")
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
        show_detailed_results: Optional[bool] = typer.Option(
            None, "--show-detailed-results", "-s",
            help="Explicitly show/hide detailed results in console (overrides prompt)."
        ),
):
    _validate_agent_name_cli(agent_name)
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

            should_show_console_results = False
            if show_detailed_results is True:
                should_show_console_results = True
            elif show_detailed_results is False:
                should_show_console_results = False
            else:
                should_show_console_results = typer.confirm("\nShow detailed results in console?", default=False,
                                                            abort=False)

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