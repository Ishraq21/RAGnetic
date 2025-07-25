import typer
import uvicorn
import os
import shutil
import yaml
import glob
import configparser
import logging
import logging.config
from multiprocessing import Process
import json
import pandas as pd
from datetime import datetime
import asyncio
from typing import Optional, List, Dict, Any
import secrets
from pathlib import Path
from dotenv import load_dotenv
import alembic.config
import alembic.command
from alembic.runtime.migration import MigrationContext
import subprocess

# IMPORTS for inspect_agent dynamic vector store loading
from langchain_community.vectorstores import FAISS, Chroma
from langchain_qdrant import Qdrant
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_mongodb import MongoDBAtlasVectorSearch
from pinecone import Pinecone as PineconeClient

from app.core.embed_config import get_embedding_model
from langchain_core.documents import Document as LangChainDocument
from app.agents.config_manager import load_agent_config, load_agent_from_yaml_file

# IMPORTS for connection checks
from sqlalchemy import create_engine, text, select
from sqlalchemy.engine.url import make_url

from sqlalchemy.exc import SQLAlchemyError
import requests
from urllib.parse import urlparse
from app.schemas.agent import DataSource
from google.oauth2 import service_account

# Import core components from the application
from app.core.config import get_path_settings, get_api_key, get_server_api_keys, get_log_storage_config, \
    get_memory_storage_config, get_db_connection, get_cors_settings
from app.evaluation.dataset_generator import generate_test_set
from app.evaluation.benchmark import run_benchmark
from app.pipelines.embed import embed_agent_data
from app.watcher import start_watcher
import pytest
from app.core.validation import is_valid_agent_name_cli
from app.db.models import metadata, agent_runs, chat_sessions_table, users_table
from app.db.models import metadata, agent_runs, chat_sessions_table, users_table, agent_run_steps, workflows_table, workflow_runs_table
from app.workflows.engine import WorkflowEngine
from app.schemas.workflow import Workflow
import subprocess

# --- Centralized Path Configuration ---
_APP_PATHS = get_path_settings()
_PROJECT_ROOT = _APP_PATHS["PROJECT_ROOT"]
_LOGS_DIR = _APP_PATHS["LOGS_DIR"]
_DATA_DIR = _APP_PATHS["DATA_DIR"]
_AGENTS_DIR = _APP_PATHS["AGENTS_DIR"]
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]
_MEMORY_DIR = _APP_PATHS["MEMORY_DIR"]
_CONFIG_FILE = _APP_PATHS["CONFIG_FILE_PATH"]
_RAGNETIC_DIR = _APP_PATHS["RAGNETIC_DIR"]
_TEMP_CLONES_DIR = _APP_PATHS["TEMP_CLONES_DIR"]
_BENCHMARK_DIR = _APP_PATHS["BENCHMARK_DIR"]
_WORKFLOWS_DIR = _APP_PATHS["WORKFLOWS_DIR"]
# _SKILLS_DIR = _APP_PATHS["SKILLS_DIR"]

# --- Load Environment Variables ---
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

# --- Custom SUCCESS Log Level ---
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kws)

logging.Logger.success = success

class TyperLogger(logging.Handler):
    """A custom logging handler that uses typer.secho for colored output."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.level_styles = {
            logging.INFO: {"fg": typer.colors.BLUE},
            SUCCESS_LEVEL: {"fg": typer.colors.GREEN, "bold": True},
            logging.WARNING: {"fg": typer.colors.YELLOW},
            logging.ERROR: {"fg": typer.colors.RED},
            logging.CRITICAL: {"fg": typer.colors.RED, "bold": True},
        }
        self.level_prefixes = {
            logging.INFO: "INFO     ",
            SUCCESS_LEVEL: "SUCCESS  ",
            logging.WARNING: "WARNING  ",
            logging.ERROR: "ERROR    ",
            logging.CRITICAL: "CRITICAL ",
        }

    def emit(self, record):
        try:
            msg = self.format(record)
            style = self.level_styles.get(record.levelno, {})
            prefix = self.level_prefixes.get(record.levelno, "")
            if "\n" in msg:
                lines = msg.splitlines()
                first_line = prefix + lines[0]
                indented_lines = [f"   {line}" for line in lines[1:]]
                final_msg = "\n".join([first_line] + indented_lines)
            else:
                final_msg = prefix + msg
            typer.secho(final_msg, **style)
        except Exception:
            self.handleError(record)


def setup_cli_logging():
    """Configures logging for clear, colored CLI command feedback."""

    # This works whether the script is run directly or as an installed package,
    # removing the dependency on a hardcoded package name.
    handler_class_path = f"{TyperLogger.__module__}.{TyperLogger.__name__}"

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'cli_formatter': {
                'format': '%(message)s'
            }
        },
        'handlers': {
            'typer_handler': {
                'class': handler_class_path,  # Use the new, dynamic path
                'formatter': 'cli_formatter',
                'level': 'INFO'
            }
        },
        'loggers': {
            'ragnetic': {'handlers': ['typer_handler'], 'level': 'INFO', 'propagate': False},
            '__main__': {'handlers': ['typer_handler'], 'level': 'INFO', 'propagate': False},
            'app': {'handlers': ['typer_handler'], 'level': 'INFO', 'propagate': False},
            'alembic': {'handlers': ['typer_handler'], 'level': 'WARNING', 'propagate': False},
            'sqlalchemy': {'handlers': ['typer_handler'], 'level': 'WARNING', 'propagate': False},
            'langchain': {'handlers': ['typer_handler'], 'level': 'WARNING', 'propagate': False},
            'langchain_core': {'handlers': ['typer_handler'], 'level': 'WARNING', 'propagate': False},
            'langchain_community': {'handlers': ['typer_handler'], 'level': 'WARNING', 'propagate': False},
            'uvicorn': {'handlers': ['typer_handler'], 'level': 'INFO', 'propagate': False},
            'uvicorn.access': {'handlers': ['typer_handler'], 'level': 'WARNING', 'propagate': False},
            'celery': {'handlers': ['typer_handler'], 'level': 'INFO', 'propagate': False},
            'celery.app': {'handlers': ['typer_handler'], 'level': 'WARNING', 'propagate': False},
        },
        'root': {
            'handlers': ['typer_handler'],
            'level': 'INFO'
        }
    }
    logging.config.dictConfig(logging_config)

# --- Initialize Logging and Typer App ---
logger = logging.getLogger(__name__)


def _update_env_file(env_vars: Dict[str, str]):
    """
    Reads an existing .env file, updates or adds the specified
    environment variables, and writes it back securely.
    """
    env_file_path = _PROJECT_ROOT / ".env"
    env_lines = []
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r') as f:
            env_lines = f.readlines()

    for key, value in env_vars.items():
        formatted_value = f"'{value}'" if any(c in value for c in ' #=') else value
        found = False
        for i, line in enumerate(env_lines):
            if line.strip().startswith(f"{key}="):
                env_lines[i] = f"{key}={formatted_value}\n"
                found = True
                break
        if not found:
            if env_lines and not env_lines[-1].endswith('\n'):
                env_lines.append('\n')
            env_lines.append(f"{key}={formatted_value}\n")

    with open(env_file_path, 'w') as f:
        f.writelines(env_lines)


def _validate_agent_name_cli(agent_name: str):
    if not is_valid_agent_name_cli(agent_name):
        error_message = (
            f"Error: Invalid agent_name '{agent_name}'. Name must be 3-50 characters "
            "and can only contain letters, numbers, underscores, and hyphens."
        )
        typer.secho(error_message, fg=typer.colors.RED)
        raise typer.Exit(code=1)




def _get_sync_db_engine():
    """Helper to get a synchronous SQLAlchemy engine."""
    mem_cfg = get_memory_storage_config()
    log_cfg = get_log_storage_config()
    conn_name = mem_cfg.get("connection_name") if mem_cfg.get("type") in ["db", "sqlite"] else log_cfg.get(
        "connection_name")
    if not conn_name:
        typer.secho("Error: No database connection is configured in config.ini.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    conn_str = get_db_connection(conn_name)
    sync_conn_str = conn_str.replace('+aiosqlite', '').replace('+asyncpg', '')
    return create_engine(sync_conn_str)


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

run_app = typer.Typer(name="run", help="Commands for running workflows and other processes.")
app.add_typer(run_app)


@run_app.command(name="workflow", help="Triggers a workflow to run via the API.")
def run_workflow_cli(
        workflow_name: str = typer.Argument(..., help="The name of the workflow to run."),
        initial_input_json: Optional[str] = typer.Option(
            None, "--input", "-i", help="Initial JSON input for the workflow."
        ),
):
    """Triggers a workflow run by calling the local API endpoint."""
    # Read server host and port from config file
    config = configparser.ConfigParser()
    config.read(_CONFIG_FILE)
    host = config.get('SERVER', 'host', fallback='127.0.0.1')
    port = config.get('SERVER', 'port', fallback='8000')

    url = f"http://{host}:{port}/api/v1/workflows/{workflow_name}/trigger"
    headers = {"Content-Type": "application/json"}

    initial_input = {}
    if initial_input_json:
        try:
            initial_input = json.loads(initial_input_json)
        except json.JSONDecodeError:
            typer.secho("Error: Invalid JSON input.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    try:
        typer.secho(f"Attempting to trigger workflow '{workflow_name}' via API...", fg=typer.colors.CYAN)
        response = requests.post(url, headers=headers, json=initial_input, timeout=10)
        response.raise_for_status()

        # Check for accepted status
        if response.status_code == 202:
            typer.secho(f"Successfully triggered workflow '{workflow_name}'.", fg=typer.colors.GREEN)
            typer.secho(f"API response: {response.json()}", fg=typer.colors.BRIGHT_WHITE)
        else:
            typer.secho(f"Workflow trigger failed with status code {response.status_code}.", fg=typer.colors.RED)
            typer.echo(f"Response: {response.text}")
            raise typer.Exit(code=1)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error: Could not connect to the RAGnetic server at {url}", fg=typer.colors.RED)
        typer.echo(f"Please ensure the server is running with 'ragnetic start-server'.")
        typer.echo(f"Detailed error: {e}")
        raise typer.Exit(code=1)

MODEL_PROVIDERS = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google (Gemini)": "GOOGLE_API_KEY",
    "Pinecone": "PINECONE_API_KEY",
    "MongoDB Atlas": "MONGODB_CONN_STRING",
    "Brave Search": "BRAVE_SEARCH_API_KEY",
    "Hugging Face": None,
    "Ollama (Local LLMs)": None,
}


def _is_db_configured() -> bool:
    """Checks if a database is configured for either memory or logging."""
    mem_config = get_memory_storage_config()
    log_config = get_log_storage_config()
    return (mem_config.get("type") in ["db", "sqlite"] and mem_config.get("connection_name")) or \
        (log_config.get("type") == "db" and log_config.get("connection_name"))


@app.command(name="reset-db", help="DANGEROUS: Drops all tables from the database.")
def reset_db(
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    """
    Connects to the database and drops all known application tables
    and the Alembic versioning table to ensure a clean slate.
    """
    typer.secho("--- DANGEROUS: Database Reset ---", fg=typer.colors.RED, bold=True)

    if not force:
        typer.secho("This will permanently delete all data from the configured database.", fg=typer.colors.YELLOW)
        if not typer.confirm("Are you absolutely sure you want to proceed?", default=False):
            typer.echo("Operation cancelled.")
            raise typer.Exit()

    try:
        mem_cfg = get_memory_storage_config()
        log_cfg = get_log_storage_config()
        conn_name = mem_cfg.get("connection_name") if mem_cfg.get("type") in ["db", "sqlite"] else log_cfg.get(
            "connection_name")

        if not conn_name:
            typer.secho("Error: No database connection is configured in config.ini.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        conn_str = get_db_connection(conn_name)
        typer.echo(f"Connecting to database via connection '{conn_name}'...")

        conn_str_sync = conn_str.replace('+aiosqlite', '').replace('+asyncpg', '')
        engine = create_engine(conn_str_sync)

        with engine.connect() as connection:
            typer.echo("Dropping all application tables defined in the metadata...")
            metadata.drop_all(engine, checkfirst=True)
            typer.secho("Application tables dropped successfully.", fg=typer.colors.GREEN)

            typer.echo("Attempting to drop the 'alembic_version' table...")
            try:
                with connection.begin():
                    connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
                typer.secho("'alembic_version' table dropped successfully.", fg=typer.colors.GREEN)
            except Exception as e:
                typer.secho(f"Could not drop 'alembic_version' table (this is likely okay): {e}",
                            fg=typer.colors.YELLOW)

            typer.secho("\nDatabase reset complete.", fg=typer.colors.GREEN, bold=True)

    except Exception as e:
        typer.secho(f"An error occurred during database reset: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command(name="makemigrations", help="Autogenerate a new database migration script.")
def makemigrations(
        message: str = typer.Option(..., "-m", "--message", help="A descriptive message for the new migration."),
):
    """Wraps Alembic's 'revision --autogenerate' command."""
    if not _is_db_configured():
        typer.secho("No explicit database configured. Using default SQLite for migrations.", fg=typer.colors.YELLOW)

    typer.echo(f"Generating new migration script with message: '{message}'...")

    alembic_ini_path = str(_PROJECT_ROOT / "alembic.ini")
    command = [
        "alembic", "-c", alembic_ini_path, "revision", "--autogenerate", "-m", message
    ]

    typer.secho(f"Running command: {' '.join(command)}", fg=typer.colors.YELLOW)
    result = subprocess.run(command, capture_output=True, text=True, cwd=_PROJECT_ROOT)

    if result.returncode == 0:
        typer.echo(result.stdout)
        typer.secho("Migration script generated successfully.", fg=typer.colors.GREEN)
    else:
        typer.secho("Error generating migration:", fg=typer.colors.RED)
        typer.echo("--- STDOUT ---")
        typer.echo(result.stdout)
        typer.echo("--- STDERR ---")
        typer.echo(result.stderr)
        raise typer.Exit(code=1)


@app.command(name="migrate", help="Applies database migrations.")
def migrate(
        revision: str = typer.Argument("head", help="The revision to migrate to (e.g., 'head' for latest).")
):
    """Wraps Alembic's 'upgrade' command."""
    if not _is_db_configured():
        typer.secho("No explicit database configured. Using default SQLite for migrations.", fg=typer.colors.YELLOW)

    typer.echo(f"Applying database migrations to revision: {revision}...")
    alembic_cfg = alembic.config.Config(str(_PROJECT_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("loglevel", "INFO")
    alembic.command.upgrade(alembic_cfg, revision)
    typer.secho("Database migration complete.", fg=typer.colors.GREEN)


@app.command(name="sync", help="Manually stamps the database with a specific migration revision.")
def sync_db_revision(
        revision: str = typer.Argument("head", help="The revision to stamp the database with.")
):
    """Wraps Alembic's 'stamp' command. Use with extreme caution!"""
    if not _is_db_configured():
        typer.secho("No explicit database configured. Using default SQLite for sync.", fg=typer.colors.YELLOW)

    typer.secho("\n--- WARNING: Using 'ragnetic sync' (alembic stamp) ---", fg=typer.colors.RED, bold=True)
    typer.secho("This command updates the database's migration history WITHOUT running any SQL.", fg=typer.colors.RED)
    if not typer.confirm(f"Are you absolutely sure you want to stamp the database to revision '{revision}'?",
                         default=False):
        typer.secho("Synchronization cancelled.", fg=typer.colors.YELLOW)
        raise typer.Exit()

    typer.echo(f"Stamping database to revision: {revision}...")
    alembic_cfg = alembic.config.Config(str(_PROJECT_ROOT / "alembic.ini"))
    alembic.command.stamp(alembic_cfg, revision)
    typer.secho(f"Database successfully stamped to revision '{revision}'.", fg=typer.colors.GREEN)


@app.command(help="Configure system settings, databases, and secrets.")
def configure():
    """
    An intelligent configuration wizard that prevents duplicate entries,
    promotes reuse, and helps clean up unused database connections.
    """
    typer.secho("--- RAGnetic System Configuration ---", bold=True)
    config = configparser.ConfigParser()
    config.read(_CONFIG_FILE)

    # --- SERVER SETTINGS ---
    if typer.confirm("\nDo you want to configure SERVER settings (host, port, etc.)?", default=True):
        if 'SERVER' not in config: config.add_section('SERVER')
        host = typer.prompt("Enter server host", default=config.get('SERVER', 'host', fallback='127.0.0.1'))
        port = typer.prompt("Enter server port", default=config.get('SERVER', 'port', fallback='8000'))
        config.set('SERVER', 'host', host)
        config.set('SERVER', 'port', str(port))

        ws_mode = typer.prompt("Enter WebSocket mode (memory, redis)",
                               default=config.get('SERVER', 'websocket_mode', fallback='memory'))
        while ws_mode not in ['memory', 'redis']:
            typer.secho("Invalid selection. Please choose 'memory' or 'redis'.", fg=typer.colors.RED)
            ws_mode = typer.prompt("Enter WebSocket mode (memory, redis)",
                                   default=config.get('SERVER', 'websocket_mode', fallback='memory'))
        config.set('SERVER', 'websocket_mode', ws_mode)

        if typer.confirm("Configure CORS allowed origins?", default=False):
            current_origins = os.environ.get("CORS_ALLOWED_ORIGINS",
                                             config.get('SERVER', 'cors_allowed_origins', fallback='*'))
            origins_str = typer.prompt("Enter comma-separated origins", default=current_origins)
            if typer.confirm("Save CORS settings to the local .env file (recommended)?", default=True):
                _update_env_file({"CORS_ALLOWED_ORIGINS": origins_str})
                if config.has_option('SERVER', 'cors_allowed_origins'): config.remove_option('SERVER',
                                                                                             'cors_allowed_origins')
                typer.secho("CORS settings saved to .env file.", fg=typer.colors.GREEN)
            else:
                config.set('SERVER', 'cors_allowed_origins', origins_str)
        typer.secho("Server settings updated.", fg=typer.colors.GREEN)

    # --- DATABASE CONNECTIONS ---
    if typer.confirm("\nDo you want to configure SYSTEM database connections?", default=True):
        if 'DATABASE_CONNECTIONS' not in config: config.add_section('DATABASE_CONNECTIONS')
        DIALECT_MAP = {"postgresql": "postgresql+psycopg2", "mysql": "mysql+mysqlconnector", "sqlite": "sqlite"}

        while True:
            existing_conns = [c.strip() for c in config.get('DATABASE_CONNECTIONS', 'names', fallback='').split(',') if
                              c.strip()]
            if existing_conns:
                typer.echo(f"Existing connections: {', '.join(existing_conns)}")

            if not typer.confirm("Add or update a database connection?", default=True):
                break

            conn_name = typer.prompt("Enter a unique name for this connection (e.g., 'prod_db')")

            existing_match = next((c for c in existing_conns if c.lower() == conn_name.lower()), None)
            if existing_match and not typer.confirm(f"Connection '{existing_match}' already exists. Overwrite it?",
                                                    default=False):
                continue

            typer.secho(f"\n--- Configuring '{conn_name}' ---", bold=True)
            db_type = typer.prompt(f"Database type? Choose from: {list(DIALECT_MAP.keys())}", default="sqlite")
            while db_type not in DIALECT_MAP:
                typer.secho("Invalid selection.", fg=typer.colors.RED)
                db_type = typer.prompt(f"Database type? Choose from: {list(DIALECT_MAP.keys())}", default="sqlite")

            section_name = f"DATABASE_{conn_name}"
            if db_type == "sqlite":
                db_path = typer.prompt("Enter path for the SQLite file", default="memory/ragnetic.db")
                config[section_name] = {'dialect': DIALECT_MAP[db_type], 'database_path': db_path}
            else:
                username = typer.prompt("Username")
                host = typer.prompt("Host", default="localhost")
                port = typer.prompt("Port", default="5432" if db_type == "postgresql" else "3306")
                database = typer.prompt("Database Name")
                password = typer.prompt(f"Enter password for user '{username}'", hide_input=True)
                config[section_name] = {'dialect': DIALECT_MAP[db_type], 'username': username, 'host': host,
                                        'port': port, 'database': database}
                if password:
                    _update_env_file({f"{conn_name.upper()}_PASSWORD": password})
                    typer.secho("Password saved to the .env file.", fg=typer.colors.GREEN)

            if not existing_match:
                existing_conns.append(conn_name)
                config.set('DATABASE_CONNECTIONS', 'names', ','.join(sorted(existing_conns)))
            typer.secho(f"\nConnection '{conn_name}' configured successfully.", fg=typer.colors.GREEN)

        if typer.confirm("\nDo you want to clean up unused database connections?", default=True):
            current_conns = [c.strip() for c in config.get('DATABASE_CONNECTIONS', 'names', fallback='').split(',') if
                             c.strip()]
            conns_to_keep = []
            for conn in current_conns:
                if typer.confirm(f"Keep connection '{conn}'?", default=True):
                    conns_to_keep.append(conn)
                else:
                    if config.has_section(f"DATABASE_{conn}"):
                        config.remove_section(f"DATABASE_{conn}")
                        typer.secho(f"Removed connection '{conn}'.", fg=typer.colors.YELLOW)
            config.set('DATABASE_CONNECTIONS', 'names', ','.join(sorted(conns_to_keep)))

    # --- STORAGE CONFIGURATION (MEMORY & LOGS) ---
    all_connections = [c.strip() for c in config.get('DATABASE_CONNECTIONS', 'names', fallback='').split(',') if
                       c.strip()]

    def configure_storage(storage_name: str, section_name: str, valid_types: List[str]):
        if not typer.confirm(f"\nDo you want to configure {storage_name} storage?", default=True):
            return
        if section_name not in config: config.add_section(section_name)

        storage_type = typer.prompt(f"Enter {storage_name} storage type ({', '.join(valid_types)})",
                                    default=config.get(section_name, 'type', fallback=valid_types[0]))
        config.set(section_name, 'type', storage_type)

        if storage_type == 'file':
            if config.has_option(section_name, 'connection_name'):
                config.remove_option(section_name, 'connection_name')
            if config.has_option(section_name, 'log_table_name'):
                config.remove_option(section_name, 'log_table_name')

        elif storage_type in ['db', 'sqlite'] and all_connections:
            typer.echo("Available database connections:")
            for i, name in enumerate(all_connections, 1):
                typer.echo(f"  [{i}] {name}")

            choice = typer.prompt("Choose a connection to use", type=int, default=1)
            if 1 <= choice <= len(all_connections):
                chosen_conn = all_connections[choice - 1]
                config.set(section_name, 'connection_name', chosen_conn)
                typer.secho(f"{storage_name} storage set to use '{chosen_conn}'.", fg=typer.colors.CYAN)
            else:
                typer.secho("Invalid choice.", fg=typer.colors.RED)
        elif storage_type == 'db':
            typer.secho("No database connections configured yet. Please add one first.", fg=typer.colors.YELLOW)

    configure_storage("MEMORY (chat history)", "MEMORY_STORAGE", ["sqlite", "db"])
    configure_storage("LOG", "LOG_STORAGE", ["file", "db"])

    if config.has_section('LOG_STORAGE') and config.get('LOG_STORAGE', 'type') == 'db':
        table_name = typer.prompt("Enter the table name for logs",
                                  default=config.get('LOG_STORAGE', 'log_table_name', fallback='ragnetic_logs'))
        config.set('LOG_STORAGE', 'log_table_name', table_name)

    with open(_CONFIG_FILE, 'w') as configfile:
        config.write(configfile)
    typer.secho("\nConfiguration saved successfully to .ragnetic/config.ini", fg=typer.colors.GREEN)


@app.command(name="show-config", help="Displays the current system configurations.")
def show_config():
    typer.secho("--- Current RAGnetic Configuration ---", bold=True)
    if not os.path.exists(_CONFIG_FILE):
        typer.secho(f"Configuration file not found at: {_CONFIG_FILE}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    config = configparser.ConfigParser()
    config.read(_CONFIG_FILE)
    for section in config.sections():
        typer.secho(f"\n[{section}]", fg=typer.colors.CYAN, bold=True)
        for key, value in config.items(section):
            typer.echo(f"  {key} = {value}")
    typer.echo("")


@app.command(name="check-system-db", help="Verifies and inspects connections to configured system databases.")
def check_system_db():
    typer.secho("--- Checking System Database Connections ---", bold=True)
    log_config = get_log_storage_config()
    memory_config = get_memory_storage_config()
    connections_to_check = {}

    if log_config.get("type") == "db" and log_config.get("connection_name"):
        conn_name = log_config["connection_name"]
        connections_to_check[conn_name] = connections_to_check.get(conn_name, []) + ["Logging"]
    if memory_config.get("type") in ["db", "sqlite"] and memory_config.get("connection_name"):
        conn_name = memory_config["connection_name"]
        connections_to_check[conn_name] = connections_to_check.get(conn_name, []) + ["Memory"]

    if not connections_to_check:
        typer.secho("No system database connections are configured.", fg=typer.colors.YELLOW)
        raise typer.Exit()

    has_failure = False
    alembic_cfg = alembic.config.Config(str(_PROJECT_ROOT / "alembic.ini"))
    script = alembic.script.ScriptDirectory.from_config(alembic_cfg)

    for conn_name, purposes in connections_to_check.items():
        typer.secho(f"\nInspecting '{conn_name}' (used for: {', '.join(purposes)})", fg=typer.colors.CYAN, bold=True)
        try:
            conn_str = get_db_connection(conn_name)
            engine_args = {'echo': False}
            if 'sqlite' in conn_str:
                conn_str = conn_str.replace("+aiosqlite", "")
            else:
                engine_args['connect_args'] = {'connect_timeout': 5}
            engine = create_engine(conn_str, **engine_args)

            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                typer.secho("  - Connectivity: [PASS]", fg=typer.colors.GREEN)

                if connection.dialect.name != 'sqlite':
                    db_version = connection.dialect.get_server_version_info(connection)
                    typer.echo(f"  - DB Type: {connection.dialect.name}")
                    typer.echo(f"  - DB Version: {'.'.join(map(str, db_version))}")
                    parsed_url = urlparse(conn_str)
                    if parsed_url.hostname:
                        typer.echo(f"  - Host: {parsed_url.hostname}")
                        typer.echo(f"  - Port: {parsed_url.port}")
                        typer.echo(f"  - Database: {parsed_url.path.lstrip('/')}")
                else:
                    typer.echo(f"  - DB Type: {connection.dialect.name}")
                    typer.echo(f"  - Path: {urlparse(conn_str).path}")

                migration_context = MigrationContext.configure(connection)
                current_rev = migration_context.get_current_revision()
                head_rev = script.get_current_head()
                typer.echo(f"  - Current DB Revision: {current_rev}")
                typer.echo(f"  - Latest Code Revision: {head_rev}")

                if current_rev == head_rev:
                    typer.secho("  - Migration Status: [UP-TO-DATE]", fg=typer.colors.GREEN)
                elif not current_rev:
                    typer.secho("  - Migration Status: [NEEDS MIGRATION] - Database is empty.", fg=typer.colors.YELLOW)
                else:
                    typer.secho("  - Migration Status: [NEEDS MIGRATION] - Run 'ragnetic migrate'.",
                                fg=typer.colors.YELLOW)
            engine.dispose()

        except Exception as e:
            typer.secho(f"  - Connectivity: [FAIL]", fg=typer.colors.RED)
            typer.secho(f"    Error: {e}", fg=typer.colors.RED)
            has_failure = True

    typer.secho("\n--- Check Complete ---", bold=True)
    if has_failure:
        raise typer.Exit(code=1)


@app.command(help="Initialize a new RAGnetic project.")
def init():
    typer.secho("Initializing new RAGnetic project...", bold=True)
    paths_to_create = {
        "DATA_DIR", "AGENTS_DIR", "VECTORSTORE_DIR", "MEMORY_DIR",
        "LOGS_DIR", "TEMP_CLONES_DIR", "RAGNETIC_DIR", "BENCHMARK_DIR", "WORKFLOWS_DIR"
    }
    for key, path in _APP_PATHS.items():
        if key in paths_to_create and not os.path.exists(path):
            os.makedirs(path, mode=0o750, exist_ok=True)
            typer.echo(f"  - Created directory: {path}")

    if not os.path.exists(_CONFIG_FILE):
        config = configparser.ConfigParser()
        config['SERVER'] = {
            'host': '127.0.0.1',
            'port': '8000',
            'cors_allowed_origins': '*',
            'websocket_mode': 'memory'
        }
        default_db_path = _MEMORY_DIR / "ragnetic.db"
        config['DATABASE_CONNECTIONS'] = {'names': 'default_sqlite'}
        config['DATABASE_default_sqlite'] = {
            'dialect': 'sqlite+aiosqlite',
            'database_path': str(default_db_path.relative_to(_PROJECT_ROOT))
        }
        config['MEMORY_STORAGE'] = {'type': 'sqlite', 'connection_name': 'default_sqlite'}
        with open(_CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        typer.echo(f"  - Created config file: {_CONFIG_FILE}")

    typer.secho("\nProject initialized successfully!", fg=typer.colors.GREEN)
    typer.secho("\n--- SECURITY NOTICE: CORS ---", fg=typer.colors.YELLOW, bold=True)
    typer.secho("By default, RAGnetic allows requests from all origins ('*').", fg=typer.colors.YELLOW)
    typer.secho("For production, you should restrict this to your frontend's domain.", fg=typer.colors.YELLOW)
    typer.echo("You can change this using: " + typer.style("ragnetic configure", bold=True))
    typer.secho("\nNext steps:", bold=True)
    typer.echo("  1. Set your API keys: " + typer.style("ragnetic set-api-key", bold=True))
    typer.echo("  2. Configure a different database (optional): " + typer.style("ragnetic configure", bold=True))
    typer.echo("  3. Secure your server: " + typer.style("ragnetic set-server-key", bold=True))


@app.command(name="set-server-key", help="Generate and set a secret key for the server API.")
def set_server_key():
    new_key = secrets.token_hex(32)
    _update_env_file({"RAGNETIC_API_KEYS": new_key})
    typer.secho("Successfully set a new server API key in the .env file.", fg=typer.colors.GREEN)
    typer.echo("Your new key is: " + typer.style(new_key, bold=True))


@app.command(name="set-api-key", help="Set and save API keys to the secure .env file.")
def set_api():
    typer.secho("--- External Service API Key Configuration ---", bold=True)
    typer.secho("This wizard saves keys securely to the project's local .env file.", fg=typer.colors.CYAN)
    while True:
        typer.echo("\nPlease select a provider to configure:")
        for i, provider in enumerate(MODEL_PROVIDERS.keys(), 1):
            typer.echo(f"  [{i}] {provider}")
        try:
            choice_index = int(typer.prompt("Enter the number of your choice")) - 1
            selected_provider = list(MODEL_PROVIDERS.keys())[choice_index]
            env_key_name = MODEL_PROVIDERS[selected_provider]

            if not env_key_name:
                typer.secho(f"\n{selected_provider} models run locally and do not require an API key.",
                            fg=typer.colors.GREEN)
                continue

            api_key = typer.prompt(f"Enter your {selected_provider} API Key", hide_input=True)
            if not api_key:
                typer.secho("Error: API Key cannot be empty.", fg=typer.colors.RED)
                continue

            _update_env_file({env_key_name: api_key})
            typer.secho(f"Successfully saved {selected_provider} API key to the .env file.", fg=typer.colors.GREEN)

        except (ValueError, IndexError):
            typer.secho("Error: Invalid selection.", fg=typer.colors.RED)

        if not typer.confirm("Do you want to set another API key?", default=False):
            break
    typer.echo("\nAPI key configuration complete.")


@auth_app.command("gdrive", help="Authenticate with Google Drive securely.")
def auth_gdrive():
    typer.secho("--- Google Drive Authentication Setup ---", bold=True)
    json_path_str = typer.prompt("Path to your service account JSON key file")
    json_path = Path(json_path_str)

    if not json_path.is_file() or not json_path.name.endswith('.json'):
        typer.secho(f"Error: File not found or not a .json file at '{json_path_str}'", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        creds_file_path = _RAGNETIC_DIR / "gdrive_credentials.json"
        shutil.copy(json_path, creds_file_path)
        if os.name == 'posix':
            os.chmod(creds_file_path, 0o600)

        _update_env_file({"GOOGLE_APPLICATION_CREDENTIALS": str(creds_file_path.resolve())})
        typer.secho(f"\nCredentials file secured at: {creds_file_path}", fg=typer.colors.GREEN)
        typer.secho("Authentication setup is complete.", bold=True)

    except Exception as e:
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(help="Starts the RAGnetic server and background worker.")
def start_server(
        host: str = typer.Option(None, help="Server host. Overrides config."),
        port: int = typer.Option(None, help="Server port. Overrides config."),
        reload: bool = typer.Option(False, "--reload", help="Enable auto-reloading for development."),
):
    """
    Starts the RAGnetic server (Uvicorn) and the Celery background worker.
    """
    config = configparser.ConfigParser()
    config.read(_CONFIG_FILE)
    final_host = host or config.get('SERVER', 'host', fallback="127.0.0.1")
    final_port = port or config.getint('SERVER', 'port', fallback=8000)

    # --- Warnings ---
    if not get_server_api_keys():
        typer.secho("SECURITY WARNING: Server starting without an API key.", fg=typer.colors.YELLOW, bold=True)
    if get_cors_settings() == ["*"]:
        typer.secho("SECURITY WARNING: Server is allowing requests from all origins ('*').", fg=typer.colors.YELLOW,
                    bold=True)

    if reload:
        # For development, use reloaders for both processes
        typer.secho("Starting server and worker in --reload mode...", fg=typer.colors.YELLOW, bold=True)

        uvicorn_cmd = ["uvicorn", "app.main:app", "--host", final_host, "--port", str(final_port), "--reload"]

        # Use 'watchmedo' (from watchdog) to auto-restart the Celery worker on file changes
        celery_cmd = [
            "watchmedo", "auto-restart",
            "--directory=./app",  # Watch the 'app' directory
            "--pattern=*.py",  # For any Python file change
            "--recursive",  # Include subdirectories
            "--",  # Separator for the command to run
            "celery", "-A", "app.workflows.tasks", "worker",
        ]

        uvicorn_process = subprocess.Popen(uvicorn_cmd)
        celery_process = subprocess.Popen(celery_cmd)

        try:
            uvicorn_process.wait()
        except KeyboardInterrupt:
            typer.echo("\nShutting down...")
        finally:
            celery_process.terminate()
            celery_process.wait()

    else:
        # For production, run them as standard background/foreground processes
        worker_process = None
        try:
            typer.secho("Starting Celery worker...", fg=typer.colors.BLUE, bold=True)
            worker_process = subprocess.Popen(["celery", "-A", "app.workflows.tasks", "worker", "--loglevel=info"])

            typer.secho(f"Starting Uvicorn server on http://{final_host}:{final_port}...", fg=typer.colors.BLUE,
                        bold=True)
            subprocess.run(["uvicorn", "app.main:app", "--host", final_host, "--port", str(final_port)], check=True)

        except KeyboardInterrupt:
            typer.echo("\nShutting down processes...")
        except Exception as e:
            typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        finally:
            if worker_process:
                worker_process.terminate()
                worker_process.wait()

    typer.secho("Shutdown complete.", fg=typer.colors.GREEN)


@app.command(help="Lists all configured agents.")
def list_agents():
    if not os.path.exists(_AGENTS_DIR):
        logging.error(f"Error: Directory '{_AGENTS_DIR}' not found. Have you run 'ragnetic init'?")
        raise typer.Exit(code=1)
    agents = [f.split(".")[0] for f in os.listdir(_AGENTS_DIR) if f.endswith((".yaml", ".yml"))]
    if not agents:
        logging.info("No agents found in the 'agents_data' directory.")
        return
    typer.echo("Available Agents:")
    for agent_name in agents:
        typer.echo(f"  - {agent_name}")


@app.command(name="deploy", help="Deploys an agent by its name, processing its data sources.")
def deploy_agent_by_name(
        agent_name: str = typer.Argument(..., help="The name of the agent to deploy."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation and overwrite existing data."),
):
    _validate_agent_name_cli(agent_name)
    logger = logging.getLogger(__name__)
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
        show_documents_metadata: bool = typer.Option(False, "--metadata", "-m",
                                                     help="Display detailed metadata for ingested documents."),
        check_connections: bool = typer.Option(False, "--check-connections", "-c",
                                               help="Verify connectivity for each configured external source."),
        num_docs: int = typer.Option(5, "--num-docs", help="Number of sample documents to retrieve and display."),
):
    _validate_agent_name_cli(agent_name)
    errors = 0
    typer.echo(f"Inspecting configuration for agent: '{agent_name}'")

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

    if show_documents_metadata:
        typer.secho(f"\n--- Inspecting Document Metadata for '{agent_name}' ---", bold=True)
        vectorstore_path = _VECTORSTORE_DIR / agent_name

        if not os.path.exists(vectorstore_path):
            typer.secho(f"Error: Vector store not found at {vectorstore_path}. Please deploy the agent first.",
                        fg=typer.colors.RED)
            errors += 1
        else:
            async def _inspect_vector_store_async(k_value: int):
                embeddings = get_embedding_model(agent_config.embedding_model)
                typer.echo(f"Loading vector store: {agent_config.vector_store.type}")
                cls_map = {'faiss': FAISS, 'chroma': Chroma, 'qdrant': Qdrant, 'pinecone': PineconeLangChain,
                           'mongodb_atlas': MongoDBAtlasVectorSearch}
                db_type = agent_config.vector_store.type
                if db_type not in cls_map:
                    typer.secho(f"Unsupported vector store type '{db_type}'", fg=typer.colors.RED)
                    return False
                if db_type in ('pinecone', 'mongodb_atlas', 'qdrant'):
                    key_map = {'pinecone': 'pinecone', 'mongodb_atlas': 'mongodb', 'qdrant': 'qdrant'}
                    key = get_api_key(key_map[db_type])
                    if not key:
                        typer.secho(f"Missing API key for {db_type}", fg=typer.colors.YELLOW)
                        return False
                    if db_type == 'pinecone': PineconeClient(api_key=key)
                if db_type == 'faiss':
                    db = await asyncio.to_thread(FAISS.load_local, str(vectorstore_path), embeddings,
                                                 allow_dangerous_deserialization=True)
                elif db_type == 'chroma':
                    db = await asyncio.to_thread(Chroma, persist_directory=str(vectorstore_path),
                                                 embedding_function=embeddings)
                elif db_type == 'qdrant':
                    cfg = agent_config.vector_store
                    db = await asyncio.to_thread(Qdrant, client=None, collection_name=agent_name, embeddings=embeddings,
                                                 host=cfg.qdrant_host, port=cfg.qdrant_port, prefer_grpc=True)
                elif db_type == 'pinecone':
                    idx = agent_config.vector_store.pinecone_index_name
                    db = await asyncio.to_thread(PineconeLangChain.from_existing_index, index_name=idx,
                                                 embedding=embeddings)
                else:
                    vs = agent_config.vector_store
                    db = await asyncio.to_thread(MongoDBAtlasVectorSearch.from_connection_string,
                                                 get_api_key("mongodb"), vs.mongodb_db_name, vs.mongodb_collection_name,
                                                 embeddings, vs.mongodb_index_name)

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
                if not success: errors += 1
            except Exception as e:
                typer.secho(f"Error during metadata inspection: {e}", fg=typer.colors.RED)
                errors += 1

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
                        connect_args = {'connect_timeout': 5} if not source.db_connection.startswith("sqlite") else {}
                        eng = create_engine(source.db_connection, connect_args=connect_args)
                        with eng.connect() as conn:
                            conn.execute(text("SELECT 1"))
                        status = "[PASS]"
                        eng.dispose()
                    elif source.type in ("url", "api") and source.url:
                        requests.head(source.url, timeout=5).raise_for_status()
                        status = "[PASS]"
                    elif source.type == "gdoc":
                        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                            status = "[PASS] GDrive creds found in environment"
                        else:
                            status = "[FAIL] Missing GDrive creds. Run 'ragnetic auth gdrive'."
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


@app.command(help="Runs the entire test suite using pytest.")
def test():
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
):
    _validate_agent_name_cli(agent_name)
    logger = logging.getLogger(__name__)
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
        show_detailed_results: Optional[bool] = typer.Option(None, "--show-detailed-results", "-s",
                                                             help="Explicitly show/hide detailed results in console."),
):
    _validate_agent_name_cli(agent_name)
    logger = logging.getLogger(__name__)
    logger.info(f"--- Running Benchmark for Agent: '{agent_name}' ---")

    try:
        agent_config = load_agent_from_yaml_file(_AGENTS_DIR / f"{agent_name}.yaml")
        with open(test_set_file, 'r') as f:
            test_set = json.load(f)

        results_df = run_benchmark(agent_config, test_set)

        if not results_df.empty:
            results_filename = _BENCHMARK_DIR / f"benchmark_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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

            should_show_console_results = show_detailed_results if show_detailed_results is not None else typer.confirm(
                "\nShow detailed results in console?", default=False, abort=False)

            if should_show_console_results:
                display_columns = ["question", "generated_answer", "key_fact_recalled", "faithfulness", "total_tokens",
                                   "estimated_cost_usd", "retrieval_f1"]
                existing_columns = [col for col in display_columns if col in results_df.columns]
                typer.echo(results_df[existing_columns].to_string())
        else:
            logger.error("Benchmark failed to produce results.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during benchmark: {e}", exc_info=True)
        raise typer.Exit(code=1)


audit_app = typer.Typer(name="audit", help="Commands for inspecting agent audit trails.")
app.add_typer(audit_app)



@audit_app.command("inspect", help="Inspect a specific agent run and its steps.")
def inspect_run(
        run_id: str = typer.Argument(..., help="The unique ID of the run to inspect."),
        details: bool = typer.Option(False, "--details", "-d",
                                     help="Show detailed JSON inputs and outputs for each step."),
):
    """Fetches and displays the details for a single agent run and all of its steps."""
    if not _is_db_configured():
        typer.secho("Audit trails require a database. Please configure one using 'ragnetic configure'.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    engine = _get_sync_db_engine()

    try:
        with engine.connect() as connection:
            # 1. Fetch the main run details
            run_stmt = (
                select(agent_runs, chat_sessions_table.c.agent_name, users_table.c.user_id.label("user_identifier"))
                .join(chat_sessions_table, agent_runs.c.session_id == chat_sessions_table.c.id)
                .join(users_table, chat_sessions_table.c.user_id == users_table.c.id)
                .where(agent_runs.c.run_id == run_id)
            )
            run = connection.execute(run_stmt).first()

            if not run:
                typer.secho(f"Error: Run with ID '{run_id}' not found.", fg=typer.colors.RED)
                raise typer.Exit(code=1)

            # 2. Fetch the steps for that run
            steps_stmt = (
                select(agent_run_steps)
                .where(agent_run_steps.c.agent_run_id == run.id)
                .order_by(agent_run_steps.c.start_time.asc())
            )
            steps = connection.execute(steps_stmt).fetchall()

        # --- Display the results ---
        typer.secho(f"\n--- Audit Trail for Run: {run.run_id} ---", bold=True)

        # Display Run Summary
        status_color = typer.colors.GREEN if run.status == 'completed' else (
            typer.colors.YELLOW if run.status == 'running' else typer.colors.RED)
        duration = (run.end_time - run.start_time).total_seconds() if run.end_time else "N/A"

        typer.echo(f"  {'Status:':<12} {typer.style(run.status, fg=status_color)}")
        typer.echo(f"  {'Agent:':<12} {run.agent_name}")
        typer.echo(f"  {'User ID:':<12} {run.user_identifier}")
        typer.echo(f"  {'Start Time:':<12} {run.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        typer.echo(f"  {'End Time:':<12} {run.end_time.strftime('%Y-%m-%d %H:%M:%S UTC') if run.end_time else 'N/A'}")
        typer.echo(f"  {'Duration:':<12} {duration:.2f}s")

        # Display Steps
        typer.secho("\n--- Steps ---", bold=True)
        if not steps:
            typer.secho("  No steps found for this run.", fg=typer.colors.YELLOW)
        else:
            for i, step in enumerate(steps, 1):
                step_duration = (step.end_time - step.start_time).total_seconds() if step.end_time else "N/A"
                step_status_color = typer.colors.GREEN if step.status == 'completed' else typer.colors.RED
                typer.secho(
                    f"  {i}. Node: {typer.style(step.node_name, bold=True)} ({step_duration:.2f}s) - {typer.style(step.status, fg=step_status_color)}")
                if details:
                    if step.inputs:
                        typer.echo(typer.style("    Inputs:", fg=typer.colors.BLUE))
                        typer.echo(f"      {json.dumps(step.inputs, indent=6)}")
                    if step.outputs:
                        typer.echo(typer.style("    Outputs:", fg=typer.colors.MAGENTA))
                        typer.echo(f"      {json.dumps(step.outputs, indent=6)}")

    except Exception as e:
        typer.secho(f"An error occurred while inspecting the run: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@audit_app.command("runs", help="Lists recent agent runs.")
def list_runs(
        agent_name: Optional[str] = typer.Option(None, "--agent", "-a", help="Filter runs by a specific agent name."),
        limit: int = typer.Option(20, "--limit", "-n", help="Number of recent runs to display."),
):
    """Connects to the database and lists the most recent agent runs."""
    if not _is_db_configured():
        typer.secho("Audit trails require a database. Please configure one using 'ragnetic configure'.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    engine = _get_sync_db_engine()

    stmt = (
        select(
            agent_runs.c.run_id,
            agent_runs.c.status,
            agent_runs.c.start_time,
            chat_sessions_table.c.agent_name,
            chat_sessions_table.c.topic_name,
        )
        .join(chat_sessions_table, agent_runs.c.session_id == chat_sessions_table.c.id)
    )

    # If an agent name is provided, add a filter to the query
    if agent_name:
        stmt = stmt.where(chat_sessions_table.c.agent_name == agent_name)

    # Add ordering and limit to the final statement
    stmt = stmt.order_by(agent_runs.c.start_time.desc()).limit(limit)

    try:
        with engine.connect() as connection:
            results = connection.execute(stmt).fetchall()

        if not results:
            message = "No agent runs found in the database."
            if agent_name:
                message = f"No runs found for agent: '{agent_name}'."
            typer.secho(message, fg=typer.colors.YELLOW)
            return

        title = f"--- Showing Last {len(results)} Agent Runs ---"
        if agent_name:
            title = f"--- Showing Last {len(results)} Runs for Agent: '{agent_name}' ---"
        typer.secho(title, bold=True)

        header = ["Run ID", "Status", "Agent", "Topic", "Start Time (UTC)"]
        rows = []
        for run in results:
            status_color = typer.colors.GREEN if run.status == 'completed' else (
                typer.colors.YELLOW if run.status == 'running' else typer.colors.RED)
            rows.append([
                typer.style(run.run_id, fg=typer.colors.CYAN),
                typer.style(run.status, fg=status_color),
                run.agent_name,
                run.topic_name or "New Chat",
                run.start_time.strftime("%Y-%m-%d %H:%M:%S")
            ])

        raw_rows_for_width_calc = []
        for run in results:
            raw_rows_for_width_calc.append([
                run.run_id,
                run.status,
                run.agent_name,
                run.topic_name or "New Chat",
                run.start_time.strftime("%Y-%m-%d %H:%M:%S")
            ])

        col_widths = [max(len(str(item)) for item in col) for col in zip(*([header] + raw_rows_for_width_calc))]

        typer.echo(" | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths)))
        typer.echo("-|-".join("-" * w for w in col_widths))

        for row in rows:
            # This complex formatting handles the invisible color codes from typer.style
            # to ensure columns align correctly.
            styled_row = [
                f"{row[0]:<{col_widths[0] + len(row[0]) - len(results[rows.index(row)].run_id)}}",
                f"{row[1]:<{col_widths[1] + len(row[1]) - len(results[rows.index(row)].status)}}",
                f"{row[2]:<{col_widths[2]}}",
                f"{row[3]:<{col_widths[3]}}",
                f"{row[4]:<{col_widths[4]}}",
            ]
            typer.echo(" | ".join(styled_row))

    except Exception as e:
        typer.secho(f"An error occurred while fetching agent runs: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@audit_app.command("list-workflows", help="Lists recent workflow runs.")
def list_workflow_runs(
        limit: int = typer.Option(20, "--limit", "-n", help="Number of recent workflow runs to display."),
):
    """Connects to the database and lists the most recent workflow runs."""
    if not _is_db_configured():
        typer.secho("Auditing requires a database. Please configure one using 'ragnetic configure'.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    engine = _get_sync_db_engine()

    # Query the workflow_runs_table and join with workflows_table to get the name
    stmt = (
        select(
            workflow_runs_table.c.run_id,
            workflow_runs_table.c.status,
            workflow_runs_table.c.start_time,
            workflow_runs_table.c.end_time,
            workflows_table.c.name.label("workflow_name"),
        )
        .join(workflows_table, workflow_runs_table.c.workflow_id == workflows_table.c.id)
        .order_by(workflow_runs_table.c.start_time.desc())
        .limit(limit)
    )

    try:
        with engine.connect() as connection:
            results = connection.execute(stmt).fetchall()

        if not results:
            typer.secho("No workflow runs found in the database.", fg=typer.colors.YELLOW)
            return

        typer.secho(f"--- Showing Last {len(results)} Workflow Runs ---", bold=True)

        header = ["Run ID", "Workflow Name", "Status", "Start Time (UTC)", "Duration (s)"]
        rows = []
        for run in results:
            status_color = typer.colors.GREEN if run.status == 'completed' else (
                typer.colors.YELLOW if run.status in ['running', 'paused'] else typer.colors.RED)

            duration = "N/A"
            if run.end_time and run.start_time:
                duration = f"{(run.end_time - run.start_time).total_seconds():.2f}"

            rows.append([
                run.run_id,
                run.workflow_name,
                typer.style(run.status, fg=status_color),
                run.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                duration
            ])

        # Simple print for alignment
        for row in [header] + rows:
            print(" | ".join(f"{item:<36}" if i == 0 else f"{item:<20}" for i, item in enumerate(row)))

    except Exception as e:
        typer.secho(f"An error occurred while fetching workflow runs: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@audit_app.command("inspect-workflow", help="Inspect a specific workflow run and its I/O.")
def inspect_workflow_run(
        run_id: str = typer.Argument(..., help="The unique ID of the workflow run to inspect."),
):
    """Fetches and displays the details for a single workflow run."""
    setup_cli_logging()

    if not _is_db_configured():
        typer.secho("Auditing requires a database. Please configure one using 'ragnetic configure'.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    engine = _get_sync_db_engine()

    try:
        with engine.connect() as connection:
            # Fetch the main run details, joining with the workflow table to get the name
            stmt = (
                select(
                    workflow_runs_table,
                    workflows_table.c.name.label("workflow_name")
                )
                .join(workflows_table, workflow_runs_table.c.workflow_id == workflows_table.c.id)
                .where(workflow_runs_table.c.run_id == run_id)
            )
            run = connection.execute(stmt).first()

            if not run:
                typer.secho(f"Error: Workflow Run with ID '{run_id}' not found.", fg=typer.colors.RED)
                raise typer.Exit(code=1)

        # --- Display the results ---
        typer.secho(f"\n--- Audit Trail for Workflow Run: {run.run_id} ---", bold=True)

        status_color = typer.colors.GREEN if run.status == 'completed' else (
            typer.colors.YELLOW if run.status in ['running', 'paused'] else typer.colors.RED)

        duration = "N/A"
        if run.end_time and run.start_time:
            duration = f"{(run.end_time - run.start_time).total_seconds():.2f}s"

        typer.echo(f"  {'Status:':<15} {typer.style(run.status.upper(), fg=status_color)}")
        typer.echo(f"  {'Workflow:':<15} {run.workflow_name}")
        typer.echo(f"  {'Start Time:':<15} {run.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        typer.echo(f"  {'End Time:':<15} {run.end_time.strftime('%Y-%m-%d %H:%M:%S UTC') if run.end_time else 'N/A'}")
        typer.echo(f"  {'Duration:':<15} {duration}")

        # Display the detailed JSON blobs
        if run.initial_input:
            typer.secho("\n--- Initial Input ---", bold=True, fg=typer.colors.BLUE)
            typer.echo(json.dumps(run.initial_input, indent=2))

        if run.final_output:
            typer.secho("\n--- Final Output (Context) ---", bold=True, fg=typer.colors.MAGENTA)
            typer.echo(json.dumps(run.final_output, indent=2))

        if run.status == 'paused' and run.last_execution_state:
            typer.secho("\n--- Paused State ---", bold=True, fg=typer.colors.YELLOW)
            typer.echo(json.dumps(run.last_execution_state, indent=2))

    except Exception as e:
        typer.secho(f"An error occurred while inspecting the workflow run: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)



if __name__ == "__main__":
    setup_cli_logging()
    app()