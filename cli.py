import ast
import asyncio
import configparser
import glob
import json
import logging
import logging.config
import os
import re
import secrets
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from urllib.parse import urlparse

import alembic.command
import alembic.config
import pandas as pd
import pytest
import redis
import requests
import typer
import yaml
from alembic.runtime.migration import MigrationContext
from dotenv import load_dotenv
# IMPORTS for inspect_agent dynamic vector store loading
from langchain_community.vectorstores import FAISS, Chroma
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_qdrant import Qdrant
from pinecone import Pinecone as PineconeClient
# IMPORTS for connection checks
from sqlalchemy import create_engine, text, select, func, case
from sqlalchemy.sql import expression

import app.db as db
from app.agents.config_manager import load_agent_config, load_agent_from_yaml_file
# Import core components from the application
from app.core.config import get_path_settings, get_api_key, get_server_api_keys, get_log_storage_config, \
    get_memory_storage_config, get_db_connection, get_cors_settings
from app.core.embed_config import get_embedding_model
from app.core.tasks import get_beat_db_uri
from app.core.validation import is_valid_agent_name_cli
from app.db.models import conversation_metrics_table
from app.db.models import metadata, agent_runs, chat_sessions_table, users_table, agent_run_steps, workflows_table, \
    workflow_runs_table
from app.evaluation.benchmark import run_benchmark
from app.evaluation.dataset_generator import generate_test_set
from app.executors.docker_executor import LocalDockerExecutor
from app.pipelines.embed import embed_agent_data
from app.schemas.data_prep import DatasetPreparationConfig
from app.schemas.fine_tuning import FineTuningJobConfig, FineTuningStatus
from app.schemas.lambda_tool import LambdaRequestPayload, LambdaResourceSpec
from app.schemas.orchestrator import OrchestratorConfig
from app.training.data_prep.conversational_jsonl_loader import ConversationalJsonlLoader
from app.training.data_prep.jsonl_instruction_loader import JsonlInstructionLoader

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
_TRAINING_CONFIGS_DIR = _APP_PATHS["TRAINING_CONFIGS_DIR"]
_DATA_PREPARED_DIR = _APP_PATHS["DATA_PREPARED_DIR"]
_DATA_RAW_DIR = _APP_PATHS["DATA_RAW_DIR"]
_FINE_TUNED_MODELS_BASE_DIR = _APP_PATHS["FINE_TUNED_MODELS_BASE_DIR"]
_DATA_PREP_CONFIGS = _APP_PATHS["DATA_PREP_CONFIGS"]

_CLI_CONFIG_FILE = _RAGNETIC_DIR / "cli_config.ini"

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



def _get_server_url() -> str:
    config = configparser.ConfigParser()
    config.read(_APP_PATHS["CONFIG_FILE_PATH"])
    host = config.get('SERVER', 'host', fallback='127.0.0.1')
    port = config.get('SERVER', 'port', fallback='8000')
    return f"http://{host}:{port}/api/v1"


def _get_api_key_for_cli() -> Optional[str]:
    """
    Retrieves the API key for CLI calls.
    Prioritizes the active key from cli_config.ini, then falls back to RAGNETIC_API_KEYS (master key).
    """
    cli_config = configparser.ConfigParser()
    cli_config.read(_CLI_CONFIG_FILE)

    # 1. Check for active user API key in cli_config.ini
    if cli_config.has_section('CLI_AUTH') and cli_config.has_option('CLI_AUTH', 'active_api_key'):
        active_key = cli_config.get('CLI_AUTH', 'active_api_key')
        if active_key:
            return active_key

    # 2. Fallback to the global server API key from .env or main config
    server_api_keys = get_server_api_keys()
    if server_api_keys:
        return server_api_keys[0] # Return the first configured master key

    return None # No key found


def _save_cli_config(section: str, key: str, value: str):
    """Saves a key-value pair to the CLI-specific config file."""
    cli_config = configparser.ConfigParser()
    cli_config.read(_CLI_CONFIG_FILE)

    if not cli_config.has_section(section):
        cli_config.add_section(section)

    cli_config.set(section, key, value)

    # Ensure the directory exists before writing
    _CLI_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(_CLI_CONFIG_FILE, 'w') as configfile:
        cli_config.write(configfile)


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

def _load_orchestrator_config(orchestrator_name: str) -> OrchestratorConfig:
    """Loads an orchestrator config from a YAML file and validates it."""
    config_path = _AGENTS_DIR / f"{orchestrator_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Orchestrator config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    return OrchestratorConfig(**yaml_data)


def _fetch_and_format_orchestration_tree(connection, parent_run_id: str, indent: str = "", is_last: bool = False):
    """
    Recursively fetches and formats child runs from both the workflow_runs and agent_runs tables
    for a given parent run ID, creating a unified data structure for processing.
    """
    lines = []

    # Fetch direct child runs from workflow_runs table
    workflow_children_stmt = select(
        workflow_runs_table.c.run_id,
        workflow_runs_table.c.status,
        workflows_table.c.name.label("run_name"),
        workflow_runs_table.c.start_time,
        workflow_runs_table.c.end_time,
    ).join(
        workflows_table, workflow_runs_table.c.workflow_id == workflows_table.c.id
    ).where(workflow_runs_table.c.parent_run_id == parent_run_id).order_by(workflow_runs_table.c.start_time)

    workflow_children_rows = connection.execute(workflow_children_stmt).fetchall()

    # Fetch direct child runs from agent_runs table
    agent_children_stmt = select(
        agent_runs.c.run_id,
        agent_runs.c.status,
        chat_sessions_table.c.agent_name.label("run_name"),
        agent_runs.c.start_time,
        agent_runs.c.end_time,
    ).join(
        chat_sessions_table, agent_runs.c.session_id == chat_sessions_table.c.id
    ).where(agent_runs.c.parent_run_id == parent_run_id).order_by(agent_runs.c.start_time)

    agent_children_rows = connection.execute(agent_children_stmt).fetchall()

    # Create a unified list of children with a consistent 'type' and 'name'
    all_children_unified = []
    for row in workflow_children_rows:
        all_children_unified.append({**row._mapping, 'run_type': 'Workflow'})
    for row in agent_children_rows:
        all_children_unified.append({**row._mapping, 'run_type': 'Agent'})

    # Sort all children by start time
    all_children_unified.sort(key=lambda x: x['start_time'])

    for i, child in enumerate(all_children_unified):
        is_last_child = (i == len(all_children_unified) - 1)
        prefix = "└── " if is_last_child else "├── "
        child_indent = "    " if is_last_child else "│   "

        status_color = typer.colors.GREEN if child['status'] == 'completed' else (
            typer.colors.YELLOW if child['status'] in ['running', 'paused'] else typer.colors.RED)

        duration = (child['end_time'] - child['start_time']).total_seconds() if child['end_time'] else "N/A"
        duration_str = f"{duration:.2f}s" if isinstance(duration, (int, float)) else duration

        line = f"{indent}{prefix}{child['run_type']}: {child['run_name']} | Run ID: {typer.style(child['run_id'], fg=typer.colors.CYAN)} | Status: {typer.style(child['status'], fg=status_color)} | Duration: {duration_str}"
        lines.append(line)

        # Recursively fetch grandchildren
        lines.extend(
            _fetch_and_format_orchestration_tree(connection, child['run_id'], indent + child_indent, is_last_child))

    return lines

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


run_app = typer.Typer(name="run", help="Commands for running workflows and other processes.")
app.add_typer(run_app)

user_app = typer.Typer(name="user", help="Manage user accounts.")
app.add_typer(user_app)

role_app = typer.Typer(name="role", help="Manage user roles and permissions.")
app.add_typer(role_app)

analytics_app = typer.Typer(name="analytics", help="Commands for analyzing system performance, costs, and quality.")
app.add_typer(analytics_app)

# Typer app for fine-tuning commands
training_app = typer.Typer(name="training", help="Commands for managing LLM fine-tuning jobs.")
app.add_typer(training_app)

# Typer app for dataset preparation commands
dataset_app = typer.Typer(name="dataset", help="Commands for preparing datasets for fine-tuning.")
app.add_typer(dataset_app)




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

    typer.echo(f"Preparing to apply database migrations to revision: {revision}...")

    try:
        engine = _get_sync_db_engine() # Re-using existing helper for sync engine
        with engine.connect() as connection:
            alembic_cfg = alembic.config.Config(str(_PROJECT_ROOT / "alembic.ini"))
            script = alembic.script.ScriptDirectory.from_config(alembic_cfg)
            migration_context = MigrationContext.configure(connection)
            current_rev = migration_context.get_current_revision()
            head_rev = script.get_current_head()

            typer.echo(f"  - Current DB Revision: {current_rev or 'None (empty database)'}")
            typer.echo(f"  - Latest Code Revision: {head_rev}")

            if revision == "head" and current_rev == head_rev:
                typer.secho("Database is already up-to-date. No migrations to apply.", fg=typer.colors.GREEN)
                return # Exit if already up-to-date and targeting head
            elif revision != "head" and current_rev == revision:
                typer.secho(f"Database is already at revision '{revision}'. No migrations to apply.", fg=typer.colors.GREEN)
                return # Exit if already at target revision
            elif not current_rev:
                 typer.secho("Database is empty. Applying initial migrations.", fg=typer.colors.YELLOW)
            else:
                 typer.secho(f"Database needs migration. Upgrading from {current_rev} to {revision}.", fg=typer.colors.YELLOW)

    except Exception as e:
        typer.secho(f"Error checking database migration status: {e}", fg=typer.colors.RED)
        typer.secho("Proceeding with migration attempt anyway, but consider investigating this error.", fg=typer.colors.YELLOW)
        # Don't exit here, still try to migrate if status check failed.

    alembic_cfg = alembic.config.Config(str(_PROJECT_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("loglevel", "INFO")
    try:
        alembic.command.upgrade(alembic_cfg, revision)
        typer.secho("Database migration complete.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"An error occurred during migration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


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

        # If Redis mode is chosen, let the operator set REDIS_URL (used by server)
        if ws_mode == "redis":
            current_redis = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            redis_url = typer.prompt("Enter Redis URL", default=current_redis)
            if typer.confirm("Save REDIS_URL to the local .env file (recommended)?", default=True):
                _update_env_file({"REDIS_URL": redis_url})
                typer.secho("REDIS_URL saved to .env file.", fg=typer.colors.GREEN)
            else:
                if 'SERVER' not in config: config.add_section('SERVER')
                config.set('SERVER', 'redis_url', redis_url)

        # CORS
        if typer.confirm("Configure CORS allowed origins?", default=False):
            current_origins = os.environ.get(
                "CORS_ALLOWED_ORIGINS",
                config.get(
                    'SERVER',
                    'cors_allowed_origins',
                    fallback="http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173"
                )
            )
            origins_str = typer.prompt("Enter comma-separated origins", default=current_origins)
            if typer.confirm("Save CORS settings to the local .env file (recommended)?", default=True):
                _update_env_file({"CORS_ALLOWED_ORIGINS": origins_str})
                if config.has_option('SERVER', 'cors_allowed_origins'):
                    config.remove_option('SERVER', 'cors_allowed_origins')
                typer.secho("CORS settings saved to .env file.", fg=typer.colors.GREEN)
            else:
                config.set('SERVER', 'cors_allowed_origins', origins_str)

        # NEW: Allowed hosts (TrustedHostMiddleware)
        if typer.confirm("Configure allowed hosts (host allowlist)?", default=True):
            current_hosts = os.environ.get("ALLOWED_HOSTS",
                                           config.get('SERVER', 'allowed_hosts', fallback="localhost,127.0.0.1"))
            hosts_str = typer.prompt("Enter comma-separated hosts (e.g., app.example.com,.example.com,localhost)",
                                     default=current_hosts)
            if typer.confirm("Save ALLOWED_HOSTS to the local .env file (recommended)?", default=True):
                _update_env_file({"ALLOWED_HOSTS": hosts_str})
                if config.has_option('SERVER', 'allowed_hosts'):
                    config.remove_option('SERVER', 'allowed_hosts')
                typer.secho("ALLOWED_HOSTS saved to .env file.", fg=typer.colors.GREEN)
            else:
                config.set('SERVER', 'allowed_hosts', hosts_str)
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

        if typer.confirm("\nAllow interactive DB password prompts on server start? (not recommended for production)",
                         default=False):
            _update_env_file({"ALLOW_DB_PASSWORD_PROMPT": "true"})
            typer.secho("Set ALLOW_DB_PASSWORD_PROMPT=true in .env", fg=typer.colors.YELLOW)
        else:
            _update_env_file({"ALLOW_DB_PASSWORD_PROMPT": "false"})

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

    # --- SMTP SETTINGS (New Section) ---
    if typer.confirm("\nDo you want to configure SMTP settings for the Email Tool?", default=True):
        if 'SMTP_SETTINGS' not in config: config.add_section('SMTP_SETTINGS')

        current_host = config.get('SMTP_SETTINGS', 'host', fallback=os.environ.get("SMTP_HOST", ""))
        current_port = config.get('SMTP_SETTINGS', 'port', fallback=os.environ.get("SMTP_PORT", "465"))
        current_username = config.get('SMTP_SETTINGS', 'username', fallback=os.environ.get("SMTP_USERNAME", ""))

        smtp_host = typer.prompt("SMTP Host (e.g., smtp.gmail.com)",
                                 default=current_host) or ""
        smtp_port = typer.prompt("SMTP Port (e.g., 465 for SSL)", default=current_port) or ""
        smtp_username = typer.prompt("SMTP Username (sender email address)", default=current_username) or "" # Ensure empty string if no input
        smtp_password = typer.prompt("SMTP Password (or App Password)", hide_input=True, confirmation_prompt=False) or "" # Ensure empty string if no input

        config.set('SMTP_SETTINGS', 'host', smtp_host)
        config.set('SMTP_SETTINGS', 'port', str(smtp_port))
        config.set('SMTP_SETTINGS', 'username', smtp_username)

        # Always save password to .env for security, remove from config.ini if present
        if smtp_password:
            _update_env_file({"SMTP_PASSWORD": smtp_password})
            typer.secho("SMTP password saved securely to the .env file.", fg=typer.colors.GREEN)
            if config.has_option('SMTP_SETTINGS', 'password'):  # Remove if it was there
                config.remove_option('SMTP_SETTINGS', 'password')
        else:
            typer.secho("No SMTP password provided. Email tool may not function without it.", fg=typer.colors.YELLOW)

        typer.secho("SMTP settings updated.", fg=typer.colors.GREEN)

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


@app.command(name="check-health", help="Verifies and inspects connections to all configured system components.")
def check_health():
    typer.secho("--- Checking System Component Health ---", bold=True)
    has_failure = False

    # 1. Database Check (re-using your existing logic)
    typer.secho("\nInspecting Database Connections...", fg=typer.colors.CYAN, bold=True)
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
        typer.secho("  No system database connections are configured.", fg=typer.colors.YELLOW)
    else:
        alembic_cfg = alembic.config.Config(str(_PROJECT_ROOT / "alembic.ini"))
        script = alembic.script.ScriptDirectory.from_config(alembic_cfg)
        for conn_name, purposes in connections_to_check.items():
            typer.secho(f"\n  - Connection '{conn_name}' (used for: {', '.join(purposes)})", fg=typer.colors.CYAN,
                        bold=True)
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
                    typer.secho("    - Connectivity: [PASS]", fg=typer.colors.GREEN)

                    migration_context = MigrationContext.configure(connection)
                    current_rev = migration_context.get_current_revision()
                    head_rev = script.get_current_head()
                    typer.echo(f"    - Current DB Revision: {current_rev}")
                    typer.echo(f"    - Latest Code Revision: {head_rev}")

                    if current_rev == head_rev:
                        typer.secho("    - Migration Status: [UP-TO-DATE]", fg=typer.colors.GREEN)
                    else:
                        typer.secho("    - Migration Status: [NEEDS MIGRATION] - Run 'ragnetic migrate'.",
                                    fg=typer.colors.YELLOW)
                engine.dispose()
            except Exception as e:
                typer.secho("    - Connectivity: [FAIL]", fg=typer.colors.RED)
                typer.secho(f"      Error: {e}", fg=typer.colors.RED)
                has_failure = True

    # 2. Redis/Queue Check (new logic)
    typer.secho("\nInspecting Redis/Queue Connection...", fg=typer.colors.CYAN, bold=True)
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        r = redis.from_url(redis_url)
        r.ping()
        typer.secho("  - Redis Connectivity: [PASS]", fg=typer.colors.GREEN)
        typer.echo(f"  - Redis URL: {redis_url}")
    except Exception as e:
        typer.secho("  - Redis Connectivity: [FAIL]", fg=typer.colors.RED)
        typer.secho(f"    Error: {e}", fg=typer.colors.RED)
        has_failure = True

    # 3. Vector Store Directory Check (new logic)
    typer.secho("\nInspecting Vector Store Directory...", fg=typer.colors.CYAN, bold=True)
    vs_dir = _APP_PATHS["VECTORSTORE_DIR"]
    if os.path.exists(vs_dir):
        typer.secho("  - Vector Store Directory: [PASS]", fg=typer.colors.GREEN)
        typer.echo(f"  - Path: {vs_dir}")
    else:
        typer.secho("  - Vector Store Directory: [FAIL]", fg=typer.colors.RED)
        typer.secho("    Error: Directory not found. Run 'ragnetic init' if this is a new project.",
                    fg=typer.colors.RED)
        has_failure = True

    typer.secho("\n--- Check Complete ---", bold=True)
    if has_failure:
        raise typer.Exit(code=1)

@app.command(help="Initialize a new RAGnetic project.")
def init():
    typer.secho("Initializing new RAGnetic project...", bold=True)

    # 1. Create essential directories
    paths_to_create = {
        "DATA_DIR", "AGENTS_DIR", "VECTORSTORE_DIR", "MEMORY_DIR",
        "LOGS_DIR", "TEMP_CLONES_DIR", "RAGNETIC_DIR", "BENCHMARK_DIR", "WORKFLOWS_DIR", "TRAINING_CONFIGS_DIR", "DATA_RAW_DIR", "FINE_TUNED_MODELS_BASE_DIR", "DATA_PREPARED_DIR",
        "DATA_PREP_CONFIGS"
    }
    for key, path in _APP_PATHS.items():
        if key in paths_to_create and not os.path.exists(path):
            os.makedirs(path, mode=0o750, exist_ok=True)
            typer.echo(f"  - Created directory: {path}")

    # 2. Handle default config and database setup
    if not os.path.exists(_CONFIG_FILE):
        typer.echo(f"  - Creating default config file: {_CONFIG_FILE}")

        # Define the path for the bundled DB file (source) and the working DB file (destination)
        bundled_db_path = _PROJECT_ROOT / "ragnetic.db"
        working_db_path = _MEMORY_DIR / "ragnetic.db"

        # Check if the bundled file exists before trying to copy
        if bundled_db_path.exists():
            # Ensure the destination directory exists
            _MEMORY_DIR.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy(bundled_db_path, working_db_path)
                typer.echo(f"  - Copied pre-migrated database to: {working_db_path}")
            except Exception as e:
                typer.secho(f"Error copying bundled database: {e}", fg=typer.colors.RED)
                typer.secho("The project has been initialized, but the default SQLite database was not set up.", fg=typer.colors.YELLOW)
        else:
            typer.secho(f"Warning: Bundled database file not found at {bundled_db_path}. Skipping database copy.", fg=typer.colors.YELLOW)

        # Create the  default config file with the specified content
        config = configparser.ConfigParser()
        config['SERVER'] = {
            'host': '127.0.0.1',
            'port': '8000',
            'json_logs': 'false',
            'websocket_mode': 'memory'
        }
        config['DATABASE_CONNECTIONS'] = {
            'names': 'ragnetic_db'
        }
        config['MEMORY_STORAGE'] = {
            'type': 'db',
            'connection_name': 'ragnetic_db'
        }
        config['LOG_STORAGE'] = {
            'type': 'file'
        }
        config['DATABASE_ragnetic_db'] = {
            'dialect': 'sqlite',
            'database_path': str(working_db_path.relative_to(_PROJECT_ROOT))
        }

        with open(_CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        typer.echo(f"  - Created config file: {_CONFIG_FILE}")
    else:
        typer.echo(f"  - Configuration file already exists at {_CONFIG_FILE}. Skipping default setup.")

    typer.secho("\nProject initialized successfully!", fg=typer.colors.GREEN)
    typer.secho("\nSECURITY NOTICE: CORS:", fg=typer.colors.YELLOW, bold=True)
    typer.secho("By default, RAGnetic allows only local dev origins (http://localhost:3000, http://127.0.0.1:3000, http://localhost:5173).", fg=typer.colors.YELLOW)
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


@app.command("gdrive", help="Authenticate with Google Drive securely.")
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


@app.command(help="Starts the RAGnetic server, worker, and scheduler.")
def start_server(
        host: str = typer.Option(None, help="Server host. Overrides config."),
        port: int = typer.Option(None, help="Server port. Overrides config."),
        reload: bool = typer.Option(False, "--reload", help="Enable auto-reloading for development."),
        worker_device_type: Optional[str] = typer.Option(None, "--worker-device",
                                                        help="Target device for the main worker (auto, cpu, gpu, mps). Overrides auto-detection."),
        gpu_visible_devices: Optional[str] = typer.Option(None, "--gpu-devices",
                                                         help="Comma-separated list of GPU device IDs (e.g., '0' or '0,1'). Only for NVIDIA GPUs."),
        worker_concurrency: Optional[int] = typer.Option(None, "--worker-concurrency",
                                                          help="Number of concurrent worker processes/threads (for --autoscale)."),
        worker_pool_type: str = typer.Option("solo", "--worker-pool",
                                             help="Celery worker pool type (fork, solo, prefork, spawn, gevent, eventlet). 'solo' or 'spawn' recommended for macOS."),
        tokenizers_parallelism: bool = typer.Option(True, "--tokenizers-parallelism/--no-tokenizers-parallelism",
                                                    help="Enable Hugging Face tokenizers parallelism. Set to False (--no-tokenizers-parallelism) to avoid 'fork' warnings on macOS.")
):
    """
    Starts the RAGnetic server (Uvicorn), the Celery worker(s), and the Celery Beat scheduler.
    This consolidates all background task management under one command.
    """
    config = configparser.ConfigParser()
    config.read(_CONFIG_FILE)
    final_host = host or config.get('SERVER', 'host', fallback="127.0.0.1")
    final_port = port or config.getint('SERVER', 'port', fallback=8000)

    paths = get_path_settings()
    uvicorn_log_cfg_path = paths["PROJECT_ROOT"] / "logging.uvicorn.json"

    # --- Warnings ---
    if not get_server_api_keys():
        typer.secho("SECURITY WARNING: Server starting without an API key.", fg=typer.colors.YELLOW, bold=True)
    if get_cors_settings() == ["*"]:
        typer.secho("SECURITY WARNING: Server is allowing requests from all origins ('*').", fg=typer.colors.YELLOW,
                    bold=True)

    redis_process = None
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        # Try to connect to an existing Redis instance
        r = redis.from_url(redis_url)
        r.ping()
        typer.secho("Redis is already running.", fg=typer.colors.GREEN)
    except redis.exceptions.ConnectionError:
        typer.secho("Redis not found. Attempting to start Redis server...", fg=typer.colors.YELLOW)
        try:
            # If connection fails, try to start a local Redis server
            redis_process = subprocess.Popen(["redis-server"])
            typer.secho("Redis server started successfully.", fg=typer.colors.GREEN)
            # Give Redis a moment to initialize
            time.sleep(2)
        except FileNotFoundError:
            typer.secho("ERROR: 'redis-server' command not found.", fg=typer.colors.RED, bold=True)
            typer.secho("Please install Redis (e.g., 'brew install redis') or start it manually.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    beat_db_uri = get_beat_db_uri()

    # Prepare a custom environment for Celery worker and beat processes
    celery_env = os.environ.copy() # Start with current environment variables
    celery_env["CELERY_BEAT_DBURI"] = beat_db_uri

    # Apply GPU/device visibility settings
    if gpu_visible_devices:
        if worker_device_type and worker_device_type.lower() not in ["gpu", "cuda"]:
            typer.secho("Warning: --gpu-devices specified but --worker-device is not 'gpu' or 'cuda'. "
                        "This might lead to unexpected behavior.", fg=typer.colors.YELLOW)
        celery_env["CUDA_VISIBLE_DEVICES"] = gpu_visible_devices
        typer.secho(f"Setting CUDA_VISIBLE_DEVICES to: {gpu_visible_devices}", fg=typer.colors.CYAN)
    elif worker_device_type and worker_device_type.lower() == "gpu":
        typer.secho("Info: User requested 'gpu' worker but no specific --gpu-devices. "
                    "Celery worker will attempt to use default/all available NVIDIA GPUs.", fg=typer.colors.CYAN)
    elif worker_device_type and worker_device_type.lower() == "mps":
        typer.secho("Info: User requested 'mps' worker. Ensure PyTorch is installed with MPS support.",
                    fg=typer.colors.CYAN)

    if tokenizers_parallelism is not None:
        celery_env["TOKENIZERS_PARALLELISM"] = str(tokenizers_parallelism).lower()
        # typer.secho(f"Setting TOKENIZERS_PARALLELISM to: {tokenizers_parallelism}", fg=typer.colors.CYAN)
    else:
        pass

    # Determine concurrency for Celery worker
    final_concurrency = worker_concurrency if worker_concurrency is not None else 4  # Default to 4

    # Base Celery worker command arguments
    worker_loglevel = "info" if reload else "warning"

    base_worker_args = [
        "celery", "-A", "app.core.tasks", "worker",
        f"--pool={worker_pool_type}",
        f"--loglevel={worker_loglevel}",
        "--without-mingle",
        "--without-gossip",
        "--without-heartbeat",
        f"--autoscale={final_concurrency},1",
        "-Q", "ragnetic_fine_tuning_tasks,celery,ragnetic_cleanup_tasks",
    ]

    # In reload mode, subprocesses are managed directly.
    if reload:
        typer.secho("Starting server and worker in --reload mode...", fg=typer.colors.YELLOW, bold=True)
        typer.secho(
            "Note: Celery Beat does not support auto-reloading. Restart the server to apply schedule changes from YAML files.",
            fg=typer.colors.CYAN)

        uvicorn_cmd = [
            "uvicorn", "app.main:app",
            "--host", final_host,
            "--port", str(final_port),
            "--reload",
            "--no-access-log",
            "--log-config", str(uvicorn_log_cfg_path),
        ]

        uvicorn_process = subprocess.Popen(uvicorn_cmd)
        # Pass celery_env to worker_process
        worker_process = subprocess.Popen(base_worker_args, env=celery_env)  # Pass env
        beat_process = subprocess.Popen([
            "celery", "-A", "app.core.tasks", "beat",
            "-S", "sqlalchemy_celery_beat.schedulers:DatabaseScheduler",
            "--loglevel=info"
        ], env=celery_env)  # Pass env

        try:
            uvicorn_process.wait()
        except KeyboardInterrupt:
            typer.echo("\nShutting down processes...")
        finally:
            if beat_process and beat_process.poll() is None:
                beat_process.terminate()
                beat_process.wait()
            if worker_process and worker_process.poll() is None:
                worker_process.terminate()
                worker_process.wait()
            if redis_process and redis_process.poll() is None:
                redis_process.terminate()
                redis_process.wait()

    # For production (non-reload) mode
    else:
        worker_process = None
        beat_process = None
        try:
            typer.secho("Starting Celery worker...", fg=typer.colors.BLUE, bold=True)
            # Pass celery_env to worker_process
            worker_process = subprocess.Popen(base_worker_args, env=celery_env)  # Pass env

            typer.secho("Starting Celery Beat scheduler...", fg=typer.colors.BLUE, bold=True)
            beat_loglevel = "info" if reload else "warning"
            beat_cmd = [
                "celery", "-A", "app.core.tasks", "beat",
                "-S", "sqlalchemy_celery_beat.schedulers:DatabaseScheduler",
                f"--loglevel={beat_loglevel}",
            ]
            beat_process = subprocess.Popen(beat_cmd, env=celery_env)

            typer.secho(f"Starting Uvicorn server on http://{final_host}:{final_port}...", fg=typer.colors.BLUE,
                        bold=True)
            subprocess.run(
                [
                    "uvicorn", "app.main:app",
                    "--host", final_host,
                    "--port", str(final_port),
                    "--no-access-log",
                    "--log-config", str(uvicorn_log_cfg_path),
                ],
                check=True
            )

        except KeyboardInterrupt:
            typer.echo("\nShutting down processes...")
        except Exception as e:
            typer.secho(f"An error occurred during server startup: {e}", fg=typer.colors.RED)
        finally:
            if beat_process and beat_process.poll() is None:
                beat_process.terminate()
                beat_process.wait()
            if worker_process and worker_process.poll() is None:
                worker_process.terminate()
                worker_process.wait()
            if redis_process and redis_process.poll() is None:
                typer.secho("Stopping Redis server...", fg=typer.colors.YELLOW)
                redis_process.terminate()
                redis_process.wait()


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
    # This function is defined in your file, no changes needed here
    _validate_agent_name_cli(agent_name)
    logger = logging.getLogger(__name__)
    try:
        # --- FIX: Initialize DB Connection using the imported 'db' module ---
        conn_name = get_memory_storage_config().get("connection_name") or get_log_storage_config().get(
            "connection_name")
        if not conn_name:
            typer.secho("Error: No database connection is configured in your config.yaml.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        # Call initialization directly on the imported db module
        db.initialize_db_connections(conn_name)

        config_path = _AGENTS_DIR / f"{agent_name}.yaml"
        logger.info(f"Loading agent configuration from: {config_path}")
        if not os.path.exists(config_path):
            typer.secho(f"Error: Configuration file not found at {config_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        vectorstore_path = _VECTORSTORE_DIR / agent_name
        if os.path.exists(vectorstore_path):
            if not force:
                typer.secho(f"Warning: A vector store for agent '{agent_name}' already exists.", fg=typer.colors.YELLOW)
                if not typer.confirm("Do you want to overwrite it and re-deploy the agent?"):
                    typer.echo("Deployment cancelled.")
                    raise typer.Exit()
            shutil.rmtree(vectorstore_path)
            logger.info(f"Removed existing vector store at: {vectorstore_path}")

        agent_config = load_agent_from_yaml_file(config_path)
        typer.echo(f"\nDeploying agent '{agent_config.name}' using embedding model '{agent_config.embedding_model}'...")

        async def run_embedding_with_session():
            # **THE FIX**: Call AsyncSessionLocal through the imported 'db' module
            async with db.AsyncSessionLocal() as session:
                return await embed_agent_data(config=agent_config, db=session)

        vector_store_created = asyncio.run(run_embedding_with_session())

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


@app.command("generate-test", help="Generates a test set from an agent's sources.")
def generate_test_command(
    agent_name: str = typer.Argument(..., help="The agent to build the test set from."),
    output_file: str = typer.Option("test_set.json", "--output", "-o",
                                    help="File name or path for the generated JSON."),
    num_questions: int = typer.Option(50, "--num-questions", "-n", help="Number of questions to generate."),
):
    _validate_agent_name_cli(agent_name)
    logger = logging.getLogger(__name__)
    logger.info(f"--- Generating Test Set for Agent: '{agent_name}' ---")
    try:
        agent_config = load_agent_config(agent_name)
        qa_pairs = asyncio.run(generate_test_set(agent_config, num_questions))

        # Default to BENCHMARK_DIR when a bare filename is provided
        paths = get_path_settings()
        bench_dir = Path(paths["BENCHMARK_DIR"])
        bench_dir.mkdir(parents=True, exist_ok=True)

        out_path = Path(output_file)
        if not out_path.is_absolute():
            out_path = bench_dir / out_path.name

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2)

        logger.info(f"Saved {len(qa_pairs)} Q&A pairs → '{out_path}'")
    except Exception as e:
        logger.error(f"An error occurred during test set generation: {e}", exc_info=True)
        raise typer.Exit(code=1)



@app.command("benchmark", help="Runs a retrieval quality benchmark on an agent.")
def benchmark_command(
    agent_name: str = typer.Argument(..., help="The agent to benchmark."),
    test_set_file: str = typer.Option(..., "--test-set", "-t", help="Path to a JSON test set file."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Optional run ID (auto-generated if omitted)."),
    export_csv: Optional[str] = typer.Option(None, "--export-csv", help="Optional CSV path (defaults to BENCHMARK_DIR/benchmark_{agent}_{run}.csv)."),
    show_detailed_results: Optional[bool] = typer.Option(None, "--show-detailed-results", "-s",
                                                         help="Explicitly show/hide detailed results in console."),
):
    _validate_agent_name_cli(agent_name)
    logger = logging.getLogger(__name__)
    logger.info(f"--- Running Benchmark for Agent: '{agent_name}' ---")

    try:
        agent_config = load_agent_config(agent_name)

        with open(test_set_file, "r", encoding="utf-8") as f:
            test_set = json.load(f)

        engine = db.get_sync_db_engine()
        df = run_benchmark(
            agent_config,
            test_set,
            run_id=run_id,
            dataset_id=None,
            sync_engine=engine,
            export_csv_path=export_csv,
        )

        if df.empty:
            logger.error("Benchmark failed to produce results.")
            raise typer.Exit(code=1)

        # Gather summary from the new nested columns
        rid = df["run_id"].iloc[0]
        total_items = len(df)

        avg_hit5 = float(df["retrieval"].apply(lambda r: (r or {}).get("hit@5", 0.0)).mean())
        avg_mrr = float(df["retrieval"].apply(lambda r: (r or {}).get("mrr", 0.0)).mean())
        avg_ndcg10 = float(df["retrieval"].apply(lambda r: (r or {}).get("ndcg@10", 0.0)).mean())

        total_cost = float(df["costs"].apply(lambda c: (c or {}).get("total_usd", 0.0)).sum())
        avg_ret_s = float(df["durations"].apply(lambda d: (d or {}).get("retrieval_s", 0.0)).mean())
        avg_gen_s = float(df["durations"].apply(lambda d: (d or {}).get("generation_s", 0.0)).mean())

        typer.secho("\n--- Benchmark Complete ---", bold=True)
        typer.echo(f"  Run ID: {rid}")
        typer.echo(f"  Items: {total_items}")
        typer.secho(f"  Avg hit@5:    {avg_hit5:.3f}", fg=typer.colors.GREEN)
        typer.secho(f"  Avg MRR:      {avg_mrr:.3f}", fg=typer.colors.GREEN)
        typer.secho(f"  Avg nDCG@10:  {avg_ndcg10:.3f}", fg=typer.colors.GREEN)
        typer.secho(f"  Total Cost:   ${total_cost:.6f}", fg=typer.colors.BLUE)
        typer.secho(f"  Avg Retrieval Time: {avg_ret_s:.3f}s", fg=typer.colors.YELLOW)
        typer.secho(f"  Avg Generation Time: {avg_gen_s:.3f}s", fg=typer.colors.YELLOW)

        # Resolve where the CSV is
        paths = get_path_settings()
        bench_dir = Path(paths["BENCHMARK_DIR"])
        agent_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", agent_name).strip("._")
        default_csv = bench_dir / f"benchmark_{agent_slug}_{rid}.csv"
        typer.secho(f"\nCSV: {export_csv or str(default_csv)}", fg=typer.colors.BLUE)

        # Optional detailed view
        should_show = show_detailed_results if show_detailed_results is not None else typer.confirm(
            "\nShow detailed results in console?", default=False, abort=False
        )
        if should_show:
            # Build a compact per-item view without changing the stored schema
            rows = []
            for _, row in df.iterrows():
                r = row.get("retrieval") or {}
                d = row.get("durations") or {}
                c = row.get("costs") or {}
                rows.append({
                    "idx": row.get("item_index"),
                    "q": (row.get("question") or "")[:100] + ("…" if len(row.get("question") or "") > 100 else ""),
                    "ctx": row.get("context_size"),
                    "hit@5": r.get("hit@5"),
                    "mrr": r.get("mrr"),
                    "ndcg@10": r.get("ndcg@10"),
                    "ret_s": d.get("retrieval_s"),
                    "gen_s": d.get("generation_s"),
                    "cost_usd": c.get("total_usd"),
                })
            try:
                import pandas as _pd
                view = _pd.DataFrame(rows, columns=["idx", "q", "ctx", "hit@5", "mrr", "ndcg@10", "ret_s", "gen_s", "cost_usd"])
                typer.echo(view.to_string(index=False))
            except Exception:
                # Fallback print if pandas is not available for some reason
                for r in rows:
                    typer.echo(f"[{r['idx']:>3}] hit@5={r['hit@5']} mrr={r['mrr']:.3f} ndcg@10={r['ndcg@10']:.3f} "
                               f"ret={r['ret_s']:.2f}s gen={r['gen_s']:.2f}s cost=${r['cost_usd']:.6f} :: {r['q']}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during benchmark: {e}", exc_info=True)
        raise typer.Exit(code=1)



@app.command("inspect-run", help="Inspect a specific agent run and its steps.")
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
            # First, fetch the main run details from the agent_runs table
            run_stmt = (
                select(agent_runs, chat_sessions_table.c.agent_name, users_table.c.user_id.label("user_identifier"))
                .join(chat_sessions_table, agent_runs.c.session_id == chat_sessions_table.c.id)
                .join(users_table, chat_sessions_table.c.user_id == users_table.c.id)
                .where(agent_runs.c.run_id == run_id)
            )
            run = connection.execute(run_stmt).first()

            # If not found in agent_runs, try the workflow_runs table
            if not run:
                run_stmt = (
                    select(
                        workflow_runs_table,
                        workflows_table.c.name.label("workflow_name")
                    )
                    .join(workflows_table, workflow_runs_table.c.workflow_id == workflows_table.c.id)
                    .where(workflow_runs_table.c.run_id == run_id)
                )
                run = connection.execute(run_stmt).first()
                if not run:
                    typer.secho(f"Error: Run with ID '{run_id}' not found in either agent or workflow tables.",
                                fg=typer.colors.RED)
                    raise typer.Exit(code=1)

            # 2. Fetch the steps for that run if it's an agent run
            steps = []
            if 'session_id' in run._mapping:  # Check if the run is an agent run
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
            typer.colors.YELLOW if run.status in ['running', 'paused'] else typer.colors.RED)
        duration = (run.end_time - run.start_time).total_seconds() if run.end_time else "N/A"
        duration_str = f"{duration:.2f}s" if isinstance(duration, (int, float)) else duration

        typer.echo(f"  {'Status:':<12} {typer.style(run.status, fg=status_color)}")

        # Display the name based on the type of run
        if 'workflow_name' in run._mapping:
            typer.echo(f"  {'Workflow:':<12} {run.workflow_name}")
        else:
            typer.echo(f"  {'Agent:':<12} {run.agent_name}")
            typer.echo(f"  {'User ID:':<12} {run.user_identifier}")

        typer.echo(f"  {'Start Time:':<12} {run.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        typer.echo(f"  {'End Time:':<12} {run.end_time.strftime('%Y-%m-%d %H:%M:%S UTC') if run.end_time else 'N/A'}")
        typer.echo(f"  {'Duration:':<12} {duration_str}")
        typer.echo(f"  {'Run ID:':<12} {typer.style(run.run_id, fg=typer.colors.CYAN)}")

        if 'parent_run_id' in run._mapping and run.parent_run_id:
            typer.echo(f"  {'Parent ID:':<12} {typer.style(run.parent_run_id, fg=typer.colors.BRIGHT_MAGENTA)}")

        # Display Steps
        if 'session_id' in run._mapping:  # Only display steps for agent runs
            typer.secho("\n--- Steps ---", bold=True)
            if not steps:
                typer.secho("  No steps found for this run.", fg=typer.colors.YELLOW)
            else:
                for i, step in enumerate(steps, 1):
                    step_duration = (step.end_time - step.start_time).total_seconds() if step.end_time else "N/A"
                    step_duration_str = f"{step_duration:.2f}s" if isinstance(step_duration,
                                                                              (int, float)) else step_duration
                    step_status_color = typer.colors.GREEN if step.status == 'completed' else typer.colors.RED
                    typer.secho(
                        f"  {i}. Node: {typer.style(step.node_name, bold=True)} ({step_duration_str}) - {typer.style(step.status, fg=step_status_color)}")
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


@app.command("list-runs", help="Lists recent agent runs.")
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


@app.command("list-workflows", help="Lists recent workflow runs.")
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


@app.command("inspect-workflow", help="Inspect a specific workflow run and its I/O.")
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

@app.command(name="trigger-workflow", help="Triggers a workflow to run via the API.")
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
    # Get the server API key to authenticate CLI calls
    server_api_keys = get_server_api_keys()
    if not server_api_keys:
        typer.secho("Error: No server API key configured. Please run 'ragnetic set-server-key'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    api_key = server_api_keys[0]  # Use the first configured key
    headers = {"Content-Type": "application/json", "x-api-key": api_key}

    initial_input = {}
    if initial_input_json:
        try:
            initial_input = json.loads(initial_input_json)
        except json.JSONDecodeError:
            typer.secho("Error: Invalid JSON input.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    response = None  # Initialize response
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
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)

@app.command(name="delete-workflow", help="Permanently deletes a workflow definition from the database and its YAML file.")
def delete_workflow(
    workflow_name: str = typer.Argument(..., help="The name of the workflow to permanently delete."),
    force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    """
    Deletes a workflow definition from the database and its corresponding YAML file.
    """
    typer.secho(f"--- DANGEROUS: Deleting Workflow '{workflow_name}' ---", fg=typer.colors.RED, bold=True)

    if not force:
        typer.secho("This will permanently delete the workflow definition from the database and its YAML file.", fg=typer.colors.YELLOW)
        if not typer.confirm("Are you absolutely sure you want to proceed?", default=False):
            typer.echo("Operation cancelled.")
            raise typer.Exit()

    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key}
    api_url = f"{server_url}/workflows/{workflow_name}"
    workflow_file_path = _APP_PATHS["WORKFLOWS_DIR"] / f"{workflow_name}.yaml"
    response = None

    try:
        # 1. Delete from database via API
        typer.echo(f"Attempting to delete workflow '{workflow_name}' from database via API: {api_url}...")
        response = requests.delete(api_url, headers=headers, timeout=10) # Pass headers
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        typer.secho(f"Workflow '{workflow_name}' successfully deleted from database.", fg=typer.colors.GREEN)

        # 2. Delete the corresponding YAML file
        if workflow_file_path.exists():
            typer.echo(f"Attempting to delete workflow YAML file: {workflow_file_path}...")
            os.remove(workflow_file_path)
            typer.secho(f"Workflow YAML file '{workflow_file_path.name}' deleted successfully.", fg=typer.colors.GREEN)
        else:
            typer.secho(f"Workflow YAML file '{workflow_file_path.name}' not found on disk, skipping file deletion.", fg=typer.colors.YELLOW)

        typer.secho(f"\nWorkflow '{workflow_name}' officially deleted.", fg=typer.colors.GREEN, bold=True)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error deleting workflow '{workflow_name}' from API. Please ensure the server is running and the workflow exists.", fg=typer.colors.RED)
        typer.echo(f"Detailed error: {e}")
        if response is not None and response.text: # Check if response object exists and has text
             typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)
    except OSError as e:
        typer.secho(f"Error deleting workflow YAML file '{workflow_file_path.name}': {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred during workflow deletion: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


# --- User Management Commands ---


@user_app.command("create", help="Create a new user account.")
def create_user(
    username: str = typer.Argument(..., help="Username for the new user."),
    password: str = typer.Option(..., prompt=True, hide_input=True, confirmation_prompt=True,
                                 help="Password for the new user."),
    email: Optional[str] = typer.Option(None, help="Email for the new user."),
    first_name: Optional[str] = typer.Option(None, "--first-name", "-f", help="First name of the user."),
    last_name: Optional[str] = typer.Option(None, "--last-name", "-l", help="Last name of the user."),
    is_superuser: bool = typer.Option(False, "--superuser", "-s", help="Grant superuser privileges to this user."),
    roles: Optional[List[str]] = typer.Option(None, "--role", "-r",
                                              help="Role(s) to assign to the user (e.g., --role admin --role developer)."),
    scope: str = typer.Option("viewer", "--scope", help="Default API key scope (admin, editor, viewer).")
):
    """Creates a new user account via the API, generates a scoped API key, and displays it."""
    server_url = _get_server_url()
    api_key_for_cli = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key_for_cli, "Content-Type": "application/json"}
    url = f"{server_url}/security/users"

    user_data = {
        "username": username,
        "password": password,
        "email": email,
        "first_name": first_name,
        "last_name": last_name,
        "is_superuser": is_superuser,
        "is_active": True,
        "roles": roles if roles else []
    }

    response = None
    try:
        typer.secho(f"Attempting to create user '{username}'...", fg=typer.colors.CYAN)
        response = requests.post(url, headers=headers, json=user_data, timeout=10)
        response.raise_for_status()

        created_user = response.json()
        typer.secho(f"\nUser '{created_user['username']}' (ID: {created_user['id']}) created successfully.", fg=typer.colors.GREEN)
        typer.echo(f"  Full Name: {created_user.get('first_name', '')} {created_user.get('last_name', '')}".strip())
        if created_user.get('roles'):
            typer.echo(f"  Assigned roles: {', '.join([r['name'] for r in created_user['roles']])}")
        if created_user.get('is_superuser'):
            typer.echo("  (This user is a Superuser)")

        # Automatically generate an API key with the specified scope
        api_key_url = f"{server_url}/security/users/{created_user['id']}/api-keys"
        api_key_response = requests.post(api_key_url, headers=headers, json={"scope": scope}, timeout=10)
        api_key_response.raise_for_status()
        new_key_data = api_key_response.json()

        typer.secho(f"\nAPI key with scope '{scope}' generated for user '{username}':", fg=typer.colors.BRIGHT_WHITE, bold=True)
        typer.echo(typer.style(new_key_data['access_token'], bold=True))
        typer.secho("Store this key securely! It will not be shown again.", fg=typer.colors.YELLOW)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error creating user: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


@user_app.command("assign-role", help="Assign an existing role to a user.")
def assign_role_cli(
        user_id: int = typer.Argument(..., help="The ID of the user to assign the role to."),
        role_name: str = typer.Argument(..., help="The name of the role to assign (e.g., 'editor')."),
        organization_name: str = typer.Option("default", help="The organization name for this assignment.")
):
    """Assigns a role to a user via the API."""
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    url = f"{server_url}/security/users/{user_id}/roles"

    data = {
        "role_name": role_name,
        "organization_name": organization_name
    }

    response = None
    try:
        typer.secho(f"Attempting to assign role '{role_name}' to user ID {user_id}...", fg=typer.colors.CYAN)
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()

        typer.secho(f"Role '{role_name}' successfully assigned to user ID {user_id}.", fg=typer.colors.GREEN)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error assigning role: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


@user_app.command("remove-role", help="Remove a role from a user.")
def remove_role_cli(
        user_id: int = typer.Argument(..., help="The ID of the user to remove the role from."),
        role_name: str = typer.Argument(..., help="The name of the role to remove."),
        organization_name: str = typer.Option("default", help="The organization name for this assignment.")
):
    """Removes a role from a user via the API."""
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    url = f"{server_url}/security/users/{user_id}/roles"

    data = {
        "role_name": role_name,
        "organization_name": organization_name
    }
    response = None

    try:
        typer.secho(f"Attempting to remove role '{role_name}' from user ID {user_id}...", fg=typer.colors.CYAN)
        response = requests.delete(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()

        typer.secho(f"Role '{role_name}' successfully removed from user ID {user_id}.", fg=typer.colors.GREEN)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error removing role: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)

@user_app.command("list", help="List all user accounts.")
def list_users():
    """Lists all user accounts via the API."""
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key}
    url = f"{server_url}/security/users"
    response = None  # Initialize response

    try:
        typer.secho("Fetching user list...", fg=typer.colors.CYAN)
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        users_data = response.json()
        if not users_data:
            typer.secho("No users found.", fg=typer.colors.YELLOW)
            return

        typer.secho("\n--- User Accounts ---", bold=True)
        header = ["ID", "Username", "Full Name", "Email", "Active", "Superuser", "Roles"]

        # Prepare rows for width calculation (using raw string values)
        raw_rows_for_width_calc = []
        for user in users_data:
            full_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or "N/A"
            roles_str = ", ".join([role['name'] for role in user.get('roles', [])]) or "None"
            raw_rows_for_width_calc.append([
                str(user['id']),
                user['username'],
                full_name,
                user.get('email', 'N/A'),
                "Yes" if user['is_active'] else "No",
                "Yes" if user['is_superuser'] else "No",
                roles_str
            ])


        # Calculate column widths based on raw strings
        col_widths = [max(len(str(item)) for item in col) for col in zip(*([header] + raw_rows_for_width_calc))]

        typer.echo(" | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths)))
        typer.echo("-|-".join("-" * w for w in col_widths))

        # Prepare rows for display (with styling)
        display_rows = []
        for user in users_data:
            full_name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip() or "N/A"
            roles_str = ", ".join([role['name'] for role in user.get('roles', [])]) or "None"
            display_rows.append([
                str(user['id']),
                user['username'],
                full_name,
                user.get('email', 'N/A'),
                typer.style("Yes", fg=typer.colors.GREEN) if user['is_active'] else typer.style("No", fg=typer.colors.RED),
                typer.style("Yes", fg=typer.colors.GREEN) if user['is_superuser'] else "No",
                roles_str
            ])


        for row_idx, row in enumerate(display_rows):
            # When joining, convert each item to string explicitly to handle StyledText objects
            # and ensure proper alignment by manually padding based on calculated width
            formatted_cells = []
            for col_idx, item in enumerate(row):
                # Calculate effective length by stripping ANSI escape codes for styled text
                raw_item_len = len(str(raw_rows_for_width_calc[row_idx][col_idx]))
                padding = col_widths[col_idx] - raw_item_len
                formatted_cells.append(f"{item}{' ' * padding}")
            typer.echo(" | ".join(formatted_cells))


    except requests.exceptions.RequestException as e:
        typer.secho(f"Error listing users: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)



@user_app.command("update", help="Update an existing user account.")
def update_user(
        user_id: int = typer.Argument(..., help="ID of the user to update."),
        username: Optional[str] = typer.Option(None, help="New username for the user."),
        password: Optional[str] = typer.Option(None, prompt=False, hide_input=True, confirmation_prompt=True,
                                               help="New password for the user."),
        email: Optional[str] = typer.Option(None, help="New email for the user."),
        first_name: Optional[str] = typer.Option(None, "--first-name", "-f", help="New first name of the user."),
        last_name: Optional[str] = typer.Option(None, "--last-name", "-l", help="New last name of the user."),
        is_active: Optional[bool] = typer.Option(None, "--active/--inactive", help="Set user active/inactive."),
        is_superuser: Optional[bool] = typer.Option(None, "--superuser/--no-superuser",
                                                    help="Grant/revoke superuser privileges."),
        roles: Optional[List[str]] = typer.Option(None, "--role", "-r",
                                                  help="Role(s) to assign to the user (e.g., --role admin --role developer). Overwrites existing roles if provided."),
):
    """Updates an existing user account via the API."""
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()  # Use the new _get_api_key_for_cli
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    url = f"{server_url}/security/users/{user_id}"

    update_data = {}
    if username is not None: update_data["username"] = username
    if password is not None: update_data["password"] = password
    if email is not None: update_data["email"] = email
    if first_name is not None: update_data["first_name"] = first_name  # Pass new field
    if last_name is not None: update_data["last_name"] = last_name  # Pass new field
    if is_active is not None: update_data["is_active"] = is_active
    if is_superuser is not None: update_data["is_superuser"] = is_superuser
    if roles is not None: update_data["roles"] = roles

    if not update_data:
        typer.secho("No update parameters provided.", fg=typer.colors.YELLOW)
        raise typer.Exit()
    response = None  # Initialize response

    try:
        typer.secho(f"Attempting to update user ID {user_id}...", fg=typer.colors.CYAN)
        response = requests.put(url, headers=headers, json=update_data, timeout=10)
        response.raise_for_status()

        updated_user = response.json()
        typer.secho(f"User '{updated_user['username']}' (ID: {updated_user['id']}) updated successfully.",
                    fg=typer.colors.GREEN)
        typer.echo(
            f"  Full Name: {updated_user.get('first_name', '')} {updated_user.get('last_name', '')}".strip())  # Display full name
        typer.echo(f"  Active: {updated_user['is_active']}")
        typer.echo(f"  Superuser: {updated_user['is_superuser']}")
        typer.echo(f"  Roles: {', '.join([r['name'] for r in updated_user.get('roles', [])]) or 'None'}")

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error updating user: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)

@user_app.command("delete", help="Delete a user account.")
def delete_user(
        user_id: int = typer.Argument(..., help="ID of the user to delete."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    """Deletes a user account via the API."""
    if not force:
        typer.secho(f"DANGER: This will permanently delete user ID {user_id}.", fg=typer.colors.RED)
        typer.confirm("Are you absolutely sure you want to proceed?", abort=True)

    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key}
    url = f"{server_url}/security/users/{user_id}"
    response = None  # Initialize response

    try:
        typer.secho(f"Attempting to delete user ID {user_id}...", fg=typer.colors.CYAN)
        response = requests.delete(url, headers=headers, timeout=10)
        response.raise_for_status()

        typer.secho(f"User ID {user_id} deleted successfully.", fg=typer.colors.GREEN)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error deleting user: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


@user_app.command("generate-key", help="Generate a new API key for a user.")
def generate_user_api_key(
        user_id: int = typer.Argument(..., help="ID of the user to generate an API key for."),
):
    """Generates a new API key for a user via the API."""
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key}
    url = f"{server_url}/security/users/{user_id}/api-keys"
    response = None  # Initialize response

    try:
        typer.secho(f"Attempting to generate API key for user ID {user_id}...", fg=typer.colors.CYAN)
        response = requests.post(url, headers=headers, timeout=10)
        response.raise_for_status()

        new_key_data = response.json()
        typer.secho(f"New API key generated for user ID {user_id}:", fg=typer.colors.GREEN)
        typer.echo(typer.style(new_key_data['access_token'], bold=True))
        typer.secho("Store this key securely! It will not be shown again.", fg=typer.colors.YELLOW)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error generating API key: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


@user_app.command("revoke-key", help="Revoke a user's API key.")
def revoke_user_api_key(
        api_key_str: str = typer.Argument(..., help="The API key string to revoke."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    """Revokes a user's API key via the API."""
    if not force:
        typer.secho(f"DANGER: This will permanently revoke the API key.", fg=typer.colors.RED)
        typer.confirm("Are you sure you want to proceed?", abort=True)

    server_url = _get_server_url()
    master_api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": master_api_key}
    url = f"{server_url}/security/api-keys/{api_key_str}"  # Note: /api-keys/{api_key_str}
    response = None  # Initialize response

    try:
        typer.secho(f"Attempting to revoke API key '{api_key_str[:8]}...'...", fg=typer.colors.CYAN)
        response = requests.delete(url, headers=headers, timeout=10)
        response.raise_for_status()

        typer.secho(f"API key '{api_key_str[:8]}...' revoked successfully.", fg=typer.colors.GREEN)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error revoking API key: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


# --- Role Management Commands ---

@role_app.command("create", help="Create a new role.")
def create_role(
        name: str = typer.Argument(..., help="Name for the new role."),
        description: Optional[str] = typer.Option(None, help="Description for the new role."),
):
    """Creates a new role via the API."""
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    url = f"{server_url}/security/roles"

    role_data = {
        "name": name,
        "description": description
    }

    response = None  # Initialize response

    try:
        typer.secho(f"Attempting to create role '{name}'...", fg=typer.colors.CYAN)
        response = requests.post(url, headers=headers, json=role_data, timeout=10)
        response.raise_for_status()

        created_role = response.json()
        typer.secho(f"Role '{created_role['name']}' (ID: {created_role['id']}) created successfully.",
                    fg=typer.colors.GREEN)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error creating role: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


@role_app.command("list", help="List all roles.")
def list_roles():
    """Lists all roles via the API."""
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key}
    url = f"{server_url}/security/roles"
    response = None  # Initialize response

    try:
        typer.secho("Fetching role list...", fg=typer.colors.CYAN)
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        roles_data = response.json()
        if not roles_data:
            typer.secho("No roles found.", fg=typer.colors.YELLOW)
            return

        typer.secho("\n--- Roles ---", bold=True)
        header = ["ID", "Name", "Description", "Permissions"]
        rows = []
        for role in roles_data:
            permissions_str = ", ".join(role.get('permissions', [])) or "None"
            rows.append([
                str(role['id']),
                role['name'],
                role.get('description', 'N/A'),
                permissions_str
            ])

        # Simple table printing
        col_widths = [max(len(str(item)) for item in col) for col in zip(*([header] + rows))]
        typer.echo(" | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths)))
        typer.echo("-|-".join("-" * w for w in col_widths))
        for row in rows:
            typer.echo(" | ".join(f"{item:<{col_widths[i]}}" for i, item in enumerate(row)))

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error listing roles: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


@role_app.command("delete", help="Delete a role.")
def delete_role(
        role_id: int = typer.Argument(..., help="ID of the role to delete."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation prompt."),
):
    """Deletes a role via the API."""
    if not force:
        typer.secho(f"DANGER: This will permanently delete role ID {role_id}.", fg=typer.colors.RED)
        typer.confirm("Are you sure you want to proceed?", abort=True)

    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key}
    url = f"{server_url}/security/roles/{role_id}"
    response = None  # Initialize response

    try:
        typer.secho(f"Attempting to delete role ID {role_id}...", fg=typer.colors.CYAN)
        response = requests.delete(url, headers=headers, timeout=10)
        response.raise_for_status()

        typer.secho(f"Role ID {role_id} deleted successfully.", fg=typer.colors.GREEN)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error deleting role: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


@role_app.command("assign-permission", help="Assign a permission to a role.")
def assign_permission_to_role(
        role_id: int = typer.Argument(..., help="ID of the role to assign permission to."),
        permission: str = typer.Argument(..., help="The permission string to assign (e.g., 'agent:create')."),
):
    """Assigns a permission to a role via the API."""
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    url = f"{server_url}/security/roles/{role_id}/permissions"

    data = {"permission": permission}
    response = None  # Initialize response

    try:
        typer.secho(f"Attempting to assign permission '{permission}' to role ID {role_id}...", fg=typer.colors.CYAN)
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()

        typer.secho(f"Permission '{permission}' assigned to role ID {role_id} successfully.", fg=typer.colors.GREEN)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error assigning permission: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


@role_app.command("remove-permission", help="Remove a permission from a role.")
def remove_permission_from_role(
        role_id: int = typer.Argument(..., help="ID of the role to remove permission from."),
        permission: str = typer.Argument(..., help="The permission string to remove."),
):
    """Removes a permission from a role via the API."""
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    url = f"{server_url}/security/roles/{role_id}/permissions"

    data = {"permission": permission}  # DELETE with body is non-standard but FastAPI allows it
    response = None  # Initialize response

    try:
        typer.secho(f"Attempting to remove permission '{permission}' from role ID {role_id}...", fg=typer.colors.CYAN)
        response = requests.delete(url, headers=headers, json=data, timeout=10)  # Pass json=data for DELETE with body
        response.raise_for_status()

        typer.secho(f"Permission '{permission}' removed from role ID {role_id} successfully.", fg=typer.colors.GREEN)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error removing permission: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


@app.command("login", help="Log in a user to activate their API key for CLI commands.")
def login(
        username: str = typer.Argument(..., help="Username of the user."),
        password: str = typer.Option(..., prompt=True, hide_input=True, help="Password for the user."),
):
    """Logs in a user and saves their API key for subsequent CLI commands."""
    server_url = _get_server_url()
    url = f"{server_url}/security/login"
    headers = {"Content-Type": "application/json"}
    login_data = {"username": username, "password": password}
    response = None

    try:
        typer.secho(f"Attempting to log in user '{username}'...", fg=typer.colors.CYAN)
        response = requests.post(url, headers=headers, json=login_data, timeout=10)
        response.raise_for_status()

        token_data = response.json()
        api_key_to_save = token_data.get("access_token")

        if api_key_to_save:
            _save_cli_config('CLI_AUTH', 'active_api_key', api_key_to_save)
            _save_cli_config('CLI_AUTH', 'active_username', username)
            typer.secho(
                f"Successfully logged in as '{username}'. Your CLI commands will now use this user's permissions.",
                fg=typer.colors.GREEN)
        else:
            typer.secho("Login failed: No access token received.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error during login: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


@app.command("logout", help="Clear the active API key, effectively logging out the CLI user.")
def logout():
    """Clears the active API key from the CLI config."""
    cli_config = configparser.ConfigParser()
    cli_config.read(_CLI_CONFIG_FILE)

    if cli_config.has_section('CLI_AUTH'):
        cli_config.remove_section('CLI_AUTH')
        with open(_CLI_CONFIG_FILE, 'w') as configfile:
            cli_config.write(configfile)
        typer.secho("Successfully logged out. No active user key configured for CLI.", fg=typer.colors.GREEN)
    else:
        typer.secho("No active user session found to log out from.", fg=typer.colors.YELLOW)


@app.command("whoami", help="Display the currently active CLI user and their permissions.")
def whoami():
    """Displays the currently active CLI user and their permissions by querying the /me endpoint."""
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()  # This will get the active key (user-specific or master)

    if not api_key:
        typer.secho("No active API key found. Please log in with 'ragnetic login' or set RAGNETIC_API_KEYS.",
                    fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    headers = {"X-API-Key": api_key}
    url = f"{server_url}/security/me"
    response = None

    try:
        typer.secho("Fetching current user details...", fg=typer.colors.CYAN)
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        user_info = response.json()

        typer.secho("\n--- Current CLI User ---", bold=True)
        typer.echo(f"  Username: {user_info.get('username', 'N/A')}")
        typer.echo(f"  Email:    {user_info.get('email', 'N/A')}")
        typer.echo(f"  First Name: {user_info.get('first_name', 'N/A')}")
        typer.echo(f"  Last Name: {user_info.get('last_name', 'N/A')}")
        typer.echo(
            f"  Active:   {typer.style('Yes', fg=typer.colors.GREEN) if user_info.get('is_active') else typer.style('No', fg=typer.colors.RED)}")
        typer.echo(
            f"  Superuser: {typer.style('Yes', fg=typer.colors.GREEN) if user_info.get('is_superuser') else 'No'}")

        typer.secho("\n  Roles and Permissions:", bold=True)
        roles = user_info.get('roles', [])
        if roles:
            for role in roles:
                typer.echo(f"    - Role: {role.get('name')}")
                permissions = role.get('permissions', [])
                if permissions:
                    typer.echo(f"      Permissions: {', '.join(permissions)}")
                else:
                    typer.echo(f"      Permissions: None")
        else:
            typer.echo("    No roles assigned.")

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error fetching user info: {e}", fg=typer.colors.RED)
        if response is not None and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)



@analytics_app.command(name="usage", help="Displays aggregated LLM usage and cost metrics.")
def analytics_usage_command(
        agent_name: Optional[str] = typer.Option(None, "--agent", "-a",
                                                 help="Filter metrics by a specific agent name."),
        start_time: Optional[datetime] = typer.Option(None, "--start", "-s", formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
                                                      help="Start time for metrics (YYYY-MM-DD or YYYY-MM-DDTHH:MM:S)."),
        end_time: Optional[datetime] = typer.Option(None, "--end", "-e", formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
                                                    help="End time for metrics (YYYY-MM-DD or YYYY-MM-DDTHH:MM:S)."),
        limit: int = typer.Option(50, "--limit", "-n", help="Limit the number of detailed metric entries."),
):
    """
    Retrieves and displays aggregated LLM usage and cost metrics from the database.
    """
    logger.info("Retrieving LLM usage and cost metrics...")

    if not _is_db_configured():
        typer.secho("Database is not configured. Please run 'ragnetic configure' to set up a database for metrics.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    engine = _get_sync_db_engine()

    try:
        with engine.connect() as connection:
            # Join conversation_metrics_table with chat_sessions_table and users_table
            stmt = select(
                func.sum(conversation_metrics_table.c.prompt_tokens).label("total_prompt_tokens"),
                func.sum(conversation_metrics_table.c.completion_tokens).label("total_completion_tokens"),
                func.sum(conversation_metrics_table.c.total_tokens).label("total_tokens"),
                func.sum(conversation_metrics_table.c.estimated_cost_usd).label("total_llm_cost_usd"),
                # Renamed for clarity
                func.sum(conversation_metrics_table.c.embedding_cost_usd).label("total_embedding_cost_usd"),
                func.avg(conversation_metrics_table.c.retrieval_time_s).label("avg_retrieval_time_s"),
                func.avg(conversation_metrics_table.c.generation_time_s).label("avg_generation_time_s"),
                func.count(conversation_metrics_table.c.request_id).label("total_requests"),
                chat_sessions_table.c.agent_name,
                conversation_metrics_table.c.llm_model,
                users_table.c.user_id  # User ID (username)
            ).outerjoin(
                chat_sessions_table, conversation_metrics_table.c.session_id == chat_sessions_table.c.id
            ).outerjoin(
                users_table, chat_sessions_table.c.user_id == users_table.c.id
            )

            conditions = []
            if agent_name:
                conditions.append(chat_sessions_table.c.agent_name == agent_name)
            if start_time:
                conditions.append(conversation_metrics_table.c.timestamp >= start_time)
            if end_time:
                conditions.append(conversation_metrics_table.c.timestamp <= end_time)

            if conditions:
                stmt = stmt.where(*conditions)

            # Group by Agent Name, LLM Model, and User ID (username)
            stmt = stmt.group_by(
                chat_sessions_table.c.agent_name,
                conversation_metrics_table.c.llm_model,
                users_table.c.user_id
            )

            # Order by total LLM cost descending
            stmt = stmt.order_by(func.sum(conversation_metrics_table.c.estimated_cost_usd).desc())

            if limit:
                stmt = stmt.limit(limit)

            results = connection.execute(stmt).fetchall()

        if not results:
            message = "No LLM usage metrics found in the database."
            if agent_name:
                message = f"No LLM usage metrics found for agent: '{agent_name}'. (Note: This might exclude metrics not tied to a specific chat session or user.)"
            typer.secho(message, fg=typer.colors.YELLOW)
            return

        # Prepare data for pandas DataFrame with new columns
        df = pd.DataFrame(results, columns=[
            "Total Prompt Tokens", "Total Completion Tokens", "Total Tokens",
            "Total LLM Cost (USD)", "Total Embedding Cost (USD)",
            "Avg Retrieval Time (s)", "Avg Generation Time (s)",
            "Total Requests", "Agent Name", "LLM Model", "User ID"
        ])

        # Calculate Total Estimated Cost (sum of LLM and Embedding costs)
        df["Total Estimated Cost (USD)"] = df["Total LLM Cost (USD)"] + df["Total Embedding Cost (USD)"]

        # Reorder columns for better display
        df = df[[
            "Total Prompt Tokens", "Total Completion Tokens", "Total Tokens",
            "Total LLM Cost (USD)", "Total Embedding Cost (USD)", "Total Estimated Cost (USD)",
            "Avg Retrieval Time (s)", "Avg Generation Time (s)",
            "Total Requests", "Agent Name", "LLM Model", "User ID"
        ]]

        # Handle None in 'Agent Name', 'LLM Model', and 'User ID' columns for display
        df['Agent Name'] = df['Agent Name'].fillna('N/A')
        df['LLM Model'] = df['LLM Model'].fillna('N/A')
        df['User ID'] = df['User ID'].fillna('N/A (No User)')

        # Format numerical columns for better readability
        df["Total LLM Cost (USD)"] = df["Total LLM Cost (USD)"].map(lambda x: f"${x:,.6f}")
        df["Total Embedding Cost (USD)"] = df["Total Embedding Cost (USD)"].fillna(0.0).map(lambda x: f"${x:,.6f}")
        df["Total Estimated Cost (USD)"] = df["Total Estimated Cost (USD)"].map(lambda x: f"${x:,.6f}")
        df["Avg Retrieval Time (s)"] = df["Avg Retrieval Time (s)"].map(lambda x: f"{x:.4f}")
        df["Avg Generation Time (s)"] = df["Avg Generation Time (s)"].map(lambda x: f"{x:.4f}")
        df["Total Requests"] = df["Total Requests"].astype(int)

        typer.secho("\n--- LLM Usage & Cost Metrics Summary ---", bold=True, fg=typer.colors.CYAN)
        typer.echo(df.to_string(index=False))
        typer.secho("\nNote: Costs are estimated based on configured model pricing.", fg=typer.colors.BLUE)

    except Exception as e:
        logger.error(f"An error occurred while fetching LLM usage metrics: {e}", exc_info=True)
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
@analytics_app.command(name="latency", help="Displays percentile latency metrics (p50, p95, p99) for agent runs.")
def analytics_latency_command(
        agent_name: Optional[str] = typer.Option(None, "--agent", "-a",
                                                 help="Filter latency metrics by a specific agent name."),
):
    """
    Retrieves and displays p50, p95, and p99 latency metrics for agent runs.
    """
    logger.info("Retrieving latency metrics...")

    if not _is_db_configured():
        typer.secho("Database is not configured. Please configure one using 'ragnetic configure'.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    engine = _get_sync_db_engine()

    try:
        with engine.connect() as connection:
            # Build the base query for latency
            stmt = select(
                conversation_metrics_table.c.generation_time_s
            ).select_from(
                conversation_metrics_table.join(chat_sessions_table)
            ).where(
                conversation_metrics_table.c.generation_time_s.isnot(None)
            )

            if agent_name:
                stmt = stmt.where(chat_sessions_table.c.agent_name == agent_name)

            results = connection.execute(stmt).fetchall()
            latencies = [row.generation_time_s for row in results]

        if not latencies:
            message = "No latency data found."
            if agent_name:
                message = f"No latency data found for agent: '{agent_name}'."
            typer.secho(message, fg=typer.colors.YELLOW)
            return

        # Use pandas for percentile calculation, as it's efficient
        df = pd.Series(latencies)
        p50 = df.quantile(0.50)
        p95 = df.quantile(0.95)
        p99 = df.quantile(0.99)
        avg = df.mean()

        typer.secho("\n--- Agent Run Latency Metrics ---", bold=True, fg=typer.colors.CYAN)
        typer.echo(f"  Total runs analyzed: {len(latencies)}")
        typer.echo(f"  Average Latency: {avg:.4f}s")
        typer.secho(f"  P50 Latency (Median): {p50:.4f}s", fg=typer.colors.GREEN)
        typer.secho(f"  P95 Latency: {p95:.4f}s", fg=typer.colors.YELLOW)
        typer.secho(f"  P99 Latency: {p99:.4f}s", fg=typer.colors.RED)

    except Exception as e:
        logger.error(f"An error occurred while fetching latency metrics: {e}", exc_info=True)
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

@analytics_app.command(name="benchmarks", help="Displays summaries of past benchmark runs.")
def analytics_benchmarks_command(
    agent_name: Optional[str] = typer.Option(None, "--agent", "-a", help="Filter benchmarks by agent name."),
    show_detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed per-item rows."),
    latest: bool = typer.Option(False, "--latest", "-l", help="Show only the latest benchmark run for the agent."),
):
    """
    Summarize benchmark CSVs that may contain *nested* columns:
    - retrieval, durations, costs, token_usage, judge (JSON-serializable)
    Also supports older flat CSVs (key_fact_recalled, retrieval_f1, etc.).
    """
    logger.info("Retrieving benchmark results...")

    paths = get_path_settings()
    bench_dir = Path(paths["BENCHMARK_DIR"])
    if not bench_dir.exists() or not any(bench_dir.iterdir()):
        typer.secho(f"No benchmark results found in '{bench_dir}'. Run 'ragnetic benchmark' first.",
                    fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    files = sorted(glob.glob(str(bench_dir / "*.csv")), reverse=True)
    if not files:
        typer.secho(f"No .csv benchmark files found in '{bench_dir}'.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    # Optional filter by agent from filename: benchmark_{agent}_{runOrTimestamp}.csv
    if agent_name:
        filtered = []
        for f in files:
            parts = Path(f).name.split('_')
            if len(parts) >= 3 and parts[1] == agent_name:
                filtered.append(f)
        if not filtered:
            typer.secho(f"No benchmark results found for agent: '{agent_name}'.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)
        files = filtered

    if latest and files:
        files = [files[0]]
        logger.info(f"Displaying latest benchmark run: {Path(files[0]).name}")
    elif latest and not files:
        typer.secho("No benchmark runs found to display the latest.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    def _maybe_parse(x):
        if isinstance(x, (dict, list)):
            return x
        if not isinstance(x, str):
            return None
        s = x.strip()
        if not s:
            return None
        # Prefer JSON; fall back to literal_eval (handles pandas dict-string with single quotes)
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return None

    all_dfs = []
    for f_path in files:
        try:
            df = pd.read_csv(f_path)
            df["benchmark_id"] = Path(f_path).name.replace(".csv", "")

            # Parse nested columns if present
            for col in ("retrieval", "durations", "costs", "token_usage", "judge"):
                if col in df.columns:
                    df[col] = df[col].apply(_maybe_parse)

            # Derive unified numeric columns (robust to new/old schema)
            # Retrieval metrics
            df["hit@5"] = df["retrieval"].apply(lambda r: (r or {}).get("hit@5", None)) if "retrieval" in df.columns else None
            df["mrr"] = df["retrieval"].apply(lambda r: (r or {}).get("mrr", None)) if "retrieval" in df.columns else None
            df["ndcg@10"] = df["retrieval"].apply(lambda r: (r or {}).get("ndcg@10", None)) if "retrieval" in df.columns else None

            # Durations
            df["retrieval_time_s_new"] = df["durations"].apply(lambda d: (d or {}).get("retrieval_s", None)) if "durations" in df.columns else None
            df["generation_time_s_new"] = df["durations"].apply(lambda d: (d or {}).get("generation_s", None)) if "durations" in df.columns else None

            # Costs
            df["total_cost_usd_new"] = df["costs"].apply(lambda c: (c or {}).get("total_usd", None)) if "costs" in df.columns else None

            # Token usage (LLM tokens only; embedding tokens shown separately)
            def _sum_llm_tokens(tu):
                if not isinstance(tu, dict):
                    return None
                def _tok(d):
                    if not isinstance(d, dict):
                        return 0
                    if d.get("total_tokens") is not None:
                        return d.get("total_tokens") or 0
                    return (d.get("prompt_tokens", 0) or 0) + (d.get("completion_tokens", 0) or 0)
                return _tok(tu.get("agent")) + _tok(tu.get("judge"))

            df["total_llm_tokens_new"] = df["token_usage"].apply(_sum_llm_tokens) if "token_usage" in df.columns else None
            df["embedding_tokens"] = df["token_usage"].apply(lambda t: (t or {}).get("embedding_tokens_query", 0)) if "token_usage" in df.columns else 0

            # Judge metrics (new) and fallbacks (old)
            df["faithfulness_new"] = df["judge"].apply(lambda j: (j or {}).get("faithfulness", None)) if "judge" in df.columns else None
            df["relevance_new"] = df["judge"].apply(lambda j: (j or {}).get("relevance", None)) if "judge" in df.columns else None
            df["conciseness_new"] = df["judge"].apply(lambda j: (j or {}).get("conciseness", None)) if "judge" in df.columns else None
            df["coherence_new"] = df["judge"].apply(lambda j: (j or {}).get("coherence", None)) if "judge" in df.columns else None

            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read benchmark file '{f_path}': {e}", exc_info=True)
            typer.secho(f"Warning: Could not read '{Path(f_path).name}'. Skipping.", fg=typer.colors.YELLOW)

    if not all_dfs:
        typer.secho("No readable benchmark files found after filtering.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    df = pd.concat(all_dfs, ignore_index=True)

    # Unified rollups with fallbacks for old flat columns
    def _mean(col_new, col_old=None, default=0.0):
        if col_new in df.columns and pd.api.types.is_numeric_dtype(df[col_new]):
            s = df[col_new].dropna()
            if len(s):
                return float(s.mean())
        if col_old and col_old in df.columns and pd.api.types.is_numeric_dtype(df[col_old]):
            s = df[col_old].dropna()
            if len(s):
                return float(s.mean())
        return float(default)

    def _sum(col_new, col_old=None, default=0.0):
        if col_new in df.columns and pd.api.types.is_numeric_dtype(df[col_new]):
            return float(df[col_new].fillna(0).sum())
        if col_old and col_old in df.columns and pd.api.types.is_numeric_dtype(df[col_old]):
            return float(df[col_old].fillna(0).sum())
        return float(default)

    total_items = int(df.shape[0])

    avg_hit5   = _mean("hit@5")
    avg_mrr    = _mean("mrr")
    avg_ndcg10 = _mean("ndcg@10")

    avg_ret_s = _mean("retrieval_time_s_new", "retrieval_time_s")
    avg_gen_s = _mean("generation_time_s_new", "generation_time_s")

    total_cost_usd = _sum("total_cost_usd_new", "estimated_cost_usd")
    total_llm_tokens = _sum("total_llm_tokens_new", "total_tokens")
    total_embed_tokens = _sum("embedding_tokens")

    # Judge metrics new → fallback to old columns if present
    avg_faith = _mean("faithfulness_new", "faithfulness")
    avg_relev = _mean("relevance_new", "answer_relevance")
    avg_conci = _mean("conciseness_new", "conciseness_score")
    avg_coher = _mean("coherence_new", "coherence_score")

    if show_detailed:
        typer.secho("\n--- Detailed Benchmark Results ---", bold=True, fg=typer.colors.CYAN)
        # Build a compact view from unified columns
        view_cols = []
        rows = []

        for _, r in df.iterrows():
            retr = r.get("retrieval") if "retrieval" in df.columns else None
            durs = r.get("durations") if "durations" in df.columns else None
            costs = r.get("costs") if "costs" in df.columns else None
            judge = r.get("judge") if "judge" in df.columns else None

            row = {
                "benchmark_id": r.get("benchmark_id"),
                "idx": r.get("item_index", None),
                "question": (str(r.get("question") or "")[:100] + ("…" if len(str(r.get("question") or "")) > 100 else "")),
                "ctx": r.get("context_size", None),
                "hit@5": (retr or {}).get("hit@5") if isinstance(retr, dict) else r.get("retrieval_f1", None),
                "mrr": (retr or {}).get("mrr") if isinstance(retr, dict) else None,
                "ndcg@10": (retr or {}).get("ndcg@10") if isinstance(retr, dict) else None,
                "ret_s": (durs or {}).get("retrieval_s") if isinstance(durs, dict) else r.get("retrieval_time_s", None),
                "gen_s": (durs or {}).get("generation_s") if isinstance(durs, dict) else r.get("generation_time_s", None),
                "cost_usd": (costs or {}).get("total_usd") if isinstance(costs, dict) else r.get("estimated_cost_usd", None),
                "faith": (judge or {}).get("faithfulness") if isinstance(judge, dict) else r.get("faithfulness", None),
                "relevance": (judge or {}).get("relevance") if isinstance(judge, dict) else r.get("answer_relevance", None),
                "conciseness": (judge or {}).get("conciseness") if isinstance(judge, dict) else r.get("conciseness_score", None),
                "coherence": (judge or {}).get("coherence") if isinstance(judge, dict) else r.get("coherence_score", None),
            }
            rows.append(row)

        try:
            detailed = pd.DataFrame(rows, columns=[
                "benchmark_id","idx","question","ctx","hit@5","mrr","ndcg@10","ret_s","gen_s","cost_usd",
                "faith","relevance","conciseness","coherence"
            ])
            typer.echo(detailed.to_string(index=False))
        except Exception:
            for r in rows:
                typer.echo(
                    f"[{r.get('idx')}] hit@5={r.get('hit@5')} mrr={r.get('mrr')} ndcg@10={r.get('ndcg@10')} "
                    f"ret={r.get('ret_s')}s gen={r.get('gen_s')}s cost=${r.get('cost_usd')} :: {r.get('question')}"
                )
        return

    # Summary
    typer.secho("\n--- Benchmark Summary ---", bold=True, fg=typer.colors.CYAN)
    typer.secho(f"Agent(s): {agent_name if agent_name else 'All'}", fg=typer.colors.WHITE)
    typer.secho(f"Number of Benchmarks Included: {len(files)}", fg=typer.colors.WHITE)
    typer.echo(f"  - Total Test Cases Evaluated: {total_items}")
    typer.secho(f"  - Avg hit@5: {avg_hit5:.3f}", fg=typer.colors.GREEN)
    typer.secho(f"  - Avg MRR:   {avg_mrr:.3f}", fg=typer.colors.GREEN)
    typer.secho(f"  - Avg nDCG@10: {avg_ndcg10:.3f}", fg=typer.colors.GREEN)

    typer.echo(f"  - Avg Faithfulness: {avg_faith:.3f}")
    typer.echo(f"  - Avg Relevance:    {avg_relev:.3f}")
    typer.echo(f"  - Avg Conciseness:  {avg_conci:.3f}")
    typer.echo(f"  - Avg Coherence:    {avg_coher:.3f}")

    typer.secho(f"  - Total Estimated Cost: ${total_cost_usd:.6f}", fg=typer.colors.BLUE)
    typer.echo(f"  - Total LLM Tokens: {int(total_llm_tokens)}")
    typer.echo(f"  - Total Embedding Tokens: {int(total_embed_tokens)}")
    typer.secho(f"  - Avg Retrieval Time: {avg_ret_s:.3f}s", fg=typer.colors.YELLOW)
    typer.secho(f"  - Avg Generation Time: {avg_gen_s:.3f}s", fg=typer.colors.YELLOW)

    # Sample config values (present in new CSVs)
    def _sample(col, default="N/A"):
        return df[col].dropna().iloc[0] if col in df.columns and not df[col].dropna().empty else default

    typer.echo("\nConfig (sample from results):")
    typer.echo(f"  - Agent LLM Model:      {_sample('agent_llm_model')}")
    typer.echo(f"  - Agent Embedding Model: {_sample('agent_embedding_model')}")
    typer.echo(f"  - Chunking Mode:         {_sample('chunking_mode')}")
    typer.echo(f"  - Chunk Size:            {_sample('chunk_size', 0)}")
    typer.echo(f"  - Chunk Overlap:         {_sample('chunk_overlap', 0)}")



@analytics_app.command(name="workflow-runs", help="Displays aggregated workflow run metrics.")
def analytics_workflow_runs_command(
        workflow_name: Optional[str] = typer.Option(None, "--workflow", "-w",
                                                    help="Filter metrics by a specific workflow name."),
        status: Optional[str] = typer.Option(None, "--status", "-s",
                                             help="Filter by workflow status (running, completed, failed, paused)."),
        start_time: Optional[datetime] = typer.Option(None, "--start", "-S", formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
                                                      help="Start time for metrics (YYYY-MM-DD or YYYY-MM-DDTHH:MM:S)."),
        end_time: Optional[datetime] = typer.Option(None, "--end", "-E", formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
                                                    help="End time for metrics (YYYY-MM-DD or YYYY-MM-DDTHH:MM:S)."),
        limit: int = typer.Option(20, "--limit", "-n", help="Limit the number of aggregated workflow results."),
):
    """
    Retrieves and displays aggregated workflow run metrics from the database.
    """
    logger.info("Retrieving workflow run metrics...")

    if not _is_db_configured():
        typer.secho("Database is not configured. Please run 'ragnetic configure' to set up a database for metrics.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    engine = _get_sync_db_engine()

    try:
        with engine.connect() as connection:
            stmt = select(
                workflows_table.c.name.label("workflow_name"),
                func.count(workflow_runs_table.c.run_id).label("total_runs"),
                func.avg(
                    func.julianday(workflow_runs_table.c.end_time) - func.julianday(workflow_runs_table.c.start_time)
                ).label("avg_duration_days"), # SQLite specific for duration
                func.sum(
                    case((workflow_runs_table.c.status == 'completed', 1), else_=0)
                ).label("completed_runs"),
                func.sum(
                    case((workflow_runs_table.c.status == 'failed', 1), else_=0)
                ).label("failed_runs"),
                func.sum(
                    case((workflow_runs_table.c.status == 'paused', 1), else_=0)
                ).label("paused_runs")
            ).join(
                workflows_table, workflow_runs_table.c.workflow_id == workflows_table.c.id
            )

            conditions = []
            if workflow_name:
                conditions.append(workflows_table.c.name == workflow_name)
            if status:
                conditions.append(workflow_runs_table.c.status == status)
            if start_time:
                conditions.append(workflow_runs_table.c.start_time >= start_time)
            if end_time:
                conditions.append(workflow_runs_table.c.end_time <= end_time)

            if conditions:
                stmt = stmt.where(*conditions)

            stmt = stmt.group_by(workflows_table.c.name)
            stmt = stmt.order_by(func.count(workflow_runs_table.c.run_id).desc())
            if limit:
                stmt = stmt.limit(limit)

            results = connection.execute(stmt).fetchall()

        if not results:
            message = "No workflow run metrics found in the database."
            if workflow_name:
                message = f"No workflow run metrics found for workflow: '{workflow_name}'."
            typer.secho(message, fg=typer.colors.YELLOW)
            return

        df = pd.DataFrame(results, columns=[
            "Workflow Name", "Total Runs", "Avg Duration (Days)",
            "Completed Runs", "Failed Runs", "Paused Runs"
        ])

        # Convert avg_duration_days to seconds for better readability
        # 1 day = 86400 seconds
        df["Avg Duration (s)"] = df["Avg Duration (Days)"].fillna(0) * 86400
        df["Avg Duration (s)"] = df["Avg Duration (s)"].map(lambda x: f"{x:.2f}")

        # Calculate percentages
        df["Success Rate"] = df["Completed Runs"] / df["Total Runs"]
        df["Failure Rate"] = df["Failed Runs"] / df["Total Runs"]
        df["Paused Rate"] = df["Paused Runs"] / df["Total Runs"]

        # Handle NaN from division by zero for rates (e.g., if Total Runs is 0 for a group)
        df = df.fillna(0)

        # Format percentages
        df["Success Rate"] = df["Success Rate"].map(lambda x: f"{x:.2%}")
        df["Failure Rate"] = df["Failure Rate"].map(lambda x: f"{x:.2%}")
        df["Paused Rate"] = df["Paused Rate"].map(lambda x: f"{x:.2%}")

        # Reorder columns for final display
        df = df[[
            "Workflow Name", "Total Runs", "Success Rate", "Failure Rate", "Paused Rate",
            "Avg Duration (s)", "Completed Runs", "Failed Runs", "Paused Runs"
        ]]

        typer.secho("\n--- Workflow Run Metrics Summary ---", bold=True, fg=typer.colors.CYAN)
        typer.echo(df.to_string(index=False))
        typer.secho(
            "\nNote: Average duration calculation is based on SQL's `julianday` function, which may vary slightly across database types (e.g., PostgreSQL).",
            fg=typer.colors.BLUE)

    except Exception as e:
        logger.error(f"An error occurred while fetching workflow run metrics: {e}", exc_info=True)
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@analytics_app.command(name="agent-steps", help="Displays aggregated agent step metrics.")
def analytics_agent_steps_command(
        agent_name: Optional[str] = typer.Option(None, "--agent", "-a",
                                                 help="Filter metrics by a specific agent name."),
        node_name: Optional[str] = typer.Option(None, "--node", "-n",
                                                help="Filter by a specific node name (e.g., 'agent', 'retriever')."),
        status: Optional[str] = typer.Option(None, "--status", "-s",
                                             help="Filter by step status (running, completed, failed)."),
        start_time: Optional[datetime] = typer.Option(None, "--start", "-S", formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
                                                      help="Start time for metrics (YYYY-MM-DD or YYYY-MM-DDTHH:MM:S)."),
        end_time: Optional[datetime] = typer.Option(None, "--end", "-E", formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
                                                    help="End time for metrics (YYYY-MM-DD or YYYY-MM-DDTHH:MM:S)."),
        limit: int = typer.Option(20, "--limit", "-l", help="Limit the number of aggregated node results."),
):
    """
    Retrieves and displays aggregated agent step metrics from the database.
    """
    logger.info("Retrieving agent step metrics...")

    if not _is_db_configured():
        typer.secho("Database is not configured. Please run 'ragnetic configure' to set up a database for metrics.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    engine = _get_sync_db_engine()

    try:
        with engine.connect() as connection:
            stmt = select(
                chat_sessions_table.c.agent_name,
                agent_run_steps.c.node_name,
                func.count(agent_run_steps.c.id).label("total_calls"),
                func.avg(
                    func.julianday(agent_run_steps.c.end_time) - func.julianday(agent_run_steps.c.start_time)
                ).label("avg_duration_days"),  # SQLite specific for duration
                func.sum(
                    expression.case((agent_run_steps.c.status == 'completed', 1), else_=0)

                ).label("completed_calls"),
                func.sum(
                    expression.case((agent_run_steps.c.status == 'failed', 1), else_=0)
                ).label("failed_calls")
            ).join(
                agent_runs, agent_run_steps.c.agent_run_id == agent_runs.c.id
            ).join(
                chat_sessions_table, agent_runs.c.session_id == chat_sessions_table.c.id
                # Link to chat_sessions for agent_name
            )

            conditions = []
            if agent_name:
                conditions.append(chat_sessions_table.c.agent_name == agent_name)
            if node_name:
                conditions.append(agent_run_steps.c.node_name == node_name)
            if status:
                conditions.append(agent_run_steps.c.status == status)
            if start_time:
                conditions.append(agent_run_steps.c.start_time >= start_time)
            if end_time:
                conditions.append(agent_run_steps.c.end_time <= end_time)

            if conditions:
                stmt = stmt.where(*conditions)

            stmt = stmt.group_by(
                chat_sessions_table.c.agent_name,
                agent_run_steps.c.node_name
            )
            stmt = stmt.order_by(func.count(agent_run_steps.c.id).desc())
            if limit:
                stmt = stmt.limit(limit)

            results = connection.execute(stmt).fetchall()

        if not results:
            message = "No agent step metrics found in the database."
            if agent_name:
                message += f" for agent: '{agent_name}'"
            if node_name:
                message += f" and node: '{node_name}'"
            typer.secho(message, fg=typer.colors.YELLOW)
            return

        df = pd.DataFrame(results, columns=[
            "Agent Name", "Node Name", "Total Calls", "Avg Duration (Days)",
            "Completed Calls", "Failed Calls"
        ])

        df["Avg Duration (s)"] = df["Avg Duration (Days)"].fillna(0) * 86400
        df["Avg Duration (s)"] = df["Avg Duration (s)"].map(lambda x: f"{x:.4f}")  # More precision for steps

        df["Success Rate"] = df["Completed Calls"] / df["Total Calls"]
        df["Failure Rate"] = df["Failed Calls"] / df["Total Calls"]

        df = df.fillna(0)  # Handle NaN from division by zero or missing data

        df["Success Rate"] = df["Success Rate"].map(lambda x: f"{x:.2%}")
        df["Failure Rate"] = df["Failure Rate"].map(lambda x: f"{x:.2%}")

        df = df[[
            "Agent Name", "Node Name", "Total Calls", "Success Rate", "Failure Rate",
            "Avg Duration (s)", "Completed Calls", "Failed Calls"
        ]]

        typer.secho("\n--- Agent Step Metrics Summary ---", bold=True, fg=typer.colors.CYAN)
        typer.echo(df.to_string(index=False))
        typer.secho(
            "\nNote: Average duration calculation is based on SQL's `julianday` function, which may vary slightly across database types (e.g., PostgreSQL).",
            fg=typer.colors.BLUE)

    except Exception as e:
        logger.error(f"An error occurred while fetching agent step metrics: {e}", exc_info=True)
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@analytics_app.command(name="agent-runs", help="Displays aggregated agent run metrics.")
def analytics_agent_runs_command(
        agent_name: Optional[str] = typer.Option(None, "--agent", "-a",
                                                 help="Filter metrics by a specific agent name."),
        status: Optional[str] = typer.Option(None, "--status", "-s",
                                             help="Filter by run status (running, completed, failed)."),
        start_time: Optional[datetime] = typer.Option(None, "--start", "-S", formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
                                                      help="Start time for metrics (YYYY-MM-DD or YYYY-MM-DDTHH:MM:S)."),
        end_time: Optional[datetime] = typer.Option(None, "--end", "-E", formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"],
                                                    help="End time for metrics (YYYY-MM-DD or YYYY-MM-DDTHH:MM:S)."),
        limit: int = typer.Option(20, "--limit", "-l", help="Limit the number of aggregated agent run results."),
):
    """
    Retrieves and displays aggregated agent run metrics from the database.
    """
    logger.info("Retrieving agent run metrics...")

    if not _is_db_configured():
        typer.secho("Database is not configured. Please run 'ragnetic configure' to set up a database for metrics.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    engine = _get_sync_db_engine()

    try:
        with engine.connect() as connection:
            stmt = select(
                chat_sessions_table.c.agent_name,
                func.count(agent_runs.c.run_id).label("total_runs"),
                func.avg(
                    func.julianday(agent_runs.c.end_time) - func.julianday(agent_runs.c.start_time)
                ).label("avg_duration_days"),  # SQLite specific for duration
                func.sum(
                    expression.case((agent_runs.c.status == 'completed', 1), else_=0)
                ).label("completed_runs"),
                func.sum(
                    expression.case((agent_runs.c.status == 'failed', 1), else_=0)
                ).label("failed_runs")
            ).join(
                chat_sessions_table, agent_runs.c.session_id == chat_sessions_table.c.id
            )

            conditions = []
            if agent_name:
                conditions.append(chat_sessions_table.c.agent_name == agent_name)
            if status:
                conditions.append(agent_runs.c.status == status)
            if start_time:
                conditions.append(agent_runs.c.start_time >= start_time)
            if end_time:
                conditions.append(agent_runs.c.end_time <= end_time)

            if conditions:
                stmt = stmt.where(*conditions)

            stmt = stmt.group_by(chat_sessions_table.c.agent_name)
            stmt = stmt.order_by(func.count(agent_runs.c.run_id).desc())
            if limit:
                stmt = stmt.limit(limit)

            results = connection.execute(stmt).fetchall()

        if not results:
            message = "No agent run metrics found in the database."
            if agent_name:
                message += f" for agent: '{agent_name}'"
            typer.secho(message, fg=typer.colors.YELLOW)
            return

        df = pd.DataFrame(results, columns=[
            "Agent Name", "Total Runs", "Avg Duration (Days)",
            "Completed Runs", "Failed Runs"
        ])

        df["Avg Duration (s)"] = df["Avg Duration (Days)"].fillna(0) * 86400
        df["Avg Duration (s)"] = df["Avg Duration (s)"].map(lambda x: f"{x:.2f}")

        df["Success Rate"] = df["Completed Runs"] / df["Total Runs"]
        df["Failure Rate"] = df["Failed Runs"] / df["Total Runs"]

        df = df.fillna(0)  # Handle NaN from division by zero or missing data

        df["Success Rate"] = df["Success Rate"].map(lambda x: f"{x:.2%}")
        df["Failure Rate"] = df["Failure Rate"].map(lambda x: f"{x:.2%}")

        df = df[[
            "Agent Name", "Total Runs", "Success Rate", "Failure Rate",
            "Avg Duration (s)", "Completed Runs", "Failed Runs"
        ]]

        typer.secho("\n--- Agent Run Metrics Summary ---", bold=True, fg=typer.colors.CYAN)
        typer.echo(df.to_string(index=False))
        typer.secho(
            "\nNote: Average duration calculation is based on SQL's `julianday` function, which may vary slightly across database types (e.g., PostgreSQL).",
            fg=typer.colors.BLUE)

    except Exception as e:
        logger.error(f"An error occurred while fetching agent run metrics: {e}", exc_info=True)
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

@training_app.command("apply", help="Submit a fine-tuning job defined in a YAML configuration file.")
def training_apply_command(
    config_file: Path = typer.Option(..., "--file", "-f", exists=True, file_okay=True, dir_okay=False,
                                       help="Path to the fine-tuning job YAML configuration file (e.g., training_configs/my_job.yaml)."),
):
    """
    Reads a YAML configuration file for a fine-tuning job, validates it,
    and submits it to the RAGnetic API for asynchronous processing.
    """
    try:
        # 1. Load YAML content
        typer.secho(f"Loading fine-tuning configuration from: {config_file}", fg=typer.colors.CYAN)
        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)

        # 2. Validate YAML against Pydantic schema
        job_config = FineTuningJobConfig(**yaml_content)
        typer.secho(f"Configuration loaded and validated for job: '{job_config.job_name}'", fg=typer.colors.GREEN)

        # 3. Get API key and server URL
        server_url = _get_server_url()
        api_key = _get_api_key_for_cli()
        headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        url = f"{server_url}/training/apply"

        # 4. Send to API
        typer.secho(f"Submitting fine-tuning job '{job_config.job_name}' to RAGnetic API...", fg=typer.colors.CYAN)
        response = requests.post(url, headers=headers, json=job_config.model_dump(), timeout=120) # Use model_dump() for JSON serialization
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        # 5. Process API response
        job_info = response.json()
        typer.secho(f"Fine-tuning job '{job_info['job_name']}' (ID: {job_info['adapter_id']}) dispatched successfully.", fg=typer.colors.GREEN)
        typer.echo(f"Status: {job_info['training_status']}")
        typer.echo(f"Model will be saved to: {job_info['adapter_path']}")
        typer.echo("You can monitor its progress with: " + typer.style(f"ragnetic training status <adapter_id>", bold=True))

    except FileNotFoundError:
        typer.secho(f"Error: Configuration file not found at '{config_file}'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except yaml.YAMLError as e:
        typer.secho(f"Error parsing YAML file '{config_file}': {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        if isinstance(e, requests.exceptions.HTTPError):
            typer.echo(f"API Response Error: {e.response.text}")
        raise typer.Exit(code=1)


@training_app.command("status", help="Check the status of a specific fine-tuning job by its adapter ID.")
def training_status_command(
    adapter_id: str = typer.Argument(..., help="The unique adapter ID of the fine-tuning job to check."),
):
    """
    Retrieves and displays the current status and detailed metadata for a specific
    fine-tuning job from the RAGnetic API.
    """
    try:
        server_url = _get_server_url()
        api_key = _get_api_key_for_cli()
        headers = {"X-API-Key": api_key}
        url = f"{server_url}/training/jobs/{adapter_id}"

        typer.secho(f"Fetching status for fine-tuning job '{adapter_id}'...", fg=typer.colors.CYAN)
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        job_info = response.json()
        typer.secho(f"\n--- Fine-tuning Job Details for '{job_info.get('job_name', 'N/A')}' (ID: {job_info['adapter_id']}) ---", bold=True, fg=typer.colors.CYAN)

        # Pretty print job details
        for key, value in job_info.items():
            if key == "hyperparameters" and value:
                typer.echo(f"  {key.replace('_', ' ').title()}:")
                for hp_key, hp_val in value.items():
                    typer.echo(f"    - {hp_key}: {hp_val}")
            elif key in ["final_loss", "validation_loss", "gpu_hours_consumed", "estimated_training_cost_usd"] and value is not None:
                if "cost_usd" in key:
                    typer.echo(f"  {key.replace('_', ' ').title()}: ${value:.6f}")
                elif "hours" in key:
                    typer.echo(f"  {key.replace('_', ' ').title()}: {value:.2f} hours")
                else:
                    typer.echo(f"  {key.replace('_', ' ').title()}: {value:.4f}")
            elif isinstance(value, str) and ("_at" in key or "timestamp" in key) and len(value) >= 19 and "T" in value: # Basic ISO format check
                try:
                    dt_object = datetime.fromisoformat(value.replace('Z', '+00:00')) # Handle 'Z' for UTC
                    typer.echo(f"  {key.replace('_', ' ').title()}: {dt_object.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                except ValueError:
                    typer.echo(f"  {key.replace('_', ' ').title()}: {value}") # Fallback
            else:
                typer.echo(f"  {key.replace('_', ' ').title()}: {value}")

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error fetching job status: {e}", fg=typer.colors.RED)
        if hasattr(e, 'response') and e.response is not None and e.response.text:
            typer.echo(f"API Response: {e.response.text}")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@training_app.command("list-models", help="List all available fine-tuned models (completed jobs).")
def training_list_models_command(
    status_filter: Optional[FineTuningStatus] = typer.Option(None, "--status", "-s", help="Filter by training status (e.g., 'completed', 'failed')."),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-b", help="Filter by the base LLM model name (case-insensitive partial match)."),
    job_name: Optional[str] = typer.Option(None, "--job-name", "-j", help="Filter by the user-defined job name (case-insensitive partial match)."),
    limit: int = typer.Option(100, "--limit", help="Maximum number of models to return."),
    offset: int = typer.Option(0, "--offset", help="Number of models to skip (for pagination)."),
):
    """
    Retrieves and displays a list of all fine-tuned models from the RAGnetic API,
    with options for filtering and pagination.
    """
    try:
        server_url = _get_server_url()
        api_key = _get_api_key_for_cli()
        headers = {"X-API-Key": api_key}
        url = f"{server_url}/training/models"

        params = {"limit": limit, "offset": offset}
        if status_filter:
            params["status_filter"] = status_filter.value
        if base_model:
            params["base_model_name"] = base_model
        if job_name:
            params["job_name"] = job_name

        typer.secho("Fetching fine-tuned models...", fg=typer.colors.CYAN)
        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()

        models = response.json()
        if not models:
            typer.secho("No fine-tuned models found matching the criteria.", fg=typer.colors.YELLOW)
            return

        typer.secho("\n--- Available Fine-Tuned Models ---", bold=True, fg=typer.colors.CYAN)
        df = pd.DataFrame(models)

        # Select and reorder columns for display
        display_cols = [
            "adapter_id", "job_name", "base_model_name", "training_status",
            "final_loss", "validation_loss", "gpu_hours_consumed",
            "estimated_training_cost_usd", "created_at", "adapter_path"
        ]
        # Filter to only columns that exist in the DataFrame
        df_display = df[[col for col in display_cols if col in df.columns]]

        # Format numerical columns for better readability
        if "estimated_training_cost_usd" in df_display.columns:
            df_display["estimated_training_cost_usd"] = df_display["estimated_training_cost_usd"].apply(lambda x: f"${x:.6f}" if pd.notna(x) else "N/A")
        if "final_loss" in df_display.columns:
            df_display["final_loss"] = df_display["final_loss"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        if "validation_loss" in df_display.columns:
            df_display["validation_loss"] = df_display["validation_loss"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        if "gpu_hours_consumed" in df_display.columns:
            df_display["gpu_hours_consumed"] = df_display["gpu_hours_consumed"].apply(lambda x: f"{x:.2f} hrs" if pd.notna(x) else "N/A")

        # Format datetime columns
        for col in ["created_at", "updated_at"]:
            if col in df_display.columns:
                df_display[col] = pd.to_datetime(df_display[col]).dt.strftime('%Y-%m-%d %H:%M UTC')

        typer.echo(df_display.to_string(index=False))

    except requests.exceptions.RequestException as e:
        typer.secho(f"Error listing fine-tuned models: {e}", fg=typer.colors.RED)
        if hasattr(e, 'response') and e.response is not None and e.response.text:
            typer.echo(f"API Response: {e.response.text}")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)



@dataset_app.command("prepare", help="Prepare a raw dataset for fine-tuning using a YAML configuration.")
def dataset_prepare_command(
    # CHANGE HERE: From typer.Argument to typer.Option
    config_file: Path = typer.Option(..., "--file", "-f", exists=True, file_okay=True, dir_okay=False,
                                       help="Path to the dataset preparation YAML configuration file (e.g., data_prep_configs/my_prep.yaml)."),
):
    """
    Reads a YAML configuration file for dataset preparation, validates it,
    and processes the raw input data into a format suitable for fine-tuning.
    """
    try:
        # 1. Load YAML content
        typer.secho(f"Loading dataset preparation configuration from: {config_file}", fg=typer.colors.CYAN)
        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)

        # 2. Validate YAML against Pydantic schema
        prep_config = DatasetPreparationConfig(**yaml_content)
        typer.secho(f"Configuration loaded and validated for dataset preparation: '{prep_config.prep_name}'",
                    fg=typer.colors.GREEN)

        # Ensure output directory exists
        output_path = Path(prep_config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Choose and execute the appropriate loader/processor
        if prep_config.format_type == "jsonl-instruction":
            loader = JsonlInstructionLoader(prep_config.input_file)
            prepared_data = loader.load()  # This will raise ValueError if format is wrong

            with open(output_path, 'w', encoding='utf-8') as f:
                for record in prepared_data:
                    f.write(json.dumps(record) + "\n")

            typer.secho(f"Successfully prepared {len(prepared_data)} records and saved to '{output_path}'.",
                        fg=typer.colors.GREEN)
        elif prep_config.format_type == "conversational-jsonl":
            loader = ConversationalJsonlLoader(prep_config.input_file)
            prepared_data = loader.load()

            with open(output_path, 'w', encoding='utf-8') as f:
                for record in prepared_data:
                    f.write(json.dumps(record) + "\n")  # Conversational data is also typically JSONL

            typer.secho(f"Successfully prepared {len(prepared_data)} records and saved to '{output_path}'.",
                        fg=typer.colors.GREEN)
        else:
            typer.secho(f"Error: Unsupported format type '{prep_config.format_type}'.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    except FileNotFoundError:
        typer.secho(f"Error: Configuration or input file not found. Check paths in '{config_file}'.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except yaml.YAMLError as e:
        typer.secho(f"Error parsing YAML file '{config_file}': {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred during dataset preparation: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="deploy-orchestrator", help="Deploys an orchestrator agent and all its sub-agents.")
def deploy_orchestrator(
        orchestrator_name: str = typer.Argument(..., help="The name of the orchestrator agent to deploy."),
        force: bool = typer.Option(False, "--force", "-f", help="Bypass confirmation and overwrite existing data."),
):
    """
    Deploys an orchestrator agent by deploying all agents listed in its roster.
    """
    _validate_agent_name_cli(orchestrator_name)
    logger = logging.getLogger(__name__)

    try:
        # Load the orchestrator's configuration using the new helper function
        orchestrator_config = _load_orchestrator_config(orchestrator_name)
    except FileNotFoundError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"Error loading orchestrator config '{orchestrator_name}': {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"\n--- Deploying Orchestrator: '{orchestrator_name}' ---", bold=True, fg=typer.colors.CYAN)

    # Deploy the orchestrator itself
    deploy_agent_by_name(orchestrator_name, force)

    # Iterate through the roster and deploy each sub-agent
    if orchestrator_config.roster:
        typer.secho("\n--- Deploying Sub-Agents from Roster ---", bold=True, fg=typer.colors.CYAN)
        for sub_agent_name in orchestrator_config.roster:
            typer.echo(f"\nFound sub-agent '{sub_agent_name}' in the roster. Starting deployment...")
            try:
                # Reuse the existing deploy function
                deploy_agent_by_name(sub_agent_name, force)
                typer.secho(f"Successfully deployed sub-agent '{sub_agent_name}'.", fg=typer.colors.GREEN)
            except typer.Exit as e:
                typer.secho(f"Deployment of sub-agent '{sub_agent_name}' failed with code {e.code}. Skipping remaining sub-agents.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            except Exception as e:
                typer.secho(f"An unexpected error occurred during deployment of sub-agent '{sub_agent_name}': {e}. Skipping remaining sub-agents.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
    else:
        typer.secho("No sub-agents found in the orchestrator's roster. Deployment complete.", fg=typer.colors.YELLOW)

    typer.secho(f"\n--- Orchestrator '{orchestrator_name}' and all sub-agents deployed successfully! ---", bold=True, fg=typer.colors.GREEN)


@app.command(name="inspect-orchestration", help="Inspects a full orchestration, showing all sub-runs in a tree view.")
def inspect_orchestration(
        run_id: str = typer.Argument(..., help="The unique ID of the top-level orchestration run to inspect.")
):
    """
    Fetches and displays a hierarchical view of a nested orchestration,
    starting from a top-level workflow or agent run ID.
    """
    if not _is_db_configured():
        typer.secho("Audit trails require a database. Please configure one using 'ragnetic configure'.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    engine = _get_sync_db_engine()

    try:
        with engine.connect() as connection:
            top_run_dict = None
            run_type = None

            # First, try to fetch the top-level run from the workflow_runs table
            top_run_stmt = select(
                workflow_runs_table.c.run_id,
                workflow_runs_table.c.status,
                workflows_table.c.name.label("run_name"),
                workflow_runs_table.c.start_time,
                workflow_runs_table.c.end_time,
            ).join(
                workflows_table, workflow_runs_table.c.workflow_id == workflows_table.c.id
            ).where(workflow_runs_table.c.run_id == run_id)

            top_run_row = connection.execute(top_run_stmt).first()

            # If not found in workflow_runs, try the agent_runs table
            if top_run_row:
                top_run_dict = {**top_run_row._mapping}
                run_type = "Workflow"
            else:
                top_run_stmt = select(
                    agent_runs.c.run_id,
                    agent_runs.c.status,
                    chat_sessions_table.c.agent_name.label("run_name"),
                    agent_runs.c.start_time,
                    agent_runs.c.end_time,
                ).join(
                    chat_sessions_table, agent_runs.c.session_id == chat_sessions_table.c.id
                ).where(agent_runs.c.run_id == run_id)

                top_run_row = connection.execute(top_run_stmt).first()
                if top_run_row:
                    top_run_dict = {**top_run_row._mapping}
                    run_type = "Agent"

            if not top_run_dict:
                typer.secho(f"Error: Top-level run with ID '{run_id}' not found in either workflow or agent tables.", fg=typer.colors.RED)
                raise typer.Exit(code=1)

            typer.secho(f"\n--- Orchestration Tree for Run: {top_run_dict['run_id']} ---", bold=True)
            status_color = typer.colors.GREEN if top_run_dict['status'] == 'completed' else (
                typer.colors.YELLOW if top_run_dict['status'] in ['running', 'paused'] else typer.colors.RED)
            duration = (top_run_dict['end_time'] - top_run_dict['start_time']).total_seconds() if top_run_dict['end_time'] else "N/A"
            duration_str = f"{duration:.2f}s" if isinstance(duration, (int, float)) else duration

            typer.echo(f"Root Run ID: {typer.style(top_run_dict['run_id'], fg=typer.colors.CYAN)}")
            typer.echo(f"  Type: {run_type}")
            typer.echo(f"  Name: {top_run_dict['run_name']}")
            typer.echo(f"  Status: {typer.style(top_run_dict['status'], fg=status_color)}")
            typer.echo(f"  Duration: {duration_str}")

            # Use the new recursive helper function to build the tree from the top-level run ID
            child_tree_lines = _fetch_and_format_orchestration_tree(connection, run_id, indent="")
            for line in child_tree_lines:
                typer.echo(line)

    except Exception as e:
        typer.secho(f"An error occurred while inspecting the orchestration: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

@app.command(name="build-sandbox", help="Builds the Docker image for the LambdaTool execution sandbox.")
def build_sandbox_image():
    """
    Builds the Docker image for the LambdaTool.
    """
    typer.secho("Building RAGnetic LambdaTool sandbox image...", bold=True)
    try:
        executor = LocalDockerExecutor()
        executor._build_image("ragnetic-lambda:py310-cpu")
    except Exception as e:
        typer.secho(f"Failed to build sandbox image: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command(name="test-lambda", help="Submits a test job to the LambdaTool and displays the result.")
def test_lambda_tool(
        job_name: str = typer.Argument("simple_test_job", help="A name for the test job."),
        code: str = typer.Option(
            "import math; result = math.sqrt(144); print(f'The result is {result}')",
            "--code", "-c",
            help="The Python code to execute in the sandbox."
        ),
):
    """
    Submits a test job to the LambdaTool via the API and displays the final output.
    """
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    if not api_key:
        typer.secho("Error: No API key found. Please log in or set a master key.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    url = f"{server_url}/lambda/execute"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    # Use the LambdaRequestPayload schema to build a valid request
    test_payload = LambdaRequestPayload(
        mode="code",
        code=code,
        resource_spec=LambdaResourceSpec(cpu="1", memory_gb=1),
    )

    response = None
    try:
        typer.secho(f"Submitting test job '{job_name}'...", fg=typer.colors.CYAN)
        response = requests.post(url, headers=headers, json=test_payload.model_dump(), timeout=15)
        response.raise_for_status()
        run_id = response.json()["run_id"]
        typer.secho(f"Job submitted successfully. Run ID: {run_id}", fg=typer.colors.GREEN)
        typer.echo("Waiting for job to complete...")

        # Poll the API to check job status
        status_url = f"{server_url}/lambda/runs/{run_id}"
        max_retries = 30
        for i in range(max_retries):
            time.sleep(1)  # Wait 1 second between polls
            status_response = requests.get(status_url, headers=headers)
            status_response.raise_for_status()
            run_status = status_response.json()["status"]

            if run_status in ["completed", "failed"]:
                typer.secho(f"\nJob finished with status: {run_status.upper()}", bold=True,
                            fg=typer.colors.GREEN if run_status == "completed" else typer.colors.RED)

                # Display final state
                run_details = status_response.json()
                if run_details.get("final_state"):
                    typer.secho("\n--- Final Output ---", bold=True)
                    typer.echo(json.dumps(run_details["final_state"], indent=2))



                return  # Exit successfully

        typer.secho("\nJob timed out. Check logs for details.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    except requests.exceptions.RequestException as e:
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        if response and response.text:
            typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)


@app.command(name="inspect-lambda", help="Inspect a specific LambdaTool run and its details.")
def inspect_lambda_run(
    run_id: str = typer.Argument(..., help="The unique ID of the LambdaTool run to inspect.")
):
    """
    Fetches and displays the details for a single LambdaTool run, including logs.
    """
    server_url = _get_server_url()
    api_key = _get_api_key_for_cli()
    if not api_key:
        typer.secho("Error: No API key found. Please log in or set a master key.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    url = f"{server_url}/lambda/runs/{run_id}"
    headers = {"X-API-Key": api_key}
    response = None

    try:
        typer.secho(f"Fetching details for LambdaTool run: {run_id}...", fg=typer.colors.CYAN)
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        run_data = response.json()

        typer.secho("\nLambdaTool Run Details:", bold=True)
        typer.echo(f"  Run ID: {typer.style(run_data.get('run_id'), fg=typer.colors.CYAN)}")
        typer.echo(f"  Status: {typer.style(run_data.get('status', 'N/A'), fg=typer.colors.GREEN if run_data.get('status') == 'completed' else typer.colors.RED)}")
        typer.echo(f"  User ID: {run_data.get('user_id', 'N/A')}")
        typer.echo(f"  Start Time: {run_data.get('start_time', 'N/A')}")
        typer.echo(f"  End Time: {run_data.get('end_time', 'N/A')}")
        typer.echo(f"  Error Message: {run_data.get('error_message', 'None')}")

        if run_data.get('initial_request'):
            typer.secho("\nInitial Request:", bold=True, fg=typer.colors.MAGENTA)
            typer.echo(json.dumps(run_data['initial_request'], indent=2))

        if run_data.get('final_state'):
            typer.secho("\nFinal State:", bold=True, fg=typer.colors.GREEN)
            typer.echo(json.dumps(run_data['final_state'], indent=2))

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            typer.secho(f"Error: Run with ID '{run_id}' not found.", fg=typer.colors.RED)
        else:
            typer.secho(f"HTTP Error: {e}", fg=typer.colors.RED)
            if response and response.text:
                typer.echo(f"API Response: {response.text}")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An unexpected error occurred while inspecting the run: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    setup_cli_logging()
    app()