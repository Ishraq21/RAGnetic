# app/cli.py
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
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url

from sqlalchemy.exc import SQLAlchemyError
import requests
from urllib.parse import urlparse
from app.schemas.agent import DataSource
from google.oauth2 import service_account

# Initialize a default logger for CLI startup FIRST, so it's always bound.
logger = logging.getLogger(__name__)

from app.core.config import get_path_settings, get_api_key, get_server_api_keys, get_log_storage_config, \
    get_memory_storage_config, get_db_connection, get_cors_settings
from app.core.structured_logging import JSONFormatter
from app.evaluation.dataset_generator import generate_test_set
from app.evaluation.benchmark import run_benchmark
from app.pipelines.embed import embed_agent_data
from app.watcher import start_watcher
import pytest
from app.core.validation import is_valid_agent_name_cli
from app.db.models import metadata

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

# --- Load Environment Variables ---
# Load .env file from the project root for all CLI commands
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


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
        # Wrap value in quotes if it contains spaces or special characters
        formatted_value = f"'{value}'" if any(c in value for c in ' #=') else value
        found = False
        for i, line in enumerate(env_lines):
            if line.strip().startswith(f"{key}="):
                env_lines[i] = f"{key}={formatted_value}\n"
                found = True
                break
        if not found:
            # Ensure there's a newline before adding a new key
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
    else:
        root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    if json_format:
        formatter = JSONFormatter()
        file_handler = TimedRotatingFileHandler(
            os.path.join(_LOGS_DIR, "ragnetic_app.jsonl"), when="midnight", interval=1, backupCount=7, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
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
    setup_logging()
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

        # Use a synchronous-compatible dialect for the check
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
    """
    Wraps Alembic's 'revision --autogenerate' command by calling it in a separate subprocess
    to ensure a clean execution environment.
    """
    setup_logging(json_format=False)
    if not _is_db_configured():
        typer.secho("No explicit database configured. Using default SQLite for migrations.", fg=typer.colors.YELLOW)

    typer.echo(f"Generating new migration script with message: '{message}'...")

    alembic_ini_path = str(_PROJECT_ROOT / "alembic.ini")
    command = [
        "alembic",
        "-c", alembic_ini_path,
        "revision",
        "--autogenerate",
        "-m", message
    ]

    typer.secho(f"Running command: {' '.join(command)}", fg=typer.colors.YELLOW)

    # We run this from the project root to ensure all paths are resolved correctly.
    result = subprocess.run(command, capture_output=True, text=True, cwd=_PROJECT_ROOT)

    if result.returncode == 0:
        # Alembic's output goes to stdout on success
        typer.echo(result.stdout)
        typer.secho("Migration script generated successfully.", fg=typer.colors.GREEN)
    else:
        # On failure, errors are often in stderr
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
    """
    Wraps Alembic's 'upgrade' command.
    """
    setup_logging(json_format=False)
    if not _is_db_configured():
        typer.secho("No explicit database configured. Using default SQLite for migrations.", fg=typer.colors.YELLOW)

    typer.echo(f"Applying database migrations to revision: {revision}...")
    alembic_cfg = alembic.config.Config(str(_PROJECT_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("loglevel", "DEBUG")
    alembic.command.upgrade(alembic_cfg, revision)
    typer.secho("Database migration complete.", fg=typer.colors.GREEN)


@app.command(name="sync", help="Manually stamps the database with a specific migration revision.")
def sync_db_revision(
        revision: str = typer.Argument("head",
                                       help="The revision to stamp the database with (e.g., 'head' for latest).")
):
    """
    Wraps Alembic's 'stamp' command. Use with extreme caution!
    """
    setup_logging(json_format=False)
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
    alembic_cfg.set_main_option("loglevel", "DEBUG")
    alembic.command.stamp(alembic_cfg, revision)
    typer.secho(f"Database successfully stamped to revision '{revision}'.", fg=typer.colors.GREEN)


@app.command(help="Configure system settings, databases, and secrets.")
def configure():
    setup_logging()
    typer.secho("--- RAGnetic System Configuration ---", bold=True)

    config = configparser.ConfigParser()
    config.read(_CONFIG_FILE)

    # --- Configure Server Settings ---
    if typer.confirm("\nDo you want to configure SERVER settings (host, port, CORS)?", default=True):
        if 'SERVER' not in config: config.add_section('SERVER')
        host = typer.prompt("Enter server host", default=config.get('SERVER', 'host', fallback='127.0.0.1'))
        port = typer.prompt("Enter server port", default=config.get('SERVER', 'port', fallback='8000'))
        json_logs = typer.confirm("Use JSON format for console logs?",
                                  default=config.getboolean('SERVER', 'json_logs', fallback=False))

        if typer.confirm("Configure CORS allowed origins?", default=True):
            current_origins = os.environ.get("CORS_ALLOWED_ORIGINS",
                                             config.get('SERVER', 'cors_allowed_origins', fallback='*'))
            origins_str = typer.prompt(
                "Enter comma-separated origins (e.g., http://localhost:3000,https://my-app.com)",
                default=current_origins
            )
            if typer.confirm("Save CORS settings to the local .env file (recommended)?", default=True):
                _update_env_file({"CORS_ALLOWED_ORIGINS": origins_str})
                if config.has_option('SERVER', 'cors_allowed_origins'):
                    config.remove_option('SERVER', 'cors_allowed_origins')  # Prefer .env
                typer.secho("CORS settings saved to .env file.", fg=typer.colors.GREEN)
            else:
                config.set('SERVER', 'cors_allowed_origins', origins_str)

        config.set('SERVER', 'host', host)
        config.set('SERVER', 'port', str(port))
        config.set('SERVER', 'json_logs', str(json_logs).lower())
        typer.secho("Server settings updated.", fg=typer.colors.GREEN)

    # --- Configure System Database Connections ---
    if typer.confirm("\nDo you want to configure SYSTEM database connections?", default=False):
        if 'DATABASE_CONNECTIONS' not in config: config.add_section('DATABASE_CONNECTIONS')

        DIALECT_MAP = {"postgresql": "postgresql+psycopg2", "mysql": "mysql+mysqlconnector", "sqlite": "sqlite"}

        while True:
            if not typer.confirm("Add or update a database connection?", default=True):
                break

            conn_name = typer.prompt("Enter a unique name for this connection (e.g., 'prod_db')")
            typer.secho(f"\n--- Configuring '{conn_name}' ---", bold=True)
            db_type = typer.prompt(f"Database type? Choose from: {list(DIALECT_MAP.keys())}", default="postgresql")
            while db_type not in DIALECT_MAP:
                typer.secho("Invalid selection.", fg=typer.colors.RED)
                db_type = typer.prompt(f"Database type? Choose from: {list(DIALECT_MAP.keys())}", default="postgresql")

            dialect = DIALECT_MAP[db_type]
            section_name = f"DATABASE_{conn_name}"

            if db_type == "sqlite":
                db_path = typer.prompt("Enter the path for the SQLite file (e.g., memory/ragnetic.db)")
                config[section_name] = {'dialect': dialect, 'database_path': db_path}
            else:
                username = typer.prompt("Username")
                host = typer.prompt("Host", default="localhost")
                port = typer.prompt("Port", default="5432" if db_type == "postgresql" else "3306")
                database = typer.prompt("Database Name")
                password = typer.prompt(f"Enter password for user '{username}'", hide_input=True)

                config[section_name] = {'dialect': dialect, 'username': username, 'host': host, 'port': port,
                                        'database': database}

                if password:
                    if typer.confirm(f"Save password for '{conn_name}' to the local .env file?", default=True):
                        password_env_var = f"{conn_name.upper()}_PASSWORD"
                        _update_env_file({password_env_var: password})
                        typer.secho(f"Password saved to the .env file.", fg=typer.colors.GREEN)
                else:
                    typer.secho("Password not provided. It will be prompted for at runtime.", fg=typer.colors.YELLOW)

            existing_conns = {c.strip() for c in config.get('DATABASE_CONNECTIONS', 'names', fallback='').split(',') if
                              c.strip()}
            existing_conns.add(conn_name)
            config.set('DATABASE_CONNECTIONS', 'names', ','.join(sorted(list(existing_conns))))
            typer.secho(f"\nConnection '{conn_name}' configured successfully.", fg=typer.colors.GREEN)

    # --- Configure Memory and Log Storage ---
    if typer.confirm("\nDo you want to configure MEMORY (chat history) storage?", default=False):
        if 'MEMORY_STORAGE' not in config: config.add_section('MEMORY_STORAGE')
        mem_type = typer.prompt("Enter memory storage type (sqlite, db)",
                                default=config.get('MEMORY_STORAGE', 'type', fallback='sqlite'))
        config.set('MEMORY_STORAGE', 'type', mem_type)
        if mem_type == 'db' or (
                mem_type == 'sqlite' and typer.confirm("Use a named connection for SQLite?", default=False)):
            conn_name = typer.prompt("Enter the database connection name to use",
                                     default=config.get('MEMORY_STORAGE', 'connection_name', fallback=""))
            config.set('MEMORY_STORAGE', 'connection_name', conn_name)

    if typer.confirm("\nDo you want to configure LOG storage?", default=False):
        if 'LOG_STORAGE' not in config: config.add_section('LOG_STORAGE')
        log_type = typer.prompt("Enter log storage type (file, db)",
                                default=config.get('LOG_STORAGE', 'type', fallback='file'))
        config.set('LOG_STORAGE', 'type', log_type)
        if log_type == 'db':
            conn_name = typer.prompt("Enter the database connection name to use",
                                     default=config.get('LOG_STORAGE', 'connection_name', fallback=""))
            table_name = typer.prompt("Enter the table name for logs",
                                      default=config.get('LOG_STORAGE', 'log_table_name', fallback='ragnetic_logs'))
            config.set('LOG_STORAGE', 'connection_name', conn_name)
            config.set('LOG_STORAGE', 'log_table_name', table_name)

    with open(_CONFIG_FILE, 'w') as configfile:
        config.write(configfile)
    typer.secho("\nConfiguration saved successfully to .ragnetic/config.ini", fg=typer.colors.GREEN)


@app.command(name="show-config", help="Displays the current system configurations.")
def show_config():
    """Reads and prints the contents of the .ragnetic/config.ini file."""
    setup_logging()
    typer.secho("--- Current RAGnetic Configuration ---", bold=True)

    if not os.path.exists(_CONFIG_FILE):
        typer.secho(f"Configuration file not found at: {_CONFIG_FILE}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    config = configparser.ConfigParser()
    config.read(_CONFIG_FILE)
    for section in config.sections():
        typer.secho(f"\n[{section}]", fg=typer.colors.CYAN, bold=True)
        for key, value in config.items(section):
            if not key.startswith('#'):
                typer.echo(f"  {key} = {value}")
    typer.echo("")


@app.command(name="check-system-db", help="Verifies and inspects connections to configured system databases.")
def check_system_db():
    """
    Checks database connectivity, displays server info, and verifies Alembic migration status.
    """
    setup_logging(json_format=False)
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

            # Use a synchronous-compatible driver for the check
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
    setup_logging()
    typer.secho("Initializing new RAGnetic project...", bold=True)
    paths_to_create = {
        "DATA_DIR", "AGENTS_DIR", "VECTORSTORE_DIR", "MEMORY_DIR",
        "LOGS_DIR", "TEMP_CLONES_DIR", "RAGNETIC_DIR", "BENCHMARK_DIR"
    }
    for key, path in _APP_PATHS.items():
        if key in paths_to_create and not os.path.exists(path):
            os.makedirs(path, mode=0o750, exist_ok=True)
            typer.echo(f"  - Created directory: {path}")

    if not os.path.exists(_CONFIG_FILE):
        config = configparser.ConfigParser()
        config['SERVER'] = {'host': '127.0.0.1', 'port': '8000', 'json_logs': 'false', 'cors_allowed_origins': '*'}
        # --- Bootstrap Default SQLite Database Configuration ---
        default_db_path = _MEMORY_DIR / "ragnetic.db"

        config['DATABASE_CONNECTIONS'] = {'names': 'default_sqlite'}
        config['DATABASE_default_sqlite'] = {
            'dialect': 'sqlite+aiosqlite',
            'database_path': str(default_db_path.relative_to(_PROJECT_ROOT))
        }
        config['MEMORY_STORAGE'] = {
            'type': 'sqlite',
            'connection_name': 'default_sqlite'
        }
        # ----------------------------------------------------

        with open(_CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
        typer.echo(f"  - Created config file: {_CONFIG_FILE}")

    typer.secho("\nProject initialized successfully!", fg=typer.colors.GREEN)

    # --- CORS WARNING ---
    typer.secho("\n--- SECURITY NOTICE: CORS ---", fg=typer.colors.YELLOW, bold=True)
    typer.secho(
        "By default, RAGnetic allows requests from all origins ('*') for ease of development.",
        fg=typer.colors.YELLOW
    )
    typer.secho(
        "For production, you should restrict this to your frontend's domain.",
        fg=typer.colors.YELLOW
    )
    typer.echo("You can change this using: " + typer.style("ragnetic configure", bold=True))

    typer.secho("\nNext steps:", bold=True)
    typer.echo("  1. Set your API keys: " + typer.style("ragnetic set-api-key", bold=True))
    typer.echo("  2. Configure a different database (optional): " + typer.style("ragnetic configure", bold=True))
    typer.echo("  3. Secure your server: " + typer.style("ragnetic set-server-key", bold=True))


@app.command(name="set-server-key", help="Generate and set a secret key for the server API.")
def set_server_key():
    setup_logging()
    new_key = secrets.token_hex(32)
    _update_env_file({"RAGNETIC_API_KEYS": new_key})
    typer.secho("Successfully set a new server API key in the .env file.", fg=typer.colors.GREEN)
    typer.echo("Your new key is: " + typer.style(new_key, bold=True))


@app.command(name="set-api-key", help="Set and save API keys to the secure .env file.")
def set_api():
    setup_logging()
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
    setup_logging()
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


@app.command(help="Starts the RAGnetic server.")
def start_server(
        host: str = typer.Option(None, help="Server host. Overrides config."),
        port: int = typer.Option(None, help="Server port. Overrides config."),
        reload: bool = typer.Option(False, "--reload", help="Enable auto-reloading for development."),
        json_logs: bool = typer.Option(None, "--json-logs", help="Output logs in JSON format. Overrides config."),
):
    config = configparser.ConfigParser()
    config.read(_CONFIG_FILE)
    final_host = host or config.get('SERVER', 'host', fallback="127.0.0.1")
    final_port = port or config.getint('SERVER', 'port', fallback=8000)
    final_json_logs = json_logs if json_logs is not None else config.getboolean('SERVER', 'json_logs', fallback=False)

    setup_logging(final_json_logs)

    if not get_server_api_keys():
        typer.secho("SECURITY WARNING: Server starting without an API key.", fg=typer.colors.YELLOW, bold=True)
        typer.secho("Run 'ragnetic set-server-key' to secure the API.", fg=typer.colors.YELLOW)

    # CORS Warning on server start
    if get_cors_settings() == ["*"]:
        typer.secho("SECURITY WARNING: Server is allowing requests from all origins ('*').", fg=typer.colors.YELLOW,
                    bold=True)
        typer.secho("This is not recommended for production. Use 'ragnetic configure' to set allowed origins.",
                    fg=typer.colors.YELLOW)

    if reload:
        logger.warning("Running in --reload mode.")

    uvicorn.run("app.main:app", host=final_host, port=final_port, reload=reload)


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
                            connect_args['connect_timeout'] = 5
                        elif not conn_str.startswith("sqlite"):
                            connect_args['timeout'] = 5

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