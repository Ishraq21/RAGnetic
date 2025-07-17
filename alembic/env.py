# alembic/env.py
import sys
from pathlib import Path

print(">>> ALEMBIC ENV.PY LOADING <<<")

# --- PATH CORRECTION (MUST BE FIRST) ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- END PATH CORRECTION ---

import logging
from logging.config import fileConfig
from sqlalchemy import create_engine, pool
from alembic import context

# Now that sys.path is fixed, import your app’s models
from app.db.models import metadata

# Alembic Config object
config = context.config

# Set up logging from alembic.ini (or fall back to DEBUG)
if config.config_file_name:
    fileConfig(config.config_file_name)
else:
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

logger = logging.getLogger(__name__)

# --- Determine the database URL dynamically ---
from app.core.config import get_db_connection, get_path_settings, get_memory_storage_config, get_log_storage_config

paths = get_path_settings()
mem_cfg = get_memory_storage_config()
log_cfg = get_log_storage_config()
conn_name = mem_cfg.get("connection_name") if mem_cfg.get("type") in ["db", "sqlite"] else log_cfg.get("connection_name")

if conn_name:
    db_url = get_db_connection(conn_name).replace("+aiosqlite", "") # Use sync driver
    logger.debug("Using DB connection from config: %s", conn_name)
else:
    # Fallback to SQLite file
    sqlite_path = paths["MEMORY_DIR"] / "ragnetic.db"
    db_url = f"sqlite:///{sqlite_path.resolve()}"
    logger.debug("No DB config found; falling back to SQLite: %s", sqlite_path)

config.set_main_option("sqlalchemy.url", db_url)

# This is the metadata Alembic will compare against
target_metadata = metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (apply to DB)."""
    connectable = create_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # It enables "batch mode" for altering tables.
            render_as_batch=True
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


# Choose the correct run function based on mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()