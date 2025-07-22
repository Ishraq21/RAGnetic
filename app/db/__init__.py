from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from app.core.config import get_db_connection, get_db_connection_config, get_path_settings
from app.db.models import metadata
from urllib.parse import urlparse, urlunparse
from pathlib import Path
import logging
from sqlalchemy import create_engine


logger = logging.getLogger(__name__)

# Asynchronous engine for FastAPI endpoints
async_engine = None
AsyncSessionLocal: async_sessionmaker[AsyncSession] = None
DATABASE_URL_SYNC: str = None  # Store the sync URL for Alembic setup if needed


def initialize_db_connections(conn_name: str):
    """Initializes async and sync DB connections based on config."""
    global async_engine, AsyncSessionLocal, DATABASE_URL_SYNC

    conn_str = get_db_connection(conn_name)
    DATABASE_URL_SYNC = conn_str

    parsed_url = urlparse(conn_str)

    async_scheme = ""
    # Corrected logic for handling SQLite URLs with proper path formatting
    if parsed_url.scheme.startswith("sqlite"):
        async_scheme = "sqlite+aiosqlite"
        db_config = get_db_connection_config()
        if not db_config:
            raise ValueError("SQLite database configuration not found.")
        db_path = Path(db_config.get('database_path', ''))
        project_root = get_path_settings()["PROJECT_ROOT"]
        if not db_path.is_absolute():
            db_path = project_root / db_path
        # Ensure the correct URL format for absolute paths
        async_db_url = f"{async_scheme}:///{db_path.resolve()}"
    elif parsed_url.scheme.startswith("postgresql"):
        async_scheme = "postgresql+asyncpg"
        async_db_url = urlunparse(parsed_url._replace(scheme=async_scheme))
    elif parsed_url.scheme.startswith("mysql"):
        async_scheme = "mysql+aiomysql"
        async_db_url = urlunparse(parsed_url._replace(scheme=async_scheme))
    else:
        # For other sync DBs
        async_scheme = parsed_url.scheme
        async_db_url = urlunparse(parsed_url._replace(scheme=async_scheme))
        logger.warning(
            f"Using synchronous DB URL for async operations. Consider an async driver for full async benefits.")

    if not async_db_url:
        raise ValueError("Could not determine asynchronous database URL from configuration.")

    async_engine = create_async_engine(async_db_url, echo=False)
    AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=async_engine)
    logger.info(f"Database connections initialized for '{conn_name}'.")


async def get_db() -> AsyncSession:
    """Dependency to provide an asynchronous database session."""
    if AsyncSessionLocal is None:
        raise RuntimeError("Database connections not initialized. Call initialize_db_connections first.")
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()



def get_sync_db_engine():
    """Helper to get a synchronous SQLAlchemy engine for background tasks."""
    from app.core.config import get_memory_storage_config, get_log_storage_config, get_db_connection

    mem_cfg = get_memory_storage_config()
    log_cfg = get_log_storage_config()
    conn_name = (
        mem_cfg.get("connection_name")
        if mem_cfg.get("type") in ["db", "sqlite"]
        else log_cfg.get("connection_name")
    )
    if not conn_name:
        raise RuntimeError("No database connection is configured for background worker.")

    conn_str = get_db_connection(conn_name)
    sync_conn_str = conn_str.replace('+aiosqlite', '').replace('+asyncpg', '')
    return create_engine(sync_conn_str)