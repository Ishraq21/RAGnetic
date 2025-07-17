# app/db/__init__.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from app.core.config import get_db_connection
from app.db.models import metadata
from urllib.parse import urlparse, urlunparse

import logging

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
    if parsed_url.scheme.startswith("postgresql"):
        async_scheme = "postgresql+asyncpg"
    elif parsed_url.scheme.startswith("mysql"):
        async_scheme = "mysql+aiomysql"
    else:
        # For SQLite or other sync DBs, no conversion is needed
        async_scheme = parsed_url.scheme
        logger.warning(
            f"Using synchronous DB URL for async operations. Consider an async driver for full async benefits.")

    # Rebuild URL with the correct async scheme and original netloc, path, etc.
    async_db_url = urlunparse(parsed_url._replace(scheme=async_scheme))

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

