import json
import logging
import logging.config
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import create_engine, event, insert, Table
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from app.core.config import get_db_connection, get_log_storage_config, get_path_settings
from app.db.models import ragnetic_logs_table

LOGGING_QUEUE = None


class JSONFormatter(logging.Formatter):
    """Formats log records into a flat JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        obj: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # merge any extra_data
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            obj.update(extra)

        if record.exc_info:
            obj["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(obj)


class DatabaseLogHandler(logging.Handler):
    """Writes structured logs into a database table with robust SQLite concurrency."""

    def __init__(self, connection_name: str, table: Table):
        super().__init__()
        self.table = table
        self.engine = None

        if not connection_name:
            logging.getLogger(__name__).warning(
                "DatabaseLogHandler initialized without connection name; disabling."
            )
            return

        # Build a pure‐sync connection string
        raw = get_db_connection(connection_name)
        sync_dsn = raw.replace("+aiosqlite", "").replace("+asyncpg", "")

        # For SQLite, we want check_same_thread=False and a higher timeout
        connect_args: Dict[str, Any] = {}
        poolclass = None # Use default pool for Postgres/MySQL
        if sync_dsn.startswith("sqlite"):
            connect_args = {
                "timeout": 20,
                "check_same_thread": False
            }
            # NullPool = no recycling of connections (avoids stale locked handles)
            poolclass = NullPool

        self.engine = create_engine(
            sync_dsn,
            connect_args=connect_args,
            poolclass=poolclass,
        )

        # Ensure WAL is enabled on every new SQLite connection
        @event.listens_for(self.engine, "connect")
        def _enable_wal(dbapi_conn, conn_record):
            if isinstance(dbapi_conn, sqlite3.Connection):
                cursor = dbapi_conn.cursor()
                try:
                    cursor.execute("PRAGMA journal_mode=WAL;")
                finally:
                    cursor.close()

        logging.getLogger(__name__).info("DatabaseLogHandler engine initialized (WAL + NullPool).")

    def emit(self, record: logging.LogRecord):
        if not self.engine:
            return

        entry: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            entry["details"] = extra

        # drop None values
        entry = {k: v for k, v in entry.items() if v is not None}

        try:
            # each log goes in its own transaction
            with self.engine.begin() as conn:
                conn.execute(insert(self.table).values(entry))

        except OperationalError as e:
            # retry once after a short sleep if locked
            if "locked" in str(e).lower():
                time.sleep(0.05)
                try:
                    with self.engine.begin() as conn:
                        conn.execute(insert(self.table).values(entry))
                except Exception as e2:
                    print(f"DatabaseLogHandler retry failed: {e2}")
            else:
                print(f"DatabaseLogHandler operational error: {e}")

        except Exception as e:
            print(f"DatabaseLogHandler unexpected error: {e}")


def get_logging_config() -> Dict[str, Any]:
    """
    Returns a dictConfig-compatible config that sets up console and optional
    FILE logging. Database logging is now handled manually in main.py.
    """
    paths = get_path_settings()
    paths["LOGS_DIR"].mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            "json": {"()": JSONFormatter},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default"
            },
            "metrics_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "json",
                "filename": str(paths["LOGS_DIR"] / "ragnetic_metrics.json"),
                "when": "midnight",
                "backupCount": 7,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "ragnetic": {"handlers": ["console"], "level": "INFO", "propagate": False},
            "app.workflows": {"handlers": ["console"], "level": "INFO", "propagate": False},
            "ragnetic.metrics": {"handlers": ["metrics_file"], "level": "INFO", "propagate": False},
            "uvicorn": {"level": "INFO"},
            "sqlalchemy.engine": {"level": "WARNING"},
            "langchain": {"level": "WARNING"},
        },
        "root": {"handlers": ["console"], "level": "INFO"},
    }

    storage = get_log_storage_config()
    if storage.get("type") == "file":
        cfg["handlers"]["app_file"] = {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "default",
            "filename": str(paths["LOGS_DIR"] / "ragnetic_app.log"),
            "when": "midnight",
            "backupCount": 7,
            "encoding": "utf-8",
        }
        cfg["loggers"]["ragnetic"]["handlers"].append("app_file")
        cfg["loggers"]["app.workflows"]["handlers"].append("app_file")


    return cfg
