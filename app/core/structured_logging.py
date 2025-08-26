import json
import logging
import logging.config
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session, sessionmaker
from app.core.config import get_db_connection, get_debug_mode
from sqlalchemy import insert, create_engine

LOGGING_QUEUE = None

_ASYNC_TO_SYNC_DRIVERS = {
    "+asyncpg": "+psycopg2",   # Postgres
    "+aiomysql": "+pymysql",   # MySQL
    "+aiosqlite": "",          # SQLite
}

def _to_sync_url(url: str) -> str:
    out = url
    for a, b in _ASYNC_TO_SYNC_DRIVERS.items():
        if a in out:
            out = out.replace(a, b)
    return out

def _build_sync_engine_for_logging(connection_name: str):
    """
    Build a SQLAlchemy *sync* engine from our configured DB connection name,
    converting async drivers to sync ones.
    """
    url_async = get_db_connection(connection_name)
    url_sync = _to_sync_url(url_async)

    engine_kwargs: Dict[str, Any] = {"pool_pre_ping": True, "pool_recycle": 1800}
    if url_sync.startswith("sqlite:///"):
        # allow usage from background threads/handlers
        engine_kwargs["connect_args"] = {"check_same_thread": False}

    return create_engine(url_sync, **engine_kwargs)


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
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            obj.update(extra)

        if record.exc_info:
            obj["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(obj)


class DatabaseLogHandler(logging.Handler):
    def __init__(self, connection_name: str, table: Any, **kwargs):
        super().__init__(**kwargs)
        self.table = table
        self.engine = _build_sync_engine_for_logging(connection_name)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)

    RESERVED = {
        "timestamp", "level", "message", "module", "function", "line",
        "exc_info", "details"
    }

    @staticmethod
    def _safe_jsonable(v):
        try:
            json.dumps(v)
            return v
        except Exception:
            return str(v)

    def emit(self, record):
        db_session = self.SessionLocal()
        try:
            log_entry = {
                "timestamp": datetime.utcfromtimestamp(record.created),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "exc_info": self.format_exception(record.exc_info) if record.exc_info else None,
                "details": getattr(record, 'details', None),
            }

            extra = getattr(record, "extra_data", None)
            if isinstance(extra, dict):
                details = log_entry.get("details")
                if not isinstance(details, dict):
                    details = {}
                for k, v in extra.items():
                    if k not in self.RESERVED:
                        details[k] = self._safe_jsonable(v)
                log_entry["details"] = details

            db_session.execute(insert(self.table).values(**log_entry))
            db_session.commit()
        except Exception:
            db_session.rollback()
            import sys, traceback
            traceback.print_exc(file=sys.stderr)
        finally:
            db_session.close()

    def format_exception(self, exc_info):
        import traceback
        if exc_info:
            return ''.join(traceback.format_exception(*exc_info))
        return None


def get_logging_config(json_logs: bool = False, log_level: Optional[str] = None):
    """
    Build logging config. If log_level is provided, use it.
    Otherwise derive from config.ini via get_debug_mode().
    """
    if log_level is None:
        log_level = "DEBUG" if get_debug_mode() else "INFO"
    else:
        log_level = str(log_level).upper()

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            "json": {"()": "app.core.structured_logging.JSONFormatter"},
            "structured": {"format": "%(asctime)s %(levelname)s %(name)s [%(correlation_id)s] %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": log_level,
            }
        },
        "loggers": {
            "ragnetic": {"handlers": ["console"], "level": log_level, "propagate": False},
            "uvicorn": {"handlers": ["console"], "level": log_level, "propagate": False},
            "uvicorn.error": {"handlers": ["console"], "level": log_level, "propagate": False},
            # keep access quieter unless debug is on
            "uvicorn.access": {"handlers": ["console"], "level": ("DEBUG" if log_level == "DEBUG" else "WARNING"), "propagate": False},
            "sqlalchemy": {"handlers": ["console"], "level": ("INFO" if log_level == "DEBUG" else "WARNING"), "propagate": False},
            "celery": {"handlers": ["console"], "level": log_level, "propagate": False},
            "httpx": {"handlers": ["console"], "level": ("INFO" if log_level == "DEBUG" else "WARNING"), "propagate": False},
            "faiss": {"handlers": ["console"], "level": ("INFO" if log_level == "DEBUG" else "WARNING"), "propagate": False},
            "app": {"handlers": ["console"], "level": log_level, "propagate": False},
        },
        "root": {"handlers": ["console"], "level": log_level},
    }
    if json_logs:
        log_config["handlers"]["console"]["formatter"] = "json"
    return log_config

