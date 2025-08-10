import json
import logging
import logging.config
import os
import sqlite3
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import create_engine, event, insert, Table
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
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
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            obj.update(extra)

        if record.exc_info:
            obj["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(obj)


class DatabaseLogHandler(logging.Handler):
    """A logging handler that stores logs in the database."""

    def __init__(self, connection_name: str, table: Any, **kwargs):
        super().__init__(**kwargs)
        self.connection_name = connection_name
        self.table = table
        self.engine = None
        self._db_session: Optional[Session] = None

    def emit(self, record):
        try:
            from app.db import get_sync_db_session
            db_session = get_sync_db_session(self.connection_name)

            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "exc_info": self.format_exception(record.exc_info) if record.exc_info else None,
                "details": getattr(record, 'details', None),
                "correlation_id": getattr(record, 'correlation_id', None),
                "request_id": getattr(record, 'request_id', None),
                "user_id": getattr(record, 'user_id', None),
            }
            stmt = insert(self.table).values(**log_entry)
            db_session.execute(stmt)
            db_session.commit()
            db_session.close()
        except Exception:
            import sys
            import traceback
            traceback.print_exc(file=sys.stderr)

    def format_exception(self, exc_info):
        import traceback
        if exc_info:
            return ''.join(traceback.format_exception(*exc_info))
        return None


def get_logging_config(json_logs: bool = False, log_level: str = "INFO"):
    """
    Returns a logging configuration dictionary.
    Includes a custom handler for database logging.
    """
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'json': {
                '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                'format': '%(asctime)s %(levelname)s %(name)s %(funcName)s %(lineno)d %(message)s'
            },
            'structured': {
                'format': '%(asctime)s %(levelname)s %(name)s [%(correlation_id)s] %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': log_level,
            },
            'db_handler': {
                'class': 'app.core.structured_logging.DatabaseLogHandler',
                'formatter': 'structured',
                'level': 'INFO',
                'connection_name': os.environ.get("LOG_DB_CONNECTION_NAME") or "ragnetic_db",
                'table': ragnetic_logs_table
            }
        },
        'loggers': {
            'ragnetic': {
                'handlers': ['console'],
                'level': log_level,
                'propagate': False
            },
            # --- NEW: Quiet third-party loggers ---
            'uvicorn': {'handlers': ['console'], 'level': 'INFO', 'propagate': False},
            'uvicorn.access': {'handlers': ['console'], 'level': 'WARNING', 'propagate': False},
            'sqlalchemy': {'handlers': ['console'], 'level': 'WARNING', 'propagate': False},
            'celery': {'handlers': ['console'], 'level': 'INFO', 'propagate': False},
            'httpx': {'handlers': ['console'], 'level': 'WARNING', 'propagate': False},
            'faiss': {'handlers': ['console'], 'level': 'WARNING', 'propagate': False},
        },
        'root': {
            'handlers': ['console'],
            'level': log_level
        }
    }

    if json_logs:
        log_config['handlers']['console']['formatter'] = 'json'
    return log_config