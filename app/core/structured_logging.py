import logging
import json
from datetime import datetime
from sqlalchemy import create_engine, text, exc
from app.core.config import get_db_connection


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class DatabaseLogHandler(logging.Handler):
    """A logging handler that writes log records to a database table."""

    def __init__(self, connection_name: str, table_name: str = 'ragnetic_logs'):
        super().__init__()
        self.table_name = table_name
        self.engine = None
        try:
            conn_str = get_db_connection(connection_name)
            self.engine = create_engine(conn_str)
        except Exception as e:
            # Use a print statement here because the logging system might not be fully configured yet
            print(f"CRITICAL: Failed to initialize database logging connection '{connection_name}'. Error: {e}")

    def emit(self, record):
        if not self.engine:
            return

        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "exc_info": self.formatException(record.exc_info) if record.exc_info else None,
        }

        insert_stmt = text(f"""
            INSERT INTO {self.table_name} (timestamp, level, message, module, function, line, exc_info)
            VALUES (:timestamp, :level, :message, :module, :function, :line, :exc_info)
        """)

        try:
            with self.engine.connect() as connection:
                connection.execute(insert_stmt, log_entry)
                connection.commit()
        except exc.SQLAlchemyError as e:
            print(f"Failed to write log to database: {e}")