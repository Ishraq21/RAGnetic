import logging
import json
from datetime import datetime
from sqlalchemy import create_engine, insert, Table, Column, MetaData, exc, JSON
from app.core.config import get_db_connection


class JSONFormatter(logging.Formatter):
    """
    Formats log records into a single, flat JSON object.
    If extra data is passed to the logger, it is merged into the root of the JSON object.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Create a base dictionary with standard log attributes
        log_object = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # If the log call includes 'extra' data, merge it into the main object
        if hasattr(record, 'extra_data'):
            log_object.update(record.extra_data)

        if record.exc_info:
            log_object['exc_info'] = self.formatException(record.exc_info)

        return json.dumps(log_object)


class DatabaseLogHandler(logging.Handler):
    """A logging handler that writes log records to a database table."""

    def __init__(self, connection_name: str, table_name: str = 'ragnetic_logs'):
        super().__init__()
        self.table_name = table_name
        self.engine = None
        try:
            conn_str = get_db_connection(connection_name)
            # Use a synchronous-compatible driver for the handler
            sync_conn_str = conn_str.replace('+aiosqlite', '').replace('+asyncpg', '')
            self.engine = create_engine(sync_conn_str)
            # Define the table structure for SQLAlchemy Core
            metadata = MetaData()
            self.log_table = Table(table_name, metadata, autoload_with=self.engine)
        except Exception as e:
            # Use a print statement here because the logging system might not be fully configured yet
            print(f"CRITICAL: Failed to initialize database logging connection '{connection_name}'. Error: {e}")

    def emit(self, record):
        if not self.engine or not hasattr(self, 'log_table'):
            return

        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "exc_info": self.formatException(record.exc_info) if record.exc_info else None,
        }

        # If extra data (like our metrics) is present, add it to the log entry.
        # This assumes your database log table has a 'details' or 'extra' column
        # of a JSON or TEXT type.
        if hasattr(record, 'extra_data'):
            # Check if the 'details' column exists in the table model
            if 'details' in self.log_table.c:
                log_entry['details'] = record.extra_data
            else:
                # Fallback: if no 'details' column, serialize it into the message
                log_entry['message'] += f" | DETAILS: {json.dumps(record.extra_data)}"

        # Use SQLAlchemy's insert() function for a safe, consistent query
        insert_stmt = insert(self.log_table).values(log_entry)

        try:
            with self.engine.connect() as connection:
                connection.execute(insert_stmt)
                connection.commit()
        except exc.SQLAlchemyError as e:
            print(f"Failed to write log to database: {e}")
