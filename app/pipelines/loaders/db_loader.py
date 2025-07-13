import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.documents import Document
from typing import List
import asyncio  # NEW: Added import for asynchronous operations

logger = logging.getLogger(__name__)


# Define synchronous helpers to run blocking SQLAlchemy operations in a thread
def _create_engine_blocking(conn_str: str):
    return create_engine(conn_str)


def _inspect_engine_blocking(engine):
    return inspect(engine)


def _get_table_names_blocking(inspector):
    return inspector.get_table_names()


def _get_columns_blocking(inspector, table_name: str):
    return inspector.get_columns(table_name)


def _execute_query_blocking(engine, query_text: str):
    with engine.connect() as connection:
        return connection.execute(
            text(query_text)).fetchall()  # .fetchall() to get all results before closing connection


async def load(connection_string: str) -> List[Document]:  # MODIFIED: Changed to async def
    """
    Connects to a SQL database, inspects its schema, and creates a
    Document for each table with its schema and sample rows.
    Includes connection string validation and emphasizes safe SQL practices.
    Now supports asynchronous loading.
    """
    docs = []
    try:
        if not connection_string:
            logger.error("Error: Database connection string is required.")
            return []

        supported_dialects = ["sqlite://", "postgresql://", "mysql://", "oracle://", "mssql+pyodbc://", "mongodb://"]
        if not any(connection_string.startswith(d) for d in supported_dialects):
            logger.error(f"Unsupported or invalid database connection string dialect: {connection_string[:50]}...")
            raise ValueError("Unsupported or invalid database connection string dialect.")

        # MODIFIED: Run create_engine in a separate thread
        engine = await asyncio.to_thread(_create_engine_blocking, connection_string)

        # MODIFIED: Run inspect(engine) in a separate thread
        inspector = await asyncio.to_thread(_inspect_engine_blocking, engine)
        logger.info(f"Successfully connected to database and started inspecting schema.")

        # MODIFIED: Run get_table_names in a separate thread
        table_names = await asyncio.to_thread(_get_table_names_blocking, inspector)

        for table_name in table_names:
            # MODIFIED: Run get_columns in a separate thread
            columns = await asyncio.to_thread(_get_columns_blocking, inspector, table_name)

            schema_info = f"Table Name: {table_name}\nColumns:\n"
            for column in columns:
                schema_info += f"- {column['name']} ({column['type']})\n"

            sample_rows_content = ""
            try:
                # MODIFIED: Run query execution in a separate thread
                query_sql = f"SELECT * FROM \"{table_name}\" LIMIT 3"
                rows = await asyncio.to_thread(_execute_query_blocking, engine, query_sql)

                sample_rows = "Sample Rows:\n"
                for row in rows:  # Iterate over the fetched rows
                    sample_rows += str(row._asdict()) + "\n"
                sample_rows_content = sample_rows
            except SQLAlchemyError as sqla_e:
                sample_rows_content = f"Could not fetch sample rows: {sqla_e}\n"
                logger.warning(f"Warning: Could not fetch sample rows for table {table_name}: {sqla_e}", exc_info=True)
            except Exception as e:
                sample_rows_content = f"Could not fetch sample rows (unexpected error): {e}\n"
                logger.warning(f"Warning: Unexpected error fetching sample rows for table {table_name}: {e}",
                               exc_info=True)

            page_content = schema_info + "\n" + sample_rows_content

            doc = Document(
                page_content=page_content,
                metadata={
                    "source": connection_string,
                    "source_type": "database",
                    "table_name": table_name
                }
            )
            docs.append(doc)

        logger.info(f"Loaded schema for {len(docs)} tables from the database.")
        return docs

    except SQLAlchemyError as e:
        logger.error(f"Database connection or schema inspection failed: {e}", exc_info=True)
        return []
    except ValueError as e:
        logger.error(f"Validation error for database connection string: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during database loading: {e}", exc_info=True)
        return []