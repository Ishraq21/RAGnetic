import logging  # Added import
from sqlalchemy import create_engine, inspect, text  # Added 'text' for parameterized queries
from sqlalchemy.exc import SQLAlchemyError  # Specific import for database errors
from langchain_core.documents import Document
from typing import List

logger = logging.getLogger(__name__)  # Added logger initialization


def load(connection_string: str) -> List[Document]:
    """
    Connects to a SQL database, inspects its schema, and creates a
    Document for each table with its schema and sample rows.
    Includes connection string validation and emphasizes safe SQL practices.
    """
    docs = []
    try:
        if not connection_string:
            logger.error("Error: Database connection string is required.")
            return []

        # Basic validation of connection string format to catch obvious malformed strings
        # This is not exhaustive but helps. You might use a more sophisticated validator.
        supported_dialects = ["sqlite://", "postgresql://", "mysql://", "oracle://", "mssql+pyodbc://", "mongodb://"]
        if not any(connection_string.startswith(d) for d in supported_dialects):
            logger.error(f"Unsupported or invalid database connection string dialect: {connection_string[:50]}...")
            raise ValueError("Unsupported or invalid database connection string dialect.")

        engine = create_engine(connection_string)
        inspector = inspect(engine)
        logger.info(f"Successfully connected to database and started inspecting schema.")  # Changed from print()

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)

            # Get table schema as a CREATE TABLE statement
            schema_info = f"Table Name: {table_name}\nColumns:\n"
            for column in columns:
                schema_info += f"- {column['name']} ({column['type']})\n"

            # Get a few sample rows
            sample_rows_content = ""
            try:
                with engine.connect() as connection:
                    # --- IMPORTANT: SQL INJECTION PREVENTION ---
                    # When constructing SQL queries with dynamic parts (like table_name here),
                    # always use parameterized queries or ensure the dynamic part comes from
                    # a trusted, whitelisted source. For table_name obtained from inspector,
                    # it's generally safe. However, if 'table_name' were user input, it
                    # would require strict validation against a whitelist.
                    # For values, use text() with .bind_params() or a dict in execute().

                    # For SELECT * FROM {table_name}, it's safer if table_name
                    # is *not* directly from user input. Since inspector.get_table_names()
                    # provides trusted names, this specific use is less risky.
                    # However, always be mindful if any other user input is ever concatenated.

                    # Example of safe query with dynamic table name *if* table_name is trusted:
                    # (Note: LIMIT 3 is generally safe)
                    result = connection.execute(
                        text(f"SELECT * FROM \"{table_name}\" LIMIT 3"))  # Quote table name for safety
                    sample_rows = "Sample Rows:\n"
                    for row in result:
                        sample_rows += str(row._asdict()) + "\n" # <<< FIX APPLIED HERE: Changed str(dict(row)) to str(row._asdict())
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

        logger.info(f"Loaded schema for {len(docs)} tables from the database.")  # Changed from print()
        return docs

    except SQLAlchemyError as e:  # Catch SQLAlchemy specific errors
        logger.error(f"Database connection or schema inspection failed: {e}", exc_info=True)
        return []
    except ValueError as e:  # Catch validation errors we raise
        logger.error(f"Validation error for database connection string: {e}")
        return []
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during database loading: {e}", exc_info=True)
        return []