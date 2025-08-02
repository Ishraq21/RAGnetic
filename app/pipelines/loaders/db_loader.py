# app/pipelines/db_loader.py

import logging
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.documents import Document
from typing import List, Optional
import asyncio
from datetime import datetime

from app.schemas.agent import AgentConfig, DataSource
from app.pipelines.data_policy_utils import apply_data_policies
from app.pipelines.metadata_utils import generate_base_metadata

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
            text(query_text)).fetchall()


async def load(connection_string: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]:
    docs = []
    engine = None
    try:
        if not connection_string:
            logger.error("Error: Database connection string is required.")
            return []

        supported_dialects = ["sqlite://", "postgresql://", "mysql://", "oracle://", "mssql+pyodbc://", "mongodb://"]
        if not any(connection_string.startswith(d) for d in supported_dialects):
            logger.error(f"Unsupported or invalid database connection string dialect: {connection_string[:50]}...")
            raise ValueError("Unsupported or invalid database connection string dialect.")

        engine = await asyncio.to_thread(_create_engine_blocking, connection_string)

        inspector = await asyncio.to_thread(_inspect_engine_blocking, engine)
        logger.info(f"Successfully connected to database and started inspecting schema.")

        table_names = await asyncio.to_thread(_get_table_names_blocking, inspector)

        for table_name in table_names:
            columns = await asyncio.to_thread(_get_columns_blocking, inspector, table_name)

            schema_info = f"Table Name: {table_name}\nColumns:\n"
            for column in columns:
                schema_info += f"- {column['name']} ({column['type']})\n"

            sample_rows_content = ""
            try:
                query_sql = f"SELECT * FROM \"{table_name}\" LIMIT 3"
                rows = await asyncio.to_thread(_execute_query_blocking, engine, query_sql)

                sample_rows = "Sample Rows:\n"
                for row in rows:
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

            processed_text = page_content
            document_blocked = False

            if agent_config and agent_config.data_policies:
                logger.debug(f"Applying data policies to table '{table_name}'...")
                processed_text, document_blocked = apply_data_policies(page_content, agent_config.data_policies, policy_context="table entry")

            if document_blocked:
                logger.warning(f"Table '{table_name}' was completely blocked by a data policy and will not be processed.")
                continue

            if processed_text.strip():
                metadata = generate_base_metadata(source, source_context=table_name, source_type="database")
                # Add database-specific keys
                metadata["source_db_connection_string"] = connection_string
                metadata["table_name"] = table_name

                doc = Document(
                    page_content=processed_text,
                    metadata={**metadata}
                )
                docs.append(doc)
            else:
                logger.debug(f"Table '{table_name}' had no content after policy application or was empty.")

        logger.info(f"Loaded schema for {len(docs)} processed tables from the database with enriched metadata.")
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
    finally:
        if engine:
            engine.dispose()
            logger.debug(f"Disposed of database engine for {connection_string[:30]}...")