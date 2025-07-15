import logging
import re # Added for PII redaction
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.documents import Document
from typing import List, Optional
import asyncio

from app.schemas.agent import AgentConfig, DataPolicy

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


def _apply_data_policies(text: str, policies: List[DataPolicy]) -> tuple[str, bool]:
    """
    Applies data policies (redaction/filtering) to the text content.
    Returns the processed text and a boolean indicating if the document (table entry) was blocked.
    """
    processed_text = text
    document_blocked = False

    for policy in policies:
        if policy.type == 'pii_redaction' and policy.pii_config:
            pii_config = policy.pii_config
            for pii_type in pii_config.types:
                pattern = None
                if pii_type == 'email':
                    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                elif pii_type == 'phone':
                    # Common phone number formats (adjust or enhance regex as needed for international formats)
                    pattern = r'\b(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
                elif pii_type == 'ssn':
                    pattern = r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'
                elif pii_type == 'credit_card':
                    # Basic credit card pattern (major issuers, e.g., Visa, Mastercard, Amex, Discover)
                    pattern = r'\b(?:4\d{3}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|5[1-5]\d{2}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|3[47]\d{13}|6(?:011|5\d{2})[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b'
                elif pii_type == 'name':
                    # Name redaction is complex and context-dependent.
                    logger.warning(f"PII type '{pii_type}' (name) is complex and not fully implemented for regex-based redaction. Skipping for now.")
                    continue

                if pattern:
                    replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 8) # Generic length
                    if pii_type == 'credit_card': replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 16) # Specific length for CC
                    processed_text = re.sub(pattern, replacement, processed_text)
                    logger.debug(f"Applied {pii_type} redaction policy. Replaced with: {replacement}")

        elif policy.type == 'keyword_filter' and policy.keyword_filter_config:
            kw_config = policy.keyword_filter_config
            for keyword in kw_config.keywords:
                if keyword in processed_text:
                    if kw_config.action == 'redact':
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.debug(f"Applied keyword redaction for '{keyword}'. Replaced with: {replacement}")
                    elif kw_config.action == 'block_chunk':
                        # At this stage (document/table entry level), we can't block just a chunk directly.
                        # We'll redact and log a warning for now, indicating this entry contains content
                        # that should ideally be split and then blocked at chunking.
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.warning(f"Keyword '{keyword}' found. This entry contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document': # In DB loader, 'document' refers to a 'table entry'
                        logger.warning(f"Keyword '{keyword}' found. Table entry is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked # Return empty content and blocked flag

    return processed_text, document_blocked


async def load(connection_string: str, agent_config: Optional[AgentConfig] = None) -> List[Document]:  # MODIFIED: Added agent_config
    """
    Connects to a SQL database, inspects its schema, and creates a
    Document for each table with its schema and sample rows.
    Includes connection string validation, data policy application, and emphasizes safe SQL practices.
    Now supports asynchronous loading.
    """
    docs = []
    engine = None # Initialize engine for finally block
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
                # MODIFIED: Run query execution in a separate thread
                # Using LIMIT 3 to get a small, representative sample, avoiding large data pulls.
                query_sql = f"SELECT * FROM \"{table_name}\" LIMIT 3"
                rows = await asyncio.to_thread(_execute_query_blocking, engine, query_sql)

                sample_rows = "Sample Rows:\n"
                for row in rows:  # Iterate over the fetched rows
                    sample_rows += str(row._asdict()) + "\n" # Convert row to dict for better representation
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

            # Apply data policies if provided in agent_config
            if agent_config and agent_config.data_policies:
                logger.debug(f"Applying data policies to table '{table_name}'...")
                processed_text, document_blocked = _apply_data_policies(page_content, agent_config.data_policies)

            if document_blocked:
                logger.warning(f"Table '{table_name}' was completely blocked by a data policy and will not be processed.")
                continue # Skip this table if it's blocked

            if processed_text.strip(): # Ensure there's actual content left after policies
                doc = Document(
                    page_content=processed_text,
                    metadata={
                        "source": connection_string,
                        "source_type": "database",
                        "table_name": table_name,
                        "db_connection_string": connection_string # Added for consistency
                    }
                )
                docs.append(doc)
            else:
                logger.debug(f"Table '{table_name}' had no content after policy application or was empty.")


        logger.info(f"Loaded schema for {len(docs)} processed tables from the database.")
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
            # Dispose of the engine to close all connections in the pool
            # This is important in async contexts where connections might persist
            # if not explicitly closed.
            engine.dispose()
            logger.debug(f"Disposed of database engine for {connection_string[:30]}...")