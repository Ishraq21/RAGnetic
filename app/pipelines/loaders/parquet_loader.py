import logging
import os
import re
from pathlib import Path
from typing import List, Optional
import asyncio
import pandas as pd
from datetime import datetime

# NEW: Import get_path_settings from centralized config
from app.core.config import get_path_settings
# NEW: Import AgentConfig, DataPolicy, DataSource for policy application and lineage
from app.schemas.agent import AgentConfig, DataPolicy, DataSource
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"]
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"]
logger.info(
    f"Loaded allowed data directories for Parquet loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")


# --- End Centralized Configuration ---


def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    """
    Validates if the input_path resolves to a location within the configured
    allowed data directories. Raises ValueError if unsafe.
    Returns the resolved absolute Path if safe.
    """
    resolved_path = Path(input_path).resolve()

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED:
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path


def _apply_data_policies(text: str, policies: List[DataPolicy]) -> tuple[str, bool]:
    """
    Applies data policies (redaction/filtering) to the text content.
    Returns the processed text and a boolean indicating if the document (row) was blocked.
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
                    pattern = r'\b(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
                elif pii_type == 'ssn':
                    pattern = r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'
                elif pii_type == 'credit_card':
                    pattern = r'\b(?:4\d{3}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|5[1-5]\d{2}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|3[47]\d{13}|6(?:011|5\d{2})[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b'
                elif pii_type == 'name':
                    logger.warning(
                        f"PII type '{pii_type}' (name) is complex and not fully implemented for regex-based redaction. Skipping for now.")
                    continue

                if pattern:
                    replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 8)
                    if pii_type == 'credit_card': replacement = pii_config.redaction_placeholder or (
                            pii_config.redaction_char * 16)
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
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.warning(
                            f"Keyword '{keyword}' found. This row contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document':
                        logger.warning(
                            f"Keyword '{keyword}' found. Row is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked

    return processed_text, document_blocked


async def load(file_path: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[
    Document]:  # MODIFIED: Added source parameter
    """
    Loads a Parquet file and creates a Document for each row,
    with path safety validation, data policy application, and standardized error logging.
    Now supports asynchronous loading and enriched metadata.
    """
    docs = []
    try:
        # First, validate the file_path itself
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

        if not safe_file_path.exists():
            logger.error(f"Error: Parquet file not found at {safe_file_path}")
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []
        if safe_file_path.suffix.lower() not in ['.parquet', '.orc']:  # Added .orc support
            logger.error(f"Error: Provided file '{safe_file_path}' is not a .parquet or .orc file.")
            return []

        logger.info(f"Attempting to load Parquet/ORC file: {safe_file_path}")

        # Run pd.read_parquet/read_orc in a separate thread because it's blocking I/O
        def _read_parquet_blocking():
            if safe_file_path.suffix.lower() == '.parquet':
                return pd.read_parquet(safe_file_path)
            elif safe_file_path.suffix.lower() == '.orc':
                return pd.read_orc(safe_file_path)  # Requires pyarrow[orc] or fastparquet[orc]

        df = await asyncio.to_thread(_read_parquet_blocking)

        for index, row in df.iterrows():
            # Convert row to a dictionary for easier formatting
            row_dict = row.to_dict()

            # Attempt to use a meaningful identifier for the record
            record_id = str(row_dict.get(df.columns[0], index))

            # Format each column-value pair on a new line
            row_details = "\n".join([f"- {str(col).replace('_', ' ').strip()}: {val}" for col, val in row_dict.items()])

            page_content = f"Record ID '{record_id}':\n{row_details}"

            processed_text = page_content
            document_blocked = False

            # Apply data policies if provided in agent_config
            if agent_config and agent_config.data_policies:
                logger.debug(
                    f"Applying data policies to row {index + 1} of {safe_file_path.name} (Record ID: {record_id})...")
                processed_text, document_blocked = _apply_data_policies(page_content, agent_config.data_policies)

            if document_blocked:
                logger.warning(
                    f"Row {index + 1} of '{safe_file_path.name}' (Record ID: {record_id}) was completely blocked by a data policy and will not be processed.")
                continue

            if processed_text.strip():
                # Create base metadata from file itself
                metadata = {
                    "source_path": str(safe_file_path.resolve()),
                    "file_name": safe_file_path.name,
                    "file_type": safe_file_path.suffix.lower(),
                    "load_timestamp": datetime.now().isoformat(),  # NEW: Add load timestamp
                    "row_number": index + 1,  # Specific for CSV/Parquet rows
                    "record_id": record_id  # Specific for CSV/Parquet
                }
                # Add general source info if available from the DataSource object
                if source:  # NEW: Add info from DataSource object for lineage
                    metadata["source_type_config"] = source.model_dump()  # Store entire DataSource config
                    if source.url: metadata["source_url"] = source.url
                    if source.db_connection: metadata["source_db_connection"] = source.db_connection
                    if source.folder_id: metadata["source_gdoc_folder_id"] = source.folder_id
                    if source.document_ids: metadata["source_gdoc_document_ids"] = source.document_ids

                doc = Document(
                    page_content=processed_text,
                    metadata={**metadata}  # Use the enriched metadata
                )
                docs.append(doc)
            else:
                logger.debug(
                    f"Row {index + 1} of '{safe_file_path.name}' (Record ID: {record_id}) had no content after policy application or was empty.")

        logger.info(f"Loaded {len(docs)} processed rows from {safe_file_path.name} with enriched metadata.")
        return docs
    except ValueError as e:
        logger.error(f"Security or validation error during Parquet/ORC file loading: {e}")
        return []
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error(f"Error parsing Parquet/ORC file {file_path}: {e}. Check file format or content.", exc_info=True)
        return []
    except ImportError as e:
        logger.error(
            f"Missing dependency for Parquet/ORC: {e}. Please ensure 'pyarrow' is installed ('pip install pyarrow'). If loading ORC, you might need 'pyarrow[orc]' or 'fastparquet[orc]'.",
            exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error loading Parquet/ORC file {file_path}: {e}", exc_info=True)
        return []