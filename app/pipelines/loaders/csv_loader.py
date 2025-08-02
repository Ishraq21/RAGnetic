# app/pipelines/csv_loader.py

import pandas as pd
import logging
import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
import asyncio
from datetime import datetime

from app.core.config import get_path_settings
from app.schemas.agent import AgentConfig, DataSource
from app.pipelines.data_policy_utils import apply_data_policies
from app.pipelines.metadata_utils import generate_base_metadata

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"]
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"]
logger.debug(
    f"Loaded allowed data directories for CSV loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
# --- End Centralized Configuration ---


def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    resolved_path = Path(input_path).resolve()

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED:
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path


async def load(file_path: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]:
    docs = []
    try:
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

        if not safe_file_path.exists():
            logger.error(f"Error: CSV file not found at {safe_file_path}")
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []
        if safe_file_path.suffix.lower() not in ['.csv']:
            logger.error(f"Error: Provided file '{safe_file_path}' is not a .csv file.")
            return []

        logger.info(f"Attempting to load CSV file: {safe_file_path}")

        def _read_csv_blocking():
            return pd.read_csv(safe_file_path)

        df = await asyncio.to_thread(_read_csv_blocking)

        for index, row in df.iterrows():
            row_dict = row.to_dict()

            record_id = str(row_dict.get(df.columns[0], index))

            row_details = "\n".join([f"- {str(col).replace('_', ' ').strip()}: {val}" for col, val in row_dict.items()])

            page_content = f"Record ID '{record_id}':\n{row_details}"

            processed_text = page_content
            document_blocked = False

            if agent_config and agent_config.data_policies:
                logger.debug(
                    f"Applying data policies to row {index + 1} of {safe_file_path.name} (Record ID: {record_id})...")
                processed_text, document_blocked = apply_data_policies(page_content, agent_config.data_policies, policy_context="row")

            if document_blocked:
                logger.warning(
                    f"Row {index + 1} of '{safe_file_path.name}' (Record ID: {record_id}) was completely blocked by a data policy and will not be processed.")
                continue

            if processed_text.strip():
                metadata = generate_base_metadata(source, source_context=safe_file_path.name, source_type="file")
                # Add CSV-specific keys
                metadata["source_path"] = str(safe_file_path.resolve())
                metadata["file_name"] = safe_file_path.name
                metadata["file_type"] = safe_file_path.suffix.lower()
                metadata["row_number"] = index + 1
                metadata["record_id"] = record_id
                metadata["chunk_id"] = index + 1
                metadata["document_name"] = safe_file_path.name

                doc = Document(
                    page_content=processed_text,
                    metadata={**metadata}
                )
                docs.append(doc)
            else:
                logger.debug(
                    f"Row {index + 1} of '{safe_file_path.name}' (Record ID: {record_id}) had no content after policy application or was empty.")

        logger.info(f"Loaded {len(docs)} processed rows from {safe_file_path.name} with enriched metadata.")
        return docs
    except ValueError as e:
        logger.error(f"Security or validation error during CSV file loading: {e}")
        return []
    except pd.errors.EmptyDataError:
        logger.warning(f"CSV file {file_path} is empty. No data loaded.")
        return []
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}. Check file format or content.", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}", exc_info=True)
        return []