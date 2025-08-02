# app/pipelines/parquet_loader.py

import logging
import os
from pathlib import Path
from typing import List, Optional
import asyncio
import pandas as pd
from datetime import datetime

from app.core.config import get_path_settings
from app.schemas.agent import AgentConfig, DataSource
from langchain_core.documents import Document
from app.pipelines.data_policy_utils import apply_data_policies
from app.pipelines.metadata_utils import generate_base_metadata

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"]
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"]
logger.debug(
    f"Loaded allowed data directories for Parquet loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
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
            logger.error(f"Error: Parquet file not found at {safe_file_path}")
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []
        if safe_file_path.suffix.lower() not in ['.parquet', '.orc']:
            logger.error(f"Error: Provided file '{safe_file_path}' is not a .parquet or .orc file.")
            return []

        logger.info(f"Attempting to load Parquet/ORC file: {safe_file_path}")

        def _read_parquet_blocking():
            if safe_file_path.suffix.lower() == '.parquet':
                return pd.read_parquet(safe_file_path)
            elif safe_file_path.suffix.lower() == '.orc':
                return pd.read_orc(safe_file_path)

        df = await asyncio.to_thread(_read_parquet_blocking)

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
                metadata = generate_base_metadata(
                    source,
                    source_context=safe_file_path.name,
                    source_type="file",
                )

                metadata.update(
                    {
                        "source_path": str(safe_file_path.resolve()),
                        "file_name": safe_file_path.name,
                        "file_type": safe_file_path.suffix.lower(),
                        "row_number": index + 1,
                        "record_id": record_id,
                        "doc_name": safe_file_path.name,
                        "source_name": safe_file_path.name,
                        "chunk_index": index,
                    }
                )

                # reproducible row-level ID (optional but helpful)
                doc_id = f"{safe_file_path.stem}-{index}"

                doc = Document(page_content=processed_text, metadata=metadata, id=doc_id)
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