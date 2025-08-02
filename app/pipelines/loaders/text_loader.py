# app/pipelines/text_loader.py

import logging
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import TextLoader
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
logger.debug(f"Loaded allowed data directories for text loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
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


async def load(file_path: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]:
    try:
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

        if not safe_file_path.exists():
            logger.error(f"Error: Text file not found at {safe_file_path}")
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []
        if safe_file_path.suffix.lower() not in ['.txt']:
            logger.error(f"Error: Provided file '{safe_file_path}' is not a plain text (.txt) file.")
            return []

        logger.info(f"Attempting to load text file: {safe_file_path}")

        loader = TextLoader(str(safe_file_path), encoding="utf-8")
        raw_documents = await asyncio.to_thread(loader.load)

        if not raw_documents:
            return []

        processed_text = raw_documents[0].page_content
        document_blocked = False

        if agent_config and agent_config.data_policies:
            logger.info(f"Applying data policies to {safe_file_path.name}...")
            processed_text, document_blocked = apply_data_policies(processed_text, agent_config.data_policies, policy_context="document")

        if document_blocked:
            logger.warning(f"Document '{safe_file_path.name}' was completely blocked by a data policy and will not be processed.")
            return []

        metadata = generate_base_metadata(source, source_context=safe_file_path.name, source_type="file")
        # Add text-specific keys
        metadata["source_path"] = str(safe_file_path.resolve())
        metadata["file_name"] = safe_file_path.name
        metadata["file_type"] = safe_file_path.suffix.lower()

        existing_metadata = raw_documents[0].metadata if raw_documents else {}
        processed_doc = Document(
            page_content=processed_text,
            metadata={**existing_metadata, **metadata}
        )

        logger.info(f"Successfully loaded and processed text file: {safe_file_path.name} with enriched metadata.")
        return [processed_doc]

    except ValueError as e:
        logger.error(f"Security or validation error during text file loading: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load text file {file_path}: {e}", exc_info=True)
        return []