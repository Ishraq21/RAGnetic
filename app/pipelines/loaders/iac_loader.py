import os
import logging
from typing import List
from pathlib import Path
import asyncio

# NEW: Import get_path_settings from centralized config
from app.core.config import get_path_settings

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
# MODIFIED: Get settings from the central configuration function
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"] # Store project root if needed
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"] # Store resolved allowed dirs
logger.info(f"Loaded allowed data directories for IaC loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
# --- End Centralized Configuration ---


def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    """
    Validates if the input_path resolves to a location within the configured
    allowed data directories. Raises ValueError if unsafe.
    Returns the resolved absolute Path if safe.
    """
    resolved_path = Path(input_path).resolve()

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED: # This variable now comes from central config
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path


async def load(file_path: str) -> List[Document]:
    """
    Loads Infrastructure-as-Code (IaC) files like Terraform and Kubernetes YAML,
    then splits them into syntax-aware chunks, with path safety validation.
    Now supports asynchronous loading.
    """
    try:
        # First, validate the file_path itself
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

        # Validate file existence and type
        if not safe_file_path.exists():
            logger.error(f"Error: IaC file not found at {safe_file_path}")
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []

        file_extension = safe_file_path.suffix.lower()
        splitter = None

        if file_extension in [".tf", ".tfvars"]:
            logger.info(f"Loading Terraform file: {safe_file_path}")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""],
            )
        elif file_extension in [".yaml", ".yml"]:
            logger.info(f"Loading YAML file: {safe_file_path}")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n---\n", "\n\n", "\n", " ", ""],
            )
        else:
            logger.warning(f"Unsupported IaC file type: {safe_file_path.name}. Skipping.")
            return []

        loader = TextLoader(str(safe_file_path), encoding="utf-8")
        raw_docs = await asyncio.to_thread(loader.load)

        chunks = splitter.split_documents(raw_docs)

        for chunk in chunks:
            chunk.metadata['source_type'] = 'infrastructure_as_code'
            chunk.metadata['file_name'] = safe_file_path.name

        logger.info(f"Successfully chunked {safe_file_path.name} into {len(chunks)} IaC chunks.")
        return chunks

    except ValueError as e:
        logger.error(f"Security or validation error during IaC file loading: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load or process IaC file {file_path}: {e}", exc_info=True)