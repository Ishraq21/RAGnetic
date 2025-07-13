import logging
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import NotebookLoader
import asyncio

# NEW: Import get_path_settings from centralized config
from app.core.config import get_path_settings

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"] # Store project root if needed
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"] # Store resolved allowed dirs
logger.info(f"Loaded allowed data directories for notebook loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
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


async def load_notebook(path: str) -> List[Document]:
    """
    Loads a Jupyter Notebook (.ipynb) file from the given path,
    with path safety validation and standardized error logging.
    Now supports asynchronous loading.
    """
    try:
        # First, validate the path itself for safety
        safe_path = _is_path_safe_and_within_allowed_dirs(path)

        # Validate file existence and type
        if not safe_path.exists():
            logger.error(f"Error: Notebook file not found at {safe_path}")
            return []
        if not safe_path.is_file():
            logger.error(f"Error: Provided path '{safe_path}' is not a file.")
            return []
        if safe_path.suffix.lower() != '.ipynb':
            logger.error(f"Error: Provided file '{safe_path}' is not a Jupyter Notebook (.ipynb).")
            return []

        logger.info(f"Attempting to load Jupyter Notebook from safe path: {safe_path}")

        loader = NotebookLoader(
            str(safe_path),  # Ensure path is a string for NotebookLoader
            include_outputs=True,
            max_output_length=100,
            remove_newline=True
        )
        # MODIFIED: Run loader.load() in a separate thread because it's blocking I/O
        documents = await asyncio.to_thread(loader.load)

        logger.info(f"Successfully loaded {len(documents)} cells from notebook: {safe_path.name}")
        return documents
    except ValueError as e:
        logger.error(f"Security or validation error during notebook loading: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load notebook {path}. Error: {e}", exc_info=True)
        return []