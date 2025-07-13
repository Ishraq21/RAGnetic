import os
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
import asyncio

# NEW: Import get_path_settings from centralized config
from app.core.config import get_path_settings

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"] # Store project root if needed
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"] # Store resolved allowed dirs
logger.info(f"Loaded allowed data directories for directory loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
# --- End Centralized Configuration ---


def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    """
    Validates if the input_path resolves to a location within the configured
    allowed data directories. Raises ValueError if unsafe.
    Returns the resolved absolute Path if safe.
    """
    resolved_path = Path(input_path).resolve() # Resolve '..' and get absolute path

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED: # This variable now comes from central config
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path


async def load(folder_path: str) -> List[Document]:
    """
    Loads all files from a local directory and creates a Document for each, ensuring path safety.
    Now supports asynchronous loading.
    """
    docs = []
    try:
        # First, validate the folder_path itself
        safe_folder_path = _is_path_safe_and_within_allowed_dirs(folder_path)

        if not safe_folder_path.exists():
            logger.warning(f"Directory not found: {safe_folder_path}. Skipping loading.")
            return []
        if not safe_folder_path.is_dir():
            logger.warning(f"Path is not a directory: {safe_folder_path}. Skipping loading.")
            return []

        logger.info(f"Loading documents from safe directory: {safe_folder_path}")

        # Define a synchronous helper function to encapsulate the blocking I/O
        def _load_directory_blocking(current_folder_path: Path) -> List[Document]:
            local_docs = []
            for file in current_folder_path.rglob("*.*"):
                if file.is_dir():
                    continue
                try:
                    # For simplicity, this example only loads .txt files directly here.
                    # If you have specific other loaders (like pdf_loader.load) that are
                    # themselves async, you would need to call them with `await` *outside*
                    # of this `_load_directory_blocking` function, perhaps by collecting
                    # their paths and then awaiting them concurrently using asyncio.gather
                    # in the main async `load` function.
                    if file.suffix.lower() == '.txt':
                        with open(file, "r", encoding="utf-8", errors='ignore') as f:
                            text = f.read()
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "source": str(file.resolve()),
                                    "source_type": "local_directory"
                                }
                            )
                            local_docs.append(doc)
                except Exception as e:
                    logger.error(f"Error reading file {file}: {e}", exc_info=True)
            return local_docs

        # MODIFIED: Run the blocking helper function in a separate thread
        docs = await asyncio.to_thread(_load_directory_blocking, safe_folder_path)

        logger.info(f"Successfully loaded {len(docs)} documents from {folder_path}.")
        return docs
    except ValueError as e:
        logger.error(f"Security error during local directory loading: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading from directory {folder_path}: {e}", exc_info=True)
        return []