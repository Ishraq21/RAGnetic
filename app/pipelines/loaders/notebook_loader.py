import logging
import os  # Added import
from pathlib import Path  # Added import
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import NotebookLoader

logger = logging.getLogger(__name__)

# --- Configuration for Allowed Data Directories (copied for consistency) ---
# IMPORTANT: Adjust these paths to correctly reflect where your 'data' and 'agents_data'
# directories are located relative to your project's root when the application runs.
# os.getcwd() assumes the script is run from the project root.
_PROJECT_ROOT = Path(os.getcwd())  # This should be your RAGnetic project's base directory
_ALLOWED_DATA_DIRS = [
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT / "agents_data"  # If agent configs or related files can be loaded via 'local' source type
    # Add any other directories that are explicitly allowed for local data sources
]
# Resolve all allowed directories to their absolute, canonical form once at startup
_ALLOWED_DATA_DIRS_RESOLVED = [d.resolve() for d in _ALLOWED_DATA_DIRS]
logger.info(f"Configured allowed data directories for notebook loader: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")


# --- End Configuration ---

def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    """
    Validates if the input_path resolves to a location within the configured
    allowed data directories. Raises ValueError if unsafe.
    Returns the resolved absolute Path if safe.
    """
    resolved_path = Path(input_path).resolve()  # Resolve '..' and get absolute path

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED:
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path


def load_notebook(path: str) -> List[Document]:
    """
    Loads a Jupyter Notebook (.ipynb) file from the given path,
    with path safety validation and standardized error logging.
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
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} cells from notebook: {safe_path.name}")
        return documents
    except ValueError as e:  # Catches validation errors from _is_path_safe_and_within_allowed_dirs or file checks
        logger.error(f"Security or validation error during notebook loading: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load notebook {path}. Error: {e}", exc_info=True)
        return []