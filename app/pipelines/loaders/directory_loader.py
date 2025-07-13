import os
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- Configuration for Allowed Data Directories ---
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
logger.info(f"Configured allowed data directories: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")


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


def load(folder_path: str) -> List[Document]:  # Changed parameter name from base_path to folder_path for consistency
    """
    Loads all files from a local directory and creates a Document for each, ensuring path safety.
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

        # Iterate over files in the safe directory
        for file in safe_folder_path.rglob("*.*"):  # Use pathlib's rglob directly on the safe Path object
            if file.is_dir():
                continue
            try:
                # You might need to add specific loaders for different file types here,
                # ensuring that each called loader also validates its input path if it's constructed dynamically.
                # For this example, we'll keep the basic text file loading.
                if file.suffix.lower() == '.txt':  # Example: only load text files
                    with open(file, "r", encoding="utf-8", errors='ignore') as f:
                        text = f.read()
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": str(file.resolve()),
                                "source_type": "local_directory"
                            }
                        )
                        docs.append(doc)
                # Add logic for other file types here, e.g.:
                # elif file.suffix.lower() == '.pdf':
                #    docs.extend(pdf_loader.load(str(file))) # Ensure pdf_loader also validates its path
                # ...
            except Exception as e:
                logger.error(f"Error reading file {file}: {e}", exc_info=True)
                # Continue processing other files even if one fails

        logger.info(f"Successfully loaded {len(docs)} documents from {folder_path}.")
        return docs
    except ValueError as e:  # Catch ValueError from _is_path_safe_and_within_allowed_dirs
        logger.error(f"Security error during local directory loading: {e}")
        return []
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred while loading from directory {folder_path}: {e}", exc_info=True)
        return []