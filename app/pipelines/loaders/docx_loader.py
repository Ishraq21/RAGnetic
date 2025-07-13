import docx
import logging
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
import asyncio  # NEW: Added import for asynchronous operations

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
logger.info(f"Configured allowed data directories for DOCX loader: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")


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


async def load(file_path: str) -> List[Document]:  # MODIFIED: Changed to async def
    """
    Loads a .docx file and creates a single Document from its content,
    with path safety validation and standardized error logging.
    Now supports asynchronous loading.
    """
    docs = []
    try:
        # First, validate the file_path itself
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

        # Validate file existence, type, and if it's a file
        if not safe_file_path.exists():
            logger.error(f"Error: DOCX file not found at {safe_file_path}")
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []
        if safe_file_path.suffix.lower() not in ['.docx']:
            logger.error(f"Error: Provided file '{safe_file_path}' is not a .docx file.")
            return []

        logger.info(f"Opening DOCX file: {safe_file_path}")

        # MODIFIED: Run docx.Document() in a separate thread because it's blocking I/O
        # Define a helper function to perform the blocking operation
        def _open_docx_blocking():
            return docx.Document(safe_file_path)

        document = await asyncio.to_thread(_open_docx_blocking)

        full_text = "\n".join([para.text for para in document.paragraphs])

        if full_text.strip():
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": str(safe_file_path.resolve()),
                    "source_type": "docx"
                }
            )
            logger.info(f"Loaded content from {safe_file_path.name}")
            return [doc]
        else:
            logger.warning(f"DOCX file {safe_file_path.name} contained no readable text. Skipping document creation.")
            return []
    except Exception as e:
        logger.error(f"Error loading .docx file {file_path}: {e}", exc_info=True)
        return []