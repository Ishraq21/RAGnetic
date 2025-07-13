import fitz  # This is the PyMuPDF library
import logging  # Added import
import os
from pathlib import Path  # Added import
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)  # Added logger initialization

# --- Configuration for Allowed Data Directories (copied from directory_loader for consistency) ---
# IMPORTANT: Adjust these paths to correctly reflect where your 'data' and 'agents_data'
# directories are located relative to your project's root when the application runs.
_PROJECT_ROOT = Path(os.getcwd())  # This should be your RAGnetic project's base directory
_ALLOWED_DATA_DIRS = [
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT / "agents_data"  # If agent configs or related files can be loaded via 'local' source type
    # Add any other directories that are explicitly allowed for local data sources
]
# Resolve all allowed directories to their absolute, canonical form once at startup
_ALLOWED_DATA_DIRS_RESOLVED = [d.resolve() for d in _ALLOWED_DATA_DIRS]
logger.info(f"Configured allowed data directories for PDF loader: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")


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


def load(file_path: str) -> List[Document]:
    """
    Loads a PDF file and creates a Document for each page,
    with path safety validation and standardized error logging.
    """
    docs = []
    try:
        # First, validate the file_path itself
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

        if not safe_file_path.exists():
            logger.error(f"Error: File not found at {safe_file_path}")  # Changed from print()
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []
        if safe_file_path.suffix.lower() != '.pdf':
            logger.error(f"Error: Provided file '{safe_file_path}' is not a PDF.")
            return []

        logger.info(f"Opening PDF file: {safe_file_path}")
        pdf_document = fitz.open(safe_file_path)

        for page_num, page in enumerate(pdf_document):
            text = page.get_text()
            if text:  # Only create a document if there's text on the page
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": str(safe_file_path.resolve()),  # Use resolved safe path
                        "source_type": "pdf",
                        "page_number": page_num + 1,
                    }
                )
                docs.append(doc)

        pdf_document.close()  # Ensure the document is closed

        logger.info(f"Loaded {len(docs)} pages from {safe_file_path.name}")  # Changed from print()
        return docs

    except ValueError as e:  # Catches validation errors from _is_path_safe_and_within_allowed_dirs
        logger.error(f"Security error during PDF loading: {e}")
        return []
    except fitz.FileDataError as e:
        logger.error(f"Error opening or reading PDF file {file_path}: {e}. File might be corrupted or not a valid PDF.",
                     exc_info=True)
        return []
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred while loading PDF {file_path}: {e}", exc_info=True)
        return []