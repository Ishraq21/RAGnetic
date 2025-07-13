import fitz  # This is the PyMuPDF library
import logging
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
import asyncio  # NEW: Added import for asynchronous operations

logger = logging.getLogger(__name__)

# --- Configuration for Allowed Data Directories (copied from directory_loader for consistency) ---
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


async def load(file_path: str) -> List[Document]:  # MODIFIED: Changed to async def
    """
    Loads a PDF file and creates a Document for each page,
    with path safety validation and standardized error logging.
    Now supports asynchronous loading.
    """
    docs = []
    pdf_document = None  # Initialize to None for finally block
    try:
        # First, validate the file_path itself
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

        if not safe_file_path.exists():
            logger.error(f"Error: File not found at {safe_file_path}")
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []
        if safe_file_path.suffix.lower() != '.pdf':
            logger.error(f"Error: Provided file '{safe_file_path}' is not a PDF.")
            return []

        logger.info(f"Opening PDF file: {safe_file_path}")

        # MODIFIED: Run fitz.open in a separate thread because it's blocking I/O
        # We need a helper function to pass to asyncio.to_thread
        def _open_pdf_blocking():
            return fitz.open(safe_file_path)

        pdf_document = await asyncio.to_thread(_open_pdf_blocking)

        for page_num, page in enumerate(pdf_document):
            # get_text() can also be blocking for large pages, but typically less than opening the file.
            # For simplicity, we'll keep it synchronous here. For extreme performance on huge PDFs,
            # this loop could also be run in asyncio.to_thread if page processing is very heavy.
            text = page.get_text()
            if text:
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": str(safe_file_path.resolve()),
                        "source_type": "pdf",
                        "page_number": page_num + 1,
                    }
                )
                docs.append(doc)

        logger.info(f"Loaded {len(docs)} pages from {safe_file_path.name}")
        return docs

    except ValueError as e:
        logger.error(f"Security error during PDF loading: {e}")
        return []
    except fitz.FileDataError as e:
        logger.error(f"Error opening or reading PDF file {file_path}: {e}. File might be corrupted or not a valid PDF.",
                     exc_info=True)
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading PDF {file_path}: {e}", exc_info=True)
        return []
    finally:
        # Ensure the document is closed even if errors occur
        if pdf_document:
            pdf_document.close()
            logger.debug(f"Closed PDF document: {safe_file_path.name}")