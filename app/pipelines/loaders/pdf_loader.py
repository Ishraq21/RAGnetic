import fitz  # This is the PyMuPDF library
import logging
import os
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
logger.info(f"Loaded allowed data directories for PDF loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
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
    Loads a PDF file and creates a Document for each page,
    with path safety validation and standardized error logging.
    Now supports asynchronous loading.
    """
    docs = []
    pdf_document = None # Initialize to None for finally block
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

        def _open_pdf_blocking():
            return fitz.open(safe_file_path)

        pdf_document = await asyncio.to_thread(_open_pdf_blocking)

        for page_num, page in enumerate(pdf_document):
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
        if pdf_document:
            pdf_document.close()
            logger.debug(f"Closed PDF document: {safe_file_path.name}")