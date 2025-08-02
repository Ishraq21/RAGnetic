# app/pipelines/pdf_loader.py

import fitz
import logging
from pathlib import Path
from typing import List, Optional
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
logger.debug(f"Loaded allowed data directories for PDF loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
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
    docs = []
    pdf_document = None
    try:
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
                processed_text = text
                page_blocked = False

                if agent_config and agent_config.data_policies:
                    logger.debug(f"Applying data policies to page {page_num + 1} of {safe_file_path.name}...")
                    processed_text, page_blocked = apply_data_policies(text, agent_config.data_policies, policy_context="page")

                if page_blocked:
                    logger.warning(f"Page {page_num + 1} of '{safe_file_path.name}' was completely blocked by a data policy and will not be processed.")
                    continue

                if processed_text.strip():
                    metadata = generate_base_metadata(source, source_context=safe_file_path.name, source_type="file")
                    # Add PDF-specific keys
                    metadata["source_path"] = str(safe_file_path.resolve())
                    metadata["file_name"] = safe_file_path.name
                    metadata["file_type"] = safe_file_path.suffix.lower()
                    metadata["page_number"] = page_num + 1
                    metadata["num_pages"] = pdf_document.page_count

                    doc = Document(
                        page_content=processed_text,
                        metadata={**metadata}
                    )
                    docs.append(doc)
                else:
                    logger.debug(f"Page {page_num + 1} of '{safe_file_path.name}' had no content after policy application or was empty.")

        logger.info(f"Loaded {len(docs)} processed pages from {safe_file_path.name} with enriched metadata.")
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