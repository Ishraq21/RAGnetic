import fitz  # This is the PyMuPDF library
import logging
import os
import re # Added for PII redaction
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
import asyncio

# NEW: Import get_path_settings from centralized config
from app.core.config import get_path_settings
# NEW: Import AgentConfig and DataPolicy for policy application
from app.schemas.agent import AgentConfig, DataPolicy

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

def _apply_data_policies(text: str, policies: List[DataPolicy]) -> tuple[str, bool]:
    """
    Applies data policies (redaction/filtering) to the text content.
    Returns the processed text and a boolean indicating if the document (page) was blocked.
    """
    processed_text = text
    document_blocked = False # Renamed from original 'document_blocked' to 'page_blocked' for clarity in PDF context

    for policy in policies:
        if policy.type == 'pii_redaction' and policy.pii_config:
            pii_config = policy.pii_config
            for pii_type in pii_config.types:
                pattern = None
                if pii_type == 'email':
                    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                elif pii_type == 'phone':
                    # Common phone number formats (adjust or enhance regex as needed for international formats)
                    pattern = r'\b(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
                elif pii_type == 'ssn':
                    pattern = r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'
                elif pii_type == 'credit_card':
                    # Basic credit card pattern (major issuers, e.g., Visa, Mastercard, Amex, Discover)
                    pattern = r'\b(?:4\d{3}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|5[1-5]\d{2}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|3[47]\d{13}|6(?:011|5\d{2})[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b'
                elif pii_type == 'name':
                    # Name redaction is complex and context-dependent.
                    # For initial phase, might require a simple list of common names or a rule.
                    # More advanced NLP or entity recognition would be needed for robust name redaction.
                    logger.warning(f"PII type '{pii_type}' (name) is complex and not fully implemented for regex-based redaction. Skipping for now.")
                    continue

                if pattern:
                    replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 8) # Generic length
                    if pii_type == 'credit_card': replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 16) # Specific length for CC
                    processed_text = re.sub(pattern, replacement, processed_text)
                    logger.debug(f"Applied {pii_type} redaction policy. Replaced with: {replacement}")

        elif policy.type == 'keyword_filter' and policy.keyword_filter_config:
            kw_config = policy.keyword_filter_config
            for keyword in kw_config.keywords:
                if keyword in processed_text:
                    if kw_config.action == 'redact':
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.debug(f"Applied keyword redaction for '{keyword}'. Replaced with: {replacement}")
                    elif kw_config.action == 'block_chunk':
                        # At this stage (page level), we can't block just a chunk directly.
                        # We'll redact and log a warning for now, indicating this page contains content
                        # that should ideally be split and then blocked at chunking.
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.warning(f"Keyword '{keyword}' found. This page contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document': # In PDF loader, 'document' refers to a 'page'
                        logger.warning(f"Keyword '{keyword}' found. Page is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked # Return empty content and blocked flag

    return processed_text, document_blocked


async def load(file_path: str, agent_config: Optional[AgentConfig] = None) -> List[Document]:
    """
    Loads a PDF file and creates a Document for each page,
    with path safety validation, data policy application, and standardized error logging.
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
                processed_text = text
                page_blocked = False

                # Apply data policies if provided in agent_config
                if agent_config and agent_config.data_policies:
                    logger.debug(f"Applying data policies to page {page_num + 1} of {safe_file_path.name}...")
                    processed_text, page_blocked = _apply_data_policies(text, agent_config.data_policies)

                if page_blocked:
                    logger.warning(f"Page {page_num + 1} of '{safe_file_path.name}' was completely blocked by a data policy and will not be processed.")
                    continue # Skip this page if it's blocked

                # Only add the document if it's not blocked and has content after processing
                if processed_text.strip(): # Ensure there's actual content left
                    doc = Document(
                        page_content=processed_text,
                        metadata={
                            "source": str(safe_file_path.resolve()),
                            "source_type": "pdf",
                            "page_number": page_num + 1,
                            "file_name": safe_file_path.name, # Added for consistency
                            "file_path": str(safe_file_path)  # Added for consistency
                        }
                    )
                    docs.append(doc)
                else:
                    logger.debug(f"Page {page_num + 1} of '{safe_file_path.name}' had no content after policy application or was empty.")


        logger.info(f"Loaded {len(docs)} processed pages from {safe_file_path.name}")
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