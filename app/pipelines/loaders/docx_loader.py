import docx
import logging
import os
import re # Added for PII redaction
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
import asyncio
from datetime import datetime

# NEW: Import get_path_settings from centralized config
from app.core.config import get_path_settings
# NEW: Import AgentConfig, DataPolicy, DataSource for policy application and lineage
from app.schemas.agent import AgentConfig, DataPolicy, DataSource

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"] # Store project root if needed
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"] # Store resolved allowed dirs
logger.info(f"Loaded allowed data directories for DOCX loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
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
    Returns the processed text and a boolean indicating if the document was blocked.
    """
    processed_text = text
    document_blocked = False

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
                        # At this stage (document level), we can't block just a chunk directly.
                        # We'll redact and log a warning for now, indicating this document contains content
                        # that should ideally be split and then blocked at chunking.
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.warning(f"Keyword '{keyword}' found. This document contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document':
                        logger.warning(f"Keyword '{keyword}' found. Document is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked # Return empty content and blocked flag

    return processed_text, document_blocked


async def load(file_path: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]:
    """
    Loads a .docx file and creates a single Document from its content,
    with path safety validation, data policy application, and standardized error logging.
    Now supports asynchronous loading and enriched metadata.
    """
    try:
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

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

        def _open_docx_blocking():
            return docx.Document(safe_file_path)

        document = await asyncio.to_thread(_open_docx_blocking)

        full_text = "\n".join([para.text for para in document.paragraphs])

        processed_text = full_text
        document_blocked = False

        if agent_config and agent_config.data_policies:
            logger.info(f"Applying data policies to {safe_file_path.name}...")
            processed_text, document_blocked = _apply_data_policies(full_text, agent_config.data_policies)

        if document_blocked:
            logger.warning(f"DOCX document '{safe_file_path.name}' was completely blocked by a data policy and will not be processed.")
            return []

        if processed_text.strip():
            # Create base metadata from file itself
            metadata = {
                "source_path": str(safe_file_path.resolve()),
                "file_name": safe_file_path.name,
                "file_type": safe_file_path.suffix.lower(),
                "load_timestamp": datetime.now().isoformat(),
                # No page_number for DOCX as it's typically treated as single content block before chunking
            }
            # Add general source info if available from the DataSource object
            if source:
                metadata["source_type_config"] = source.model_dump() # Store entire DataSource config
                if source.url: metadata["source_url"] = source.url
                if source.db_connection: metadata["source_db_connection"] = source.db_connection
                if source.folder_id: metadata["source_gdoc_folder_id"] = source.folder_id
                if source.document_ids: metadata["source_gdoc_document_ids"] = source.document_ids

            doc = Document(
                page_content=processed_text,
                metadata={**metadata} # Use the enriched metadata
            )
            logger.info(f"Loaded and processed content from {safe_file_path.name} with enriched metadata.")
            return [doc]
        else:
            logger.warning(f"DOCX file {safe_file_path.name} contained no readable text or all content was redacted/filtered. Skipping document creation.")
            return []
    except Exception as e:
        logger.error(f"Error loading .docx file {file_path}: {e}", exc_info=True)
        return []