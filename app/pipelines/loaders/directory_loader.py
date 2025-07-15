import os
import logging
import re # Added for PII redaction
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
import asyncio

from app.core.config import get_path_settings
from app.schemas.agent import AgentConfig, DataPolicy

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

def _apply_data_policies(text: str, policies: List[DataPolicy]) -> tuple[str, bool]:
    """
    Applies data policies (redaction/filtering) to the text content.
    Returns the processed text and a boolean indicating if the document (file) was blocked.
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
                        # At this stage (file level), we can't block just a chunk directly.
                        # We'll redact and log a warning for now, indicating this file contains content
                        # that should ideally be split and then blocked at chunking.
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.warning(f"Keyword '{keyword}' found. This file contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document': # In directory loader, 'document' refers to a 'file'
                        logger.warning(f"Keyword '{keyword}' found. File is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked # Return empty content and blocked flag

    return processed_text, document_blocked


async def load(folder_path: str, agent_config: Optional[AgentConfig] = None) -> List[Document]: # Added agent_config
    """
    Loads all files from a local directory and creates a Document for each, ensuring path safety.
    Applies data policies to each file's content.
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
        def _load_directory_blocking(current_folder_path: Path, policies: Optional[List[DataPolicy]]) -> List[Document]: # Added policies
            local_docs = []
            for file in current_folder_path.rglob("*.*"):
                if file.is_dir():
                    continue
                try:
                    # This loader primarily handles .txt files directly.
                    # For other file types (PDF, DOCX, CSV), you would typically
                    # call their respective specialized loaders (pdf_loader.load, etc.)
                    # and pass the agent_config to them.
                    # This current implementation focuses on .txt files for direct processing here.
                    if file.suffix.lower() == '.txt':
                        with open(file, "r", encoding="utf-8", errors='ignore') as f:
                            text = f.read()

                        processed_text = text
                        document_blocked = False

                        # Apply data policies if provided
                        if policies:
                            logger.debug(f"Applying data policies to {file.name} in directory loader...")
                            processed_text, document_blocked = _apply_data_policies(text, policies)

                        if document_blocked:
                            logger.warning(f"File '{file.name}' was completely blocked by a data policy and will not be processed.")
                            continue # Skip this file if it's blocked

                        if processed_text.strip(): # Ensure there's actual content left
                            doc = Document(
                                page_content=processed_text,
                                metadata={
                                    "source": str(file.resolve()),
                                    "source_type": "local_directory",
                                    "file_name": file.name, # Added for consistency
                                    "file_path": str(file) # Added for consistency
                                }
                            )
                            local_docs.append(doc)
                        else:
                            logger.debug(f"File '{file.name}' had no content after policy application or was empty.")
                    else:
                        # Placeholder for integrating other specific loaders.
                        # For a comprehensive directory loader, you'd likely map file
                        # extensions to specific loader functions and await them.
                        logger.debug(f"Skipping file {file.name} (unsupported type {file.suffix}) for direct loading in directory_loader. Consider integrating specific loaders for this type.")


                except Exception as e:
                    logger.error(f"Error reading file {file}: {e}", exc_info=True)
            return local_docs

        # MODIFIED: Pass agent_config.data_policies to the blocking helper function
        docs = await asyncio.to_thread(_load_directory_blocking, safe_folder_path, agent_config.data_policies if agent_config else None)

        logger.info(f"Successfully loaded {len(docs)} documents from {folder_path}.")
        return docs
    except ValueError as e:
        logger.error(f"Security error during local directory loading: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading from directory {folder_path}: {e}", exc_info=True)
        return []