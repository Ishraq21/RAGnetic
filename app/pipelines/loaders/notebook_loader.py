import logging
import os
import re # Added for PII redaction
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import NotebookLoader
import asyncio
from datetime import datetime

from app.core.config import get_path_settings
from app.schemas.agent import AgentConfig, DataPolicy, DataSource

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"] # Store project root if needed
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"] # Store resolved allowed dirs
logger.debug(f"Loaded allowed data directories for notebook loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
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
    Returns the processed text and a boolean indicating if the document (notebook cell) was blocked.
    """
    processed_text = text
    document_blocked = False # In notebook loader, 'document' refers to a 'cell'

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
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.warning(f"Keyword '{keyword}' found. This cell contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document':
                        logger.warning(f"Keyword '{keyword}' found. Cell is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked

    return processed_text, document_blocked


async def load_notebook(path: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]:
    """
    Loads a Jupyter Notebook (.ipynb) file from the given path,
    with path safety validation, data policy application, and standardized error logging.
    Now supports asynchronous loading and enriched metadata.
    """
    processed_documents = []
    try:
        safe_path = _is_path_safe_and_within_allowed_dirs(path)

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
            str(safe_path),
            include_outputs=True,
            max_output_length=100,
            remove_newline=True
        )
        documents = await asyncio.to_thread(loader.load)

        for doc_index, doc in enumerate(documents): # Iterate with index for cell metadata
            processed_text = doc.page_content
            document_blocked = False

            if agent_config and agent_config.data_policies:
                logger.debug(f"Applying data policies to notebook cell {doc_index + 1} from {safe_path.name}...")
                processed_text, document_blocked = _apply_data_policies(doc.page_content, agent_config.data_policies)

            if document_blocked:
                logger.warning(f"Notebook cell {doc_index + 1} from '{safe_path.name}' was completely blocked by a data policy and will not be processed.")
                continue

            if processed_text.strip():
                # Create base metadata for the cell
                metadata = {
                    "source_path": str(safe_path.resolve()), # Full path for lineage
                    "file_name": safe_path.name,
                    "file_type": safe_path.suffix.lower(),
                    "load_timestamp": datetime.now().isoformat(), # NEW: Add load timestamp
                    "cell_number": doc_index + 1, # Specific to notebooks
                    **doc.metadata # Preserve existing metadata from NotebookLoader (e.g., cell type, execution order)
                }
                # Add granular DataSource info if available
                if source: # NEW: Add info from DataSource object for lineage
                    metadata["source_type_config"] = source.model_dump()
                    if source.url: metadata["source_url"] = source.url
                    if source.db_connection: metadata["source_db_connection"] = source.db_connection
                    if source.folder_id: metadata["source_gdoc_folder_id"] = source.folder_id
                    if source.document_ids: metadata["source_gdoc_document_ids"] = source.document_ids


                doc.page_content = processed_text # Update the document with processed text
                doc.metadata = metadata # Overwrite with enriched metadata (or merge)
                processed_documents.append(doc)
            else:
                logger.debug(f"Notebook cell {doc_index + 1} from '{safe_path.name}' had no content after policy application or was empty.")


        logger.info(f"Loaded {len(processed_documents)} processed cells from notebook: {safe_path.name} with enriched metadata.")
        return processed_documents
    except ValueError as e:
        logger.error(f"Security or validation error during notebook loading: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load notebook {path}. Error: {e}", exc_info=True)
        return []