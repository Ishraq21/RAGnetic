import os
import logging
import re # Added for PII redaction
from typing import List, Optional
from pathlib import Path
import asyncio
from datetime import datetime # NEW: Import datetime for load_timestamp

from app.core.config import get_path_settings
from app.schemas.agent import AgentConfig, DataPolicy, DataSource

from langchain_community.document_loaders import TextLoader# Re-confirming TextLoader import
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"]
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"]
logger.info(f"Loaded allowed data directories for IaC loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
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
                    pattern = r'\b(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
                elif pii_type == 'ssn':
                    pattern = r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'
                elif pii_type == 'credit_card':
                    pattern = r'\b(?:4\d{3}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|5[1-5]\d{2}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|3[47]\d{13}|6(?:011|5\d{2})[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b'
                elif pii_type == 'name':
                    logger.warning(f"PII type '{pii_type}' (name) is complex and not fully implemented for regex-based redaction. Skipping for now.")
                    continue

                if pattern:
                    replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 8)
                    if pii_type == 'credit_card': replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 16)
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
                        logger.warning(f"Keyword '{keyword}' found. This file contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document':
                        logger.warning(f"Keyword '{keyword}' found. File is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked

    return processed_text, document_blocked


async def load(file_path: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]: # MODIFIED: Added source parameter
    """
    Loads Infrastructure-as-Code (IaC) files like Terraform and Kubernetes YAML,
    applies data policies, then splits them into syntax-aware chunks, with path safety validation.
    Now supports asynchronous loading and enriched metadata.
    """
    try:
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

        if not safe_file_path.exists():
            logger.error(f"Error: IaC file not found at {safe_file_path}")
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []

        file_extension = safe_file_path.suffix.lower()
        splitter = None

        if file_extension in [".tf", ".tfvars"]:
            logger.info(f"Loading Terraform file: {safe_file_path}")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""],
            )
        elif file_extension in [".yaml", ".yml"]:
            logger.info(f"Loading YAML file: {safe_file_path}")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n---\n", "\n\n", "\n", " ", ""],
            )
        else:
            logger.warning(f"Unsupported IaC file type: {safe_file_path.name}. Skipping.")
            return []

        loader = TextLoader(str(safe_file_path), encoding="utf-8")
        raw_docs = await asyncio.to_thread(loader.load)

        if not raw_docs:
            return []

        processed_text = raw_docs[0].page_content
        document_blocked = False

        if agent_config and agent_config.data_policies:
            logger.info(f"Applying data policies to IaC file: {safe_file_path.name}...")
            processed_text, document_blocked = _apply_data_policies(processed_text, agent_config.data_policies)

        if document_blocked:
            logger.warning(f"IaC file '{safe_file_path.name}' was completely blocked by a data policy and will not be processed.")
            return []

        if not processed_text.strip():
            logger.warning(f"IaC file {safe_file_path.name} contained no readable text after policy application. Skipping chunking.")
            return []

        raw_docs[0].page_content = processed_text

        chunks = splitter.split_documents(raw_docs)

        for chunk_idx, chunk in enumerate(chunks): # MODIFIED: Iterate with index
            # Create base metadata for the chunk
            metadata = {
                "source_path": str(safe_file_path.resolve()),
                "file_name": safe_file_path.name,
                "file_type": safe_file_path.suffix.lower(),
                "load_timestamp": datetime.now().isoformat(), # NEW: Add load timestamp
                "chunk_index": chunk_idx, # NEW: Add chunk index for multi-chunk documents
                "source_type": 'infrastructure_as_code', # Set specific source type
            }
            # Add general source info if available from the DataSource object
            if source: # NEW: Add info from DataSource object for lineage
                metadata["source_type_config"] = source.model_dump()
                if source.url: metadata["source_url"] = source.url
                if source.db_connection: metadata["source_db_connection"] = source.db_connection
                if source.folder_id: metadata["source_gdoc_folder_id"] = source.folder_id
                if source.document_ids: metadata["source_gdoc_document_ids"] = source.document_ids

            # Merge with existing metadata from chunk (if any)
            chunk.metadata = {**chunk.metadata, **metadata} # Merge, with new metadata taking precedence

        logger.info(f"Successfully chunked {safe_file_path.name} into {len(chunks)} IaC chunks with enriched metadata.")
        return chunks

    except ValueError as e:
        logger.error(f"Security or validation error during IaC file loading: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load or process IaC file {file_path}: {e}", exc_info=True)
        return []