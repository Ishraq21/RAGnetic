import os
import logging
import re
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
import asyncio
import concurrent.futures
from datetime import datetime

from app.core.config import get_path_settings
from app.schemas.agent import AgentConfig, DataPolicy, DataSource  # MODIFIED: Import DataSource

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"]
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"]
logger.info(
    f"Loaded allowed data directories for directory loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")


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
                    logger.warning(
                        f"PII type '{pii_type}' (name) is complex and not fully implemented for regex-based redaction. Skipping for now.")
                    continue

                if pattern:
                    replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 8)
                    if pii_type == 'credit_card': replacement = pii_config.redaction_placeholder or (
                                pii_config.redaction_char * 16)
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
                        logger.warning(
                            f"Keyword '{keyword}' found. This file contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document':
                        logger.warning(
                            f"Keyword '{keyword}' found. File is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked

    return processed_text, document_blocked


def _process_single_file(file_path: Path, policies: Optional[List[DataPolicy]], source: Optional[DataSource]) -> \
Optional[Document]:  # MODIFIED: Added source parameter
    """
    Synchronously processes a single text file from a directory.
    This function is designed to be run in a ThreadPoolExecutor.
    """
    if file_path.is_dir():
        return None  # Skip directories

    try:
        if file_path.suffix.lower() == '.txt':
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                text = f.read()

            processed_text = text
            document_blocked = False

            if policies:
                logger.debug(f"Applying data policies to {file_path.name} in directory loader (worker)...")
                processed_text, document_blocked = _apply_data_policies(text, policies)

            if document_blocked:
                logger.warning(
                    f"File '{file_path.name}' was completely blocked by a data policy and will not be processed.")
                return None

            if processed_text.strip():
                # Create base metadata for the document
                metadata = {
                    "source_path": str(file_path.resolve()),  # Full path for lineage
                    "file_name": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "load_timestamp": datetime.now().isoformat(),  # Add load timestamp
                }
                # Add general source info if available from the DataSource object
                if source:  # NEW: Add info from DataSource object for lineage
                    metadata["source_type_config"] = source.model_dump()  # Store entire DataSource config
                    if source.url: metadata["source_url"] = source.url
                    if source.db_connection: metadata["source_db_connection"] = source.db_connection
                    if source.folder_id: metadata["source_gdoc_folder_id"] = source.folder_id
                    if source.document_ids: metadata["source_gdoc_document_ids"] = source.document_ids

                # Note: For directory loader, source_type is typically 'local',
                # but we're pulling specific file types (like .txt) for processing here.
                # The primary source type comes from the DataSource object itself.
                metadata["source_type"] = source.type if source else "local_directory"  # Use DataSource type or default

                return Document(
                    page_content=processed_text,
                    metadata={**metadata}  # Use the enriched metadata
                )
            else:
                logger.debug(f"File '{file_path.name}' had no content after policy application or was empty.")
                return None
        else:
            logger.debug(
                f"Skipping file {file_path.name} (unsupported type {file_path.suffix}) for direct loading in directory_loader.")
            return None

    except Exception as e:
        logger.error(f"Error processing file {file_path} in parallel: {e}", exc_info=True)
        return None


async def load(folder_path: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> \
List[Document]:  # MODIFIED: Added source parameter
    """
    Loads all files from a local directory, applying data policies.
    Supports parallel loading of files within the directory based on agent_config.scaling.
    """
    docs = []
    try:
        safe_folder_path = _is_path_safe_and_within_allowed_dirs(folder_path)

        if not safe_folder_path.exists():
            logger.warning(f"Directory not found: {safe_folder_path}. Skipping loading.")
            return []
        if not safe_folder_path.is_dir():
            logger.warning(f"Path is not a directory: {safe_folder_path}. Skipping loading.")
            return []

        logger.info(f"Loading documents from safe directory: {safe_folder_path}")

        policies = agent_config.data_policies if agent_config and agent_config.data_policies else []

        num_workers = os.cpu_count() or 1
        if agent_config and agent_config.scaling and agent_config.scaling.num_ingestion_workers:
            num_workers = agent_config.scaling.num_ingestion_workers

        all_files_in_dir = [f for f in safe_folder_path.rglob("*") if f.is_file()]

        if agent_config and agent_config.scaling and agent_config.scaling.parallel_ingestion:
            logger.info(f"Using parallel ingestion with {num_workers} workers for directory: {safe_folder_path}")

            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                tasks = [
                    loop.run_in_executor(executor, _process_single_file, file_path, policies, source)
                    # <-- Passed source
                    for file_path in all_files_in_dir
                ]
                processed_results = await asyncio.gather(*tasks)

            docs = [doc for doc in processed_results if doc is not None]

        else:
            logger.info(f"Using sequential ingestion for directory: {safe_folder_path}")
            # Sequential processing
            for file_path in all_files_in_dir:
                # MODIFIED: Pass 'source' object to _process_single_file
                doc = await asyncio.to_thread(_process_single_file, file_path, policies, source)  # <-- Passed source
                if doc is not None:
                    docs.append(doc)

        logger.info(f"Successfully loaded {len(docs)} documents from {folder_path} with enriched metadata.")
        return docs
    except ValueError as e:
        logger.error(f"Security error during local directory loading: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading from directory {folder_path}: {e}", exc_info=True)
        return []