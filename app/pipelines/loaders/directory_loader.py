# app/pipelines/directory_loader.py

import os
import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
import asyncio
import concurrent.futures
from datetime import datetime

from app.core.config import get_path_settings
from app.schemas.agent import AgentConfig, DataSource, DataPolicy
from app.pipelines.data_policy_utils import apply_data_policies
from app.pipelines.metadata_utils import generate_base_metadata

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"]
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"]
logger.debug(
    f"Loaded allowed data directories for directory loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
# --- End Centralized Configuration ---


def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    resolved_path = Path(input_path).resolve()

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED:
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path


def _process_single_file(file_path: Path, policies: Optional[List[DataPolicy]], source: Optional[DataSource]) -> \
Optional[Document]:
    """
    Synchronously processes a single text file from a directory.
    This function is designed to be run in a ThreadPoolExecutor.
    """
    if file_path.is_dir():
        return None

    try:
        if file_path.suffix.lower() == '.txt':
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                text = f.read()

            processed_text = text
            document_blocked = False

            if policies:
                logger.debug(f"Applying data policies to {file_path.name} in directory loader (worker)...")
                processed_text, document_blocked = apply_data_policies(text, policies, policy_context="file")

            if document_blocked:
                logger.warning(
                    f"File '{file_path.name}' was completely blocked by a data policy and will not be processed.")
                return None

            if processed_text.strip():
                metadata = generate_base_metadata(source, source_context=file_path.name, source_type="file")
                # Add directory-specific keys
                metadata["source_path"] = str(file_path.resolve())
                metadata["file_name"] = file_path.name
                metadata["file_type"] = file_path.suffix.lower()

                return Document(
                    page_content=processed_text,
                    metadata={**metadata}
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
List[Document]:
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
                    for file_path in all_files_in_dir
                ]
                processed_results = await asyncio.gather(*tasks)

            docs = [doc for doc in processed_results if doc is not None]

        else:
            logger.info(f"Using sequential ingestion for directory: {safe_folder_path}")
            for file_path in all_files_in_dir:
                doc = await asyncio.to_thread(_process_single_file, file_path, policies, source)
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