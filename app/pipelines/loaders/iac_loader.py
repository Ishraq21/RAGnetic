# app/pipelines/iac_loader.py

import os
import logging
from typing import List, Optional
from pathlib import Path
import asyncio
from datetime import datetime

from app.core.config import get_path_settings
from app.schemas.agent import AgentConfig, DataSource
from app.pipelines.data_policy_utils import apply_data_policies
from app.pipelines.metadata_utils import generate_base_metadata

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_PATH_SETTINGS = get_path_settings()
_PROJECT_ROOT_FROM_CONFIG = _PATH_SETTINGS["PROJECT_ROOT"]
_ALLOWED_DATA_DIRS_RESOLVED = _PATH_SETTINGS["ALLOWED_DATA_DIRS"]
logger.debug(f"Loaded allowed data directories for IaC loader from central config: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
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


async def load(file_path: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]:
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
            processed_text, document_blocked = apply_data_policies(processed_text, agent_config.data_policies, policy_context="file")

        if document_blocked:
            logger.warning(f"IaC file '{safe_file_path.name}' was completely blocked by a data policy and will not be processed.")
            return []

        if not processed_text.strip():
            logger.warning(f"IaC file {safe_file_path.name} contained no readable text after policy application. Skipping chunking.")
            return []

        raw_docs[0].page_content = processed_text

        chunks = splitter.split_documents(raw_docs)

        for chunk_idx, chunk in enumerate(chunks):
            metadata = generate_base_metadata(source, source_context=safe_file_path.name, source_type="infrastructure_as_code")
            # Add IaC-specific keys
            metadata["source_path"] = str(safe_file_path.resolve())
            metadata["file_name"] = safe_file_path.name
            metadata["file_type"] = safe_file_path.suffix.lower()
            metadata["chunk_index"] = chunk_idx

            chunk.metadata = {**chunk.metadata, **metadata}

        logger.info(f"Successfully chunked {safe_file_path.name} into {len(chunks)} IaC chunks with enriched metadata.")
        return chunks

    except ValueError as e:
        logger.error(f"Security or validation error during IaC file loading: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load or process IaC file {file_path}: {e}", exc_info=True)
        return []