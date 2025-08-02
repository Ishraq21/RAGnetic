# app/pipelines/notebook_loader.py

import logging
import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import NotebookLoader
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
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED:
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path


async def load_notebook(path: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]:
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

        for doc_index, doc in enumerate(documents):
            processed_text = doc.page_content
            document_blocked = False

            if agent_config and agent_config.data_policies:
                logger.debug(f"Applying data policies to notebook cell {doc_index + 1} from {safe_path.name}...")
                processed_text, document_blocked = apply_data_policies(doc.page_content, agent_config.data_policies, policy_context="notebook cell")

            if document_blocked:
                logger.warning(f"Notebook cell {doc_index + 1} from '{safe_path.name}' was completely blocked by a data policy and will not be processed.")
                continue

            if processed_text.strip():
                metadata = generate_base_metadata(
                    source,
                    source_context=safe_path.name,
                    source_type="file",
                )

                # notebook-specific + chunk-identity fields
                metadata.update({
                    "source_path": str(safe_path.resolve()),
                    "file_name": safe_path.name,
                    "file_type": safe_path.suffix.lower(),
                    "cell_number": doc_index + 1,

                    "doc_name": safe_path.name,
                    "source_name": safe_path.name,
                    "chunk_index": doc_index,  # 0-based cell index
                })

                # reproducible, row-level ID
                doc_id = f"{safe_path.stem}-cell{doc_index}"
                metadata.setdefault("original_doc_id", doc_id)

                doc.page_content = processed_text
                doc.metadata = {**doc.metadata, **metadata}  # merge any existing meta first
                doc.id = doc_id  # give the chunk its stable id
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