import os
import logging
from typing import List
from pathlib import Path
import asyncio # NEW: Added import for asynchronous operations

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

logger = logging.getLogger(__name__)

# --- Configuration for Allowed Data Directories (copied for consistency) ---
# IMPORTANT: Adjust these paths to correctly reflect where your 'data' and 'agents_data'
# directories are located relative to your project's root when the application runs.
# os.getcwd() assumes the script is run from the project root.
_PROJECT_ROOT = Path(os.getcwd()) # This should be your RAGnetic project's base directory
_ALLOWED_DATA_DIRS = [
    _PROJECT_ROOT / "data",
    _PROJECT_ROOT / "agents_data" # If agent configs or related files can be loaded via 'local' source type
    # Add any other directories that are explicitly allowed for local data sources
]
# Resolve all allowed directories to their absolute, canonical form once at startup
_ALLOWED_DATA_DIRS_RESOLVED = [d.resolve() for d in _ALLOWED_DATA_DIRS]
logger.info(f"Configured allowed data directories for IaC loader: {[str(d) for d in _ALLOWED_DATA_DIRS_RESOLVED]}")
# --- End Configuration ---

def _is_path_safe_and_within_allowed_dirs(input_path: str) -> Path:
    """
    Validates if the input_path resolves to a location within the configured
    allowed data directories. Raises ValueError if unsafe.
    Returns the resolved absolute Path if safe.
    """
    resolved_path = Path(input_path).resolve() # Resolve '..' and get absolute path

    is_safe = False
    for allowed_dir in _ALLOWED_DATA_DIRS_RESOLVED:
        if resolved_path.is_relative_to(allowed_dir):
            is_safe = True
            break

    if not is_safe:
        raise ValueError(f"Attempted to access path outside allowed directories: {resolved_path}")

    return resolved_path

async def load(file_path: str) -> List[Document]: # MODIFIED: Changed to async def
    """
    Loads Infrastructure-as-Code (IaC) files like Terraform and Kubernetes YAML,
    then splits them into syntax-aware chunks, with path safety validation.
    Now supports asynchronous loading.
    """
    try:
        # First, validate the file_path itself
        safe_file_path = _is_path_safe_and_within_allowed_dirs(file_path)

        # Validate file existence and type
        if not safe_file_path.exists():
            logger.error(f"Error: IaC file not found at {safe_file_path}")
            return []
        if not safe_file_path.is_file():
            logger.error(f"Error: Provided path '{safe_file_path}' is not a file.")
            return []

        file_extension = safe_file_path.suffix.lower() # Use suffix from Path object
        splitter = None

        if file_extension in [".tf", ".tfvars"]:
            logger.info(f"Loading Terraform file: {safe_file_path}")
            # Separators for HCL/Terraform code
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""],
            )
        elif file_extension in [".yaml", ".yml"]:
            logger.info(f"Loading YAML file: {safe_file_path}")
            # Manually define separators that work well for YAML
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n---\n", "\n\n", "\n", " ", ""],
            )
        else:
            logger.warning(f"Unsupported IaC file type: {safe_file_path.name}. Skipping.")
            return []

        # Load the raw text content of the file
        # TextLoader should handle the safe_file_path as a string
        loader = TextLoader(str(safe_file_path), encoding="utf-8")
        raw_docs = await asyncio.to_thread(loader.load) # MODIFIED: Use asyncio.to_thread

        # Use the appropriate splitter
        chunks = splitter.split_documents(raw_docs) # This is CPU-bound, no need for to_thread

        # Add metadata to each chunk
        for chunk in chunks:
            chunk.metadata['source_type'] = 'infrastructure_as_code'
            chunk.metadata['file_name'] = safe_file_path.name # Use name from Path object

        logger.info(f"Successfully chunked {safe_file_path.name} into {len(chunks)} IaC chunks.")
        return chunks

    except ValueError as e:
        logger.error(f"Security or validation error during IaC file loading: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load or process IaC file {file_path}: {e}", exc_info=True)
        return []