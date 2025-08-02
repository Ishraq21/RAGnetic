# app/pipelines/metadata_utils.py

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from app.schemas.agent import DataSource, AgentConfig

logger = logging.getLogger(__name__)

def generate_base_metadata(
    source: DataSource,
    source_context: str, # e.g., "file.pdf", "table_name", "url"
    source_type: Optional[str] = None # Optional override for source_type if needed
) -> Dict[str, Any]:
    """
    Generates a consistent set of base metadata for a document chunk, regardless of the source type.

    This function enforces a standard set of keys to improve downstream processing,
    UI rendering, and audit trails.

    Args:
        source (DataSource): The DataSource object from the AgentConfig.
        source_context (str): A human-readable identifier for the specific item being processed,
                              e.g., the file name, table name, or specific URL.
        source_type (str, optional): An optional override for the source_type. If not provided,
                                     it defaults to source.type.

    Returns:
        Dict[str, Any]: A dictionary containing standardized metadata.
    """
    if not source:
        logger.warning("Attempted to generate metadata with a None DataSource object.")
        return {"source_type": "unknown", "source_name": "unknown", "load_timestamp": datetime.now().isoformat()}

    # Standardize metadata keys
    standard_metadata = {
        "source_type": source_type if source_type else source.type,
        "source_name": source_context,
        "load_timestamp": datetime.now().isoformat(),
        # Store the entire DataSource object for full lineage
        "source_config": source.model_dump()
    }

    # Add optional context-specific keys
    if source.path:
        standard_metadata["source_path"] = source.path
    if source.url:
        standard_metadata["source_url"] = source.url
    if source.db_connection:
        standard_metadata["source_db_connection"] = source.db_connection
    if source.folder_id:
        standard_metadata["source_gdoc_folder_id"] = source.folder_id
    if source.document_ids:
        standard_metadata["source_gdoc_document_ids"] = source.document_ids

    return standard_metadata