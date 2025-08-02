# app/pipelines/url_loader.py
import hashlib
import logging
from pathlib import PurePosixPath

import trafilatura
from typing import List, Optional
from langchain_core.documents import Document
from urllib.parse import urlparse
from datetime import datetime
import asyncio

from app.schemas.agent import AgentConfig, DataSource
from app.pipelines.data_policy_utils import apply_data_policies
from app.pipelines.metadata_utils import generate_base_metadata

logger = logging.getLogger(__name__)


async def load(url: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]:
    try:
        # --- Input Validation: URL Scheme ---
        if not url:
            logger.error("Error: A starting URL is required for the web crawler.")
            return []

        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logger.error(f"Attempted to load from unsupported URL scheme: {parsed_url.scheme} in {url}. Only 'http' and 'https' are allowed for security reasons.")
            raise ValueError("Unsupported URL scheme for security.")
        if not parsed_url.netloc:
            logger.error(f"Invalid URL format: missing domain/host in {url}.")
            raise ValueError("Invalid URL format.")

        logger.info(f"Attempting to download and extract content from URL: {url}")
        downloaded = await asyncio.to_thread(trafilatura.fetch_url, url)

        if not downloaded:
            logger.warning(f"No content downloaded from {url}. It might be empty, inaccessible, or an error occurred during fetch.")
            return []

        text = await asyncio.to_thread(trafilatura.extract, downloaded)

        processed_text = text
        document_blocked = False

        if agent_config and agent_config.data_policies:
            logger.info(f"Applying data policies to content from URL: {url}...")
            processed_text, document_blocked = apply_data_policies(text, agent_config.data_policies, policy_context="webpage")

        if document_blocked:
            logger.warning(f"Content from URL '{url}' was completely blocked by a data policy and will not be processed.")
            return []

        if processed_text.strip():
            metadata = generate_base_metadata(source, source_context=url, source_type="url")
            # URL-specific keys
            metadata["source_url"] = url
            metadata["file_name"] = parsed_url.netloc + parsed_url.path.replace('/', '_')
            metadata["file_path"] = url

            # ---- identity keys & stable ID ----
            rel_path = PurePosixPath(url).as_posix()  # canonical form of the URL
            metadata.update(
                {
                    "doc_name": rel_path,  # group everything from this URL
                    "source_name": rel_path,  # used by _generate_chunk_id
                    "chunk_index": 0,  # single-chunk document
                }
            )

            full_hash = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()
            short_id = full_hash[:8]  # compact but unique

            metadata.setdefault("original_doc_id", full_hash)

            doc = Document(
                page_content=processed_text,
                metadata=metadata,
                id=short_id  # set .id for downstream use
            )
            logger.info(f"Successfully extracted and processed content from {url} with enriched metadata.")
            return [doc]
        else:
            logger.warning(f"Content extraction from {url} resulted in empty text after policy application. Skipping document creation.")
            return []
    except ValueError as e:
        logger.error(f"Validation error for URL {url}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading URL {url}: {e}", exc_info=True)
        return []