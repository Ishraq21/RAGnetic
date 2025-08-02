# app/pipelines/web_crawler_loader.py

import logging
import trafilatura
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from urllib.parse import urlparse
from datetime import datetime
import asyncio

from app.schemas.agent import AgentConfig, DataSource
from app.pipelines.data_policy_utils import apply_data_policies
from app.pipelines.metadata_utils import generate_base_metadata

logger = logging.getLogger(__name__)


def _trafilatura_extractor(html: str) -> str:
    """A custom extractor function that uses trafilatura to get main text content."""
    try:
        extracted_text = trafilatura.extract(html)
        return extracted_text or ""
    except Exception as e:
        logger.warning(f"Error during Trafilatura HTML extraction: {e}", exc_info=True)
        return ""


async def load(url: str, max_depth: int = 2, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]:
    processed_docs = []
    try:
        # --- Input Validation: URL Scheme ---
        if not url:
            logger.error("Error: A starting URL is required for the web crawler.")
            return []

        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logger.error(f"Attempted to crawl unsupported URL scheme: {parsed_url.scheme} in {url}. Only 'http' and 'https' are allowed for security reasons.")
            raise ValueError("Unsupported URL scheme for web crawling.")
        if not parsed_url.netloc:
            logger.error(f"Invalid URL format for crawling: missing domain/host in {url}.")
            raise ValueError("Invalid URL format for crawling.")

        logger.info(f"Starting web crawl from URL: {url} with max_depth: {max_depth}")

        loader = RecursiveUrlLoader(
            url=url,
            max_depth=max_depth,
            extractor=_trafilatura_extractor,
            prevent_outside=True,
            use_async=True,
            timeout=60,
            check_response_status=True
        )

        docs_raw = await asyncio.to_thread(loader.load)

        for doc in docs_raw:
            processed_text = doc.page_content
            document_blocked = False

            if agent_config and agent_config.data_policies:
                logger.debug(f"Applying data policies to crawled page '{doc.metadata.get('url', 'unknown')}'...")
                processed_text, document_blocked = apply_data_policies(doc.page_content, agent_config.data_policies, policy_context="webpage")

            if document_blocked:
                logger.warning(f"Crawled page from '{doc.metadata.get('url', 'unknown')}' was completely blocked by a data policy and will not be processed.")
                continue

            if processed_text.strip():
                source_context = doc.metadata.get('url', url)
                metadata = generate_base_metadata(source, source_context=source_context, source_type="web_crawler")
                # Add web crawler-specific keys
                metadata["source_url"] = doc.metadata.get('url', url)
                metadata["file_name"] = urlparse(doc.metadata.get('url', url)).netloc + urlparse(doc.metadata.get('url', url)).path.replace('/', '_')
                metadata["file_path"] = doc.metadata.get('url', url)
                metadata["crawl_depth"] = doc.metadata.get('depth', 0)
                metadata["page_title"] = doc.metadata.get('title', 'N/A')

                doc.page_content = processed_text
                doc.metadata = {**doc.metadata, **metadata}
                processed_docs.append(doc)
            else:
                logger.debug(f"Crawled page from '{doc.metadata.get('url', 'unknown')}' had no content after policy application or was empty.")

        logger.info(f"Crawled and loaded {len(processed_docs)} processed pages starting from {url} with enriched metadata.")
        return processed_docs

    except ValueError as e:
        logger.error(f"Validation error for web crawling URL {url}: {e}")
        return []
    except Exception as e:
        logger.error(f"An error occurred while crawling website {url}: {e}", exc_info=True)
        return []