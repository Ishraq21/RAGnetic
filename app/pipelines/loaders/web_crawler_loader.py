# app/pipelines/web_crawler_loader.py
import hashlib
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
                source_context = doc.metadata.get("url", url)

                metadata = generate_base_metadata(
                    source,
                    source_context=source_context,
                    source_type="web_crawler",
                )
                metadata.update(
                    {
                        "source_url": source_context,
                        "file_name": urlparse(source_context).netloc
                                     + urlparse(source_context).path.replace("/", "_"),
                        "file_path": source_context,
                        "crawl_depth": doc.metadata.get("depth", 0),
                        "page_title": doc.metadata.get("title", "N/A"),

                        "doc_name": source_context,  # group by URL
                        "source_name": source_context,  # used by _generate_chunk_id
                        "chunk_index": 0,  # one chunk per page
                    }
                )

                # reproducible, stable ID (hash of URL works nicely)
                doc_id = hashlib.sha256(source_context.encode("utf-8")).hexdigest()[:12]
                metadata.setdefault("original_doc_id", doc_id)

                doc.id = doc_id
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