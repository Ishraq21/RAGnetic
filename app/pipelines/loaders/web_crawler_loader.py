import logging
import re # Added for PII redaction
import trafilatura
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup # Imported for RecursiveUrlLoader if it's used with it
from urllib.parse import urlparse # Added import for URL scheme validation
from datetime import datetime # NEW: Import datetime for load_timestamp

# NEW: Import DataSource for lineage
from app.schemas.agent import AgentConfig, DataPolicy, DataSource # MODIFIED: Import DataSource

logger = logging.getLogger(__name__) # Added logger initialization

def _trafilatura_extractor(html: str) -> str:
    """A custom extractor function that uses trafilatura to get main text content."""
    try:
        extracted_text = trafilatura.extract(html)
        return extracted_text or ""
    except Exception as e:
        logger.warning(f"Error during Trafilatura HTML extraction: {e}", exc_info=True)
        return ""

def _apply_data_policies(text: str, policies: List[DataPolicy]) -> tuple[str, bool]:
    """
    Applies data policies (redaction/filtering) to the text content.
    Returns the processed text and a boolean indicating if the document (webpage) was blocked.
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
                    # Common phone number formats (adjust or enhance regex as needed for international formats)
                    pattern = r'\b(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
                elif pii_type == 'ssn':
                    pattern = r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'
                elif pii_type == 'credit_card':
                    # Basic credit card pattern (major issuers, e.g., Visa, Mastercard, Amex, Discover)
                    pattern = r'\b(?:4\d{3}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|5[1-5]\d{2}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}|3[47]\d{13}|6(?:011|5\d{2})[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b'
                elif pii_type == 'name':
                    # Name redaction is complex and context-dependent.
                    logger.warning(f"PII type '{pii_type}' (name) is complex and not fully implemented for regex-based redaction. Skipping for now.")
                    continue

                if pattern:
                    replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 8) # Generic length
                    if pii_type == 'credit_card': replacement = pii_config.redaction_placeholder or (pii_config.redaction_char * 16) # Specific length for CC
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
                        logger.warning(f"Keyword '{keyword}' found. This document contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document':
                        logger.warning(f"Keyword '{keyword}' found. Webpage is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked

    return processed_text, document_blocked


async def load(url: str, max_depth: int = 2, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]: # MODIFIED: Added source parameter
    """
    Crawls a website starting from the given URL up to a max_depth,
    ingesting the content of each page found, applies data policies,
    with URL validation and proper logging.
    Now supports enriched metadata for lineage.
    """
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
        if not parsed_url.netloc: # Basic check for domain/host presence
            logger.error(f"Invalid URL format for crawling: missing domain/host in {url}.")
            raise ValueError("Invalid URL format for crawling.")

        logger.info(f"Starting web crawl from URL: {url} with max_depth: {max_depth}")

        # Initialize the loader with our custom text extractor and settings
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=max_depth,
            extractor=_trafilatura_extractor,
            prevent_outside=True,  # IMPORTANT: Prevents crawling external sites
            use_async=True,  # Uses asyncio for faster crawling
            timeout=60,  # Timeout for each request
            check_response_status=True
        )

        docs_raw = loader.load() # Renamed to docs_raw to clarify

        for doc in docs_raw: # Iterate over raw documents
            processed_text = doc.page_content
            document_blocked = False

            # Apply data policies if provided in agent_config
            if agent_config and agent_config.data_policies:
                logger.debug(f"Applying data policies to crawled page '{doc.metadata.get('url', 'unknown')}'...")
                processed_text, document_blocked = _apply_data_policies(doc.page_content, agent_config.data_policies)

            if document_blocked:
                logger.warning(f"Crawled page from '{doc.metadata.get('url', 'unknown')}' was completely blocked by a data policy and will not be processed.")
                continue

            if processed_text.strip(): # Ensure there's actual content left after policies
                # Create base metadata for the webpage
                metadata = {
                    "source_url": doc.metadata.get('url', url), # Original crawled URL
                    "file_name": urlparse(doc.metadata.get('url', url)).netloc + urlparse(doc.metadata.get('url', url)).path.replace('/', '_'), # Basic file name
                    "file_path": doc.metadata.get('url', url), # Treat URL as path for consistency
                    "load_timestamp": datetime.now().isoformat(), # NEW: Add load timestamp
                    "crawl_depth": doc.metadata.get('depth', 0), # Specific to web crawler
                    "page_title": doc.metadata.get('title', 'N/A'), # Page title from metadata
                }
                # Add general source info if available from the DataSource object
                if source: # NEW: Add info from DataSource object for lineage
                    metadata["source_type_config"] = source.model_dump() # Store entire DataSource config
                    if source.path: metadata["source_path"] = source.path
                    if source.db_connection: metadata["source_db_connection"] = source.db_connection
                    if source.folder_id: metadata["source_gdoc_folder_id"] = source.folder_id
                    if source.document_ids: metadata["source_gdoc_document_ids"] = source.document_ids

                metadata["source_type"] = source.type if source else "web_crawler" # Use DataSource type or default

                doc.page_content = processed_text # Update the document with processed text
                doc.metadata = {**doc.metadata, **metadata} # Merge existing LangChain metadata with new enriched data
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