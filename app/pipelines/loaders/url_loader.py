import logging
import re # Added for PII redaction
import trafilatura
from typing import List, Optional
from langchain_core.documents import Document
from urllib.parse import urlparse # Added import for URL scheme validation
from datetime import datetime

from app.schemas.agent import AgentConfig, DataPolicy, DataSource

logger = logging.getLogger(__name__) # Added logger initialization

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


async def load(url: str, agent_config: Optional[AgentConfig] = None, source: Optional[DataSource] = None) -> List[Document]: # MODIFIED: Added source parameter
    """
    Loads a webpage and creates a single Document from its main content,
    with URL scheme validation, data policy application, and standardized error logging.
    Now supports enriched metadata for lineage.
    """
    try:
        # --- Input Validation: URL Scheme ---
        if not url:
            logger.error("Error: A starting URL is required for the web crawler.")
            return []

        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logger.error(f"Attempted to load from unsupported URL scheme: {parsed_url.scheme} in {url}. Only 'http' and 'https' are allowed for security reasons.")
            raise ValueError("Unsupported URL scheme for security.")
        if not parsed_url.netloc: # Basic check for domain/host presence
            logger.error(f"Invalid URL format: missing domain/host in {url}.")
            raise ValueError("Invalid URL format.")

        logger.info(f"Attempting to download and extract content from URL: {url}")
        # trafilatura.fetch_url handles network errors internally, often returning None or raising exceptions
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            logger.warning(f"No content downloaded from {url}. It might be empty, inaccessible, or an error occurred during fetch.")
            return []

        text = trafilatura.extract(downloaded)

        processed_text = text
        document_blocked = False

        if agent_config and agent_config.data_policies:
            logger.info(f"Applying data policies to content from URL: {url}...")
            processed_text, document_blocked = _apply_data_policies(text, agent_config.data_policies)

        if document_blocked:
            logger.warning(f"Content from URL '{url}' was completely blocked by a data policy and will not be processed.")
            return []

        if processed_text.strip(): # Ensure there's actual content left after policies
            # Create base metadata for the webpage
            metadata = {
                "source_url": url, # Full URL for lineage
                "file_name": parsed_url.netloc + parsed_url.path.replace('/', '_'), # Basic file name
                "file_path": url, # Treat URL as path for consistency
                "load_timestamp": datetime.now().isoformat(), # NEW: Add load timestamp
            }
            # Add general source info if available from the DataSource object
            if source: # NEW: Add info from DataSource object for lineage
                metadata["source_type_config"] = source.model_dump() # Store entire DataSource config
                if source.path: metadata["source_path"] = source.path
                if source.db_connection: metadata["source_db_connection"] = source.db_connection
                if source.folder_id: metadata["source_gdoc_folder_id"] = source.folder_id
                if source.document_ids: metadata["source_gdoc_document_ids"] = source.document_ids

            metadata["source_type"] = source.type if source else "url" # Use DataSource type or default

            doc = Document(
                page_content=processed_text,
                metadata={**metadata} # Use the enriched metadata
            )
            logger.info(f"Successfully extracted and processed content from {url} with enriched metadata.")
            return [doc]
        else:
            logger.warning(f"Content extraction from {url} resulted in empty text after policy application. Skipping document creation.")
            return []
    except trafilatura.errors.NonSuccessfulRequest as e:
        logger.error(f"Trafilatura failed to fetch URL {url} due to non-successful request: {e}", exc_info=True)
        return []
    except ValueError as e:
        logger.error(f"Validation error for URL {url}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading URL {url}: {e}", exc_info=True)
        return []