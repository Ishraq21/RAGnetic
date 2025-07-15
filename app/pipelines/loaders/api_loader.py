import requests
import json
import jsonpointer
import logging
import re # Added for PII redaction
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import asyncio
from datetime import datetime

# NEW: Import DataSource for lineage
from app.schemas.agent import AgentConfig, DataPolicy, DataSource

logger = logging.getLogger(__name__)


# Define a synchronous helper function to encapsulate the blocking requests calls
def _make_request_blocking(url: str, method: str, headers: Dict, params: Dict, payload: Dict, request_timeout: int):
    response = None
    if method.upper() == 'POST':
        response = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
    else:  # Default to GET
        response = requests.get(url, headers=headers, params=params, timeout=request_timeout)
    response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    return response.json()

def _apply_data_policies(text: str, policies: List[DataPolicy]) -> tuple[str, bool]:
    """
    Applies data policies (redaction/filtering) to the text content.
    Returns the processed text and a boolean indicating if the document (API record) was blocked.
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
                        logger.warning(f"Keyword '{keyword}' found. This API record contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document':
                        logger.warning(f"Keyword '{keyword}' found. API record is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked

    return processed_text, document_blocked


async def load(
        url: str,
        method: str = 'GET',
        headers: Dict = None,
        params: Dict = None,
        payload: Dict = None,
        json_pointer: str = None,
        agent_config: Optional[AgentConfig] = None,
        source: Optional[DataSource] = None # MODIFIED: Added source parameter
) -> List[Document]:
    """
    Fetches data from a REST API endpoint using GET or POST and creates a Document for each record,
    with robust URL validation, timeouts, data policy application, standardized error logging,
    asynchronous operations, and enriched metadata for lineage.
    """
    docs = []
    try:
        # --- Input Validation: URL Scheme and Format ---
        if not url:
            logger.error("Validation Error: An API URL is required.")
            return []

        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logger.error(
                f"Validation Error: Unsupported URL scheme for API: {parsed_url.scheme} in {url}. Only 'http' and 'https' are allowed for security.")
            raise ValueError("Unsupported URL scheme for API.")
        if not parsed_url.netloc:
            logger.error(f"Validation Error: Invalid API URL format: missing domain/host in {url}.")
            raise ValueError("Invalid API URL format.")

        logger.info(f"Making {method.upper()} request to {url}")

        request_timeout = 30  # Standardized timeout

        data = await asyncio.to_thread(
            _make_request_blocking,
            url=url,
            method=method,
            headers=headers if headers is not None else {},
            params=params if params is not None else {},
            payload=payload if payload is not None else {},
            request_timeout=request_timeout
        )

        # If a json_pointer is provided, navigate to the specific part of the JSON response
        if json_pointer:
            try:
                records = jsonpointer.resolve_pointer(data, json_pointer)
            except jsonpointer.JsonPointerException as e:
                logger.error(f"Error resolving JSON pointer '{json_pointer}' for URL {url}: {e}", exc_info=True)
                return []
        else:
            records = data

        # Ensure we're working with a list, even if the API returns a single object
        if not isinstance(records, list):
            records = [records]

        for record_idx, record in enumerate(records): # MODIFIED: Iterate with index for record number
            content = json.dumps(record, indent=2)

            processed_text = content
            document_blocked = False

            if agent_config and agent_config.data_policies:
                logger.debug(f"Applying data policies to API record {record_idx + 1} from {url}...")
                processed_text, document_blocked = _apply_data_policies(content, agent_config.data_policies)

            if document_blocked:
                logger.warning(f"API record {record_idx + 1} from '{url}' was completely blocked by a data policy and will not be processed.")
                continue

            if processed_text.strip():
                # Create base metadata for the API record
                metadata = {
                    "source_url": url, # Full URL for lineage
                    "api_method": method.upper(), # API Method (GET/POST)
                    "json_pointer_path": json_pointer if json_pointer else "root", # JSON Pointer used
                    "record_number": record_idx + 1, # Specific for API records
                    "load_timestamp": datetime.now().isoformat(), # NEW: Add load timestamp
                }
                # Add general source info if available from the DataSource object
                if source: # NEW: Add info from DataSource object for lineage
                    metadata["source_type_config"] = source.model_dump() # Store entire DataSource config
                    if source.path: metadata["source_path"] = source.path # e.g. for local OpenAPI spec file
                    if source.db_connection: metadata["source_db_connection"] = source.db_connection
                    if source.folder_id: metadata["source_gdoc_folder_id"] = source.folder_id
                    if source.document_ids: metadata["source_gdoc_document_ids"] = source.document_ids

                metadata["source_type"] = source.type if source else "api" # Use DataSource type or default

                doc = Document(
                    page_content=processed_text,
                    metadata={**metadata} # Use the enriched metadata
                )
                docs.append(doc)
            else:
                logger.debug(f"API record {record_idx + 1} from '{url}' had no content after policy application or was empty.")


        logger.info(f"Loaded {len(docs)} processed records from API endpoint: {url} with enriched metadata.")
        return docs

    except ValueError as e:
        logger.error(f"API Loader Validation Error: {e}")
        return []
    except requests.exceptions.Timeout:
        logger.error(f"Request to API {url} timed out after {request_timeout} seconds.")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from API {url}: {e}", exc_info=True)
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from API {url}: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred in the API loader for URL {url}: {e}", exc_info=True)

    return []