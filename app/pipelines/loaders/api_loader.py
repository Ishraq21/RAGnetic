import requests
import json
import jsonpointer
import logging
from urllib.parse import urlparse
from typing import List, Dict, Any
from langchain_core.documents import Document
import asyncio  # NEW: Added import for asynchronous operations

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


async def load(  # MODIFIED: Changed to async def
        url: str,
        method: str = 'GET',
        headers: Dict = None,
        params: Dict = None,
        payload: Dict = None,
        json_pointer: str = None
) -> List[Document]:
    """
    Fetches data from a REST API endpoint using GET or POST and creates a Document for each record,
    with robust URL validation, timeouts, standardized error logging, and asynchronous operations.
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

        # MODIFIED: Run the blocking request in a separate thread
        data = await asyncio.to_thread(
            _make_request_blocking,
            url=url,
            method=method,
            headers=headers,
            params=params,
            payload=payload,
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

        for record in records:
            content = json.dumps(record, indent=2)
            doc = Document(
                page_content=content,
                metadata={"source": url, "source_type": "api"}
            )
            docs.append(doc)

        logger.info(f"Loaded {len(docs)} records from API endpoint: {url}")
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