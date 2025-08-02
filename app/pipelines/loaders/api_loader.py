import requests
import json
import jsonpointer
import logging
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import asyncio
from datetime import datetime
import hashlib

from app.schemas.agent import AgentConfig, DataSource
from app.pipelines.data_policy_utils import apply_data_policies
from app.pipelines.metadata_utils import generate_base_metadata

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


async def load(
        url: str,
        method: str = 'GET',
        headers: Dict = None,
        params: Dict = None,
        payload: Dict = None,
        json_pointer: str = None,
        agent_config: Optional[AgentConfig] = None,
        source: Optional[DataSource] = None
) -> List[Document]:
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

        for record_idx, record in enumerate(records):
            content = json.dumps(record, indent=2)

            processed_text = content
            document_blocked = False

            if agent_config and agent_config.data_policies:
                logger.debug(f"Applying data policies to API record {record_idx + 1} from {url}...")
                processed_text, document_blocked = apply_data_policies(content, agent_config.data_policies, policy_context="api record")

            if document_blocked:
                logger.warning(f"API record {record_idx + 1} from '{url}' was completely blocked by a data policy and will not be processed.")
                continue

            if processed_text.strip():
                source_context = url
                metadata = generate_base_metadata(source, source_context=source_context, source_type="api")
                # Add API-specific keys
                metadata["source_url"] = url
                metadata["api_method"] = method.upper()
                metadata["json_pointer_path"] = json_pointer if json_pointer else "root"
                metadata["record_number"] = record_idx + 1

                source_ctx = url  # all records from this endpoint share the same group-name
                chunk_idx = record_idx  # 0-based record number

                identity_meta = {
                    "doc_name": source_ctx,
                    "source_name": source_ctx,  # used by embed.py _generate_chunk_id
                    "chunk_index": chunk_idx,
                }

                raw_id = f"{source_ctx}:{chunk_idx}"
                full_hash = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
                short_hash = full_hash[:8]  # reproducible but concise

                identity_meta["original_doc_id"] = full_hash
                metadata.update(identity_meta)
                # -------------------------------------------------

                doc = Document(
                    page_content=processed_text,
                    metadata=metadata,
                    id=short_hash  # stable chunk ID
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