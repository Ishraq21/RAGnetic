import logging
import re # Added for PII redaction
from langchain_googledrive.document_loaders import GoogleDriveLoader
from google.oauth2 import service_account
from typing import List, Optional
from langchain_core.documents import Document
import os
import json
import configparser
import asyncio
from datetime import datetime # NEW: Import datetime for load_timestamp


from app.schemas.agent import AgentConfig, DataPolicy, DataSource

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CONFIG_FILE = os.path.join(".ragnetic", "config.ini")  # Path to your config file


# Define synchronous helpers to run blocking operations in a thread
def _read_config_blocking(config_file_path: str):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    return config


def _load_credentials_blocking(creds_json_str: str, scopes: List[str]):
    creds_info = json.loads(creds_json_str)
    return service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)


def _load_gdrive_docs_blocking(loader: GoogleDriveLoader):
    return loader.load()

def _apply_data_policies(text: str, policies: List[DataPolicy]) -> tuple[str, bool]:
    """
    Applies data policies (redaction/filtering) to the text content.
    Returns the processed text and a boolean indicating if the document (GDoc) was blocked.
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
                        logger.warning(f"Keyword '{keyword}' found. Document is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked

    return processed_text, document_blocked


async def load(
        folder_id: str = None,
        document_ids: List[str] = None,
        file_types: List[str] = None,
        agent_config: Optional[AgentConfig] = None,
        source: Optional[DataSource] = None # MODIFIED: Added source parameter
) -> List[Document]:
    """
    Loads documents from Google Drive using credentials stored in the RAGnetic config file.
    Includes robust credential loading, data policy application, standardized error logging,
    asynchronous operations, and enriched metadata for lineage.
    """
    processed_docs = []
    try:
        if not folder_id and not document_ids:
            logger.error(
                "Validation Error: Must provide either a 'folder_id' or a list of 'document_ids' for Google Drive loader.")
            raise ValueError("Must provide either a 'folder_id' or a list of 'document_ids'.")

        # Read credentials directly from the central config file
        if not os.path.exists(CONFIG_FILE):
            logger.critical(
                f"Google Drive Error: Config file not found at {CONFIG_FILE}. Credentials cannot be loaded.")
            raise FileNotFoundError(f"Config file not found at {CONFIG_FILE}.")

        config = await asyncio.to_thread(_read_config_blocking, CONFIG_FILE)
        creds_json_str = config.get('GOOGLE_CREDENTIALS', 'json_key', fallback=None)

        if not creds_json_str:
            logger.critical(
                "Google Drive Error: Credentials not found in .ragnetic/config.ini. "
                "Please run 'ragnetic auth gdrive' to set them up."
            )
            raise ValueError(
                "Google Drive credentials not found. "
                "Please run 'ragnetic auth gdrive' to set them up."
            )

        try:
            credentials = await asyncio.to_thread(_load_credentials_blocking, creds_json_str, SCOPES)
            logger.info("Google Drive credentials loaded successfully from config.ini.")
        except json.JSONDecodeError as json_e:
            logger.critical(f"Google Drive Error: Invalid JSON format for credentials in config.ini: {json_e}",
                            exc_info=True)
            raise ValueError("Invalid JSON format for Google Drive credentials.") from json_e
        except Exception as cred_e:
            logger.critical(f"Google Drive Error: Failed to parse or load service account credentials: {cred_e}",
                            exc_info=True)
            raise RuntimeError("Failed to load Google Drive service account credentials.") from cred_e

        logger.info(
            f"Initializing GoogleDriveLoader with folder_id: {folder_id}, document_ids: {document_ids}, file_types: {file_types}")
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            file_ids=document_ids,
            credentials=credentials,
            file_types=file_types,
            recursive=False
        )

        docs_raw = await asyncio.to_thread(_load_gdrive_docs_blocking, loader) # Renamed to docs_raw to clarify

        for doc in docs_raw: # Iterate over raw documents
            processed_text = doc.page_content
            document_blocked = False

            if agent_config and agent_config.data_policies:
                logger.debug(f"Applying data policies to GDoc '{doc.metadata.get('title', doc.metadata.get('id', 'unknown'))}'...")
                processed_text, document_blocked = _apply_data_policies(doc.page_content, agent_config.data_policies)

            if document_blocked:
                logger.warning(f"GDoc '{doc.metadata.get('title', doc.metadata.get('id', 'unknown'))}' was completely blocked by a data policy and will not be processed.")
                continue

            if processed_text.strip():
                # Create base metadata for the document
                metadata = {
                    "source_gdoc_id": doc.metadata.get('id', 'N/A'), # Specific GDoc ID
                    "source_gdoc_title": doc.metadata.get('title', 'N/A'), # Specific GDoc Title
                    "file_type": doc.metadata.get('mimeType', 'N/A'), # Specific MIME Type
                    "load_timestamp": datetime.now().isoformat(), # NEW: Add load timestamp
                }
                # Add general source info from the DataSource object
                if source: # NEW: Add info from DataSource object for lineage
                    metadata["source_type_config"] = source.model_dump() # Store entire DataSource config
                    if source.url: metadata["source_url"] = source.url # This would be the GDoc URL
                    if source.folder_id: metadata["source_gdoc_folder_id"] = source.folder_id
                    if source.document_ids: metadata["source_gdoc_document_ids"] = source.document_ids

                # Merge with existing metadata from raw_documents (Langchain's GDriveLoader adds some)
                doc.page_content = processed_text # Update the document with processed text
                doc.metadata = {**doc.metadata, **metadata} # Merge, with new metadata taking precedence
                processed_docs.append(doc)
            else:
                logger.debug(f"GDoc '{doc.metadata.get('title', doc.metadata.get('id', 'unknown'))}' had no content after policy application or was empty.")


        source_info = folder_id or f"[{', '.join(document_ids)}]"
        logger.info(f"Loaded {len(processed_docs)} processed documents from Google Drive source: {source_info} with enriched metadata.")
        return processed_docs

    except ValueError as e:
        logger.error(f"Google Drive Loader Validation Error: {e}")
        return []
    except FileNotFoundError as e:
        logger.error(f"Google Drive Loader Configuration Error: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading from Google Drive: {e}", exc_info=True)
        return []