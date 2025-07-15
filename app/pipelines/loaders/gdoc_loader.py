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

from app.schemas.agent import AgentConfig, DataPolicy

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
                        # At this stage (document level), we can't block just a chunk directly.
                        # We'll redact and log a warning for now, indicating this document contains content
                        # that should ideally be split and then blocked at chunking.
                        replacement = kw_config.redaction_placeholder or (kw_config.redaction_char * len(keyword))
                        processed_text = processed_text.replace(keyword, replacement)
                        logger.warning(f"Keyword '{keyword}' found. This document contains content that should ideally be blocked at chunk level. Currently redacting.")
                    elif kw_config.action == 'block_document': # In GDoc loader, 'document' refers to the entire GDoc
                        logger.warning(f"Keyword '{keyword}' found. Document is marked for blocking by policy. Content will be discarded.")
                        document_blocked = True
                        return "", document_blocked # Return empty content and blocked flag

    return processed_text, document_blocked


async def load(
        folder_id: str = None,
        document_ids: List[str] = None,
        file_types: List[str] = None,
        agent_config: Optional[AgentConfig] = None # NEW: Added agent_config
) -> List[Document]:
    """
    Loads documents from Google Drive using credentials stored in the RAGnetic config file.
    Includes robust credential loading, data policy application, standardized error logging, and asynchronous operations.
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
            credentials=credentials,  # Provide credentials directly to the loader
            file_types=file_types,
            recursive=False # Set to True if you want to recursively load from subfolders
        )

        docs = await asyncio.to_thread(_load_gdrive_docs_blocking, loader)

        for doc in docs:
            doc.metadata['source_type'] = 'gdoc'
            # Ensure metadata has consistent file_name and file_path (using id or title if available)
            if 'id' in doc.metadata:
                doc.metadata['file_id'] = doc.metadata['id']
            if 'title' in doc.metadata:
                doc.metadata['file_name'] = doc.metadata['title']
                doc.metadata['file_path'] = f"gdrive://{doc.metadata['id']}" # Conceptual path for GDrive

            processed_text = doc.page_content
            document_blocked = False

            # Apply data policies if provided in agent_config
            if agent_config and agent_config.data_policies:
                logger.debug(f"Applying data policies to GDoc '{doc.metadata.get('title', doc.metadata.get('id', 'unknown'))}'...")
                processed_text, document_blocked = _apply_data_policies(doc.page_content, agent_config.data_policies)

            if document_blocked:
                logger.warning(f"GDoc '{doc.metadata.get('title', doc.metadata.get('id', 'unknown'))}' was completely blocked by a data policy and will not be processed.")
                continue # Skip this document if it's blocked

            if processed_text.strip(): # Ensure there's actual content left after policies
                doc.page_content = processed_text # Update the document with processed text
                processed_docs.append(doc)
            else:
                logger.debug(f"GDoc '{doc.metadata.get('title', doc.metadata.get('id', 'unknown'))}' had no content after policy application or was empty.")


        source_info = folder_id or f"[{', '.join(document_ids)}]"
        logger.info(f"Loaded {len(processed_docs)} processed documents from Google Drive source: {source_info}")
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