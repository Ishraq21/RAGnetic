# app/pipelines/gdoc_loader.py

import logging
from langchain_googledrive.document_loaders import GoogleDriveLoader
from google.oauth2 import service_account
from typing import List, Optional
from langchain_core.documents import Document
import os
import json
import configparser
import asyncio
from datetime import datetime

from app.schemas.agent import AgentConfig, DataSource
from app.pipelines.data_policy_utils import apply_data_policies
from app.pipelines.metadata_utils import generate_base_metadata

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CONFIG_FILE = os.path.join(".ragnetic", "config.ini")


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


async def load(
        folder_id: str = None,
        document_ids: List[str] = None,
        file_types: List[str] = None,
        agent_config: Optional[AgentConfig] = None,
        source: Optional[DataSource] = None
) -> List[Document]:
    processed_docs = []
    try:
        if not folder_id and not document_ids:
            logger.error(
                "Validation Error: Must provide either a 'folder_id' or a list of 'document_ids' for Google Drive loader.")
            raise ValueError("Must provide either a 'folder_id' or a list of 'document_ids'.")

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

        docs_raw = await asyncio.to_thread(_load_gdrive_docs_blocking, loader)

        for doc in docs_raw:
            processed_text = doc.page_content
            document_blocked = False

            if agent_config and agent_config.data_policies:
                logger.debug(f"Applying data policies to GDoc '{doc.metadata.get('title', doc.metadata.get('id', 'unknown'))}'...")
                processed_text, document_blocked = apply_data_policies(doc.page_content, agent_config.data_policies, policy_context="gdoc")

            if document_blocked:
                logger.warning(f"GDoc '{doc.metadata.get('title', doc.metadata.get('id', 'unknown'))}' was completely blocked by a data policy and will not be processed.")
                continue

            if processed_text.strip():
                source_context = doc.metadata.get("title") or doc.metadata.get("id", "unknown")

                metadata = generate_base_metadata(
                    source,
                    source_context=source_context,
                    source_type="gdoc",
                )
                metadata.update(
                    {
                        "source_gdoc_id": doc.metadata.get("id"),
                        "source_gdoc_title": doc.metadata.get("title"),
                        "file_type": doc.metadata.get("mimeType"),
                        # ---- finished-chunk identity keys ----
                        "doc_name": source_context,
                        "source_name": source_context,
                        "chunk_index": 0,
                    }
                )

                # reproducible, stable ID (good default = Google file id)
                doc_id = doc.metadata.get("id", source_context)
                metadata.setdefault("original_doc_id", doc_id)

                doc.id = doc_id
                doc.page_content = processed_text
                doc.metadata = {**doc.metadata, **metadata}
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