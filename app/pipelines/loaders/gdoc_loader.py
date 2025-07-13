import logging
from langchain_googledrive.document_loaders import GoogleDriveLoader
from google.oauth2 import service_account
from typing import List
from langchain_core.documents import Document
import os
import json
import configparser
import asyncio  # NEW: Added import for asynchronous operations

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


async def load(  # MODIFIED: async def
        folder_id: str = None,
        document_ids: List[str] = None,
        file_types: List[str] = None
) -> List[Document]:
    """
    Loads documents from Google Drive using credentials stored in the RAGnetic config file.
    Includes robust credential loading, standardized error logging, and asynchronous operations.
    """
    try:
        if not folder_id and not document_ids:
            logger.error(
                "Validation Error: Must provide either a 'folder_id' or a list of 'document_ids' for Google Drive loader.")
            raise ValueError("Must provide either a 'folder_id' or a list of 'document_ids'.")

        # Read credentials directly from the central config file
        # MODIFIED: Run config reading in a separate thread
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
            # MODIFIED: Run credential loading in a separate thread
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
            recursive=False
        )

        # MODIFIED: Run loader.load() in a separate thread
        docs = await asyncio.to_thread(_load_gdrive_docs_blocking, loader)

        for doc in docs:
            doc.metadata['source_type'] = 'gdoc'

        source_info = folder_id or f"[{', '.join(document_ids)}]"
        logger.info(f"Loaded {len(docs)} documents from Google Drive source: {source_info}")
        return docs

    except ValueError as e:
        logger.error(f"Google Drive Loader Validation Error: {e}")
        return []
    except FileNotFoundError as e:
        logger.error(f"Google Drive Loader Configuration Error: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading from Google Drive: {e}", exc_info=True)
        return []