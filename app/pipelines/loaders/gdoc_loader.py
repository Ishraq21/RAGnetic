import logging # Added import
from langchain_googledrive.document_loaders import GoogleDriveLoader
from google.oauth2 import service_account
from typing import List
from langchain_core.documents import Document
import os
import json
import configparser

logger = logging.getLogger(__name__) # Added logger initialization

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CONFIG_FILE = os.path.join(".ragnetic", "config.ini")  # Path to your config file


def load(
        folder_id: str = None,
        document_ids: List[str] = None,
        file_types: List[str] = None
) -> List[Document]:
    """
    Loads documents from Google Drive using credentials stored in the RAGnetic config file.
    Includes robust credential loading and standardized error logging.
    """
    try:
        if not folder_id and not document_ids:
            logger.error("Validation Error: Must provide either a 'folder_id' or a list of 'document_ids' for Google Drive loader.")
            raise ValueError("Must provide either a 'folder_id' or a list of 'document_ids'.")

        # Read credentials directly from the central config file
        config = configparser.ConfigParser()
        if not os.path.exists(CONFIG_FILE):
            logger.critical(f"Google Drive Error: Config file not found at {CONFIG_FILE}. Credentials cannot be loaded.")
            raise FileNotFoundError(f"Config file not found at {CONFIG_FILE}.")

        config.read(CONFIG_FILE)
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
            # Load the credentials from the string stored in the config
            creds_info = json.loads(creds_json_str)
            credentials = service_account.Credentials.from_service_account_info(
                creds_info, scopes=SCOPES
            )
            logger.info("Google Drive credentials loaded successfully from config.ini.")
        except json.JSONDecodeError as json_e:
            logger.critical(f"Google Drive Error: Invalid JSON format for credentials in config.ini: {json_e}", exc_info=True)
            raise ValueError("Invalid JSON format for Google Drive credentials.") from json_e
        except Exception as cred_e:
            logger.critical(f"Google Drive Error: Failed to parse or load service account credentials: {cred_e}", exc_info=True)
            raise RuntimeError("Failed to load Google Drive service account credentials.") from cred_e

        logger.info(f"Initializing GoogleDriveLoader with folder_id: {folder_id}, document_ids: {document_ids}, file_types: {file_types}")
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            file_ids=document_ids,
            credentials=credentials,  # Provide credentials directly to the loader
            file_types=file_types,
            recursive=False # Set to True if you want to crawl subfolders
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata['source_type'] = 'gdoc'

        source_info = folder_id or f"[{', '.join(document_ids)}]"
        logger.info(f"Loaded {len(docs)} documents from Google Drive source: {source_info}") # Changed from print()
        return docs

    except ValueError as e: # Catches validation errors we raise
        logger.error(f"Google Drive Loader Validation Error: {e}")
        return []
    except FileNotFoundError as e: # Catches specific FileNotFoundError we might raise
        logger.error(f"Google Drive Loader Configuration Error: {e}")
        return []
    except Exception as e: # Catch any other unexpected errors during the process
        logger.error(f"An unexpected error occurred while loading from Google Drive: {e}", exc_info=True) # Changed from print()
        return []