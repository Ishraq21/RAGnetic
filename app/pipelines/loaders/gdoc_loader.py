from langchain_googledrive.document_loaders import GoogleDriveLoader
from google.oauth2 import service_account
from typing import List
from langchain_core.documents import Document
import os
import json
import configparser

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CONFIG_FILE = os.path.join(".ragnetic", "config.ini")  # Path to your config file


def load(
        folder_id: str = None,
        document_ids: List[str] = None,
        file_types: List[str] = None
) -> List[Document]:
    """
    Loads documents from Google Drive using credentials stored in the RAGnetic config file.
    """
    if not folder_id and not document_ids:
        raise ValueError("Must provide either a 'folder_id' or a list of 'document_ids'.")

    # Read credentials directly from the central config file
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    creds_json_str = config.get('GOOGLE_CREDENTIALS', 'json_key', fallback=None)

    if not creds_json_str:
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

        loader = GoogleDriveLoader(
            folder_id=folder_id,
            file_ids=document_ids,
            credentials=credentials,  # Provide credentials directly to the loader
            file_types=file_types,
            recursive=False
        )

        docs = loader.load()

        for doc in docs:
            doc.metadata['source_type'] = 'gdoc'

        source_info = folder_id or f"[{', '.join(document_ids)}]"
        print(f"Loaded {len(docs)} documents from Google Drive source: {source_info}")
        return docs

    except Exception as e:
        print(f"An error occurred while loading from Google Drive: {e}")
        return []