from langchain_googledrive.document_loaders import GoogleDriveLoader
from google.oauth2 import service_account
from typing import List
from langchain_core.documents import Document
import os
import json

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def load(
        folder_id: str = None,
        document_ids: List[str] = None,
        file_types: List[str] = None
) -> List[Document]:
    """
    Loads documents from Google Drive using credentials stored directly
    in the GOOGLE_CREDENTIALS_JSON environment variable.
    """
    if not folder_id and not document_ids:
        raise ValueError("Must provide either a 'folder_id' or a list of 'document_ids'.")

    creds_json_str = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if not creds_json_str:
        raise ValueError(
            "The GOOGLE_CREDENTIALS_JSON environment variable is not set. "
            "Please add the full content of your service account key to the .env file."
        )

    try:
        creds_info = json.loads(creds_json_str)
        credentials = service_account.Credentials.from_service_account_info(
            creds_info, scopes=SCOPES
        )

        # The advanced loader can accept credentials directly.
        # It correctly handles file_types only when folder_id is present.
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            file_ids=document_ids,
            credentials=credentials,
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