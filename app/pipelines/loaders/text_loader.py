import logging
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load(file_path: str) -> List[Document]:
    """
    Loads a plain text file.
    """
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        logger.info(f"Successfully loaded text file: {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load text file {file_path}: {e}")
        return []