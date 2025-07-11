import os
import logging
from typing import List
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def load(file_path: str) -> List[Document]:
    """
    Loads Infrastructure-as-Code (IaC) files like Terraform and Kubernetes YAML,
    then splits them into syntax-aware chunks.
    """
    try:
        file_extension = Path(file_path).suffix
        splitter = None

        if file_extension in [".tf", ".tfvars"]:
            logger.info(f"Loading Terraform file: {file_path}")
            # Separators for HCL/Terraform code
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""],
            )
        elif file_extension in [".yaml", ".yml"]:
            logger.info(f"Loading YAML file: {file_path}")
            # Manually define separators that work well for YAML
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n---\n", "\n\n", "\n", " ", ""],
            )
        else:
            logger.warning(f"Unsupported IaC file type: {file_path}. Skipping.")
            return []

        # Load the raw text content of the file
        loader = TextLoader(file_path, encoding="utf-8")
        raw_docs = loader.load()

        # Use the appropriate splitter
        chunks = splitter.split_documents(raw_docs)

        # Add metadata to each chunk
        for chunk in chunks:
            chunk.metadata['source_type'] = 'infrastructure_as_code'
            chunk.metadata['file_name'] = os.path.basename(file_path)

        logger.info(f"Successfully chunked {file_path} into {len(chunks)} IaC chunks.")
        return chunks

    except Exception as e:
        logger.error(f"Failed to load or process IaC file {file_path}: {e}")
        return []