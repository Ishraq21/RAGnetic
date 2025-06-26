import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.schemas.agent import AgentConfig, DataSource
# Correctly import all the specific loaders
from app.pipelines.loaders import (
    directory_loader,
    url_loader,
    db_loader,
    api_loader,
    code_loader,
    gdoc_loader,
    web_crawler_loader,
    notebook_loader,
    pdf_loader,
    docx_loader,
    csv_loader
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents_from_source(source: DataSource) -> list[Document]:
    """
    Dispatcher function that calls the appropriate loader based on source type.
    """
    source_type = source.type

    if source_type == "local":
        path = source.path
        if not path or not os.path.exists(path):
            logger.warning(f"Local path not found: {path}. Skipping.")
            return []
        if os.path.isfile(path):
            if path.lower().endswith('.pdf'):
                return pdf_loader.load(path)
            elif path.lower().endswith('.docx'):
                return docx_loader.load(path)
            elif path.lower().endswith('.csv'):
                return csv_loader.load(path)
            elif path.lower().endswith('.ipynb'):  # Support for single notebook files
                return notebook_loader.load_notebook(path)
            else:
                logger.warning(f"Unsupported local file type: {path}. Skipping.")
                return []
        elif os.path.isdir(path):
            return directory_loader.load(path)
        else:
            return []
    elif source_type == "url":
        return url_loader.load(source.url)
    elif source_type == "code_repository":
        return code_loader.load(source.path)
    elif source_type == "db":
        return db_loader.load(source.db_connection)
    elif source_type == "gdoc":
        return gdoc_loader.load(folder_id=source.folder_id, document_ids=source.document_ids,
                                file_types=source.file_types)
    elif source_type == "web_crawler":
        return web_crawler_loader.load(url=source.url, max_depth=source.max_depth)
    elif source_type == "api":
        return api_loader.load(url=source.url, method=source.method, headers=source.headers,
                               params=source.params, payload=source.payload,
                               json_pointer=source.json_pointer)
    # The 'notebook' type can be handled under 'local' if it's a single file.
    # If it's meant for a directory of notebooks, that logic would be added here.
    else:
        logger.warning(f"Unknown or unsupported source type: {source_type}. Skipping.")
        return []


def embed_agent_data(config: AgentConfig, openai_api_key: str = None):
    """
    The main embedding pipeline for a RAGnetic agent.

    This function orchestrates the entire data processing workflow:
    1.  It iterates through all data sources defined in the agent's config.
    2.  It invokes the appropriate data loader based on the source type.
    3.  It collects all loaded documents into a single list.
    4.  It splits the documents into manageable chunks for embedding.
    5.  It generates embeddings for each chunk using the specified model.
    6.  It creates a FAISS vector store from the embeddings and saves it to disk.
    """
    all_docs = []
    logger.info(f"Starting data embedding process for agent: '{config.name}'")
    for source in config.sources:
        # Pass the Pydantic model directly
        loaded_docs = load_documents_from_source(source)
        if loaded_docs:
            all_docs.extend(loaded_docs)

    if not all_docs:
        logger.warning("No valid documents were loaded from any source. Aborting embedding process.")
        # We raise an error to stop the deployment if no data is found.
        raise ValueError("No valid documents found to embed from any source.")

    logger.info(f"Loaded a total of {len(all_docs)} documents from all sources.")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    logger.info(f"Split documents into {len(chunks)} chunks.")

    # Create embeddings and vector store
    try:
        logger.info(
            f"Creating embeddings with model '{config.embedding_model}'. API key provided: {bool(openai_api_key)}")
        embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=openai_api_key
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(f"vectorstore/{config.name}")
        logger.info(f"Saved FAISS vectorstore with {len(chunks)} chunks to vectorstore/{config.name}")
    except Exception as e:
        logger.error(f"Failed to create or save vector store. Error: {e}", exc_info=True)
        # Re-raise the exception to be handled by the calling process (e.g., the CLI)
        raise
