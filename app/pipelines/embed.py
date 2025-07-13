import os
import logging
from typing import List, Dict, Any
import uuid
import hashlib
import json

# LangChain and LlamaIndex have different Document objects
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding

# LangChain vector store components
from langchain_community.vectorstores import FAISS, Chroma
from langchain_qdrant import Qdrant
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_mongodb import MongoDBAtlasVectorSearch
from pinecone import Pinecone as PineconeClient

# Use forward references for type hints to avoid circular imports
from app.schemas.agent import AgentConfig, DataSource, ChunkingConfig
from app.core.embed_config import get_embedding_model
from app.core.config import get_api_key
# Import all available loaders
from app.pipelines.loaders import (
    directory_loader, url_loader, db_loader, api_loader,
    code_repository_loader, gdoc_loader, web_crawler_loader,
    notebook_loader, pdf_loader, docx_loader, csv_loader,
    text_loader, iac_loader
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreCreationError(Exception):
    """Custom exception for errors during vector store creation."""
    pass


def _get_chunks_from_documents(
        documents: List[LangChainDocument],
        chunking_config: ChunkingConfig,
        embedding_model_name: str,
        reproducible_ids: bool,
) -> List[LangChainDocument]:
    """
    Helper function to apply a chunking strategy to documents and ensure unique,
    reproducible IDs for chunks. Supports 'none', 'semantic', and 'default' modes.
    """
    chunks: List[LangChainDocument] = []
    chunking_mode = chunking_config.mode
    logger.info(f"Applying chunking mode: '{chunking_mode}'")

    # --- Handle 'none' chunking mode ---
    if chunking_mode == 'none':
        logger.info("Chunking mode is 'none'. Using documents directly as chunks.")
        chunks = documents
        for idx, chunk in enumerate(chunks):
            original_doc_id = chunk.metadata.get("original_doc_id", "")
            if reproducible_ids:
                content_hash = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
                chunk.id = f"{original_doc_id}-{idx}-{content_hash[:8]}"
            else:
                chunk.id = str(uuid.uuid4())
            chunk.metadata["chunk_id"] = chunk.id
        return chunks

    # --- Handle 'semantic' chunking mode ---
    elif chunking_mode == 'semantic':
        logger.info("Attempting semantic chunking using LlamaIndex...")
        try:
            langchain_embeddings = get_embedding_model(embedding_model_name)
            llama_embeddings = LangchainEmbedding(langchain_embeddings)

            semantic_splitter = SemanticSplitterNodeParser.from_defaults(
                embed_model=llama_embeddings,
                breakpoint_percentile_threshold=chunking_config.breakpoint_percentile_threshold,
            )
            llama_docs = [LlamaDocument(text=doc.page_content, metadata=doc.metadata) for doc in documents]
            nodes = semantic_splitter.get_nodes_from_documents(llama_docs)

            for node in nodes:
                original_doc_id = node.metadata.get("original_doc_id", "")
                content = node.get_content()
                if reproducible_ids:
                    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    generated_chunk_id = f"{original_doc_id}-{content_hash[:8]}"
                else:
                    generated_chunk_id = str(uuid.uuid4())

                metadata = {**node.metadata, "chunk_id": generated_chunk_id}
                new_doc = LangChainDocument(page_content=content, metadata=metadata)
                new_doc.id = generated_chunk_id
                chunks.append(new_doc)

            logger.info(f"Successfully applied semantic chunking. Resulted in {len(chunks)} chunks.")

        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}. Falling back to default recursive splitting.", exc_info=True)
            chunking_mode = 'default'

    # --- Handle 'default' (Recursive) chunking mode ---
    elif chunking_mode == 'default':
        logger.info(
            f"Applying default recursive character chunking with size={chunking_config.chunk_size} and overlap={chunking_config.chunk_overlap}.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.chunk_size,
            chunk_overlap=chunking_config.chunk_overlap
        )
        split_chunks = splitter.split_documents(documents)
        for idx, chunk in enumerate(split_chunks):
            original_doc_id = chunk.metadata.get("original_doc_id", "")
            if reproducible_ids:
                content_hash = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
                generated_chunk_id = f"{original_doc_id}-{idx}-{content_hash[:8]}"
            else:
                generated_chunk_id = str(uuid.uuid4())

            chunk.id = generated_chunk_id
            chunk.metadata["chunk_id"] = generated_chunk_id
            chunks.append(chunk)
        logger.info(f"Default chunking resulted in {len(chunks)} chunks.")

    if not chunks:
        raise ValueError("No chunks were generated after document splitting.")

    return chunks


def load_documents_from_source(source: DataSource, reproducible_ids: bool = False) -> list[LangChainDocument]:
    """
    Dispatcher function that calls the appropriate loader based on the source type.
    Assigns a unique 'original_doc_id' to each loaded document for traceability.
    """
    source_type = source.type
    logger.info(f"Dispatching to loader for source type: '{source_type}'")

    loaded_docs: List[LangChainDocument] = []

    # --- Comprehensive loading logic for all supported types ---
    if source_type == "local":
        path = source.path
        if not path or not os.path.exists(path):
            logger.warning(f"Local path not found: {path}. Skipping.")
            return []
        if os.path.isfile(path):
            if path.lower().endswith('.txt'):
                loaded_docs = text_loader.load(path)
            elif path.lower().endswith('.pdf'):
                loaded_docs = pdf_loader.load(path)
            elif path.lower().endswith('.docx'):
                loaded_docs = docx_loader.load(path)
            elif path.lower().endswith('.csv'):
                loaded_docs = csv_loader.load(path)
            elif path.lower().endswith('.ipynb'):
                loaded_docs = notebook_loader.load_notebook(path)
            elif path.lower().endswith(('.tf', '.tfvars', '.yaml', '.yml')):
                loaded_docs = iac_loader.load(path)
            else:
                logger.warning(f"Unsupported local file type: {path}. Skipping.")
        elif os.path.isdir(path):
            loaded_docs = directory_loader.load(path)
    elif source_type == "code_repository":
        loaded_docs = code_repository_loader.load(source.path)
    elif source_type == "url":
        loaded_docs = url_loader.load(source.url)
    elif source_type == "db":
        loaded_docs = db_loader.load(source.db_connection)
    elif source_type == "gdoc":
        loaded_docs = gdoc_loader.load(folder_id=source.folder_id, document_ids=source.document_ids,
                                       file_types=source.file_types)
    elif source_type == "web_crawler":
        loaded_docs = web_crawler_loader.load(url=source.url, max_depth=source.max_depth)
    elif source_type == "api":
        loaded_docs = api_loader.load(url=source.url, method=source.method, headers=source.headers,
                                      params=source.params, payload=source.payload, json_pointer=source.json_pointer)
    else:
        logger.warning(f"Unknown or unsupported source type: '{source_type}'. Skipping.")
        return []

    # Assign a unique and deterministic 'original_doc_id' to each loaded document.
    for idx, doc in enumerate(loaded_docs):
        source_identifier = doc.metadata.get('source', source.path or source.url or source.type)
        if reproducible_ids:
            # Create a hash based on source, index, and content for a stable ID.
            hash_input = f"{source_identifier}-{idx}-{doc.page_content}"
            doc.metadata['original_doc_id'] = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
        else:
            doc.metadata['original_doc_id'] = str(uuid.uuid4())

    return loaded_docs


def embed_agent_data(config: AgentConfig):
    """
    The main embedding pipeline for a RAGnetic agent. This function orchestrates
    loading, chunking, and storing documents in a vector database.
    """
    logger.info(f"[EMBED CONFIG] chunking.mode={config.chunking.mode}, vector_store.type={config.vector_store.type}")
    logger.info(f"Starting data embedding process for agent: '{config.name}'")
    logger.info(f"Reproducible IDs enabled for embedding: {config.reproducible_ids}")

    all_docs = []
    for source in config.sources:
        loaded_docs = load_documents_from_source(source, config.reproducible_ids)
        # Filter out any documents that might be empty after loading
        validated_docs = [doc for doc in loaded_docs if doc.page_content and doc.page_content.strip()]
        all_docs.extend(validated_docs)

    if not all_docs:
        raise ValueError("No valid documents with content were found to embed from any of the specified sources.")

    # Sort documents by their unique ID to ensure the chunking order is always the same.
    all_docs.sort(key=lambda d: d.metadata.get("original_doc_id", ""))
    logger.info(f"Loaded a total of {len(all_docs)} valid documents from all sources.")

    try:
        langchain_embeddings = get_embedding_model(config.embedding_model)

        chunks = _get_chunks_from_documents(
            documents=all_docs,
            chunking_config=config.chunking,
            embedding_model_name=config.embedding_model,
            reproducible_ids=config.reproducible_ids,
        )

        vs_config = config.vector_store
        db_type = vs_config.type
        vectorstore_path = f"vectorstore/{config.name}"
        logger.info(f"Creating vector store of type '{db_type}' for agent '{config.name}'.")

        # --- Vector Store Creation Dispatcher ---
        if db_type == 'faiss':
            db = FAISS.from_documents(chunks, langchain_embeddings)
            db.save_local(vectorstore_path)
        elif db_type == 'chroma':
            Chroma.from_documents(chunks, langchain_embeddings, persist_directory=vectorstore_path)
        elif db_type == 'qdrant':
            Qdrant.from_documents(chunks, langchain_embeddings, host=vs_config.qdrant_host, port=vs_config.qdrant_port,
                                  path=None if vs_config.qdrant_host else vectorstore_path, collection_name=config.name)
        elif db_type == 'pinecone':
            pc_api_key = get_api_key("pinecone")
            PineconeClient(api_key=pc_api_key)
            PineconeLangChain.from_documents(chunks, langchain_embeddings, index_name=vs_config.pinecone_index_name)
        elif db_type == 'mongodb_atlas':
            conn_string = get_api_key("mongodb")
            for chunk in chunks:
                chunk.metadata['_id'] = chunk.id
            MongoDBAtlasVectorSearch.from_documents(documents=chunks, embedding=langchain_embeddings,
                                                    connection_string=conn_string,
                                                    namespace=f"{vs_config.mongodb_db_name}.{vs_config.mongodb_collection_name}",
                                                    index_name=vs_config.mongodb_index_name)
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")

        logger.info(f"Successfully created and populated vector store for agent '{config.name}'.")

    except Exception as e:
        logger.error(f"Failed to create vector store for agent '{config.name}'. Error: {e}", exc_info=True)
        raise VectorStoreCreationError(f"Vector store creation failed for agent '{config.name}'.") from e
