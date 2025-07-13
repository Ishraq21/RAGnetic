import os
import logging
from typing import List, Dict, Any
import uuid
import hashlib  # Now used directly for sha256
import json

# LangChain and LlamaIndex have different Document objects
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding

from app.core.config import get_api_key
from langchain_community.vectorstores import FAISS, Chroma
from langchain_qdrant import Qdrant
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_mongodb import MongoDBAtlasVectorSearch
from pinecone import Pinecone as PineconeClient

from app.schemas.agent import AgentConfig, DataSource
from app.core.embed_config import get_embedding_model
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


def _generate_unique_id(content: str, use_hash: bool) -> str:
    """
    Generates a unique ID for a document or chunk.
    If use_hash is True, generates a hash based on content (for reproducibility).
    Otherwise, generates a random UUID.
    """
    if use_hash:
        # EXACTLY the same SHA-256 of the chunk text every time
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    else:
        return str(uuid.uuid4())


def _get_chunks_from_documents(
        documents: List[LangChainDocument],
        chunking_config: 'app.schemas.agent.ChunkingConfig',  # Use string literal for forward reference
        embedding_model_name: str,
        reproducible_ids: bool,
) -> List[LangChainDocument]:
    """
    Helper function to apply chunking strategy (semantic or default) to documents
    and ensure unique, reproducible IDs for chunks.
    """
    chunks: List[LangChainDocument] = []
    chunking_mode_effective = chunking_config.mode

    if chunking_config.mode == 'semantic':
        logger.info("Attempting semantic chunking using LlamaIndex...")
        try:
            langchain_embeddings = get_embedding_model(embedding_model_name)
            llama_embeddings = LangchainEmbedding(langchain_embeddings)

            threshold = chunking_config.breakpoint_percentile_threshold
            semantic_splitter = SemanticSplitterNodeParser.from_defaults(
                embed_model=llama_embeddings,
                breakpoint_percentile_threshold=threshold,
            )

            llama_docs = []
            for doc in documents:
                llama_doc = LlamaDocument(
                    text=doc.page_content,
                    metadata={**doc.metadata}  # Preserve all original metadata
                )
                llama_docs.append(llama_doc)

            nodes = semantic_splitter.get_nodes_from_documents(llama_docs)

            for idx, node in enumerate(nodes):
                text = node.get_content()
                if reproducible_ids:
                    salt = f"{idx}-{text}"  # Salt with index for uniqueness
                    generated_chunk_id = hashlib.sha256(salt.encode("utf-8")).hexdigest()
                else:
                    generated_chunk_id = node.id_  # Use LlamaIndex's node ID

                metadata = {**node.metadata, "chunk_id": generated_chunk_id}
                new_doc = LangChainDocument(page_content=text, metadata=metadata)
                new_doc.id = generated_chunk_id  # Set Document.id to chunk_id
                chunks.append(new_doc)

            logger.info(f"Successfully applied semantic chunking. Resulted in {len(chunks)} chunks.")

        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}. Falling back to default recursive character splitting.",
                         exc_info=True)
            chunking_mode_effective = 'default'  # Force fallback

    if chunking_mode_effective == 'default':  # Default recursive character chunking strategy (also fallback)
        logger.info(
            f"Applying default recursive character chunking with size={chunking_config.chunk_size} and overlap={chunking_config.chunk_overlap}.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.chunk_size,
            chunk_overlap=chunking_config.chunk_overlap
        )
        for idx, chunk in enumerate(splitter.split_documents(documents)):
            if reproducible_ids:
                salt = f"{idx}-{chunk.page_content}"  # Salt with index for uniqueness
                generated_chunk_id = hashlib.sha256(salt.encode("utf-8")).hexdigest()
            else:
                generated_chunk_id = str(uuid.uuid4())  # Generate random UUID

            chunk.metadata["chunk_id"] = generated_chunk_id
            chunk.metadata["original_doc_id"] = chunk.metadata.get(
                "original_doc_id")  # Ensure original_doc_id is retained
            chunk.id = generated_chunk_id  # Set Document.id to chunk_id

            chunks.append(chunk)
        logger.info(f"Default chunking resulted in {len(chunks)} chunks.")

    if not chunks:
        raise ValueError("No chunks were generated after document splitting.")

    return chunks


def load_documents_from_source(source: DataSource, reproducible_ids: bool = False) -> list[LangChainDocument]:
    """
    Dispatcher function that calls the appropriate loader based on the source type.
    Assigns a unique 'original_doc_id' to each loaded document.
    """
    source_type = source.type
    logger.info(f"Dispatching to loader for source type: '{source_type}'")

    loaded_docs: List[LangChainDocument] = []

    # --- Existing loading logic ---
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
            elif path.lower().endswith(('.tf', '.tfvars')):
                loaded_docs = iac_loader.load(path)
            elif path.lower().endswith(('.yaml', '.yml')):
                loaded_docs = iac_loader.load(path)
            else:
                logger.warning(f"Unsupported local file type: {path}. Skipping.")
                return []
        elif os.path.isdir(path):
            loaded_docs = directory_loader.load(path)
        else:
            return []
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
                                      params=source.params,
                                      payload=source.payload, json_pointer=source.json_pointer)
    else:
        logger.warning(f"Unknown or unsupported source type: '{source_type}'. Skipping.")
        return []

    # Assign a unique 'original_doc_id' to each loaded document
    for idx, doc in enumerate(loaded_docs):
        # Ensure 'source' is always present for consistent ID generation or debugging
        if 'source' not in doc.metadata:
            doc.metadata['source'] = source.path or source.url or source.type  # Fallback source
        # Verify page_content is not empty for consistent ID generation
        if not doc.page_content or not doc.page_content.strip():
            logger.warning(
                f"Document has empty page_content from source: {doc.metadata.get('source', 'N/A')}. Generating ID based on limited info.")

        # Generate original_doc_id: salt with index + hash content
        if reproducible_ids:
            # Salt with index to guarantee uniqueness for original documents too
            hash_input = f"{idx}-{doc.page_content}"
            doc.metadata['original_doc_id'] = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
        else:
            doc.metadata['original_doc_id'] = str(uuid.uuid4())

    return loaded_docs


def embed_agent_data(config: AgentConfig):
    """
    The main embedding pipeline for a RAGnetic agent with enhanced error handling
    and configurable chunking strategies. Ensures unique IDs for chunks.
    Crucially, sets `Document.id` to `chunk_id`.
    """

    logger.info(
        f"[EMBED CONFIG] chunking.mode={config.chunking.mode}, "
        f"chunk_size={config.chunking.chunk_size}, "
        f"chunk_overlap={config.chunking.chunk_overlap}, "
        f"breakpoint_percentile_threshold={config.chunking.breakpoint_percentile_threshold}; "
        f"vector_store.type={config.vector_store.type}, "
        f"retrieval_strategy={config.vector_store.retrieval_strategy}, "
        f"bm25_k={config.vector_store.bm25_k}, "
        f"semantic_k={config.vector_store.semantic_k}, "
        f"rerank_top_n={config.vector_store.rerank_top_n}"
    )

    all_docs = []
    logger.info(f"Starting data embedding process for agent: '{config.name}'")
    # Log to confirm reproducible_ids flag
    logger.info(f"Reproducible IDs enabled for embedding: {config.reproducible_ids}")

    for source in config.sources:
        # Pass reproducible_ids to document loader
        loaded_docs = load_documents_from_source(source, config.reproducible_ids)
        if not loaded_docs:
            continue

        validated_docs = []
        for doc in loaded_docs:
            if doc.page_content and doc.page_content.strip():
                validated_docs.append(doc)
            else:
                logger.warning(f"Skipping empty document from source: {doc.metadata.get('source', 'N/A')}")

        if validated_docs:
            all_docs.extend(validated_docs)

    if not all_docs:
        raise ValueError("No valid documents with content found to embed from any source.")

    # **NEW**: make the document order deterministic for chunking consistency
    all_docs.sort(key=lambda d: d.metadata["original_doc_id"])

    logger.info(f"Loaded a total of {len(all_docs)} valid documents from all sources.")

    try:
        langchain_embeddings = get_embedding_model(config.embedding_model)
        logger.info(f"Using chunking mode: '{config.chunking.mode}'")

        # Use the new shared chunking helper
        chunks = _get_chunks_from_documents(
            documents=all_docs,
            chunking_config=config.chunking,
            embedding_model_name=config.embedding_model,
            reproducible_ids=config.reproducible_ids,
        )

        if not chunks:
            raise ValueError("No chunks were generated after document splitting.")

        vs_config = config.vector_store
        db_type = vs_config.type
        vectorstore_path = f"vectorstore/{config.name}"
        # id_key_for_vs is no longer needed as Document.id will be used by default by LangChain VS implementations.

        logger.info(f"Creating vector store of type '{db_type}' for agent '{config.name}'.")

        # Vector store creation relies on Document.id being correctly set in 'chunks' list
        if db_type == 'faiss':
            # FAISS.from_documents uses Document.id by default.
            db = FAISS.from_documents(chunks, langchain_embeddings)
            db.save_local(vectorstore_path)
        elif db_type == 'chroma':
            # Chroma.from_documents also uses Document.id by default.
            Chroma.from_documents(chunks, langchain_embeddings, persist_directory=vectorstore_path)
        elif db_type == 'qdrant':
            # Qdrant.from_documents uses Document.id by default.
            Qdrant.from_documents(chunks, langchain_embeddings, host=vs_config.qdrant_host, port=vs_config.qdrant_port,
                                  path=None if vs_config.qdrant_host else vectorstore_path, collection_name=config.name)
        elif db_type == 'pinecone':
            pc_api_key = get_api_key("pinecone")
            PineconeClient(api_key=pc_api_key)
            # PineconeLangChain.from_documents uses Document.id by default.
            PineconeLangChain.from_documents(chunks, langchain_embeddings, index_name=vs_config.pinecone_index_name)
        elif db_type == 'mongodb_atlas':
            conn_string = get_api_key("mongodb")
            # MongoDBAtlasVectorSearch.from_documents relies on Document.metadata['_id'] or generates one.
            # We explicitly set `chunk.metadata['_id'] = chunk.id` to ensure consistency.
            for chunk in chunks:
                if '_id' not in chunk.metadata:  # Ensure _id is set if not already present
                    chunk.metadata['_id'] = chunk.id
            MongoDBAtlasVectorSearch.from_documents(documents=chunks, embedding=langchain_embeddings,
                                                    connection_string=conn_string,
                                                    namespace=f"{vs_config.mongodb_db_name}.{vs_config.mongodb_collection_name}",
                                                    index_name=vs_config.mongodb_index_name)
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")

        logger.info(f"Successfully created and populated vector store for agent '{config.name}'.")

    except Exception as e:
        error_details = (
            f"Agent: '{config.name}', "
            f"Vector Store Type: '{config.vector_store.type}', "
            f"Embedding Model: '{config.embedding_model}'"
        )
        logger.error(
            f"Failed to create vector store with details: {error_details}. Error: {e}",
            exc_info=True
        )
        raise VectorStoreCreationError(f"Vector store creation failed for agent '{config.name}'.") from e