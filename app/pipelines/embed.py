import os
import logging
from typing import List, Dict, Any
import uuid
import hashlib
import json
import asyncio

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
from app.pipelines.loaders import (
    directory_loader, url_loader, db_loader, api_loader,
    code_repository_loader, gdoc_loader, web_crawler_loader,
    notebook_loader, pdf_loader, docx_loader, csv_loader,
    text_loader, iac_loader, parquet_loader
)

from app.core.config import get_api_key, get_path_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_APP_PATHS = get_path_settings()
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]


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
    This function remains synchronous as it's CPU-bound after documents are loaded.
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


async def load_documents_from_source(source: DataSource, agent_config: AgentConfig, reproducible_ids: bool) -> list[LangChainDocument]:
    """
    Dispatcher function that calls the appropriate loader based on the source type,
    passing the full agent_config for policy application and scaling settings,
    and the reproducible_ids flag.
    """
    source_type = source.type
    logger.info(f"Dispatching to loader for source type: '{source_type}'")
    loaded_docs: List[LangChainDocument] = []

    # Each loader's 'load' function should now accept a 'source' argument if it needs it for metadata enrichment.
    loader_args = {'agent_config': agent_config, 'source': source}

    try:
        if source_type == "local":
            path = source.path
            if not path or not os.path.exists(path):
                logger.warning(f"Local path not found: {path}. Skipping.")
                return []
            if os.path.isfile(path):
                if path.lower().endswith('.txt'):
                    loaded_docs = await text_loader.load(path, **loader_args)
                elif path.lower().endswith('.pdf'):
                    loaded_docs = await pdf_loader.load(path, **loader_args)
                elif path.lower().endswith('.docx'):
                    loaded_docs = await docx_loader.load(path, **loader_args)
                elif path.lower().endswith('.csv'):
                    loaded_docs = await csv_loader.load(path, **loader_args)
                elif path.lower().endswith('.ipynb'):
                    loaded_docs = await notebook_loader.load_notebook(path, **loader_args)
                elif path.lower().endswith(('.tf', '.tfvars', '.yaml', '.yml')):
                    loaded_docs = await iac_loader.load(path, **loader_args)
                else:
                    logger.warning(f"Unsupported local file type: {path}. Skipping.")
            elif os.path.isdir(path):
                loaded_docs = await directory_loader.load(path, **loader_args)
        elif source_type == "code_repository":
            loaded_docs = await code_repository_loader.load(source.path, **loader_args)
        elif source_type == "url":
            loaded_docs = await url_loader.load(source.url, **loader_args)
        elif source_type == "db":
            loaded_docs = await db_loader.load(source.db_connection, **loader_args)
        elif source_type == "gdoc":
            loaded_docs = await gdoc_loader.load(folder_id=source.folder_id, document_ids=source.document_ids,
                                                 file_types=source.file_types, **loader_args)
        elif source_type == "web_crawler":
            loaded_docs = await web_crawler_loader.load(url=source.url, max_depth=source.max_depth, **loader_args)
        elif source_type == "api":
            loaded_docs = await api_loader.load(url=source.url, method=source.method, headers=source.headers,
                                                params=source.params, payload=source.payload,
                                                json_pointer=source.json_pointer, **loader_args)
        elif source_type == "parquet":
            if not source.path:
                logger.warning("Parquet source type requires a 'path'. Skipping.")
                return []
            loaded_docs = await parquet_loader.load(source.path, **loader_args)
        else:
            logger.warning(f"Unknown or unsupported source type: '{source_type}'. Skipping.")
            return []

    except Exception as e:
        logger.error(f"Error loading documents from source {source.type} (path/url: {source.path or source.url}): {e}",
                     exc_info=True)
        return []

    for idx, doc in enumerate(loaded_docs):
        # Ensure 'source' metadata is consistently added/updated from the DataSource object
        doc.metadata['source_type'] = source.type # Consistently set source_type from DataSource
        if source.path:
            doc.metadata['source_path'] = source.path # Detailed path
        if source.url:
            doc.metadata['source_url'] = source.url # Detailed URL
        if source.db_connection:
            doc.metadata['source_db_connection'] = source.db_connection # Detailed DB connection string

        # Original doc ID generation as before
        source_identifier = doc.metadata.get('source', source.path or source.url or source.type)
        if reproducible_ids:
            hash_input = f"{source_identifier}-{idx}-{doc.page_content}"
            doc.metadata['original_doc_id'] = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
        else:
            doc.metadata['original_doc_id'] = str(uuid.uuid4())
    return loaded_docs


async def embed_agent_data(config: AgentConfig) -> bool:
    """
    Main embedding pipeline. Loads, chunks, stores documents asynchronously, and saves a
    raw text copy of chunks for scalable BM25 initialization.
    Returns True if a vector store was created, False otherwise.
    """
    logger.info(f"[EMBED CONFIG] chunking.mode={config.chunking.mode}, vector_store.type={config.vector_store.type}")
    logger.info(f"Starting data embedding process for agent: '{config.name}'")

    loading_tasks = [
        load_documents_from_source(source, config, config.reproducible_ids)
        for source in config.sources or []
    ]

    all_loaded_docs_lists = await asyncio.gather(*loading_tasks, return_exceptions=True)

    all_docs = []
    for loaded_docs_list in all_loaded_docs_lists:
        if isinstance(loaded_docs_list, Exception):
            logger.error(f"A document loading task failed: {loaded_docs_list}", exc_info=True)
            continue
        validated_docs = [doc for doc in loaded_docs_list if doc.page_content and doc.page_content.strip()]
        all_docs.extend(validated_docs)

    if not all_docs:
        logger.info(
            f"No valid documents found for agent '{config.name}' from any specified sources. Skipping vector store creation.")
        return False

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

        # Ensure chunks are not empty before attempting vector store creation
        if not chunks:
            logger.info(f"No chunks generated for agent '{config.name}'. Skipping vector store creation.")
            return False

        vs_config = config.vector_store
        db_type = vs_config.type
        vectorstore_path = os.path.join(_VECTORSTORE_DIR, config.name)

        os.makedirs(vectorstore_path, exist_ok=True, mode=0o750)

        logger.info(f"Creating vector store of type '{db_type}' for agent '{config.name}' at {vectorstore_path}")

        if db_type == 'faiss':
            db = await asyncio.to_thread(FAISS.from_documents, chunks, langchain_embeddings)
            await asyncio.to_thread(db.save_local, vectorstore_path)
        elif db_type == 'chroma':
            await asyncio.to_thread(Chroma.from_documents, chunks, langchain_embeddings,
                                    persist_directory=vectorstore_path)
        elif db_type == 'qdrant':
            await asyncio.to_thread(Qdrant.from_documents, chunks, langchain_embeddings, host=vs_config.qdrant_host,
                                    port=vs_config.qdrant_port,
                                    path=None if vs_config.qdrant_host else vectorstore_path,
                                    collection_name=config.name)
        elif db_type == 'pinecone':
            pc_api_key = get_api_key("pinecone")
            PineconeClient(api_key=pc_api_key)
            await asyncio.to_thread(PineconeLangChain.from_documents, chunks, langchain_embeddings,
                                    index_name=vs_config.pinecone_index_name)
        elif db_type == 'mongodb_atlas':
            conn_string = get_api_key("mongodb")
            for chunk in chunks:
                chunk.metadata['_id'] = chunk.id
            await asyncio.to_thread(MongoDBAtlasVectorSearch.from_documents, documents=chunks,
                                    embedding=langchain_embeddings,
                                    connection_string=conn_string,
                                    namespace=f"{vs_config.mongodb_db_name}.{vs_config.mongodb_collection_name}",
                                    index_name=vs_config.mongodb_index_name)
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")

        logger.info(f"Successfully created and populated vector store for agent '{config.name}'.")

        bm25_docs_path = os.path.join(_VECTORSTORE_DIR, config.name, "bm25_documents.jsonl")

        def _write_bm25_chunks_to_file():
            with open(bm25_docs_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps({"id": chunk.id, "page_content": chunk.page_content}) + "\n")

        await asyncio.to_thread(_write_bm25_chunks_to_file)

        logger.info(f"Saved {len(chunks)} chunks for scalable BM25 retrieval at {bm25_docs_path}")
        logger.info(f"Successfully created vector store and BM25 source for agent '{config.name}'.")
        return True

    except Exception as e:
        logger.error(f"Failed to create vector store for agent '{config.name}'. Error: {e}", exc_info=True)
        raise VectorStoreCreationError(f"Vector store creation failed for agent '{config.name}'.") from e