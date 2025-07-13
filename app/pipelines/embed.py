import os
import logging
from typing import List, Dict, Any
import uuid
import hashlib
import json
import asyncio  # NEW: Import asyncio

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
# Import all available loaders (these will need to be async later)
from app.pipelines.loaders import (
    directory_loader, url_loader, db_loader, api_loader,
    code_repository_loader, gdoc_loader, web_crawler_loader,  # web_crawler_loader is already async
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
            # Note: get_embedding_model and LangchainEmbedding are synchronous
            # If they involve network calls, they should also be async eventually
            langchain_embeddings = get_embedding_model(embedding_model_name)
            llama_embeddings = LangchainEmbedding(langchain_embeddings)
            semantic_splitter = SemanticSplitterNodeParser.from_defaults(
                embed_model=llama_embeddings,
                breakpoint_percentile_threshold=chunking_config.breakpoint_percentile_threshold,
            )
            llama_docs = [LlamaDocument(text=doc.page_content, metadata=doc.metadata) for doc in documents]
            # get_nodes_from_documents is CPU-bound
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
        # split_documents is CPU-bound
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


async def load_documents_from_source(source: DataSource, reproducible_ids: bool = False) -> list[
    LangChainDocument]:  # MODIFIED: async def
    """
    Dispatcher function that calls the appropriate loader based on the source type.
    This function must await the loader calls if they are async.
    """
    source_type = source.type
    logger.info(f"Dispatching to loader for source type: '{source_type}'")
    loaded_docs: List[LangChainDocument] = []

    try:
        if source_type == "local":
            path = source.path
            if not path or not os.path.exists(path):
                logger.warning(f"Local path not found: {path}. Skipping.")
                return []
            if os.path.isfile(path):
                # These loaders need to be converted to async
                if path.lower().endswith('.txt'):
                    loaded_docs = await text_loader.load(path)  # MODIFIED: await
                elif path.lower().endswith('.pdf'):
                    loaded_docs = await pdf_loader.load(path)  # MODIFIED: await
                elif path.lower().endswith('.docx'):
                    loaded_docs = await docx_loader.load(path)  # MODIFIED: await
                elif path.lower().endswith('.csv'):
                    loaded_docs = await csv_loader.load(path)  # MODIFIED: await
                elif path.lower().endswith('.ipynb'):
                    loaded_docs = await notebook_loader.load_notebook(path)  # MODIFIED: await
                elif path.lower().endswith(('.tf', '.tfvars', '.yaml', '.yml')):
                    loaded_docs = await iac_loader.load(path)  # MODIFIED: await
                else:
                    logger.warning(f"Unsupported local file type: {path}. Skipping.")
            elif os.path.isdir(path):
                loaded_docs = await directory_loader.load(path)  # MODIFIED: await
        elif source_type == "code_repository":
            loaded_docs = await code_repository_loader.load(source.path)  # MODIFIED: await
        elif source_type == "url":
            loaded_docs = await url_loader.load(source.url)  # MODIFIED: await
        elif source_type == "db":
            loaded_docs = await db_loader.load(source.db_connection)  # MODIFIED: await
        elif source_type == "gdoc":
            loaded_docs = await gdoc_loader.load(folder_id=source.folder_id, document_ids=source.document_ids,
                                                 file_types=source.file_types)  # MODIFIED: await
        elif source_type == "web_crawler":
            loaded_docs = await web_crawler_loader.load(url=source.url, max_depth=source.max_depth)  # Already async
        elif source_type == "api":
            loaded_docs = await api_loader.load(url=source.url, method=source.method, headers=source.headers,
                                                params=source.params, payload=source.payload,
                                                json_pointer=source.json_pointer)  # MODIFIED: await
        else:
            logger.warning(f"Unknown or unsupported source type: '{source_type}'. Skipping.")
            return []

    except Exception as e:
        logger.error(f"Error loading documents from source {source.type} (path/url: {source.path or source.url}): {e}",
                     exc_info=True)
        return []

    for idx, doc in enumerate(loaded_docs):
        source_identifier = doc.metadata.get('source', source.path or source.url or source.type)
        if reproducible_ids:
            hash_input = f"{source_identifier}-{idx}-{doc.page_content}"
            doc.metadata['original_doc_id'] = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
        else:
            doc.metadata['original_doc_id'] = str(uuid.uuid4())
    return loaded_docs


async def embed_agent_data(config: AgentConfig):  # MODIFIED: async def
    """
    Main embedding pipeline. Loads, chunks, stores documents asynchronously, and saves a
    raw text copy of chunks for scalable BM25 initialization.
    """
    logger.info(f"[EMBED CONFIG] chunking.mode={config.chunking.mode}, vector_store.type={config.vector_store.type}")
    logger.info(f"Starting data embedding process for agent: '{config.name}'")

    # MODIFIED: Use asyncio.gather to load documents from sources concurrently
    loading_tasks = [
        load_documents_from_source(source, config.reproducible_ids)
        for source in config.sources
    ]

    # Run all loading tasks concurrently and collect results
    all_loaded_docs_lists = await asyncio.gather(*loading_tasks, return_exceptions=True)  # NEW: return_exceptions

    all_docs = []
    for loaded_docs_list in all_loaded_docs_lists:
        if isinstance(loaded_docs_list, Exception):
            logger.error(f"A document loading task failed: {loaded_docs_list}", exc_info=True)
            continue  # Continue with other sources even if one fails
        validated_docs = [doc for doc in loaded_docs_list if doc.page_content and doc.page_content.strip()]
        all_docs.extend(validated_docs)

    if not all_docs:
        raise ValueError("No valid documents with content found to embed from any of the specified sources.")

    all_docs.sort(key=lambda d: d.metadata.get("original_doc_id", ""))
    logger.info(f"Loaded a total of {len(all_docs)} valid documents from all sources.")

    try:
        # get_embedding_model is synchronous; if it involves network calls, it might need async version
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

        # --- Vector Store Creation Dispatcher (mostly synchronous LangChain calls) ---
        # These methods are mostly blocking for local FAISS/Chroma.
        # For remote VDBs (Qdrant, Pinecone, MongoDB Atlas), their respective
        # from_documents methods might have async equivalents or can be run in executor.
        if db_type == 'faiss':
            # FAISS.from_documents and db.save_local are blocking, consider running in ThreadPoolExecutor
            db = await asyncio.to_thread(FAISS.from_documents, chunks, langchain_embeddings)  # NEW: to_thread
            await asyncio.to_thread(db.save_local, vectorstore_path)  # NEW: to_thread
        elif db_type == 'chroma':
            await asyncio.to_thread(Chroma.from_documents, chunks, langchain_embeddings,
                                    persist_directory=vectorstore_path)  # NEW: to_thread
        elif db_type == 'qdrant':
            # Qdrant might have async client, check Qdrant.afrom_documents or similar
            await asyncio.to_thread(Qdrant.from_documents, chunks, langchain_embeddings, host=vs_config.qdrant_host,
                                    port=vs_config.qdrant_port,
                                    path=None if vs_config.qdrant_host else vectorstore_path,
                                    collection_name=config.name)  # NEW: to_thread
        elif db_type == 'pinecone':
            pc_api_key = get_api_key("pinecone")  # This is synchronous
            PineconeClient(api_key=pc_api_key)  # This is synchronous
            # PineconeLangChain.from_documents can be blocking
            await asyncio.to_thread(PineconeLangChain.from_documents, chunks, langchain_embeddings,
                                    index_name=vs_config.pinecone_index_name)  # NEW: to_thread
        elif db_type == 'mongodb_atlas':
            conn_string = get_api_key("mongodb")  # This is synchronous
            for chunk in chunks:
                chunk.metadata['_id'] = chunk.id
            # MongoDBAtlasVectorSearch.from_documents can be blocking
            await asyncio.to_thread(MongoDBAtlasVectorSearch.from_documents, documents=chunks,
                                    embedding=langchain_embeddings,
                                    connection_string=conn_string,
                                    namespace=f"{vs_config.mongodb_db_name}.{vs_config.mongodb_collection_name}",
                                    index_name=vs_config.mongodb_index_name)  # NEW: to_thread
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")

        logger.info(f"Successfully created and populated vector store for agent '{config.name}'.")

        # Save raw chunks for BM25 (file I/O, consider to_thread if large files)
        bm25_docs_path = os.path.join(vectorstore_path, "bm25_documents.jsonl")

        # For large files, this would also benefit from asyncio.to_thread

        # Original code with syntax error:
        # await asyncio.to_thread(lambda: (
        #     with open(bm25_docs_path, 'w', encoding='utf-8') as f:
        # for chunk in chunks:
        #     f.write(json.dumps({"id": chunk.id, "page_content": chunk.page_content}) + "\n")
        # ))

        # Corrected structure for the lambda or helper function
        def _write_bm25_chunks_to_file():
            with open(bm25_docs_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps({"id": chunk.id, "page_content": chunk.page_content}) + "\n")

        await asyncio.to_thread(_write_bm25_chunks_to_file)

        logger.info(f"Saved {len(chunks)} chunks for scalable BM25 retrieval at {bm25_docs_path}")

        logger.info(f"Successfully created vector store and BM25 source for agent '{config.name}'.")

    except Exception as e:
        logger.error(f"Failed to create vector store for agent '{config.name}'. Error: {e}", exc_info=True)
        raise VectorStoreCreationError(f"Vector store creation failed for agent '{config.name}'.") from e