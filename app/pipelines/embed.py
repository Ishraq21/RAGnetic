import os
import logging
from typing import List, Dict, Any, Optional
import uuid
import hashlib
import json
import asyncio

# LangChain and LlamaIndex have different Document objects
from langchain_core.documents import Document as LangChainDocument  # Using canonical import path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding

# LangChain vector store components
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_qdrant import Qdrant
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_mongodb import MongoDBAtlasVectorSearch
from pinecone import Pinecone as PineconeClient

# Use forward references for type hints to avoid circular imports
from app.schemas.agent import AgentConfig, DataSource, ChunkingConfig
from app.core.embed_config import get_embedding_model
from app.core.config import get_api_key, get_path_settings
from app.pipelines.loaders import (
    directory_loader, url_loader, db_loader, api_loader,
    code_repository_loader, gdoc_loader, web_crawler_loader,
    notebook_loader, pdf_loader, docx_loader, csv_loader,
    text_loader, iac_loader, parquet_loader
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_APP_PATHS = get_path_settings()
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]


class VectorStoreCreationError(Exception):
    """Raised when vector store creation fails."""
    pass


def _generate_chunk_id(content: str, original_id: str = "", reproducible: bool = False, idx: int = None) -> str:
    if reproducible:
        hash_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
        return f"{original_id}-{idx}-{hash_digest}" if idx is not None else f"{original_id}-{hash_digest}"
    return str(uuid.uuid4())


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
    final_chunks_list: List[LangChainDocument] = []  # Use a single list for all modes' results
    chunking_mode = chunking_config.mode
    logger.info(f"Applying chunking mode: '{chunking_mode}'")

    # Ensure input documents are valid LangChainDocuments
    validated_documents = []
    for i, doc in enumerate(documents):
        if not isinstance(doc, LangChainDocument):
            logger.error(
                f"[DEBUG] _get_chunks_from_documents (Input Validation): Input doc at #{i} is not LangChainDocument, got {type(doc)}. Skipping. Content snippet: {str(doc)[:100]}")
            continue
        validated_documents.append(doc)
    documents = validated_documents  # Use validated documents for chunking

    # --- Handle 'none' chunking mode ---
    if chunking_mode == 'none':
        logger.info("Chunking mode is 'none'. Using documents directly as chunks.")
        for idx, doc in enumerate(documents):
            # Create a new LangChainDocument instance to guarantee consistency
            new_chunk = LangChainDocument(page_content=doc.page_content, metadata=doc.metadata.copy(), id=doc.id)
            if reproducible_ids:
                content_hash = hashlib.sha256(new_chunk.page_content.encode('utf-8')).hexdigest()
                new_chunk.id = f"{new_chunk.metadata.get('original_doc_id', '')}-{idx}-{content_hash[:8]}"
            else:
                new_chunk.id = str(uuid.uuid4())
            new_chunk.metadata["chunk_id"] = new_chunk.id
            final_chunks_list.append(new_chunk)
        logger.info(f"None chunking resulted in {len(final_chunks_list)} chunks.")
        return final_chunks_list

    # --- Handle 'semantic' chunking mode ---
    elif chunking_mode == 'semantic':
        logger.info("Attempting semantic chunking using LlamaIndex...")
        try:
            embeddings = LangchainEmbedding(get_embedding_model(embedding_model_name))
            splitter = SemanticSplitterNodeParser.from_defaults(
                embed_model=embeddings,
                breakpoint_percentile_threshold=chunking_config.breakpoint_percentile_threshold,
            )
            llama_docs = [LlamaDocument(text=d.page_content, metadata=d.metadata.copy(), id_=d.id) for d in documents]
            nodes = splitter.get_nodes_from_documents(llama_docs)
            final_chunks_list = []
            for node in nodes:
                content = node.get_content()
                metadata = node.metadata.copy()

                original_doc_id = metadata.get("original_doc_id", "")
                cid = _generate_chunk_id(content, original_doc_id, reproducible_ids)

                metadata["chunk_id"] = cid
                new_doc = LangChainDocument(page_content=content, metadata=metadata, id=cid)
                final_chunks_list.append(new_doc)
            logger.info(f"Successfully applied semantic chunking. Resulted in {len(final_chunks_list)} chunks.")
            return final_chunks_list
        except Exception:
            logger.warning("Semantic chunking failed, falling back to default.", exc_info=True)
            chunking_mode = 'default'  # Fallback to default below

    # --- Handle 'default' (Recursive) chunking mode ---
    # This block will execute if chunking_mode was 'default' initially, or if 'semantic' failed and fell back here.
    if chunking_mode == 'default':
        logger.info(
            f"Applying default recursive character chunking with size={chunking_config.chunk_size} and overlap={chunking_config.chunk_overlap}.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.chunk_size,
            chunk_overlap=chunking_config.chunk_overlap
        )
        split_chunks_raw = splitter.split_documents(documents)  # This should return LangChainDocuments

        normalized_chunks: List[LangChainDocument] = []
        for idx, item in enumerate(split_chunks_raw):
            logger.debug(
                f"[DEBUG] _get_chunks_from_documents/default: raw_chunk_item #{idx} type: {type(item)}, content_snippet: {str(item)[:100]}")  # NEW DEBUG LOG

            if isinstance(item, LangChainDocument):
                normalized_chunks.append(item)
            elif isinstance(item, tuple) and len(item) >= 2:
                page_content, metadata = item[0], item[1] if isinstance(item[1], dict) else {}
                chunk_doc = LangChainDocument(page_content=page_content, metadata=metadata)
                normalized_chunks.append(chunk_doc)  # CORRECTED: Append the converted Document
                logger.debug(
                    f"[Chunk Normalize] Tuple found at #{idx} – converted to LangChainDocument in _get_chunks_from_documents/default.")
            else:
                logger.warning(
                    f"[Chunk Normalize] Unexpected type at #{idx}: {type(item)} — skipping in _get_chunks_from_documents/default. Content: {str(item)[:100]}")
                continue

        # Correctly populate final_chunks_list from normalized_chunks
        for idx, d in enumerate(normalized_chunks):
            cid = _generate_chunk_id(d.page_content, d.metadata.get('original_doc_id', ''), reproducible_ids, idx)
            final_chunks_list.append(
                LangChainDocument(
                    page_content=d.page_content,
                    metadata={**d.metadata, "chunk_id": cid},
                    id=cid
                )
            )
        logger.info(f"Default chunking resulted in {len(final_chunks_list)} chunks.")
        return final_chunks_list

    # If no mode executed successfully or yielded chunks. This return should ideally be unreachable.
    return []


async def load_documents_from_source(source: DataSource, agent_config: AgentConfig, reproducible_ids: bool) -> list[
    LangChainDocument]:
    """
    Dispatcher function that calls the appropriate loader based on the source type,
    passing the full agent_config for policy application and scaling settings,
    and the reproducible_ids flag.
    """
    source_type = source.type
    logger.info(f"Dispatching to loader for source type: '{source_type}'")
    loaded_docs: List[LangChainDocument] = []

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

    final_loaded_docs = []
    for idx, doc in enumerate(loaded_docs):
        if not isinstance(doc, LangChainDocument):
            logger.error(
                f"DEBUG: Invalid document type returned by loader: Expected LangChainDocument, got {type(doc)}. Skipping. Content snippet: {str(doc)[:100]}")
            continue

        # Create a copy of metadata to ensure we don't modify original objects if they are shared
        metadata_copy = doc.metadata.copy()

        # Ensure 'source' metadata is consistently added/updated from the DataSource object
        metadata_copy['source_type'] = source.type
        if source.path:
            metadata_copy['source_path'] = source.path
        if source.url:
            metadata_copy['source_url'] = source.url
        if source.db_connection:
            metadata_copy['source_db_connection'] = source.db_connection

        # Original doc ID generation as before
        source_identifier = metadata_copy.get('source', source.path or source.url or source.type)
        if reproducible_ids:
            hash_input = f"{source_identifier}-{idx}-{doc.page_content}"
            metadata_copy['original_doc_id'] = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
        else:
            metadata_copy['original_doc_id'] = str(uuid.uuid4())

        # Create a new LangChainDocument instance with updated metadata and original ID
        new_doc = LangChainDocument(page_content=doc.page_content, metadata=metadata_copy, id=doc.id)
        final_loaded_docs.append(new_doc)

    return final_loaded_docs


async def embed_agent_data(config: AgentConfig) -> bool:
    """
    Main pipeline: load, chunk, filter metadata, build vector store, save BM25.
    """
    logger.info(f"Embedding for agent: {config.name}")
    # 1) Load
    tasks = [load_documents_from_source(s, config, config.reproducible_ids) for s in config.sources or []]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    docs = []
    for res in results:
        if isinstance(res, Exception):
            logger.error("Loader task failed", exc_info=True)
        else:
            docs.extend(res)
    if not docs:
        logger.info("No documents to embed.")
        return False

    # 2) Chunk
    docs.sort(key=lambda d: d.metadata.get('original_doc_id', ''))
    chunks = _get_chunks_from_documents(docs, config.chunking, config.embedding_model, config.reproducible_ids)
    if not chunks:
        logger.info("No chunks generated.")
        return False

    for i, c in enumerate(chunks):
        logger.debug(f"Chunk #{i} type: {type(c)}")


    # 3) Filter metadata in batch
    # This loop applies filter_complex_metadata to each chunk individually
    # filter_complex_metadata expects a Document and returns a Document with filtered metadata.
    chunks = filter_complex_metadata(chunks)
    if not chunks:
        logger.info("No valid chunks after metadata filtering.")
        return False

    # 4) Build store
    store_dir = os.path.join(_VECTORSTORE_DIR, config.name)
    os.makedirs(store_dir, exist_ok=True)
    embeddings = get_embedding_model(config.embedding_model)
    vs = config.vector_store
    try:
        if vs.type == 'faiss':
            db = await asyncio.to_thread(FAISS.from_documents, chunks, embeddings)
            await asyncio.to_thread(db.save_local, store_dir)
        elif vs.type == 'chroma':
            chroma_dir = os.path.join(store_dir, 'chroma_db')
            db = await asyncio.to_thread(Chroma.from_documents, chunks, embeddings, persist_directory=chroma_dir)
        elif vs.type == 'qdrant':
            db = await asyncio.to_thread(Qdrant.from_documents, chunks, embeddings,
                                         host=vs.qdrant_host, port=vs.qdrant_port, collection_name=config.name)
        elif vs.type == 'pinecone':
            PineconeClient(api_key=get_api_key('pinecone'))
            db = await asyncio.to_thread(PineconeLangChain.from_documents, chunks, embeddings,
                                         index_name=vs.pinecone_index_name)
        elif vs.type == 'mongodb_atlas':
            for c in chunks:
                c.metadata['_id'] = c.id
            db = await asyncio.to_thread(
                MongoDBAtlasVectorSearch.from_documents,
                documents=chunks,
                embedding=embeddings,
                connection_string=get_api_key('mongodb'),
                namespace=f"{vs.mongodb_db_name}.{vs.mongodb_collection_name}",
                index_name=vs.mongodb_index_name
            )
        else:
            raise ValueError(f"Unsupported store: {vs.type}")
    except Exception as e:
        logger.error("Store creation failed", exc_info=True)
        raise VectorStoreCreationError from e

    # 5) Save BM25
    bm25_path = os.path.join(store_dir, 'bm25_documents.jsonl')
    with open(bm25_path, 'w', encoding='utf-8') as f:
        for c in chunks:
            f.write(json.dumps({'id': c.id, 'page_content': c.page_content}) + '\n')

    logger.info(f"Finished embedding for {config.name}")
    return True