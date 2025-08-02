import asyncio
import hashlib
import json
import logging
import os
import uuid
from itertools import groupby
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document as LangChainDocument
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_qdrant import Qdrant
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document as LlamaDocument
from llama_index.embeddings.langchain import LangchainEmbedding
from pinecone import Pinecone as PineconeClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_api_key, get_path_settings
from app.core.embed_config import get_embedding_model
from app.db.dao import create_document_chunk
from app.db.models import document_chunks_table
from app.pipelines.loaders import (
    directory_loader, url_loader, db_loader, api_loader,
    code_repository_loader, gdoc_loader, web_crawler_loader,
    notebook_loader, pdf_loader, docx_loader, csv_loader,
    text_loader, iac_loader, parquet_loader
)
from app.schemas.agent import AgentConfig
from app.schemas.agent import DataSource, ChunkingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_APP_PATHS = get_path_settings()
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]


class VectorStoreCreationError(Exception):
    """Raised when vector store creation fails."""
    pass


def _generate_chunk_id(content: str, original_id: str = "", reproducible: bool = False, idx: int = None) -> str:
    """Generates a unique ID for a chunk."""
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
    Applies a chunking strategy, correctly handling multi-page/source documents.
    Supports 'none', 'semantic', and 'default' modes.
    This version intelligently handles documents that are already pre-chunked by loaders.
    """
    final_chunks_list: List[LangChainDocument] = []
    chunking_mode = chunking_config.mode
    logger.info(f"Applying chunking strategy from config: '{chunking_mode}'")

    # Group documents by their source for consistent chunk indexing
    def get_source_key(doc):
        # Using a tuple of standardized metadata keys to create a consistent source key
        metadata_dict = doc.metadata.get("source_config", {})
        source_id = metadata_dict.get("path") or metadata_dict.get("url") or metadata_dict.get("db_connection")
        return source_id

    documents.sort(key=get_source_key)

    # Check if documents are already pre-chunked by the loader
    is_pre_chunked = all('chunk_index' in doc.metadata for doc in documents)
    if is_pre_chunked or chunking_mode == 'none':
        logger.info(
            "Documents appear to be pre-chunked by a loader or chunking mode is 'none'. Treating documents directly as chunks.")
        # Treat each document as a final chunk. We just need to ensure IDs and metadata are correctly set.
        for idx, doc in enumerate(documents):
            # If chunk_id is not already present, generate one
            if "chunk_id" not in doc.metadata or not doc.metadata["chunk_id"]:
                original_doc_id = doc.metadata.get("original_doc_id", "")
                cid = _generate_chunk_id(doc.page_content, original_doc_id, reproducible_ids, idx)
                doc.id = cid
                doc.metadata["chunk_id"] = cid

            # Ensure doc_name and chunk_index are set for DB persistence
            if "doc_name" not in doc.metadata:
                doc.metadata["doc_name"] = os.path.basename(
                    doc.metadata.get('source_path', doc.metadata.get('source_url', "Unknown Document")))
            if "chunk_index" not in doc.metadata:
                doc.metadata["chunk_index"] = idx

            final_chunks_list.append(doc)

        logger.info(f"Accepted {len(final_chunks_list)} pre-chunked documents as final chunks.")
        return final_chunks_list

    # If documents are not pre-chunked, proceed with the configured strategy
    for source_key, docs_from_source_group in groupby(documents, key=get_source_key):
        docs_from_source = list(docs_from_source_group)
        doc_name = os.path.basename(source_key) if source_key else "Unknown Document"
        current_source_chunks: List[LangChainDocument] = []

        if chunking_mode == 'semantic':
            logger.info(f"Attempting semantic chunking using LlamaIndex for source '{doc_name}'...")
            try:
                langchain_embeddings = get_embedding_model(embedding_model_name)
                llama_embeddings = LangchainEmbedding(langchain_embeddings)
                semantic_splitter = SemanticSplitterNodeParser.from_defaults(
                    embed_model=llama_embeddings,
                    breakpoint_percentile_threshold=chunking_config.breakpoint_percentile_threshold,
                )
                llama_docs = [LlamaDocument(text=doc.page_content, metadata=doc.metadata) for doc in docs_from_source]
                nodes = semantic_splitter.get_nodes_from_documents(llama_docs)

                for idx, node in enumerate(nodes):
                    content = node.get_content()
                    metadata = node.metadata

                    original_doc_id = metadata.get("original_doc_id", "")
                    generated_chunk_id = _generate_chunk_id(content, original_doc_id, reproducible_ids, idx)

                    metadata = {**metadata, "chunk_id": generated_chunk_id, "doc_name": doc_name, "chunk_index": idx}
                    new_doc = LangChainDocument(page_content=content, metadata=metadata, id=generated_chunk_id)
                    current_source_chunks.append(new_doc)
                logger.info(
                    f"Successfully applied semantic chunking for source '{doc_name}'. Resulted in {len(current_source_chunks)} chunks.")
            except Exception as e:
                logger.error(
                    f"Semantic chunking failed for source '{doc_name}': {e}. Falling back to default recursive splitting.",
                    exc_info=True)
                # Fallback to default chunking for this source
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunking_config.chunk_size,
                                                          chunk_overlap=chunking_config.chunk_overlap)
                for doc_part in docs_from_source:
                    chunks_from_page = splitter.split_documents([doc_part])
                    for sub_chunk_idx, chunk in enumerate(chunks_from_page):
                        original_doc_id = chunk.metadata.get('original_doc_id', '')
                        cid = _generate_chunk_id(chunk.page_content, original_doc_id, reproducible_ids, sub_chunk_idx)
                        chunk.id = cid
                        chunk.metadata["chunk_id"] = cid
                        chunk.metadata["doc_name"] = doc_name
                        chunk.metadata["chunk_index"] = sub_chunk_idx
                        current_source_chunks.append(chunk)

        elif chunking_mode == 'default':
            logger.info(
                f"Applying default recursive character chunking for source '{doc_name}' with size={chunking_config.chunk_size} and overlap={chunking_config.chunk_overlap}.")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunking_config.chunk_size,
                chunk_overlap=chunking_config.chunk_overlap
            )
            for doc_part in docs_from_source:
                chunks_from_page = splitter.split_documents([doc_part])
                for chunk_idx, chunk in enumerate(chunks_from_page):
                    original_doc_id = chunk.metadata.get('original_doc_id', '')
                    cid = _generate_chunk_id(chunk.page_content, original_doc_id, reproducible_ids, chunk_idx)
                    chunk.id = cid
                    chunk.metadata["chunk_id"] = cid
                    chunk.metadata["doc_name"] = doc_name
                    chunk.metadata["chunk_index"] = chunk_idx
                    current_source_chunks.append(chunk)
            logger.info(f"Default chunking for source '{doc_name}' resulted in {len(current_source_chunks)} chunks.")

        final_chunks_list.extend(current_source_chunks)

    if not final_chunks_list:
        raise ValueError("No chunks were generated after document splitting across all sources.")

    logger.info(f"Chunking resulted in {len(final_chunks_list)} total chunks across all sources.")
    return final_chunks_list


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
        elif source_type == "csv":
            loaded_docs = await csv_loader.load(source.path, **loader_args)
        elif source_type == "pdf":
            loaded_docs = await pdf_loader.load(source.path, **loader_args)
        elif source_type == "txt":
            loaded_docs = await text_loader.load(source.path, **loader_args)
        elif source_type == "docx":
            loaded_docs = await docx_loader.load(source.path, **loader_args)
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


async def embed_agent_data(config: AgentConfig, db: AsyncSession) -> bool:
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
    chunks = _get_chunks_from_documents(docs, config.chunking, config.embedding_model, config.reproducible_ids)
    if not chunks:
        logger.info("No chunks were generated from the documents.")
        return False

    logger.info(f"Saving/updating {len(chunks)} chunks in the database.")
    for chunk in chunks:
        doc_name = chunk.metadata["doc_name"]
        chunk_index = chunk.metadata["chunk_index"]
        try:
            chunk_id_from_db = await create_document_chunk(
                db=db,
                document_name=chunk.metadata.get("doc_name"),
                chunk_index=chunk.metadata.get("chunk_index"),
                content=chunk.page_content,
                page_number=chunk.metadata.get("page_number"),
                row_number=chunk.metadata.get("row_number"),

            )
            chunk.metadata['chunk_id'] = chunk_id_from_db

        except IntegrityError:
            await db.rollback()

            logger.warning(
                f"Chunk already exists, fetching ID for: {chunk.metadata.get('doc_name')}#{chunk.metadata.get('chunk_index')}")
            stmt = select(document_chunks_table.c.id).where(
                document_chunks_table.c.document_name == chunk.metadata.get("doc_name"),
                document_chunks_table.c.chunk_index == chunk.metadata.get("chunk_index"),
            )
            result = await db.execute(stmt)
            chunk_id = result.scalar_one()
            chunk.metadata["chunk_id"] = chunk_id
        except Exception as e:
            logger.error(f"Failed to save chunk for doc '{chunk.metadata.get('doc_name')}': {e}", exc_info=False)
            continue
    logger.info("Finished saving/updating chunks in the database.")

    # 3) Filter metadata in batch
    # This loop applies filter_complex_metadata to each chunk individually
    # filter_complex_metadata expects a Document and returns a Document with filtered metadata.
    chunks_for_store = filter_complex_metadata(chunks)

    chunks = filter_complex_metadata(chunks)
    if not chunks_for_store:
        logger.warning("No valid chunks remain after metadata filtering. Vector store will be empty.")
        return True  # Return True as the process didn't fail, but log the warning.

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
            f.write(json.dumps({
                'id': c.id,
                'page_content': c.page_content,
                'metadata': c.metadata
            }) + '\n')

    logger.info(f"Finished embedding for {config.name}")
    return True