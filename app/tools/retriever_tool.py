# app/tools/retriever_tool.py

import os
import logging
import json
from typing import List, Optional, Union, Dict, Any
import asyncio

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_qdrant import Qdrant
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, Tool
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from pinecone import Pinecone as PineconeClient

from app.core.embed_config import get_embedding_model
from app.core.config import get_path_settings, get_api_key
from app.schemas.agent import AgentConfig

logger = logging.getLogger(__name__)

_APP_PATHS = get_path_settings()
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]
_TEMP_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"] / "temp_chat_data"


async def get_retriever_tool(agent_config: AgentConfig) -> Tool:
    """
    Creates a LangChain Tool for document retrieval.
    This tool encapsulates the logic for combining permanent and
    dynamically loaded temporary vector stores on each invocation.
    """
    agent_name = agent_config.name
    vs_config = agent_config.vector_store

    try:
        embeddings = get_embedding_model(agent_config.embedding_model)

        permanent_vectorstore = None
        permanent_vectorstore_path = os.path.join(_VECTORSTORE_DIR, agent_name)

        if vs_config.type == 'faiss':
            if os.path.exists(permanent_vectorstore_path):
                permanent_vectorstore = await asyncio.to_thread(FAISS.load_local,
                                                                str(permanent_vectorstore_path), embeddings,
                                                                allow_dangerous_deserialization=True)
                logger.info(f"Loaded FAISS permanent vector store for agent '{agent_name}'.")
            else:
                logger.warning(
                    f"FAISS permanent store not found for '{agent_name}'. Agent may not have base knowledge.")

        elif vs_config.type == 'chroma':
            permanent_vectorstore_path_chroma = os.path.join(_VECTORSTORE_DIR, agent_name, "chroma_db")
            if os.path.exists(permanent_vectorstore_path_chroma):
                permanent_vectorstore = await asyncio.to_thread(
                    Chroma, persist_directory=permanent_vectorstore_path_chroma,
                    embedding_function=embeddings)
                logger.info(f"Loaded Chroma permanent vector store for agent '{agent_name}'.")
            else:
                logger.warning(
                    f"Chroma permanent store not found for '{agent_name}'. Agent may not have base knowledge.")

        elif vs_config.type == 'qdrant':
            if not vs_config.qdrant_host:
                logger.warning("Qdrant host not specified for permanent store. Skipping Qdrant permanent retriever.")
            else:
                permanent_vectorstore = await asyncio.to_thread(Qdrant,
                                                                client=None, collection_name=agent_name,
                                                                embeddings=embeddings,
                                                                host=vs_config.qdrant_host,
                                                                port=vs_config.qdrant_port, prefer_grpc=True
                                                                )
                logger.info(
                    f"Loaded Qdrant permanent vector store for agent '{agent_name}' on {vs_config.qdrant_host}.")

        elif vs_config.type == 'pinecone':
            if not vs_config.pinecone_index_name:
                logger.warning(
                    "Pinecone index name not specified for permanent store. Skipping Pinecone permanent retriever.")
            else:
                pinecone_api_key = get_api_key("pinecone")
                if not pinecone_api_key:
                    raise ValueError("Pinecone API key not found. Please configure it for permanent store.")
                await asyncio.to_thread(PineconeClient, api_key=pinecone_api_key)
                permanent_vectorstore = await asyncio.to_thread(PineconeLangChain.from_existing_index,
                                                                index_name=vs_config.pinecone_index_name,
                                                                embedding=embeddings
                                                                )
                logger.info(
                    f"Loaded Pinecone permanent vector store for agent '{agent_name}' index '{vs_config.pinecone_index_name}'.")

        elif vs_config.type == 'mongodb_atlas':
            if not vs_config.mongodb_db_name or not vs_config.mongodb_collection_name:
                logger.warning(
                    "MongoDB Atlas db/collection not specified for permanent store. Skipping MongoDB permanent retriever.")
            else:
                mongodb_conn_string = get_api_key("mongodb")
                if not mongodb_conn_string:
                    raise ValueError("MongoDB connection string not found. Please configure it for permanent store.")
                permanent_vectorstore = await asyncio.to_thread(MongoDBAtlasVectorSearch.from_connection_string,
                                                                mongodb_conn_string, vs_config.mongodb_db_name,
                                                                vs_config.mongodb_collection_name, embeddings,
                                                                index_name=vs_config.mongodb_index_name
                                                                )
                logger.info(
                    f"Loaded MongoDB Atlas permanent vector store for agent '{agent_name}' collection '{vs_config.mongodb_collection_name}'.")
        else:
            logger.warning(
                f"Unsupported permanent vector store type: {vs_config.type}. Agent will rely only on temporary documents if provided, or not retrieve.")
            permanent_vectorstore = None

        bm25_permanent = None
        docs_path_permanent = os.path.join(_VECTORSTORE_DIR, agent_name, "bm25_documents.jsonl")
        if os.path.exists(docs_path_permanent):
            try:
                with await asyncio.to_thread(open, docs_path_permanent, 'r', encoding='utf-8') as f:
                    bm25_docs_permanent = [
                        Document(page_content=json.loads(line)["page_content"], metadata=json.loads(line)["metadata"])
                        for line in f
                    ]

                if bm25_docs_permanent:
                    bm25_permanent = await asyncio.to_thread(
                        BM25Retriever.from_documents, bm25_docs_permanent, k=vs_config.bm25_k)
                    logger.info("Loaded BM25 retriever for permanent store.")
                else:
                    logger.warning("No documents found for BM25 retriever in bm25_documents.jsonl.")
            except Exception as e:
                logger.error(f"Error loading BM25 documents from {docs_path_permanent}: {e}")
        else:
            logger.warning(
                f"BM25 permanent source file not found at {docs_path_permanent}. Hybrid retrieval for permanent store might be impacted.")

        class DynamicRetrieverTool(BaseTool):
            name: str = "document_retriever"
            description: str = (
                "Useful for retrieving relevant documents from the knowledge base based on a user query. "
                "Optionally, it can also include temporarily uploaded documents in the search. "
                "Input should be a dictionary with 'query' (string) and optionally 'temp_document_ids' (list of strings)."
            )

            # Declare instance fields for complex objects
            permanent_semantic_retriever: Optional[Any]
            permanent_bm25_retriever: Optional[Any]
            embeddings: Any
            vs_config: Any
            agent_config: Any

            class Config:
                # Allow Pydantic to handle complex, non-serializable objects
                arbitrary_types_allowed = True

            async def _arun(self, query: str, temp_document_ids: Optional[List[str]] = None) -> List[Document]:
                logger.info(f"Retriever tool running with query: '{query}' and temp_doc_ids: {temp_document_ids}")

                current_retrievers = []

                # Use the initialized retrievers from instance fields
                if self.permanent_bm25_retriever:
                    current_retrievers.append(self.permanent_bm25_retriever)
                if self.permanent_semantic_retriever:
                    current_retrievers.append(self.permanent_semantic_retriever)

                temp_retrievers_instances: List[Any] = []

                if temp_document_ids:
                    logger.info(f"Attempting to load {len(temp_document_ids)} temporary documents for retrieval.")
                    for temp_doc_id in temp_document_ids:
                        temp_doc_vector_path = _TEMP_VECTORSTORE_DIR / temp_doc_id
                        if temp_doc_vector_path.exists() and self.vs_config.type == 'faiss':
                            try:
                                temp_vectorstore = await asyncio.to_thread(FAISS.load_local,
                                                                           str(temp_doc_vector_path), self.embeddings,
                                                                           allow_dangerous_deserialization=True)
                                temp_retrievers_instances.append(
                                    temp_vectorstore.as_retriever(search_kwargs={"k": self.vs_config.semantic_k}))
                                logger.info(f"Loaded temporary FAISS vector store for temp_doc_id: {temp_doc_id}")
                            except Exception as e:
                                logger.error(f"Failed to load temporary FAISS store {temp_doc_id}: {e}", exc_info=True)
                        else:
                            logger.warning(
                                f"Temporary document '{temp_doc_id}' not found or unsupported for type {self.vs_config.type}. Only FAISS temp stores are directly loaded via file path.")

                current_retrievers.extend(temp_retrievers_instances)

                if not current_retrievers:
                    logger.warning("No retrievers (permanent or temporary) are available. Returning empty documents.")
                    return []

                if len(current_retrievers) == 1:
                    base_retriever = current_retrievers[0]
                    logger.info(f"Using single retriever ('{current_retrievers[0].__class__.__name__}').")
                else:
                    weights = [0.5, 0.5] if len(current_retrievers) == 2 else [1.0 / len(current_retrievers)] * len(
                        current_retrievers)
                    ensemble = EnsembleRetriever(retrievers=current_retrievers, weights=weights)
                    base_retriever = ensemble
                    logger.info(f"Using EnsembleRetriever with {len(current_retrievers)} components.")

                if self.vs_config.retrieval_strategy == 'enhanced':
                    cross_encoder = await asyncio.to_thread(HuggingFaceCrossEncoder,
                                                            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
                    compressor = ContextualCompressionRetriever(
                        base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=self.vs_config.rerank_top_n),
                        base_retriever=base_retriever)
                    base_retriever = compressor
                    logger.info("Enhanced retrieval strategy enabled with Cross-Encoder Reranker.")

                retrieved_docs = await base_retriever.ainvoke(query)

                for idx, doc in enumerate(retrieved_docs):
                    logger.info(f"Retrieved Document {idx} metadata: {doc.metadata}")

                for doc in retrieved_docs:
                    if hasattr(doc, 'metadata') and doc.metadata.get('doc_name'):
                        doc.metadata['citation_source'] = doc.metadata['doc_name']
                        if doc.metadata.get('page_number'):
                            doc.metadata['citation_source'] += f" (Page {doc.metadata['page_number']})"
                    else:
                        doc.metadata['citation_source'] = "Unknown Source"

                return retrieved_docs

            def _run(self, query: str, temp_document_ids: Optional[List[str]] = None) -> List[Document]:
                logger.warning("Synchronous _run called for DynamicRetrieverTool. Using asyncio.run to call _arun.")
                return asyncio.run(self._arun(query, temp_document_ids))

        # Instantiate the tool, passing the complex objects as arguments
        return DynamicRetrieverTool(
            permanent_semantic_retriever=permanent_vectorstore.as_retriever(search_kwargs={"k": vs_config.semantic_k}) if permanent_vectorstore else None,
            permanent_bm25_retriever=bm25_permanent,
            embeddings=embeddings,
            vs_config=vs_config,
            agent_config=agent_config
        )

    except Exception as initialization_err:
        logger.error(f"Failed to initialize base retrievers for '{agent_name}': {initialization_err}", exc_info=True)

        def error_func(query: str, temp_document_ids: Optional[List[str]] = None) -> str:
            return f"Error: Could not load retriever for '{agent_name}'. Details: {initialization_err}"

        return Tool.from_function(name="retriever_error", func=error_func,
                                  description="Reports an error during retriever initialization.")