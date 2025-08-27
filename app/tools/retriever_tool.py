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
from sqlalchemy.ext.asyncio import AsyncSession

from pinecone import Pinecone as PineconeClient

import app.db as db_mod

from app.core.embed_config import get_embedding_model
from app.core.config import get_path_settings, get_api_key
from app.db.dao import get_temp_document_by_user_thread_id
from app.schemas.agent import AgentConfig

logger = logging.getLogger(__name__)

_APP_PATHS = get_path_settings()
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]
_TEMP_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"] / "temp_chat_data"


def _load_bm25_docs(path: str) -> List[Document]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(Document(page_content=obj["page_content"], metadata=obj["metadata"]))
    return out

async def get_retriever_tool(
        agent_config: AgentConfig,
        user_id: int,
        thread_id: str
) -> Tool:
    """
    Creates a LangChain Tool for document retrieval.
    This tool encapsulates the logic for combining permanent and
    dynamically loaded temporary vector stores on each invocation.

    This version now enforces isolation for temporary documents based on user_id and thread_id.
    """
    agent_name = agent_config.name
    vs_config = agent_config.vector_store

    # We will assume temporary documents are always stored locally as FAISS files.
    # The permanent store type can still be configured in the YAML.
    # We don't need a `temp_vs_config` as it's now a fixed choice.

    try:
        embeddings = get_embedding_model(agent_config.embedding_model)

        permanent_vectorstore = None
        permanent_vectorstore_path = os.path.join(_VECTORSTORE_DIR, agent_name)

        if vs_config.type == 'faiss':
            if os.path.exists(permanent_vectorstore_path):
                permanent_vectorstore = await asyncio.to_thread(
                    FAISS.load_local,
                    str(permanent_vectorstore_path),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.debug(f"Loaded FAISS permanent vector store for agent '{agent_name}'.")
            else:
                logger.warning(
                    f"FAISS permanent store not found for '{agent_name}'. Agent may not have base knowledge."
                )

        elif vs_config.type == 'chroma':
            permanent_vectorstore_path_chroma = os.path.join(_VECTORSTORE_DIR, agent_name, "chroma_db")
            if os.path.exists(permanent_vectorstore_path_chroma):
                permanent_vectorstore = await asyncio.to_thread(
                    Chroma,
                    persist_directory=permanent_vectorstore_path_chroma,
                    embedding_function=embeddings
                )
                logger.info(f"Loaded Chroma permanent vector store for agent '{agent_name}'.")
            else:
                logger.warning(
                    f"Chroma permanent store not found for '{agent_name}'. Agent may not have base knowledge."
                )

        elif vs_config.type == 'qdrant':
            if not vs_config.qdrant_host:
                logger.warning("Qdrant host not specified for permanent store. Skipping Qdrant permanent retriever.")
            else:
                permanent_vectorstore = await asyncio.to_thread(
                    Qdrant,
                    client=None,
                    collection_name=agent_name,
                    embeddings=embeddings,
                    host=vs_config.qdrant_host,
                    port=vs_config.qdrant_port,
                    prefer_grpc=True
                )
                logger.info(
                    f"Loaded Qdrant permanent vector store for agent '{agent_name}' on {vs_config.qdrant_host}."
                )

        elif vs_config.type == 'pinecone':
            if not vs_config.pinecone_index_name:
                logger.warning(
                    "Pinecone index name not specified for permanent store. Skipping Pinecone permanent retriever."
                )
            else:
                pinecone_api_key = get_api_key("pinecone")
                if not pinecone_api_key:
                    raise ValueError(
                        "Pinecone API key not found. Please configure it for permanent store."
                    )
                await asyncio.to_thread(PineconeClient, api_key=pinecone_api_key)
                permanent_vectorstore = await asyncio.to_thread(
                    PineconeLangChain.from_existing_index,
                    index_name=vs_config.pinecone_index_name,
                    embedding=embeddings
                )
                logger.info(
                    f"Loaded Pinecone permanent vector store for agent '{agent_name}' index '{vs_config.pinecone_index_name}'."
                )

        elif vs_config.type == 'mongodb_atlas':
            if not vs_config.mongodb_db_name or not vs_config.mongodb_collection_name:
                logger.warning(
                    "MongoDB Atlas db/collection not specified for permanent store. Skipping MongoDB permanent retriever."
                )
            else:
                mongodb_conn_string = get_api_key("mongodb")
                if not mongodb_conn_string:
                    raise ValueError(
                        "MongoDB connection string not found. Please configure it for permanent store."
                    )
                permanent_vectorstore = await asyncio.to_thread(
                    MongoDBAtlasVectorSearch.from_connection_string,
                    mongodb_conn_string,
                    vs_config.mongodb_db_name,
                    vs_config.mongodb_collection_name,
                    embeddings,
                    index_name=vs_config.mongodb_index_name
                )
                logger.info(
                    f"Loaded MongoDB Atlas permanent vector store for agent '{agent_name}' collection '{vs_config.mongodb_collection_name}'."
                )
        else:
            logger.warning(
                f"Unsupported permanent vector store type: {vs_config.type}. "
                f"Agent will rely only on temporary documents if provided, or not retrieve."
            )
            permanent_vectorstore = None

        bm25_permanent = None
        docs_path_permanent = os.path.join(_VECTORSTORE_DIR, agent_name, "bm25_documents.jsonl")
        if os.path.exists(docs_path_permanent):
            try:
                # offload all file IO + JSON parsing
                bm25_docs_permanent = await asyncio.to_thread(_load_bm25_docs, docs_path_permanent)

                if bm25_docs_permanent:
                    bm25_permanent = await asyncio.to_thread(
                        BM25Retriever.from_documents,
                        bm25_docs_permanent,
                        k=vs_config.bm25_k
                    )
                    logger.info("Loaded BM25 retriever for permanent store.")
                else:
                    logger.warning("No documents found for BM25 retriever in bm25_documents.jsonl.")
            except Exception as e:
                logger.error(f"Error loading BM25 documents from {docs_path_permanent}: {e}")
        else:
            logger.warning(
                f"BM25 permanent source file not found at {docs_path_permanent}. "
                f"Hybrid retrieval for permanent store might be impacted."
            )


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
            user_id: int
            thread_id: str

            class Config:
                # Allow Pydantic to handle complex, non-serializable objects
                arbitrary_types_allowed = True

            async def _arun(self, query: str, temp_document_ids: Optional[List[str]] = None) -> List[Document]:
                logger.debug(
                    f"Retriever tool running with query: '{query}' "
                    f"for thread '{self.thread_id}' (user {self.user_id}). "
                    f"Temp doc IDs: {temp_document_ids}"
                )

                current_retrievers = []
                if self.permanent_bm25_retriever:
                    current_retrievers.append(self.permanent_bm25_retriever)
                if self.permanent_semantic_retriever:
                    current_retrievers.append(self.permanent_semantic_retriever)

                temp_retrievers_instances: List[Any] = []

                if temp_document_ids:
                    logger.info(f"Attempting to load {len(temp_document_ids)} temporary documents for retrieval.")
                    async with db_mod.AsyncSessionLocal() as temp_db:


                        for temp_doc_id in temp_document_ids:
                            # ownership check
                            if not await get_temp_document_by_user_thread_id(
                                    db=temp_db,
                                    temp_doc_id=temp_doc_id,
                                    user_id=self.user_id,
                                    thread_id=self.thread_id
                            ):
                                logger.warning(f"Skipping temp doc {temp_doc_id}: not owned by this user/thread.")
                                continue

                            temp_store_path = _TEMP_VECTORSTORE_DIR / temp_doc_id
                            if not temp_store_path.exists():
                                logger.warning(f"No FAISS store found at {temp_store_path}.")
                                continue

                            # load the FAISS index only once
                            temp_vs = await asyncio.to_thread(
                                FAISS.load_local,
                                str(temp_store_path),
                                self.embeddings,
                                allow_dangerous_deserialization=True
                            )

                            # stage-1: semantic search on that one temp doc
                            temp_retriever = temp_vs.as_retriever(search_kwargs={"k": self.vs_config.semantic_k})
                            hits = await temp_retriever.ainvoke(query)
                            if not hits:
                                logger.info(f"Temp doc {temp_doc_id}: no relevant chunks. Skipping.")
                                continue

                            # only if we got hits do we add this retriever to the mix
                            logger.info(f"Temp doc {temp_doc_id}: {len(hits)} relevant chunks—adding to retrievers.")
                            current_retrievers.append(temp_retriever)

                if not current_retrievers:
                    logger.warning(
                        "No retrievers (permanent or temporary) are available. Returning empty documents."
                    )
                    return []

                if len(current_retrievers) == 1:
                    base_retriever = current_retrievers[0]
                    logger.info(
                        f"Using single retriever ('{type(base_retriever).__name__}')."
                    )
                else:
                    weights = (
                        [0.5, 0.5]
                        if len(current_retrievers) == 2
                        else [1.0 / len(current_retrievers)] * len(current_retrievers)
                    )
                    ensemble = EnsembleRetriever(
                        retrievers=current_retrievers, weights=weights
                    )
                    base_retriever = ensemble
                    logger.info(
                        f"Using EnsembleRetriever with {len(current_retrievers)} components."
                    )

                if self.vs_config.retrieval_strategy == 'enhanced':
                    cross_encoder = await asyncio.to_thread(
                        HuggingFaceCrossEncoder,
                        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
                    )
                    compressor = ContextualCompressionRetriever(
                        base_compressor=CrossEncoderReranker(
                            model=cross_encoder, top_n=self.vs_config.rerank_top_n
                        ),
                        base_retriever=base_retriever
                    )
                    base_retriever = compressor
                    logger.info(
                        "Enhanced retrieval strategy enabled with Cross-Encoder Reranker."
                    )

                # perform actual retrieval
                retrieved_docs = await base_retriever.ainvoke(query)

                for idx, doc in enumerate(retrieved_docs):
                    logger.info(f"Retrieved Document {idx} metadata: {doc.metadata}")

                # set a uniform citation_source field
                for doc in retrieved_docs:
                    if hasattr(doc, 'metadata') and doc.metadata.get('doc_name'):
                        doc.metadata['citation_source'] = doc.metadata['doc_name']
                        if doc.metadata.get('page_number'):
                            doc.metadata['citation_source'] += (
                                f" (Page {doc.metadata['page_number']})"
                            )
                    else:
                        doc.metadata['citation_source'] = "Unknown Source"

                return retrieved_docs

            def _run(
                    self,
                    query: str,
                    temp_document_ids: Optional[List[str]] = None
            ) -> List[Document]:
                """Synchronous version of the tool. Must create its own async loop and db session."""
                logger.warning(
                    "Synchronous _run called for DynamicRetrieverTool. Using new async session."
                )

                async def sync_runner():
                    return await self._arun(
                        query=query,
                        temp_document_ids=temp_document_ids
                    )

                return asyncio.run(sync_runner())

        # Instantiate the tool, passing the complex objects and new parameters
        return DynamicRetrieverTool(
            permanent_semantic_retriever=(
                permanent_vectorstore.as_retriever(
                    search_kwargs={"k": vs_config.semantic_k}
                ) if permanent_vectorstore else None
            ),
            permanent_bm25_retriever=bm25_permanent,
            embeddings=embeddings,
            vs_config=vs_config,
            agent_config=agent_config,
            user_id=user_id,
            thread_id=thread_id
        )

    except Exception as initialization_err:
        logger.error(
            f"Failed to initialize base retrievers for '{agent_name}': {initialization_err}",
            exc_info=True
        )

        def error_func(
                query: str,
                temp_document_ids: Optional[List[str]] = None
        ) -> str:
            return (
                f"Error: Could not load retriever for '{agent_name}'. "
                f"Details: {initialization_err}"
            )

        return Tool.from_function(
            name="retriever_error",
            func=error_func,
            description="Reports an error during retriever initialization."
        )
