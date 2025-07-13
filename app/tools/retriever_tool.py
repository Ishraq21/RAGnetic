# app/tools/retriever_tool.py
import os
import logging
import json
from typing import List, Dict, Union

# LangChain components for retrieval and vector stores
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_qdrant import Qdrant
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_core.tools import Tool

# Pinecone client for initialization
from pinecone import Pinecone as PineconeClient
# LangChain's native HuggingFace cross-encoder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Local application imports
from app.core.embed_config import get_embedding_model
from app.core.config import get_api_key
from app.schemas.agent import AgentConfig

logger = logging.getLogger(__name__)


def _get_all_documents_from_vectorstore(vectorstore) -> List[Document]:
    """
    A helper function to retrieve all documents from any supported vector store.
    This is needed to initialize the BM25 keyword retriever.
    Relies on Document.id being correctly populated by the embedding pipeline.
    Note: For very large vector stores, retrieving all documents might be inefficient.
    """
    all_docs: List[Document] = []

    if isinstance(vectorstore, FAISS):
        if hasattr(vectorstore, 'docstore') and hasattr(vectorstore, 'index_to_docstore_id'):
            docs_from_store = [vectorstore.docstore.search(doc_id) for doc_id in
                               vectorstore.index_to_docstore_id.values()]
            all_docs = [doc for doc in docs_from_store if doc is not None]
        else:
            logger.warning(
                "FAISS docstore not directly accessible for all documents. Falling back to similarity search.")
            all_docs = vectorstore.similarity_search("*", k=10000)
    elif isinstance(vectorstore, Chroma):
        try:
            all_docs = vectorstore.similarity_search("*", k=10000)  # Retrieve a large number
        except Exception:
            logger.warning("Chroma similarity search for all documents failed. Attempting collection get.")
            try:
                # Direct access to collection to get all documents might be better
                raw_docs = vectorstore._collection.get(ids=None, where={}, limit=10000)
                # Ensure Document.id is mapped from Chroma's internal ID
                all_docs = [Document(page_content=doc.get('document', ''), metadata={**doc.get('metadata', {})},
                                     id=doc.get('id', '')) for doc in raw_docs.get('documents', [])]
            except Exception as e:
                logger.error(f"Failed to retrieve all documents from Chroma directly: {e}")
                all_docs = []
    elif isinstance(vectorstore, Qdrant):
        try:
            points, _ = vectorstore.client.scroll(
                collection_name=vectorstore.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            # Ensure Document.id is mapped from Qdrant's point.id
            all_docs = [Document(page_content=point.payload.get('page_content', ''),
                                 metadata={**point.payload}, id=str(point.id)) for point in points]
        except Exception as e:
            logger.error(f"Failed to retrieve all documents from Qdrant: {e}")
            all_docs = []
    elif isinstance(vectorstore, PineconeVectorStore):
        logger.warning(
            "Retrieving all documents from Pinecone for BM25 is complex and can be inefficient. Consider alternative BM25 indexing or sampling from original data source.")
        all_docs = vectorstore.as_retriever(search_kwargs={"k": 10000}).invoke(" ")
    elif isinstance(vectorstore, MongoDBAtlasVectorSearch):
        logger.warning(
            "Retrieving all documents from MongoDB Atlas Vector Search for BM25 can be inefficient. Consider indexing BM25 from original data source.")
        try:
            collection = vectorstore.collection
            raw_docs = list(collection.find().limit(10000))
            # Ensure Document.id is mapped from MongoDB's _id
            all_docs = [Document(page_content=doc.get('page_content', ''), metadata={**doc}, id=str(doc.get('_id', '')))
                        for doc in raw_docs]
        except Exception as e:
            logger.error(f"Failed to retrieve all documents from MongoDB Atlas via collection: {e}")
            all_docs = vectorstore.similarity_search("", k=10000)  # Fallback

    if not all_docs:
        logger.warning(
            f"No documents retrieved from vector store '{type(vectorstore).__name__}' for BM25 initialization. This may affect hybrid search.")

    return all_docs


def tool_fn(retriever, query: Union[str, Dict[str, str]]) -> List[Document]:
    """The actual function that the tool will execute."""
    if isinstance(query, dict):
        search_query = query.get("input", "")
    else:
        search_query = query
    return retriever.invoke(search_query)


def get_retriever_tool(agent_config: AgentConfig) -> Tool:
    """
    Creates a universal retriever tool with a configurable retrieval strategy.
    Relies on Document.id being correctly populated by the embedding pipeline.
    """
    agent_name = agent_config.name
    vs_config = agent_config.vector_store
    strategy = vs_config.retrieval_strategy
    # id_key_for_vs is no longer needed in config or passed directly to VS constructors.
    # We rely on Document.id being set in embed.py.

    logger.info(
        f"[RETRIEVER CONFIG for {agent_name}] "
        f"strategy={strategy}, "
        f"bm25_k={vs_config.bm25_k}, "
        f"semantic_k={vs_config.semantic_k}, "
        f"rerank_top_n={vs_config.rerank_top_n}, "
        f"vector_store.type={vs_config.type}"  # Removed id_key_for_vector_store from log
    )

    logger.info(f"Configuring retriever for '{agent_name}' with strategy: '{strategy}'.")

    try:
        # --- Vector Store Loading ---
        db_type = vs_config.type
        vectorstore_path = f"vectorstore/{agent_name}"
        embeddings = get_embedding_model(agent_config.embedding_model)
        vectorstore = None

        logger.info(f"Loading vector store of type '{db_type}' for agent '{agent_name}'.")

        if db_type == 'faiss':
            # FAISS.load_local expects Document.id to be set for internal consistency.
            if not os.path.exists(vectorstore_path): raise FileNotFoundError(
                f"FAISS store not found for '{agent_name}'.")
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        elif db_type == 'chroma':
            if not os.path.exists(vectorstore_path): raise FileNotFoundError(
                f"Chroma store not found for '{agent_name}'.")
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=embeddings,
                collection_metadata={'hnsw:space': 'cosine'},  # Example metadata
                # collection_metadata_id_key is not needed if Document.id is consistently set
            )
        elif db_type == 'qdrant':
            import qdrant_client
            client = qdrant_client.QdrantClient(host=vs_config.qdrant_host, port=vs_config.qdrant_port,
                                                path=None if vs_config.qdrant_host else vectorstore_path)
            vectorstore = Qdrant(
                client=client,
                collection_name=agent_name,
                embeddings=embeddings,
                # id_key is not needed if Document.id is consistently set
                content_payload_key='page_content'  # Still useful to specify content key
            )
        elif db_type == 'pinecone':
            pc_api_key = get_api_key("pinecone")
            pinecone = PineconeClient(api_key=pc_api_key)
            index = pinecone.Index(vs_config.pinecone_index_name)
            # PineconeLangChain uses Document.id by default.
            vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        elif db_type == 'mongodb_atlas':
            conn_string = get_api_key("mongodb")
            from pymongo import MongoClient
            client = MongoClient(conn_string)
            db_name, collection_name = vs_config.mongodb_db_name, vs_config.mongodb_collection_name
            collection = client[db_name][collection_name]
            # MongoDBAtlasVectorSearch relies on Document.metadata['_id'] or generates one.
            # This is handled by embed.py setting chunk.metadata['_id'] = chunk.id
            vectorstore = MongoDBAtlasVectorSearch(collection=collection, embedding=embeddings,
                                                   index_name=vs_config.mongodb_index_name)
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")

        logger.info(f"Successfully loaded vector store for agent '{agent_name}'.")

        # --- Hybrid Search & Re-ranking Setup ---
        # _get_all_documents_from_vectorstore no longer needs id_key parameter
        all_docs = _get_all_documents_from_vectorstore(vectorstore)
        if not all_docs:
            raise ValueError("No documents in vector store.")

        bm25_retriever = BM25Retriever.from_documents(all_docs)

        # .as_retriever() will now implicitly use Document.id for retrieval.
        semantic_retriever = vectorstore.as_retriever()

        bm25_k = vs_config.bm25_k
        sem_k = vs_config.semantic_k
        top_n = vs_config.rerank_top_n

        bm25_retriever.k = bm25_k
        semantic_retriever.search_kwargs = {"k": sem_k}

        if strategy == 'enhanced':
            logger.info("Applying 'enhanced' retrieval strategy (with re-ranking).")
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5])

            model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
            compressor = CrossEncoderReranker(model=model, top_n=top_n)
            final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever
            )
        else:  # Default to 'hybrid'
            logger.info("Using 'hybrid' retrieval strategy (BM25 + semantic).")
            final_retriever = EnsembleRetriever(retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5])

        logger.info(f"Retriever for '{agent_name}' configured successfully.")

        return Tool(
            name="retriever",
            func=lambda q: tool_fn(final_retriever, q),
            description=(
                f"Searches the knowledge base of '{agent_name}' "
                f"using strategy='{strategy}', bm25_k={bm25_k}, sem_k={sem_k}, rerank_top_n={top_n}."
            )
        )

    except Exception as e:
        logger.error(f"Failed to initialize retriever tool for agent '{agent_name}': {e}", exc_info=True)

        def error_func(input_str: str) -> str:
            return f"Error: The retriever tool for '{agent_name}' could not be loaded. Details: {e}"

        return Tool(name="retriever_error", func=error_func,
                    description="Reports an error during retriever initialization.")