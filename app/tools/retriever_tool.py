# app/tools/retriever_tool.py
import os
import logging
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
    """
    if isinstance(vectorstore, (FAISS, Chroma)):
        if hasattr(vectorstore, 'docstore') and hasattr(vectorstore, 'index_to_docstore_id'):
            return [vectorstore.docstore.search(doc_id) for doc_id in vectorstore.index_to_docstore_id.values()]
    return vectorstore.similarity_search("*", k=10000)


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
    """
    agent_name = agent_config.name
    vs_config = agent_config.vector_store
    strategy = vs_config.retrieval_strategy

    logger.info(f"Configuring retriever for '{agent_name}' with strategy: '{strategy}'.")

    try:
        # --- Vector Store Loading ---
        db_type = vs_config.type
        vectorstore_path = f"vectorstore/{agent_name}"
        embeddings = get_embedding_model(agent_config.embedding_model)
        vectorstore = None

        logger.info(f"Loading vector store of type '{db_type}' for agent '{agent_name}'.")

        if db_type == 'faiss':
            if not os.path.exists(vectorstore_path): raise FileNotFoundError(
                f"FAISS store not found for '{agent_name}'.")
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        elif db_type == 'chroma':
            if not os.path.exists(vectorstore_path): raise FileNotFoundError(
                f"Chroma store not found for '{agent_name}'.")
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
        elif db_type == 'qdrant':
            import qdrant_client
            client = qdrant_client.QdrantClient(host=vs_config.qdrant_host, port=vs_config.qdrant_port,
                                                path=None if vs_config.qdrant_host else vectorstore_path)
            vectorstore = Qdrant(client, agent_name, embeddings)
        elif db_type == 'pinecone':
            pc_api_key = get_api_key("pinecone")
            pinecone = PineconeClient(api_key=pc_api_key)
            index = pinecone.Index(vs_config.pinecone_index_name)
            vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        elif db_type == 'mongodb_atlas':
            conn_string = get_api_key("mongodb")
            from pymongo import MongoClient
            client = MongoClient(conn_string)
            db_name, collection_name = vs_config.mongodb_db_name, vs_config.mongodb_collection_name
            collection = client[db_name][collection_name]
            vectorstore = MongoDBAtlasVectorSearch(collection=collection, embedding=embeddings,
                                                   index_name=vs_config.mongodb_index_name)
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")

        logger.info(f"Successfully loaded vector store for agent '{agent_name}'.")

        # --- Hybrid Search & Re-ranking Setup ---
        all_docs = _get_all_documents_from_vectorstore(vectorstore)
        if not all_docs:
            raise ValueError("No documents in vector store.")

        bm25_retriever = BM25Retriever.from_documents(all_docs)
        semantic_retriever = vectorstore.as_retriever()

        # Set the 'k' value on the individual retrievers before creating the ensemble.
        if strategy == 'enhanced':
            logger.info("Applying 'enhanced' retrieval strategy (with re-ranking).")
            # For enhanced search, retrieve more documents initially to give the re-ranker more options.
            bm25_retriever.k = 10
            semantic_retriever.search_kwargs = {"k": 10}

            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5])

            model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
            compressor = CrossEncoderReranker(model=model, top_n=5)
            final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever
            )
        else:  # Default to 'hybrid'
            logger.info("Using 'hybrid' retrieval strategy (BM25 + semantic).")
            # For standard hybrid search, retrieve fewer documents.
            bm25_retriever.k = 5
            semantic_retriever.search_kwargs = {"k": 5}
            final_retriever = EnsembleRetriever(retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5])

        logger.info(f"Retriever for '{agent_name}' configured successfully.")

        return Tool(
            name="retriever",
            func=lambda q: tool_fn(final_retriever, q),
            description=f"Searches the knowledge base of '{agent_name}' using its configured retrieval strategy."
        )

    except Exception as e:
        logger.error(f"Failed to initialize retriever tool for agent '{agent_name}': {e}", exc_info=True)

        def error_func(input_str: str) -> str:
            return f"Error: The retriever tool for '{agent_name}' could not be loaded. Details: {e}"

        return Tool(name="retriever_error", func=error_func,
                    description="Reports an error during retriever initialization.")