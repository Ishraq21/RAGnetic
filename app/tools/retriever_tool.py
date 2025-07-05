# app/tools/retriever_tool.py
import os
import logging
from typing import List, Dict, Union

# LangChain components for retrieval and vector stores
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS, Chroma
from langchain_qdrant import Qdrant
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_core.tools import Tool

# Pinecone client for initialization
from pinecone import Pinecone as PineconeClient

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
    # FAISS and Chroma have a similar, non-standard way to get all docs
    if isinstance(vectorstore, (FAISS, Chroma)):
        # Check if the docstore and index_to_docstore_id are available
        if hasattr(vectorstore, 'docstore') and hasattr(vectorstore, 'index_to_docstore_id'):
            return [vectorstore.docstore.search(doc_id) for doc_id in vectorstore.index_to_docstore_id.values()]

    # For other modern vector stores, the standard is to do a similarity search
    # with a high 'k' value to retrieve all documents.
    # We can use a simple query like "*" to fetch everything.
    # Note: This might be inefficient for extremely large databases.
    return vectorstore.similarity_search("*", k=10000)  # Assuming k is large enough


def tool_fn(retriever, query: Union[str, Dict[str, str]]) -> List[Document]:
    """The actual function that the tool will execute."""
    if isinstance(query, dict):
        search_query = query.get("input", "")
    else:
        search_query = query
    return retriever.invoke(search_query)


def get_retriever_tool(agent_config: AgentConfig) -> Tool:
    """
    Creates a universal HYBRID retriever tool for a given agent.

    This function reads the agent's configuration to determine which vector database
    to connect to (FAISS, Chroma, Qdrant, Pinecone, or MongoDB Atlas). It then
    initializes a hybrid search retriever combining keyword and semantic search.
    """
    agent_name = agent_config.name
    embedding_model_name = agent_config.embedding_model
    vs_config = agent_config.vector_store
    db_type = vs_config.type
    vectorstore_path = f"vectorstore/{agent_name}"  # For local DBs

    logger.info(f"Configuring HYBRID retriever for '{agent_name}' using vector store type: '{db_type}'.")

    try:
        vectorstore = None
        embeddings = get_embedding_model(embedding_model_name)

        # --- Vector Store Loading Dispatcher ---
        if db_type == 'faiss':
            if not os.path.exists(vectorstore_path): raise FileNotFoundError(
                f"FAISS store not found for agent '{agent_name}'.")
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

        elif db_type == 'chroma':
            if not os.path.exists(vectorstore_path): raise FileNotFoundError(
                f"Chroma store not found for agent '{agent_name}'.")
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)

        elif db_type == 'qdrant':
            import qdrant_client
            client = qdrant_client.QdrantClient(
                host=vs_config.qdrant_host,
                port=vs_config.qdrant_port,
                path=None if vs_config.qdrant_host else vectorstore_path  # Use path for local, host/port for remote
            )
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
            collection = client[vs_config.mongodb_db_name][vs_config.mongodb_collection_name]
            vectorstore = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=embeddings,
                index_name=vs_config.mongodb_index_name
            )
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")

        logger.info(f"Successfully loaded vector store for agent '{agent_name}'.")

        # --- Hybrid Search Setup ---
        all_docs = _get_all_documents_from_vectorstore(vectorstore)
        if not all_docs:
            raise ValueError("No documents found in the vector store. Cannot create a retriever.")

        # 1. Initialize Keyword Retriever (BM25)
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 5

        # 2. Initialize Semantic Retriever
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 3. Initialize the Ensemble Retriever to combine both
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5]
        )

        logger.info(f"Hybrid retriever for '{agent_name}' configured successfully.")

        return Tool(
            name="retriever",
            func=lambda q: tool_fn(ensemble_retriever, q),
            description=f"Searches the knowledge base of the '{agent_name}' agent using hybrid search for relevant information."
        )

    except Exception as e:
        logger.error(f"Failed to initialize retriever tool for agent '{agent_name}': {e}", exc_info=True)

        def error_func(input_str: str) -> str:
            return f"Error: The retriever tool for '{agent_name}' could not be loaded. Details: {e}"

        return Tool(
            name="retriever_error",
            func=error_func,
            description="Reports an error during retriever initialization."
        )