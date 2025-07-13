import os
import logging
import json
from typing import List, Dict, Union

# LangChain components for retrieval and vector stores
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS, Chroma, Qdrant, Pinecone as PineconeVectorStore, \
    MongoDBAtlasVectorSearch
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


def _load_bm25_documents(agent_name: str) -> List[Document]:
    """
    Loads documents for BM25 from a dedicated file, making it scalable.
    This avoids loading the entire vector store into memory.
    """
    docs_path = f"vectorstore/{agent_name}/bm25_documents.jsonl"
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"BM25 source file not found at {docs_path}. Please deploy the agent again.")

    docs = []
    with open(docs_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Recreate Document objects with the necessary 'id' and content.
            # The 'id' attribute is crucial for the evaluation framework to work correctly.
            docs.append(Document(id=data.get("id"), page_content=data.get("page_content", "")))
    return docs


def tool_fn(retriever, query: Union[str, Dict[str, str]]) -> List[Document]:
    """The actual function that the tool will execute."""
    if isinstance(query, dict):
        search_query = query.get("input", "")
    else:
        search_query = query
    return retriever.invoke(search_query)


def get_retriever_tool(agent_config: AgentConfig) -> Tool:
    """
    Creates a retriever tool with a scalable BM25 implementation.
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
        # Add other vector store loading logic here (Chroma, Qdrant, etc.)
        else:
            raise ValueError(f"Unsupported vector store type: {db_type}")

        logger.info(f"Successfully loaded vector store for agent '{agent_name}'.")

        # --- Scalable BM25 Initialization ---
        logger.info(f"Initializing BM25 retriever from scalable source file.")
        bm25_docs = _load_bm25_documents(agent_name)
        if not bm25_docs:
            raise ValueError("No documents found for BM25 retriever. The source file might be empty.")

        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = vs_config.bm25_k

        # --- Ensemble and Re-ranking Setup ---
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": vs_config.semantic_k})

        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5])

        if strategy == 'enhanced':
            logger.info("Applying 'enhanced' retrieval strategy (with re-ranking).")
            model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
            compressor = CrossEncoderReranker(model=model, top_n=vs_config.rerank_top_n)
            final_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                             base_retriever=ensemble_retriever)
        else:  # 'hybrid'
            final_retriever = ensemble_retriever

        logger.info(f"Retriever for '{agent_name}' configured successfully.")

        return Tool(name="retriever", func=lambda q: tool_fn(final_retriever, q),
                    description=f"Searches the knowledge base of '{agent_name}'.")

    except Exception as e:
        logger.error(f"Failed to initialize retriever tool for agent '{agent_name}': {e}", exc_info=True)

        def error_func(input_str: str) -> str:
            return f"Error: The retriever tool for '{agent_name}' could not be loaded. Details: {e}"

        return Tool(name="retriever_error", func=error_func,
                    description="Reports an error during retriever initialization.")