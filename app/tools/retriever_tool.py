# app/tools/retriever_tool.py
import os
import logging
from typing import List, Dict, Union
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.tools import Tool

from app.core.embed_config import get_embedding_model

logger = logging.getLogger(__name__)

def tool_fn(retriever, query: Union[str, Dict[str, str]]) -> List[Document]:
    """The actual function that the tool will execute."""
    if isinstance(query, dict):
        search_query = query.get("input", "")
    else:
        search_query = query
    return retriever.invoke(search_query)


def get_retriever_tool(agent_name: str, embedding_model_name: str) -> Tool:
    """
    Creates a HYBRID retriever tool for a given agent.

    This tool now combines keyword-based search (BM25) with semantic search (FAISS)
    to provide more accurate and relevant document retrieval.
    """
    logger.info(f"Configuring HYBRID retriever for '{agent_name}'.")

    try:
        # Load the raw documents from the agent's vector store.
        # This is a prerequisite for initializing the BM25 retriever.
        vectorstore_path = f"vectorstore/{agent_name}"
        if not os.path.exists(vectorstore_path):
            raise FileNotFoundError("Vector store not found. Please deploy the agent first.")

        embeddings = get_embedding_model(embedding_model_name)
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

        # We need the original documents to initialize the keyword retriever
        # The FAISS index has a convenient way to access them
        docstore = vectorstore.docstore
        all_docs = [docstore.search(doc_id) for doc_id in vectorstore.index_to_docstore_id.values()]

        if not all_docs:
            raise ValueError("No documents found in the vector store. Cannot create a retriever.")

        # 1. Initialize the Keyword Retriever (BM25)
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 5  # Retrieve top 5 results based on keywords

        # 2. Initialize the Semantic Retriever (FAISS)
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 3. Initialize the Ensemble Retriever to combine both
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5]  # Give equal weight to both keyword and semantic results
        )

        logger.info(f"Hybrid retriever for '{agent_name}' configured successfully.")

        # The tool's function now gets the retriever passed to it
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