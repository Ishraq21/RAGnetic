import logging
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from app.core.embed_config import get_embedding_model
from app.agents.config_manager import load_agent_config

logger = logging.getLogger(__name__)

def get_retriever_tool(agent_name: str, description: str = None) -> Tool:
    """
    Creates a retriever tool for a given agent by dynamically selecting
    the embedding model specified in the agent's configuration.
    """
    try:
        # Load the agent's config to find out which embedding model to use
        agent_config = load_agent_config(agent_name)
        embedding_model_name = agent_config.embedding_model
        logger.info(f"Configuring retriever for '{agent_name}' with embedding model '{embedding_model_name}'.")

        # Use the factory to get the correct embedding model instance
        embeddings = get_embedding_model(embedding_model_name)

        # Load the vector store with the correct embeddings
        vectordb = FAISS.load_local(
            f"vectorstore/{agent_name}",
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Configure the retriever to search all documents
        total_docs = vectordb.index.ntotal
        retriever = vectordb.as_retriever(search_kwargs={'k': total_docs})
        logger.info(f"Retriever for '{agent_name}' configured with k={total_docs} documents.")

    except Exception as e:
        logger.error(f"Failed to initialize retriever tool for agent '{agent_name}': {e}", exc_info=True)
        # Create a dummy tool that returns an error if initialization fails
        def error_fn(input_dict: dict):
            return f"Error: The retriever tool for '{agent_name}' could not be loaded. It may not be deployed correctly."
        return Tool(
            name=f"{agent_name}_retriever_error",
            description="Returns an error message indicating the retriever failed to load.",
            func=error_fn,
        )

    def tool_fn(input_dict: dict):
        if "input" not in input_dict:
            raise ValueError(f"Tool input must include 'input' key. Got: {input_dict}")
        query = input_dict["input"]
        return retriever.invoke(query)

    return Tool(
        name=f"{agent_name}_retriever",
        description=description or f"Search the embedded knowledge base for {agent_name}.",
        func=tool_fn,
    )
