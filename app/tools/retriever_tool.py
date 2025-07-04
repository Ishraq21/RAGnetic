import logging
from typing import List, Dict, Union

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.tools import Tool

from app.core.embed_config import get_embedding_model
from app.core.config import get_api_key

logger = logging.getLogger(__name__)


def get_retriever_tool(agent_name: str, embedding_model_name: str) -> Tool:
    """
    Creates a retriever tool for a given agent.

    This tool is responsible for searching the agent's knowledge base (a FAISS vector store)
    to find documents relevant to the user's query.

    Args:
        agent_name: The name of the agent, used to locate the correct vector store.
        embedding_model_name: The name of the embedding model to use.

    Returns:
        A LangChain Tool object that can be used within an agent workflow.
    """
    logger.info(f"Configuring retriever for '{agent_name}' with embedding model '{embedding_model_name}'.")

    try:
        # Initialize the embedding model using the centralized factory function
        embeddings = get_embedding_model(embedding_model_name)

        # Load the vector store from the agent's directory
        vectorstore_path = f"vectorstore/{agent_name}"
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

        # Configure the retriever to return a specific number of documents
        # Here, we set k to a dynamic value, but you can adjust as needed.
        k = len(vectorstore.index_to_docstore_id)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        logger.info(f"Retriever for '{agent_name}' configured with k={k} documents.")

    except Exception as e:
        logger.error(f"Failed to initialize retriever tool for agent '{agent_name}': {e}", exc_info=True)

        # Return a dummy tool that just reports the error
        def error_func(input_str: str) -> str:
            return f"Error: The retriever tool for '{agent_name}' could not be loaded. It may not be deployed correctly. Details: {e}"

        return Tool(
            name="retriever_error",
            func=error_func,
            description="Reports an error during retriever initialization."
        )

    # ** FIX IS HERE: This function now handles both string and dict inputs **
    def tool_fn(input_data: Union[str, Dict[str, str]]) -> List[Document]:
        """The actual function that the tool will execute."""
        if isinstance(input_data, dict):
            # If input is a dictionary, extract the value from the 'input' key
            query = input_data.get("input")
            if query is None:
                # Handle cases where the dictionary doesn't have the expected key
                raise ValueError(f"Input dictionary must include 'input' key. Got: {input_data}")
        elif isinstance(input_data, str):
            # If input is already a string, use it directly
            query = input_data
        else:
            raise TypeError(f"Tool input must be a string or a dictionary, but got {type(input_data)}")

        return retriever.invoke(query)

    return Tool(
        name="retriever",
        func=tool_fn,
        description=f"Searches the knowledge base of the '{agent_name}' agent for information relevant to the user's query."
    )