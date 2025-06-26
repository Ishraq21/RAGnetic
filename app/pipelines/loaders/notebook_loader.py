import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import NotebookLoader

logger = logging.getLogger(__name__)

def load_notebook(path: str) -> List[Document]:
    """
    Loads a Jupyter Notebook (.ipynb) file from the given path.

    This function utilizes LangChain's NotebookLoader, which intelligently
    parses the notebook, extracting content from both markdown and code cells.
    This ensures that both explanatory text and the code itself are included.

    Args:
        path: The local file path to the .ipynb file.

    Returns:
        A list of Document objects, where each document represents a cell
        from the notebook. Returns an empty list if loading fails.
    """
    logger.info(f"Attempting to load Jupyter Notebook from path: {path}")
    try:
        # NotebookLoader handles the parsing of the .ipynb file format.
        loader = NotebookLoader(
            path,
            include_outputs=True,  # Include cell outputs in the document content
            max_output_length=100, # Limit the length of cell outputs
            remove_newline=True    # Clean up excess whitespace
        )
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} cells from notebook: {path}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load notebook {path}. Error: {e}", exc_info=True)
        # Return an empty list to prevent the pipeline from crashing
        return []

