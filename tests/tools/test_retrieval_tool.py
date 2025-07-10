# In tests/tools/test_retriever_tool.py

import pytest
from unittest.mock import MagicMock, patch

from app.schemas.agent import AgentConfig, VectorStoreConfig
from app.tools.retriever_tool import get_retriever_tool
from langchain_core.documents import Document

# 1. Use a pytest "fixture" to create a reusable AgentConfig for our tests.
@pytest.fixture
def sample_agent_config():
    """Provides a sample AgentConfig for testing the retriever tool with FAISS."""
    return AgentConfig(
        name="test-retriever-agent",
        persona_prompt="Test persona",
        sources=[],
        tools=["retriever"],
        vector_store=VectorStoreConfig(type="faiss", retrieval_strategy="hybrid")
    )

# 2. Write the test function.
def test_get_retriever_tool_faiss_hybrid(sample_agent_config, mocker):
    """
    GIVEN a valid agent configuration for a FAISS vector store and hybrid retrieval
    WHEN the get_retriever_tool function is called
    THEN it should initialize and return a valid retriever tool.
    """
    # ARRANGE: Set up mocks for all external dependencies
    # Mock the embedding model loader to avoid a real model download
    mocker.patch("app.tools.retriever_tool.get_embedding_model", return_value=MagicMock())

    # Mock the file system check to pretend the vector store exists
    mocker.patch("app.tools.retriever_tool.os.path.exists", return_value=True)

    # Mock the FAISS loader to return a fake vector store object
    mock_vectorstore = MagicMock()
    # The tool calls `as_retriever()` on the vector store, so our mock needs it.
    mock_vectorstore.as_retriever.return_value = MagicMock()
    mocker.patch("app.tools.retriever_tool.FAISS.load_local", return_value=mock_vectorstore)

    # Mock the helper function that gets documents from the store, returning a fake document
    mock_documents = [Document(page_content="This is a test document.")]
    mocker.patch(
        "app.tools.retriever_tool._get_all_documents_from_vectorstore",
        return_value=mock_documents
    )

    # Mock the BM25Retriever since it's a key part of the hybrid strategy
    mocker.patch("app.tools.retriever_tool.BM25Retriever.from_documents", return_value=MagicMock())

    # Mock the final EnsembleRetriever
    mock_ensemble_retriever = MagicMock()
    mocker.patch("app.tools.retriever_tool.EnsembleRetriever", return_value=mock_ensemble_retriever)


    # ACT: Run the function we want to test
    tool = get_retriever_tool(sample_agent_config)


    # ASSERT: Check if the outcome is what we expect
    # Check that a tool was created
    assert tool is not None
    assert tool.name == "retriever"
    assert "Searches the knowledge base" in tool.description

    # Check that the FAISS loader was called with the correct path
    # This proves our logic for constructing the path is working.
    # We use .assert_called_with() to check the arguments.
    from app.tools.retriever_tool import FAISS
    FAISS.load_local.assert_called_with("vectorstore/test-retriever-agent", mocker.ANY, allow_dangerous_deserialization=True)

    # Check that the EnsembleRetriever was initialized, confirming the hybrid strategy was used
    from app.tools.retriever_tool import EnsembleRetriever
    EnsembleRetriever.assert_called_once()