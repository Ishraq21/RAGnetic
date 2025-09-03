import pytest
from unittest.mock import MagicMock, patch

from app.schemas.agent import AgentConfig, VectorStoreConfig
from app.tools.retriever_tool import get_retriever_tool
from langchain_core.documents import Document

# ADD these imports to get the centralized path settings for the test
from app.core.config import get_path_settings
import os

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

@pytest.mark.asyncio
async def test_get_retriever_tool_faiss_hybrid(sample_agent_config, mocker):
    """
    GIVEN a valid agent configuration for a FAISS vector store and hybrid retrieval
    WHEN the get_retriever_tool function is called
    THEN it should initialize and return a valid retriever tool using the scalable method.
    """
    # ARRANGE: Set up mocks for all external dependencies
    mocker.patch("app.tools.retriever_tool.get_embedding_model", return_value=MagicMock())
    mocker.patch("app.tools.retriever_tool.os.path.exists", return_value=True) # Pretend files exist

    # Mock the FAISS loader
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()
    mocker.patch("app.tools.retriever_tool.FAISS.load_local", return_value=mock_vectorstore)

    # Mock the new, scalable document loader instead of the old function
    mock_documents = [Document(page_content="This is a test document.")]
    mock_load_bm25 = mocker.patch(
        "app.tools.retriever_tool._load_bm25_docs",
        return_value=mock_documents
    )

    # Mock the BM25Retriever
    mocker.patch("app.tools.retriever_tool.BM25Retriever.from_documents", return_value=MagicMock())

    # Mock the final EnsembleRetriever
    mock_ensemble_retriever = MagicMock()
    mock_ensemble_retriever.invoke = MagicMock()
    mocker.patch("app.tools.retriever_tool.EnsembleRetriever", return_value=mock_ensemble_retriever)

    # Mock HuggingFaceCrossEncoder (needed for 'enhanced' strategy, though not directly tested here, good practice)
    mocker.patch("app.tools.retriever_tool.HuggingFaceCrossEncoder", return_value=MagicMock())


    # ACT: Run the function we want to test
    tool = await get_retriever_tool(
        agent_config=sample_agent_config, 
        user_id=1, 
        thread_id="test-thread"
    )

    # ASSERT: Check if the outcome is what we expect
    assert tool is not None
    assert tool.name == "document_retriever"
    assert "Useful for retrieving relevant documents from the knowledge base" in tool.description

    # MODIFIED: Calculate the expected absolute path using the same logic as the app
    _APP_PATHS_TEST = get_path_settings()
    _VECTORSTORE_DIR_TEST = _APP_PATHS_TEST["VECTORSTORE_DIR"]
    
    # Check that our new scalable loader was called with the expected path
    expected_docs_path = os.path.join(_VECTORSTORE_DIR_TEST, "test-retriever-agent", "bm25_documents.jsonl")
    mock_load_bm25.assert_called_once_with(expected_docs_path)
    expected_absolute_vectorstore_path = os.path.join(_VECTORSTORE_DIR_TEST, sample_agent_config.name)


    # Check that the FAISS loader was called correctly with the absolute path
    from app.tools.retriever_tool import FAISS
    FAISS.load_local.assert_called_with(
        expected_absolute_vectorstore_path, # Use the calculated absolute path
        mocker.ANY,
        allow_dangerous_deserialization=True
    )

    # The EnsembleRetriever is created dynamically during tool invocation, not during initialization
    # So we don't assert its creation here