import pytest
from unittest.mock import MagicMock, patch

from app.schemas.agent import AgentConfig, VectorStoreConfig
from app.tools.retriever_tool import get_retriever_tool
from langchain_core.documents import Document

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

def test_get_retriever_tool_faiss_hybrid(sample_agent_config, mocker):
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

    # --- FIX: Mock the new, scalable document loader instead of the old function ---
    mock_documents = [Document(page_content="This is a test document.")]
    mocker.patch(
        "app.tools.retriever_tool._load_bm25_documents",
        return_value=mock_documents
    )

    # Mock the BM25Retriever
    mocker.patch("app.tools.retriever_tool.BM25Retriever.from_documents", return_value=MagicMock())

    # Mock the final EnsembleRetriever
    mock_ensemble_retriever = MagicMock()
    # The tool now calls .invoke(), so we need to mock that.
    mock_ensemble_retriever.invoke = MagicMock()
    mocker.patch("app.tools.retriever_tool.EnsembleRetriever", return_value=mock_ensemble_retriever)

    # ACT: Run the function we want to test
    tool = get_retriever_tool(sample_agent_config)

    # ASSERT: Check if the outcome is what we expect
    assert tool is not None
    assert tool.name == "retriever"
    assert "Searches the knowledge base" in tool.description

    # Check that our new scalable loader was called
    from app.tools.retriever_tool import _load_bm25_documents
    _load_bm25_documents.assert_called_once_with("test-retriever-agent")

    # Check that the FAISS loader was called correctly
    from app.tools.retriever_tool import FAISS
    FAISS.load_local.assert_called_with("vectorstore/test-retriever-agent", mocker.ANY, allow_dangerous_deserialization=True)

    # Check that the EnsembleRetriever was initialized
    from app.tools.retriever_tool import EnsembleRetriever
    EnsembleRetriever.assert_called_once()