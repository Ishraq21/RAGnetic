# tests/test_agent_workflow.py

import pytest

from app.schemas.agent import AgentConfig
from app.agents.agent_graph import get_agent_workflow
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from unittest.mock import MagicMock, AsyncMock

@tool
def dummy_tool(query: str) -> str:
    """A dummy tool that returns a fixed string."""
    return "The result from the dummy tool."

@pytest.fixture
def sample_agent_config():
    """Provides a sample AgentConfig for the integration test."""
    return AgentConfig(
        name="test-workflow-agent",
        persona_prompt="Test persona",
        sources=[],
        tools=["retriever"],
        llm_model="gpt-4o-mini",
        # any extra fields will be ignored by Pydantic
    )

@pytest.mark.asyncio
async def test_agent_workflow_with_tool_call(sample_agent_config, mocker):
    """
    GIVEN a user query that should trigger a tool
    WHEN the agent graph is executed
    THEN it should follow the correct sequence: LLM -> Tool -> Final LLM response.
    """
    # --- Arrange: mock out the LLM and tools ---
    # 1) The first LLM turn requests a tool
    mock_tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "dummy_tool", "args": {"query": "test query"}, "id": "tool-call-123"}]
    )
    # 2) The second LLM turn returns the final answer
    mock_final_response = AIMessage(
        content="The tool said: The result from the dummy tool."
    )

    # Mock the bound model so .invoke() yields tool-call then final answer
    mock_bound_model = MagicMock()
    mock_bound_model.invoke.side_effect = [mock_tool_call, mock_final_response]

    # Mock init_chat_model to return a model whose .bind_tools() gives our bound model
    mock_chat_model = MagicMock()
    mock_chat_model.bind_tools.return_value = mock_bound_model
    mocker.patch("app.agents.agent_graph.init_chat_model", return_value=mock_chat_model)

    # Also stub out the retriever (even if not used here)
    mocker.patch("app.agents.agent_graph.get_retriever_tool", return_value=MagicMock())

    # --- Act: run through the graph ---
    agent_graph = get_agent_workflow(tools=[dummy_tool])
    runnable = agent_graph.compile()

    # seed initial state: messages + agent_config + request_id
    inputs = {
        "messages": [HumanMessage(content="Use the tool.")],
        "agent_config": sample_agent_config,
        "request_id": "test-request-id",
    }
    
    # Mock database session for test
    mock_db = AsyncMock()
    config = {
        "configurable": {
            "thread_id": "test-thread",
            "db_session": mock_db
        }
    }
    final_state = await runnable.ainvoke(inputs, config)

    # --- Assert: the last message is our final LLM response ---
    final_message = final_state["messages"][-1]
    assert isinstance(final_message, AIMessage)
    assert "The tool said: The result from the dummy tool." in final_message.content

    # And that our bound model was invoked exactly twice (tool-call + final)
    assert mock_bound_model.invoke.call_count == 2
