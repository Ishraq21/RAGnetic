# In tests/test_agent_workflow.py

import pytest
from unittest.mock import MagicMock
from typing import List

from app.schemas.agent import AgentConfig
from app.agents.agent_graph import get_agent_workflow
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


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
        extraction_examples=[]
    )


def test_agent_workflow_with_tool_call(sample_agent_config, mocker):
    """
    GIVEN a user query that should trigger a tool
    WHEN the agent graph is executed
    THEN it should follow the correct sequence: LLM -> Tool -> Final LLM response.
    """
    # ARRANGE: Set up the mocks for the LLM and the tools

    mock_tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "dummy_tool", "args": {"query": "test query"}, "id": "tool-call-123"}]
    )
    mock_final_response = AIMessage(content="The tool said: The result from the dummy tool.")


    # Create a more sophisticated mock that simulates the .bind_tools() method

    # 1. This is the mock for the final, tool-bound model.
    #    Its invoke method will have the side_effect we want.
    mock_bound_model = MagicMock()
    mock_bound_model.invoke.side_effect = [mock_tool_call, mock_final_response]

    # 2. This is the mock for the initial chat model.
    #    Its bind_tools() method will return our mock_bound_model.
    mock_chat_model = MagicMock()
    mock_chat_model.bind_tools.return_value = mock_bound_model

    # 3. Patch init_chat_model to return our initial mock_chat_model.
    mocker.patch("app.agents.agent_graph.init_chat_model", return_value=mock_chat_model)


    # We also need to mock the retriever tool, as the graph tries to initialize it
    # even if it's not used in this specific test path.
    mocker.patch("app.agents.agent_graph.get_retriever_tool", return_value=MagicMock())


    # ACT: Run a query through the full agent workflow
    agent_graph = get_agent_workflow(tools=[dummy_tool])
    runnable_agent = agent_graph.compile()

    inputs = {"messages": [HumanMessage(content="Use the tool.")]}
    config = {"configurable": {"thread_id": "test-thread", "agent_config": sample_agent_config}}

    final_state = runnable_agent.invoke(inputs, config)

    # ASSERT: Check if the workflow executed correctly
    final_message = final_state["messages"][-1]
    assert isinstance(final_message, AIMessage)
    assert "The tool said: The result from the dummy tool." in final_message.content

    # The final bound model's invoke method should have been called twice.
    assert mock_bound_model.invoke.call_count == 2