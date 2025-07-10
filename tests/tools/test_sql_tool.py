# In tests/tools/test_sql_tool.py

import pytest
from unittest.mock import MagicMock

# Import the function to be tested
from app.tools.sql_tool import create_sql_toolkit
# Import the classes that need to be mocked
from langchain_core.tools import BaseTool


def test_create_sql_toolkit(mocker):
    """
    GIVEN a database connection string
    WHEN the create_sql_toolkit function is called
    THEN it should initialize all dependencies correctly and return a list of tools.
    """
    # ARRANGE: Set up mocks for all external dependencies
    # ----------------------------------------------------

    # Mock the database engine creation to avoid a real connection
    mock_create_engine = mocker.patch("app.tools.sql_tool.create_engine")

    # Mock the SQLDatabase object that uses the engine
    mock_sql_db = mocker.patch("app.tools.sql_tool.SQLDatabase")

    # Mock the ChatOpenAI LLM to avoid a real API call
    mock_chat_openai = mocker.patch("app.tools.sql_tool.ChatOpenAI")

    # Create a mock tool instance and explicitly set its 'name' attribute
    mock_tool = MagicMock(spec=BaseTool)
    mock_tool.name = "mock_sql_tool"

    # Mock the SQLDatabaseToolkit itself to control its output
    mock_toolkit_instance = MagicMock()
    # Have its get_tools() method return a list containing our configured mock tool
    mock_toolkit_instance.get_tools.return_value = [mock_tool]

    mock_sql_toolkit = mocker.patch(
        "app.tools.sql_tool.SQLDatabaseToolkit",
        return_value=mock_toolkit_instance
    )

    # ACT: Run the function we want to test
    db_connection_string = "sqlite:///test.db"
    tools = create_sql_toolkit(db_connection_string)

    # ASSERT: Check if the outcome is what we expect
    # Check that the function returned a list containing our mock tool
    assert isinstance(tools, list)
    assert len(tools) == 1
    assert tools[0].name == "mock_sql_tool"

    # Verify that all the components were initialized correctly
    mock_create_engine.assert_called_once_with(db_connection_string)
    mock_sql_db.assert_called_once_with(mock_create_engine.return_value)
    mock_chat_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0)
    mock_sql_toolkit.assert_called_once_with(
        db=mock_sql_db.return_value,
        llm=mock_chat_openai.return_value
    )