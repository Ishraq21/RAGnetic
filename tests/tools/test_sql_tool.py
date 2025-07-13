import pytest
from unittest.mock import MagicMock

# Import the function to be tested
from app.tools.sql_tool import create_sql_toolkit
# Import the classes that need to be mocked or used for argument types
from langchain_core.tools import BaseTool
from app.core.config import get_llm_model # NEW: Import get_llm_model
from app.schemas.agent import ModelParams # NEW: Import ModelParams for dummy arguments


def test_create_sql_toolkit(mocker):
    """
    GIVEN a database connection string and LLM parameters
    WHEN the create_sql_toolkit function is called
    THEN it should initialize all dependencies correctly and return a list of tools.
    """
    # ARRANGE: Set up mocks for all external dependencies
    # ----------------------------------------------------

    # Mock the database engine creation to avoid a real connection
    mock_create_engine = mocker.patch("app.tools.sql_tool.create_engine")

    # Mock the SQLDatabase object that uses the engine
    mock_sql_db = mocker.patch("app.tools.sql_tool.SQLDatabase")

    # Mock the get_llm_model function that create_sql_toolkit now calls
    mock_llm_model = MagicMock() # This will be the mocked LLM instance
    mock_get_llm_model = mocker.patch(
        "app.tools.sql_tool.get_llm_model",
        return_value=mock_llm_model # make it return our mock LLM
    )

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
    dummy_llm_model_name = "gpt-4o-mini-test" # NEW: Dummy LLM name
    dummy_llm_model_params = ModelParams(temperature=0.0) # NEW: Dummy ModelParams

    # MODIFIED: Pass the new arguments to create_sql_toolkit
    tools = create_sql_toolkit(
        db_connection_string=db_connection_string,
        llm_model_name=dummy_llm_model_name,
        llm_model_params=dummy_llm_model_params
    )

    # ASSERT: Check if the outcome is what we expect
    # Check that the function returned a list containing our mock tool
    assert isinstance(tools, list)
    assert len(tools) == 1
    assert tools[0].name == "mock_sql_tool"

    # Verify that all the components were initialized correctly
    mock_create_engine.assert_called_once_with(db_connection_string)
    mock_sql_db.assert_called_once_with(mock_create_engine.return_value)

    # MODIFIED: Assert that get_llm_model was called correctly
    mock_get_llm_model.assert_called_once_with(
        model_name=dummy_llm_model_name,
        model_params=dummy_llm_model_params
    )

    mock_sql_toolkit.assert_called_once_with(
        db=mock_sql_db.return_value,
        llm=mock_llm_model
    )