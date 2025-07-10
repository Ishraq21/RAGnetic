# In tests/tools/test_arxiv_tool.py

import pytest
from unittest.mock import MagicMock

# Import the function to be tested
from app.tools.arxiv_tool import get_arxiv_tool

def test_get_arxiv_tool(mocker):
    """
    GIVEN no arguments
    WHEN the get_arxiv_tool function is called
    THEN it should return a list containing an instance of ArxivQueryRun.
    """
    # ARRANGE: Mock the ArxivQueryRun class to prevent any real initialization
    # or network calls. We replace the class with a simple MagicMock.
    mock_arxiv_class = mocker.patch("app.tools.arxiv_tool.ArxivQueryRun")
    mock_arxiv_class.return_value = MagicMock(name="ArxivQueryRun_instance")

    # ACT: Run the function we want to test
    tools = get_arxiv_tool()

    # ASSERT: Check if the outcome is what we expect
    # Check that the function returned a list with one item
    assert isinstance(tools, list)
    assert len(tools) == 1

    # Check that the ArxivQueryRun class was called to create an instance
    mock_arxiv_class.assert_called_once()

    # Check that the item in the list is the instance our mock created
    assert tools[0] == mock_arxiv_class.return_value