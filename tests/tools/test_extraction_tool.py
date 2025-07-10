# In tests/tools/test_extraction_tool.py

import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel

from app.schemas.agent import AgentConfig
from app.tools.extraction_tool import get_extraction_tool


# 1. This mock now simulates the nested Pydantic model structure
#    that your tool's code actually expects from the LLM chain.
class MockEntity(BaseModel):
    invoice_id: str
    vendor_name: str

class MockWrapper(BaseModel):
    data: MockEntity

    # The formatting function calls .data.dict() on the result.
    # We need to simulate this structure correctly.
    def dict(self):
        return self.data.dict()



@pytest.fixture
def sample_agent_config():
    """Provides a sample AgentConfig for testing the extractor tool."""
    return AgentConfig(
        name="test-extractor-agent",
        persona_prompt="Test persona",
        sources=[],
        tools=["extractor"],
        llm_model="gpt-4o-mini",
        extraction_schema={
            "invoice_id": "The invoice ID",
            "vendor_name": "The vendor's name",
        },
        extraction_examples=[]  # Important: This field must exist
    )


def test_get_extraction_tool_initialization(sample_agent_config):
    """
    GIVEN a valid agent configuration
    WHEN get_extraction_tool is called
    THEN it should return a valid Tool instance without errors.
    """
    tool = get_extraction_tool(sample_agent_config)
    assert tool is not None
    assert tool.name == "information_extractor"


def test_extraction_function_formats_output(sample_agent_config, mocker):
    """
    GIVEN the extraction tool's function is executed
    WHEN the (mocked) underlying LLM chain returns data
    THEN the function should return a correctly formatted markdown string.
    """
    # ARRANGE: Define the final data we want our mock LLM to return
    mock_llm_output = MockWrapper(
        data=MockEntity(invoice_id="INV-999", vendor_name="MockCorp")
    )

    # We mock the final `chain` object that is created inside the function.
    # This mock will have an `invoke` method that returns our predefined data.
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_llm_output

    # This is the key: We mock the ChatPromptTemplate's `from_messages` method.
    # We configure it to return an object that, when the `|` (__or__) operator is used,
    # returns our `mock_chain`. This bypasses the entire LLM initialization.
    mocker.patch(
        "app.tools.extraction_tool.ChatPromptTemplate.from_messages",
        return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
    )

    # ACT: Get the tool and run its internal function.
    tool = get_extraction_tool(sample_agent_config)
    result = tool.func("some dummy text to extract from")

    # ASSERT: Verify that the result is a string and contains the formatted data.
    assert isinstance(result, str)
    assert "## Invoice Details" in result
    assert "- **Invoice Id:** INV-999" in result
    assert "- **Vendor Name:** MockCorp" in result