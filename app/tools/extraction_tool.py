import logging
import uuid
import json
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.utils.function_calling import tool_example_to_messages
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama

from app.core.config import get_api_key
from app.schemas.agent import AgentConfig

logger = logging.getLogger(__name__)


def get_extraction_tool(agent_config: AgentConfig) -> Tool:
    schema_definition = agent_config.extraction_schema
    if not schema_definition:
        raise ValueError("`extraction_schema` must be defined in the agent config to use the extractor tool.")

    try:
        fields = {
            name: (Optional[str], Field(None, description=desc))
            for name, desc in schema_definition.items()
        }
        EntityModel = create_model("EntityModel", **fields)

        class WrapperModel(BaseModel):
            """Extracted entity wrapper"""
            data: EntityModel

    except Exception as e:
        logger.error(f"Failed to create Pydantic model from schema: {e}")
        raise e

    logger.info(f"Successfully created extraction schema for agent '{agent_config.name}'.")

    # Define few-shot examples (static or dynamically sampled)
    EXAMPLES: List[tuple[str, Any]] = agent_config.extraction_examples or []

    example_messages = []
    for txt, tool_obj in EXAMPLES:
        ai_response = "Detected information." if tool_obj else "No relevant data found."
        example_messages.extend(tool_example_to_messages(txt, [tool_obj], ai_response=ai_response))

    # Prompt template with example placeholder
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute, return null.",
            ),
            MessagesPlaceholder("examples"),
            ("human", "{text_to_extract_from}"),
        ]
    )

    model_name = agent_config.llm_model
    model_kwargs = agent_config.model_params.model_dump(exclude_unset=True) if agent_config.model_params else {}

    if model_name.startswith("ollama/"):
        ollama_model_name = model_name.split("/", 1)[1]
        llm = ChatOllama(model=ollama_model_name, **model_kwargs)
    else:
        provider = "openai"
        if "claude" in model_name.lower(): provider = "anthropic"
        elif "gemini" in model_name.lower(): provider = "google_genai"
        api_key = get_api_key(provider)
        llm = init_chat_model(model_name, model_provider=provider, api_key=api_key, **model_kwargs)

    structured_llm = llm.with_structured_output(schema=WrapperModel, method="function_calling")
    chain = prompt | structured_llm

    def extraction_and_formatting_function(text_to_extract_from: str) -> str:
        try:
            result = chain.invoke({
                "text_to_extract_from": text_to_extract_from,
                "examples": example_messages,
            })

            data = result.data.model_dump() if hasattr(result, "data") else {}

            section_title = agent_config.extraction_section_title or f"{agent_config.display_name} Extraction"
            output_lines = [f"## {section_title}"]
            for key, value in data.items():
                if value:
                    formatted_key = key.replace('_', ' ').title()
                    output_lines.append(f"- **{formatted_key}:** {value}")

            final_output = "\n".join(output_lines)
            logger.info("Formatted extraction output:\n%s", final_output)
            return final_output

        except Exception as e:
            logger.error("Extraction failed: %s", str(e))
            return "## Invoice Details\n- **Error:** Failed to extract data."

    return Tool(
        name="information_extractor",
        func=extraction_and_formatting_function,
        description="Extracts structured information from text and returns it as a formatted markdown string."
    )
