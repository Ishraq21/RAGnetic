import os
import logging
from typing import List, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

# LangChain's generic chat model initializer
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

from langchain_ollama import ChatOllama

# Local application imports
from app.tools.retriever_tool import get_retriever_tool
from app.core.config import get_api_key
from ollama import ResponseError

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    tool_calls: List[dict]


def call_model(state: AgentState, config: RunnableConfig):
    """
    This is the core reasoning step of the agent. It is responsible for:
    1. Retrieving context using the RAG retriever.
    2. Constructing a detailed prompt with the retrieved context.
    3. Invoking the language model with robust error handling.
    """
    try:
        # Configuration and State Validation
        logger.info("Agent 'call_model' node executing...")
        agent_config = config['configurable']['agent_config']
        tools = config['configurable'].get('tools', [])
        messages = state['messages']

        if not messages:
            logger.error("State has no messages. Cannot proceed.")
            return {"messages": [AIMessage(
                content="I'm sorry, but there was an error processing your request as the message history is empty.")]}

        query = messages[-1].content
        model_name = agent_config.llm_model
        logger.info(f"Processing query for agent '{agent_config.name}' using model '{model_name}'.")

        # RAG Retrieval
        retrieved_docs_str = ""
        if "retriever" in agent_config.tools:
            try:
                logger.info(f"Attempting to retrieve documents for query: '{query[:80]}...'")
                retriever_tool = get_retriever_tool(agent_config)
                retrieved_docs = retriever_tool.invoke({"input": query})

                if isinstance(retrieved_docs, str):
                    retrieved_docs_str = retrieved_docs
                    logger.error(f"Retriever tool returned an error string: {retrieved_docs}")
                elif retrieved_docs:
                    retrieved_docs_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    logger.info(f"Successfully retrieved {len(retrieved_docs)} documents.")
                else:
                    retrieved_docs_str = "No documents were found matching the query."
                    logger.warning("Retriever tool ran successfully but returned no documents.")
            except Exception as e:
                logger.error(f"Error during document retrieval: {e}", exc_info=True)
                retrieved_docs_str = f"An error occurred while trying to retrieve relevant documents: {e}"

        # Build the System Prompt
        context_section = f"<context>\n<retrieved_documents>\n{retrieved_docs_str}\n</retrieved_documents>\n</context>"
        persona = agent_config.persona_prompt
        system_prompt = f"""You are RAGnetic, a professional AI assistant. Your behavior and personality are defined by the user's custom instructions below.
                       ---
                       **USER'S CUSTOM INSTRUCTIONS (Your Persona):**
                       {persona}
                       ---
                       Your primary goal is to provide clear and accurate answers based *only* on the information provided to you in the "SOURCES" section.
                       **General Instructions:**
                       1.  Synthesize an answer from the information given in the "SOURCES" section below.
                       2.  Do not refer to "the context provided" or "the information I have." Respond directly and authoritatively.
                       3.  If the sources indicate an error or that no documents were found, inform the user of this fact.
                       **Instructions for Formatting:**
                       - Use Markdown for all your responses.
                       - Use headings (`##`, `###`) to structure main topics.
                       - Use bold text (`**text**`) to highlight key terms, figures, or important information.
                       - Use bullet points (`- `) or numbered lists (`1. `) for detailed points or steps.
                       **SOURCES:**
                       ---
                       {context_section}
                       ---
                       Based on the sources and your persona, please answer the user's query.
                    """
        prompt_with_history = [HumanMessage(content=system_prompt)] + messages

        # Model Initialization Logic
        model_kwargs = agent_config.model_params.model_dump(exclude_unset=True) if agent_config.model_params else {}

        if model_name.startswith("ollama/"):
            ollama_model_name = model_name.split("/", 1)[1]
            logger.info(f"Initializing local Ollama model: '{ollama_model_name}' with params: {model_kwargs}")
            model = ChatOllama(model=ollama_model_name, **model_kwargs)
        else:
            provider = "openai"
            if "claude" in model_name.lower():
                provider = "anthropic"
            elif "gemini" in model_name.lower():
                provider = "google"
            logger.info(f"Determined model provider: '{provider}' for model '{model_name}'.")
            api_key = get_api_key(provider)
            model = init_chat_model(model_name, model_provider=provider, streaming=True, api_key=api_key,
                                    **model_kwargs)

        model_with_tools = model.bind_tools(tools)

        logger.info("Invoking the language model...")
        try:
            # Attempt to invoke the model with the tool-binding
            response = model_with_tools.invoke(prompt_with_history)

        except ResponseError as e:
            if "does not support tools" in str(e):
                logger.warning(f"Model '{model_name}' does not support tools. Retrying without tools.")
                # If the model doesn't support tools, invoke it again without them
                response = model.invoke(prompt_with_history)
            else:
                raise e
        logger.info("Model invocation successful.")

        return {"messages": [response]}

    except Exception as e:
        logger.critical(f"A critical error occurred in the 'call_model' node: {e}", exc_info=True)
        error_message = AIMessage(content=f"I'm sorry, but I encountered an unexpected error. Error: {e}")
        return {"messages": [error_message]}


def should_continue(state: AgentState) -> str:
    """Routes to the tool node if the model requests it, otherwise ends."""
    if not state or not state.get('messages'):
        logger.error("Invalid state passed to 'should_continue'. Ending flow.")
        return "end"

    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info(f"Model requested tool call: {last_message.tool_calls[0]['name']}. Routing to 'call_tool'.")
        return "call_tool"

    logger.info("No tool call requested. Ending graph execution.")
    return "end"


def get_agent_workflow(tools: List[BaseTool]):
    """Builds the tool-using agent graph."""
    workflow = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    workflow.add_node("agent", call_model)
    workflow.add_node("call_tool", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"call_tool": "call_tool", "end": END})
    workflow.add_edge("call_tool", "agent")
    return workflow