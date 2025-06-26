import os
import logging
from typing import List, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model

from app.tools.retriever_tool import get_retriever_tool
from app.core.config import get_api_key
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    # 'tool_calls' is part of the standard LangGraph state for tool-using agents
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

        # Always Perform RAG Retrieval with Error Handling ---
        try:
            logger.info(f"Attempting to retrieve documents for query: '{query[:80]}...'")
            retriever_tool = get_retriever_tool(agent_config.name)
            retrieved_docs = retriever_tool.func({"input": query})
            if retrieved_docs:
                retrieved_docs_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
                logger.info(f"Successfully retrieved {len(retrieved_docs)} documents.")
            else:
                retrieved_docs_str = "No documents were found matching the query."
                logger.warning("Retriever tool ran successfully but returned no documents.")
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}", exc_info=True)
            retrieved_docs_str = "An error occurred while trying to retrieve relevant documents. The information may be unavailable."

        # Build the System Prompt ---
        context_section = f"<context>\n<retrieved_documents>\n{retrieved_docs_str}\n</retrieved_documents>\n</context>"
        system_prompt = f"""You are RAGnetic, a professional AI assistant. Your goal is to provide clear and accurate answers based *only* on the information provided to you in the "SOURCES" section.

                       **Instructions:**
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

                       Based on the sources above, please answer the user's query.
                    """
        prompt_with_history = [HumanMessage(content=system_prompt)] + messages

        api_key = get_api_key(model_name)
        provider = "openai"
        if "claude" in model_name.lower():
            provider = "anthropic"
        elif "gemini" in model_name.lower():
            provider = "google_genai"

        logger.info(f"Initializing model '{model_name}' from provider '{provider}'. API key found: {bool(api_key)}")

        model = init_chat_model(
            model_name,
            model_provider=provider,
            temperature=0,
            streaming=True,
            api_key=api_key,
        ).bind_tools(tools)

        logger.info("Invoking the language model...")
        response = model.invoke(prompt_with_history)
        logger.info("Model invocation successful.")

        return {"messages": [response]}

    except Exception as e:
        logger.critical(f"A critical error occurred in the 'call_model' node: {e}", exc_info=True)
        # Return a fallback error message to the user
        error_message = AIMessage(
            content=f"I'm sorry, but I encountered an unexpected error and cannot complete your request. Please try again later. Error: {e}")
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
    """
    Builds the tool-using agent graph.
    It accepts a pre-assembled list of tools from the main application logic.
    """
    workflow = StateGraph(AgentState)

    # The agent node is the main reasoning step
    agent_node = call_model

    # The tool node is initialized with all available tools (retriever, sql, etc.)
    tool_node = ToolNode(tools)

    workflow.add_node("agent", agent_node)
    workflow.add_node("call_tool", tool_node)

    # Define the graph's control flow
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"call_tool": "call_tool", "end": END},
    )
    workflow.add_edge("call_tool", "agent")

    # Return the uncompiled workflow, as expected by main.py
    return workflow
