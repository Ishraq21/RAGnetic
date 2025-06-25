import os
from typing import List, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model

from app.agents.config_manager import load_agent_config
from app.tools.retriever_tool import get_retriever_tool
from app.core.config import get_api_key


# --- The state definition for our agent ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    tool_calls: List[dict]


# --- This is the core agent node ---
def call_model(state: AgentState, config: RunnableConfig):
    """
    This is the core reasoning step. It first retrieves document context,
    then decides whether to respond directly or to call another tool.
    """
    # Config is passed from main.py at runtime
    agent_config = config['configurable']['agent_config']

    # The complete list of tools is assembled in main.py and passed here.
    tools = config['configurable'].get('tools', [])
    messages = state['messages']
    query = messages[-1].content
    model_name = agent_config.llm_model


    # 1. Always perform RAG retrieval to get document context.
    retriever_tool = get_retriever_tool(agent_config.name)
    retrieved_docs = retriever_tool.func({"input": query})
    retrieved_docs_str = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 2. Build the system prompt that INCLUDES the retrieved context.
    context_section = f"<context>\n<retrieved_documents>\n{retrieved_docs_str}\n</retrieved_documents>\n</context>"

    system_prompt = f"""You are RAGnetic, a professional AI assistant. Your goal is to provide clear and accurate answers based *only* on the information provided to you in the "SOURCES" section.

                   **Instructions:**
                   1.  Synthesize an answer from the information given in the "SOURCES" section below.
                   2.  Do not refer to "the context provided" or "the information I have." Respond directly and authoritatively.

                   **Instructions for Formatting:**
                   - Use Markdown for all your responses.
                   - Use headings (`##`, `###`) to structure main topics.
                   - Use bold text (`**text**`) to highlight key terms, figures, or important information.
                   - Use bullet points (`- `) or numbered lists (`1. `) for detailed points or steps.
                   - If you include code snippets, use Markdown code blocks.
                   - Keep your tone helpful and engaging.

                   **SOURCES:**
                   ---
                   {context_section}
                   ---

                   Based on the sources above, please answer the user's query.
                """

    prompt_with_history = [HumanMessage(content=system_prompt)] + messages


    # First, determine the provider from the model name string
    provider = "openai"  # Default
    if "claude" in model_name.lower():
        provider = "anthropic"
    elif "gemini" in model_name.lower():
        provider = "google_genai"
    elif "grok" in model_name.lower():
        provider = "xai"

    # Your get_api_key function ensures the correct key is loaded into the environment
    get_api_key(model_name)

    # Now, use the init_chat_model helper to create the correct model instance
    model = init_chat_model(
        model_name,
        model_provider=provider,
        temperature=0,
        streaming=True,
    ).bind_tools(tools)

    # --- End of Refinement ---
    response = model.invoke(prompt_with_history)

    # The 'add' annotation on AgentState handles appending this to history
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Routes to the tool node if the model requests it, otherwise ends."""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "call_tool"
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
