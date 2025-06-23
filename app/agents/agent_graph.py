import os
from typing import List, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from app.agents.config_manager import load_agent_config
from app.tools.retriever_tool import get_retriever_tool
from app.tools.sql_tool import create_sql_toolkit
from app.core.config import get_api_key


# Define the agent's state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    tool_calls: List[dict]


# This is the agent node
def call_model(state: AgentState, config: RunnableConfig):
    """
    The core of the agent. It performs RAG retrieval first, then gives
    the LLM the option to use SQL tools if the document context is insufficient.
    """
    agent_config = config['configurable']['agent_config']
    sql_tools = config['configurable'].get('tools', [])
    messages = state['messages']
    query = messages[-1].content

    # 1. Always perform RAG retrieval
    retriever_tool = get_retriever_tool(agent_config.name)
    retrieved_docs_str = retriever_tool.func({"input": query})

    # 2. Build the system prompt with the retrieved context
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

    prompt_with_context = [HumanMessage(content=system_prompt)] + messages

    # 3. Dynamically instantiate the LLM and bind the optional SQL tools
    api_key = get_api_key(agent_config.llm_model)
    model = ChatOpenAI(
        model=agent_config.llm_model,
        temperature=0,
        streaming=True,
        api_key=api_key
    ).bind_tools(sql_tools)

    response = model.invoke(prompt_with_context)

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Routes to the tool node if the model requests it."""
    if state['messages'][-1].tool_calls:
        return "call_tool"
    return "end"


def get_agent_workflow(agent_name: str):
    """Builds the hybrid RAG + Tools agent graph."""
    agent_config = load_agent_config(agent_name)

    sql_tools = []
    if "sql_toolkit" in agent_config.tools:
        db_source = next((s for s in agent_config.sources if s.type == 'db'), None)
        if db_source and db_source.db_connection:
            sql_tools = create_sql_toolkit(db_source.db_connection)

    workflow = StateGraph(AgentState)

    # Pass the agent_config and tools to the agent node at runtime
    agent_node = call_model.with_configurable(
        agent_config=agent_config,
        tools=sql_tools
    )

    workflow.add_node("agent", agent_node)
    workflow.add_node("call_tool", ToolNode(sql_tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"call_tool": "call_tool", "end": END})
    workflow.add_edge("call_tool", "agent")

    return workflow
