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


# --- Define the state for our hybrid agent ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]
    tool_calls: List[dict]


# --- This is the hybrid agent node with the system prompt ---
def call_model(state: AgentState, config: RunnableConfig):
    """
    This node performs RAG retrieval first, then decides if it needs to use other tools.
    """
    # Get runtime configurations that are passed during invocation
    agent_name = config['configurable']['agent_name']
    sql_tools = config['configurable'].get('tools', [])
    messages = state['messages']
    query = messages[-1].content

    # 1. Always perform RAG retrieval to get document context
    retriever_tool = get_retriever_tool(agent_name)
    retrieved_docs_str = retriever_tool.func({"input": query})

    # 2. Build the system prompt that INCLUDES the retrieved context
    context_text = f"""
            <context>
              <retrieved_documents>
                {retrieved_docs_str}
              </retrieved_documents>
            </context>
            """
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
               {context_text}
               ---

               Based on the sources above, please answer the user's query.
               """

    prompt_with_context = [HumanMessage(content=system_prompt)] + messages

    # 3. Bind the SQL tools (if any) to the model and invoke it
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True).bind_tools(sql_tools)
    response = model.invoke(prompt_with_context)

    return {"messages": [response]}


# --- This function routes between the agent and tools ---
def should_continue(state: AgentState) -> str:
    if state['messages'][-1].tool_calls:
        return "call_tool"
    return "end"


# --- This is the main workflow builder ---
def get_agent_workflow(agent_name: str) -> StateGraph:
    """Builds the hybrid RAG + Tools agent graph."""
    agent_config = load_agent_config(agent_name)

    # Assemble ONLY the SQL tools for the ToolNode.
    sql_tools = []
    if "sql_toolkit" in agent_config.tools:
        db_source = next((s for s in agent_config.sources if s.type == 'db'), None)
        if db_source and db_source.db_connection:
            sql_tools = create_sql_toolkit(db_source.db_connection)
        else:
            print(f"Warning: Agent '{agent_name}' has 'sql_toolkit' enabled but no db_connection string is configured.")

    # Define the graph
    workflow = StateGraph(AgentState)

    # The ToolNode is initialized with the SQL tools
    workflow.add_node("agent", call_model)
    workflow.add_node("call_tool", ToolNode(sql_tools))

    # Define the graph's control flow
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"call_tool": "call_tool", "end": END},
    )
    workflow.add_edge("call_tool", "agent")

    # Return the UNCOMPILED workflow. Your main.py will handle compilation.
    return workflow