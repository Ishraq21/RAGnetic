import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI

from app.agents.loader import load_agent_config
from app.tools.retriever_tool import get_retriever_tool


def build_langgraph_agent(name: str):
    config = load_agent_config(name)

    # Ensure memory directory exists
    os.makedirs("memory", exist_ok=True)

    # Create per-agent memory file
    checkpointer = SqliteSaver(f"memory/{name}.db")

    # Initialize language model
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Dynamically inject retriever tool if agent has sources
    tools = []
    if config.sources:
        tools.append(get_retriever_tool(name))

    # Create the agent with tools and memory
    agent = create_react_agent(
        tools=tools,
        model=llm,
        checkpointer=checkpointer,
    )

    return agent
