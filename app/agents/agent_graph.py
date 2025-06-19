import os
import sqlite3
from typing import List, TypedDict

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from app.agents.loader import load_agent_config
from app.tools.retriever_tool import get_retriever_tool
from app.tools.retriever_tool import get_retriever

class MessagesState(TypedDict):
    messages: List[BaseMessage]


def build_call_model_with_retriever(retriever):
    def call_model(state: MessagesState, config: RunnableConfig) -> dict:
        messages = state.get("messages", [])
        new_msg = config.get("configurable", {}).get("new_message")

        if new_msg:
            messages.append(new_msg)

        query = new_msg.content if new_msg else "Hello"
        context_docs = retriever.get_relevant_documents(query)
        context_text = "\n\n".join([doc.page_content for doc in context_docs[:3]])

        system_prompt = f"""
        You are a helpful agent. Use the following context to answer the user's query.
        Context:
        {context_text}
        """

        prompt_messages = [HumanMessage(content=system_prompt)] + messages[-30:]

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm.invoke(prompt_messages)

        return {"messages": messages + [response]}

    return call_model



def build_langgraph_agent(agent_name: str, user_id: str, thread_id: str):
    config = load_agent_config(agent_name)
    os.makedirs("memory", exist_ok=True)

    memory_path = f"memory/{agent_name}_{user_id}_{thread_id}.db"
    connection = sqlite3.connect(memory_path, check_same_thread=False)
    saver = SqliteSaver(connection)

    # Load retriever and build call_model with it
    retriever = get_retriever(agent_name)
    call_model = build_call_model_with_retriever(retriever)

    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.set_entry_point("model")

    return workflow.compile(checkpointer=saver)