# app/agents/agent_graph.py

import os
from typing import List, TypedDict

from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from app.agents.loader import load_agent_config
from app.tools.retriever_tool import get_retriever

class MessagesState(TypedDict):
    messages: List[BaseMessage]

def build_call_model_with_retriever(retriever):
    """
    This function is synchronous, but LangGraph can correctly run it
    within an async stream.
    """
    def call_model(state: MessagesState, config: RunnableConfig) -> dict:
        messages = state.get("messages", [])
        # The new message is passed via the config from the /ws endpoint
        new_msg = config.get("configurable", {}).get("new_message")

        if new_msg:
            messages.append(new_msg)

        # Use the content of the new message as the query
        query = new_msg.content if new_msg else messages[-1].content
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

        # The state is updated with the user's message AND the AI's response
        return {"messages": messages + [response]}

    return call_model


def get_agent_workflow(agent_name: str) -> StateGraph:
    """
    Builds the StateGraph for an agent but does not compile it.
    Compilation with a checkpointer must happen in an async context.
    """
    config = load_agent_config(agent_name)
    retriever = get_retriever(agent_name)
    call_model = build_call_model_with_retriever(retriever)

    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.set_entry_point("model")
    workflow.set_finish_point("model")

    return workflow