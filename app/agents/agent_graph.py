# app/agents/agent_graph.py

import os
from typing import List, TypedDict

from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from app.agents.config_manager import load_agent_config
from app.tools.retriever_tool import get_retriever


class MessagesState(TypedDict):
    messages: List[BaseMessage]


# in app/agents/agent_graph.py

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

        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        # This new prompt instructs the LLM to use Markdown for formatting.
        system_prompt = f"""
                You are RAGnetic, a helpful and friendly AI assistant.
                Your goal is to provide clear, well-structured, and conversational answers based on the provided context.

                **Instructions for Formatting:**
                - Use Markdown for all your responses.
                - Use headings (`##`, `###`) to structure main topics.
                - Use bold text (`**text**`) to highlight key terms, figures, or important information.
                - Use bullet points (`- `) or numbered lists (`1. `) for detailed points or steps.
                - If you include code snippets, use Markdown code blocks.
                - Keep your tone helpful and engaging.

                **Context to use for your answer:**
                ---
                {context_text}
                ---

                Based on the context above and our conversation history, please answer the user's query.
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