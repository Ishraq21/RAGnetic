import os
from typing import List, TypedDict, Dict

from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from app.agents.config_manager import load_agent_config
from app.tools.retriever_tool import get_retriever


class MessagesState(TypedDict):
    messages: List[BaseMessage]
    citations: List[Dict]


def build_call_model_with_retriever(retriever):
    """
    This function prepares the context and prompt for the LLM,
    including detailed source information for citations.
    """

    def call_model(state: MessagesState, config: RunnableConfig) -> dict:
        messages = state.get("messages", [])
        new_msg = config.get("configurable", {}).get("new_message")

        if new_msg:
            messages.append(new_msg)

        query = new_msg.content if new_msg else messages[-1].content
        context_docs = retriever.get_relevant_documents(query)

        context_chunks = []
        cited_docs_metadata = []
        unique_source_keys = set()

        for doc in context_docs:
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')

            # Create a detailed, human-readable citation label for the prompt
            source_name = metadata.get('title') or os.path.basename(source)
            page_num = metadata.get('page_number')
            row_num = metadata.get('row_number')
            table_name = metadata.get('table_name')

            citation_label = source_name
            if table_name:
                citation_label = f"Table: {table_name}"
            if page_num:
                citation_label += f", Page {page_num}"
            elif row_num:
                citation_label += f", Row {row_num}"

            # Use the clean label directly in the context provided to the LLM
            context_chunks.append(f"Source [{citation_label}]:\n{doc.page_content}")

            # Collect unique metadata to return to the UI
            if source and source not in unique_source_keys:
                cited_docs_metadata.append(metadata)
                unique_source_keys.add(source)

        context_text = "\n\n---\n\n".join(context_chunks)

        # Update the system prompt to match the new, cleaner citation format
        system_prompt = f"""You are RAGnetic, a professional AI assistant. Your goal is to provide clear and accurate answers based *only* on the information provided to you in the "SOURCES" section.

            **Instructions:**
            1.  Synthesize an answer from the information given in the "SOURCES" section below.
            2.  For every piece of information you use, you **must** cite the specific source by appending its reference tag, for example `[report.pdf, Page 2]` or `[Table: customers, Row 45]`, to the end of the sentence.
            3.  Do not refer to "the context provided" or "the information I have." Respond directly and authoritatively.
            4.  If the sources do not contain an answer, state that you do not have that information.
            
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

        prompt_messages = [HumanMessage(content=system_prompt)] + messages[-30:]

        llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
        response = llm.invoke(prompt_messages)

        return {
            "messages": [response],
            "citations": cited_docs_metadata
        }

    return call_model


def get_agent_workflow(agent_name: str) -> StateGraph:
    """
    Builds the StateGraph for an agent but does not compile it.
    """
    config = load_agent_config(agent_name)
    retriever = get_retriever(agent_name)
    call_model = build_call_model_with_retriever(retriever)

    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.set_entry_point("model")
    workflow.set_finish_point("model")

    return workflow