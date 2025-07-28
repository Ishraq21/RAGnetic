import os
import logging
from typing import List, TypedDict, Annotated, Optional
from operator import add
import time
import uuid
import json
import asyncio
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from app.schemas.agent import AgentConfig

# LangChain's generic chat model initializer
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from ollama import ResponseError

# Local application imports
from app.tools.retriever_tool import get_retriever_tool
from app.core.config import get_api_key, get_llm_model
from app.core.cost_calculator import calculate_cost, count_tokens
from app.db.dao import save_conversation_metrics_sync
from app.db import get_sync_db_engine

logger = logging.getLogger(__name__)
logger = logging.getLogger("ragnetic")
metrics_logger = logging.getLogger("ragnetic.metrics")

class AgentState(TypedDict):
    """
    Represents the state of our agent. It's expanded to carry metrics
    through the workflow for logging.
    """
    messages: Annotated[List[BaseMessage], add]
    tool_calls: List[dict]
    # Metrics to be logged for each request
    request_id: str
    agent_name: str
    agent_config: AgentConfig
    retrieval_time_s: float
    generation_time_s: float
    total_duration_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    retrieved_chunk_ids: List[str]
    # session_id removed from TypedDict as it's passed via config directly


async def call_model(state: AgentState, config: RunnableConfig):
    """
    Core reasoning step of the agent. Now also responsible for calculating
    and populating all performance and cost metrics into the state,
    and saving them to the database and logging to file.
    """
    sync_engine_connection = None
    try:
        start_time = time.perf_counter()
        agent_config = state['agent_config']
        tools = config['configurable'].get('tools', [])
        messages = state['messages']

        # Extract session_id and other configurable items directly from config
        session_id = config['configurable'].get("session_id")
        request_id = config['configurable'].get("request_id") or state["request_id"]
        # In a chat context, the agent name from config might be more reliable
        agent_name_from_config = config['configurable'].get("agent_name", agent_config.name)

        logger.debug(f"[call_model] Session ID from config: {session_id}")
        logger.debug(f"[call_model] Request ID from config/state: {request_id}")
        logger.debug(f"[call_model] Agent Name from config/state: {agent_name_from_config}")

        if session_id is None:  # Add a warning if session_id is still None
            logger.warning(f"[call_model] Session ID is None for request {request_id}. Metrics will be unlinked.")


        if not messages:
            logger.error("State has no messages. Cannot proceed.")
            return {"messages": [AIMessage(
                content="I'm sorry, but there was an error processing your request as the message history is empty.")]}

        query = messages[-1].content
        history = messages[:-1]

        model_name = agent_config.llm_model
        logger.info(f"Processing query for agent '{agent_name_from_config}' using model '{model_name}'.")

        retrieved_docs_str = ""
        retrieved_chunk_ids = []
        retrieved_docs = []
        t0 = time.perf_counter()
        if "retriever" in agent_config.tools:
            try:
                logger.info(f"Attempting to retrieve documents for query: '{query[:80]}...'")
                retriever_tool = get_retriever_tool(agent_config)
                retrieved_docs = await asyncio.to_thread(retriever_tool.invoke, {"input": query})
                if isinstance(retrieved_docs, str):
                    retrieved_docs_str = retrieved_docs
                    logger.error(f"Retriever tool returned an error string: {retrieved_docs}")
                elif retrieved_docs:
                    retrieved_docs_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    retrieved_chunk_ids = [doc.id for doc in retrieved_docs if hasattr(doc, 'id')]
                    logger.info(f"Successfully retrieved {len(retrieved_docs)} documents.")
                else:
                    retrieved_docs_str = "No documents were found matching the query."
                    logger.warning("Retriever tool ran successfully but returned no documents.")
            except Exception as e:
                logger.error(f"Error during document retrieval: {e}", exc_info=True)
                retrieved_docs_str = f"An error occurred while trying to retrieve relevant documents: {e}"
        retrieval_time = time.perf_counter() - t0

        if agent_config.execution_prompt:
            logger.info("Using custom 'execution_prompt' from agent configuration.")
            prompt = ChatPromptTemplate.from_template(agent_config.execution_prompt)
            system_prompt_content = prompt.format(
                user_query=query,
                retrieved_context=retrieved_docs_str,
                persona=agent_config.persona_prompt
            )
            prompt_with_history = [SystemMessage(content=system_prompt_content)] + history + [messages[-1]]

        else:
            logger.info("Using default system prompt.")
            context_section = f"<context>\n<retrieved_documents>\n{retrieved_docs_str}\n</retrieved_documents>\n</context>"
            system_prompt_content = f"""You are RAGnetic, a professional AI assistant. Your behavior and personality are defined by the user's custom instructions below.
                                   ---
                                   **USER'S CUSTOM INSTRUCTIONS (Your Persona):**
                                   {agent_config.persona_prompt}
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

                                   IMPORTANT: When explaining any mathematical expressions, YOU MUST USE LaTeX SYNTAX.
                                   IMPORTANT: Wrap inline equations in `$...$`, and display equations in `$$...$$`.
                                   IMPORTANT: DO NOT use <pre><code> or similar tags for mathematical explanations or calculations.

                                   - Surround only the actual code snippets with <pre><code> tags. Do not wrap non-code content.
                                   - Do NOT use <pre><code> tags for math or calculations.
                                   - Do NOT use <pre><code> tags for Physics and Chemistry calculations. Use LaTex instead.

                                   I REPEAT! When explaining or showing any mathematical expressions or equations, YOU MUST USE LaTeX SYNTAX.

                                   - Use LaTeX syntax for all mathematical expressions.
                                   - Inline math must be wrapped in `\\( ... \\)`.
                                   - Block math must be wrapped in `$$ ... $$`.

                                   **SOURCES:**
                                   ---
                                   {context_section}
                                   ---
                                   Based on the sources and your persona, please answer the user's query.
                                """
            prompt_with_history = [SystemMessage(content=system_prompt_content)] + messages

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
            if provider == "google":
                logger.info("Initializing ChatGoogleGenerativeAI directly.")
                model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, **model_kwargs)
            else:
                if provider == "openai" and model_kwargs.get("streaming", True):
                    # Ensure stream_usage is passed correctly to init_chat_model for OpenAI streaming
                    openai_model_kwargs = model_kwargs.copy()
                    openai_model_kwargs["stream_usage"] = True
                    model = init_chat_model(model_name, model_provider=provider, streaming=True, api_key=api_key, **openai_model_kwargs)
                else:
                    model = init_chat_model(model_name, model_provider=provider, streaming=True, api_key=api_key, **model_kwargs)


        model_with_tools = model.bind_tools(tools)
        logger.info("Invoking the language model...")
        t1 = time.perf_counter()
        try:
            response = await model_with_tools.ainvoke(prompt_with_history)
        except ResponseError as e:
            if "does not support tools" in str(e):
                logger.warning(f"Model '{model_name}' does not support tools. Retrying without tools.")
                response = await model.ainvoke(prompt_with_history)
            else:
                raise e
        generation_time = time.perf_counter() - t1
        logger.info("Model invocation successful.")

        # --- Debugging: Log all relevant response attributes ---
        logger.info(f"[call_model] Raw LLM response: {response}")  # Log the full response object
        logger.info(f"[call_model] Raw LLM response metadata: {response.response_metadata}")
        logger.info(f"[call_model] Raw LLM usage_metadata: {getattr(response, 'usage_metadata', {})}")

        prompt_tokens = 0
        completion_tokens = 0

        # Prioritize usage_metadata from AIMessage object (LangChain standard)
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = response.usage_metadata.get("input_tokens", 0)
            completion_tokens = response.usage_metadata.get("output_tokens", 0)
            logger.debug(
                f"[call_model] Retrieved tokens from usage_metadata: input={prompt_tokens}, output={completion_tokens}")

        # Fallback to response_metadata.token_usage (older LangChain or non-standard)
        if not prompt_tokens and not completion_tokens:
            token_usage_from_metadata = response.response_metadata.get("token_usage", {})
            if token_usage_from_metadata:
                prompt_tokens = token_usage_from_metadata.get("prompt_tokens", 0)
                completion_tokens = token_usage_from_metadata.get("completion_tokens", 0)
                logger.debug(
                    f"[call_model] Retrieved tokens from response_metadata.token_usage: input={prompt_tokens}, output={completion_tokens}")

        # If still no tokens, use estimation (last resort)
        if not prompt_tokens and not completion_tokens:
            logger.warning("[call_model] No explicit token usage found in metadata. Estimating tokens using fallback.")

            prompt_text_for_estimation = ""
            for msg in prompt_with_history:
                if msg.content:
                    prompt_text_for_estimation += msg.content + "\n"

            prompt_tokens = count_tokens(prompt_text_for_estimation, model_name)
            completion_tokens = count_tokens(response.content, model_name)
            logger.debug(
                f"[call_model] Estimated prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")

        # Calculate cost based on whatever tokens we derived
        cost = calculate_cost(agent_config.llm_model, prompt_tokens, completion_tokens)

        state.update({
            "messages": [response],
            "retrieval_time_s": retrieval_time,
            "generation_time_s": generation_time,
            "total_duration_s": time.perf_counter() - start_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,  # Ensure total_tokens is sum of derived values
            "estimated_cost_usd": cost,
            "retrieved_chunk_ids": retrieved_chunk_ids
        })

        current_timestamp = datetime.utcnow()

        db_metrics_data = {
            "request_id": request_id,
            "session_id": session_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,  # Ensure total_tokens here as well
            "retrieval_time_s": retrieval_time,
            "generation_time_s": generation_time,
            "estimated_cost_usd": cost,
            "timestamp": current_timestamp,
            "llm_model": model_name
        }

        log_metrics_data = {
            "request_id": request_id,
            "agent_name": agent_name_from_config,
            "llm_model": agent_config.llm_model,
            "retrieval_s": retrieval_time,
            "generation_s": generation_time,
            "total_duration_s": state["total_duration_s"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,  # Ensure total_tokens here as well
            "estimated_cost_usd": cost,
            "retrieved_chunks": retrieved_chunk_ids,
            "timestamp": current_timestamp.isoformat()
        }

        sync_engine = get_sync_db_engine()
        with sync_engine.connect() as sync_engine_connection:
            await asyncio.to_thread(save_conversation_metrics_sync, sync_engine_connection, db_metrics_data)

        metrics_logger.info(
            "Agent request metrics",
            extra={'extra_data': log_metrics_data}
        )

        return state
    except Exception as e:
        logger.critical(f"A critical error occurred in the 'call_model' node: {e}", exc_info=True)
        error_message = AIMessage(content=f"I'm sorry, but I encountered an unexpected error. Error: {e}")
        state.update({"messages": [error_message]})
        return state
    finally:
        if sync_engine_connection:
            sync_engine_connection.close()

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