# app/agents/agent_graph.py

import os
import logging
from pathlib import Path
from typing import List, TypedDict, Annotated, Optional, Dict, Any
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
from langchain_huggingface.llms import HuggingFacePipeline as LCHuggingFacePipeline
from transformers import AutoTokenizer, pipeline

from app.schemas.agent import AgentConfig

# LangChain's generic chat model initializer
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_ollama import ChatOllama
from ollama import ResponseError

# Local application imports
from app.tools.retriever_tool import get_retriever_tool  # This import will stay the same
from app.core.config import get_api_key, get_llm_model, get_path_settings
from app.core.cost_calculator import calculate_cost, count_tokens
from app.db.dao import save_conversation_metrics_sync
from app.db import get_sync_db_engine

from app.training.model_manager import FineTunedModelManager

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
    temp_document_ids: List[str]
    retrieved_documents_meta_for_citation: List[
        Dict[str, Any]]


async def call_model(state: AgentState, config: RunnableConfig):
    """
    Core reasoning step of the agent. Now also responsible for calculating
    and populating all performance and cost metrics into the state,
    and saving them to the database and logging to file.
    """
    sync_engine_connection = None
    embedding_cost_usd = 0.0
    retrieved_documents_meta_for_citation: List[Dict[str, Any]] = []  # Initialize for this scope
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

        temp_document_ids = state.get("temp_document_ids", [])
        if temp_document_ids:
            logger.info(f"[{request_id}] Found {len(temp_document_ids)} temporary documents for retrieval.")

        logger.debug(f"[call_model] Session ID from config: {session_id}")
        logger.debug(f"[call_model] Request ID from config/state: {request_id}")
        logger.debug(f"[call_model] Agent Name from config/state: {agent_name_from_config}")

        if session_id is None:
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

        embedding_model_name = agent_config.embedding_model

        t0 = time.perf_counter()
        if "retriever" in agent_config.tools:
            try:
                embedding_tokens = count_tokens(query, embedding_model_name)
                logger.info(f"[call_model] Embedding query: '{query[:50]}...'")
                logger.info(f"[call_model] Embedding model used: {embedding_model_name}")
                logger.info(f"[call_model] Calculated embedding tokens: {embedding_tokens}")

                embedding_cost_usd = calculate_cost(
                    embedding_model_name=embedding_model_name,
                    embedding_tokens=embedding_tokens
                )
                logger.info(f"[call_model] Calculated embedding cost: ${embedding_cost_usd:.6f}")
                logger.info(f"Attempting to retrieve documents for query: '{query[:80]}...'")

                retriever_tool = await get_retriever_tool(agent_config)
                tool_input = {"query": query, "temp_document_ids": temp_document_ids}
                retrieved_docs = await retriever_tool.ainvoke(tool_input)
                if isinstance(retrieved_docs, str):
                    retrieved_docs_str = retrieved_docs
                    logger.error(f"Retriever tool returned an error string: {retrieved_docs}")
                elif retrieved_docs:
                    retrieved_docs_str = ""
                    retrieved_documents_meta_for_citation = []  # Reset for current run
                    for doc in retrieved_docs:
                        doc_name = doc.metadata.get('doc_name', 'Unknown Document')
                        page_number = doc.metadata.get('page_number')
                        # Use temp_doc_id for unique identification, but display name/page for LLM readability
                        temp_doc_id = doc.metadata.get('temp_doc_id')

                        source_info = f"Source: '{doc_name}'"
                        if page_number:
                            source_info += f" Page: {page_number}"
                        if temp_doc_id:
                            source_info += f" (ID: {temp_doc_id[:8]}...)"  # Short ID for debug/backend tracing

                        retrieved_docs_str += f"Document Content ({source_info}):\n{doc.page_content}\n\n"

                        # Store metadata for later citation extraction in main.py/websocket_chat
                        retrieved_documents_meta_for_citation.append(doc.metadata)

                        # Use temp_doc_id and chunk_index for unique retrieved_chunk_ids
                        retrieved_chunk_ids.append(f"{temp_doc_id}_{doc.metadata.get('chunk_index', '')}"
                                                   if temp_doc_id else str(
                            uuid.uuid4()))  # Fallback to UUID if no temp_doc_id

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
            # MODIFIED: Enhanced system prompt with clear citation instructions
            system_prompt_content = f"""You are RAGnetic, a professional AI assistant. Your behavior and personality are defined by the user's custom instructions below.
                                   ---
                                   **USER'S CUSTOM INSTRUCTIONS (Your Persona):**
                                   {agent_config.persona_prompt}
                                   ---
                                   Your primary goal is to provide clear and accurate answers based *only* on the information provided to you in the "SOURCES" section.
                                   **General Instructions:**
                                   1.  Synthesize an answer from the information given in the "SOURCES" section below.
                                   2.  If information from a document is used, **you MUST cite the source inline**.
                                       Use the format: `[↩:OriginalFileName.ext:PageNumber]` or `[↩:OriginalFileName.ext]` if page number is not available.
                                       Example: "The report indicates a 15% increase in sales [↩:Q4_Report.pdf:2]."
                                       Example: "The new policy was outlined [↩:Policy_Doc.docx]."
                                       For code files or CSVs, if no page number, just use the filename.
                                   3.  Do not refer to "the context provided" or "the information I have." Respond directly and authoritatively.
                                   4.  If the sources indicate an error or that no documents were found, inform the user of this fact.
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
                                   {retrieved_docs_str}
                                   ---
                                   Based on the sources and your persona, please answer the user's query.
                                """
            prompt_with_history = [SystemMessage(content=system_prompt_content)] + messages

        model_kwargs = agent_config.model_params.model_dump(exclude_unset=True) if agent_config.model_params else {}
        base_llm_model = None
        if model_name.startswith("ollama/"):
            ollama_model_name = model_name.split("/", 1)[1]
            logger.info(f"Initializing local Ollama model: '{ollama_model_name}' with params: {model_kwargs}")
            base_llm_model = ChatOllama(model=ollama_model_name, **model_kwargs)
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
                base_llm_model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, **model_kwargs)
            else:
                if provider == "openai" and model_kwargs.get("streaming", True):
                    openai_model_kwargs = model_kwargs.copy()
                    openai_model_kwargs["stream_usage"] = True
                    base_llm_model = init_chat_model(model_name, model_provider=provider, streaming=True,
                                                     api_key=api_key, **openai_model_kwargs)
                else:
                    base_llm_model = init_chat_model(model_name, model_provider=provider, streaming=True,
                                                     api_key=api_key, **model_kwargs)

        final_llm_model_raw_hf = None
        final_tokenizer_hf = None

        if agent_config.fine_tuned_model_id:
            logger.info(f"Fine-tuned model ID specified: {agent_config.fine_tuned_model_id}. Attempting to load.")
            _APP_PATHS = get_path_settings()
            fine_tuned_models_base_dir = _APP_PATHS["FINE_TUNED_MODELS_BASE_DIR"]
            model_manager = FineTunedModelManager(Path(fine_tuned_models_base_dir))

            db_engine_sync = get_sync_db_engine()
            from app.db.models import fine_tuned_models_table
            from sqlalchemy import select

            adapter_record = None
            with db_engine_sync.connect() as conn:
                stmt = select(fine_tuned_models_table).where(
                    fine_tuned_models_table.c.adapter_id == agent_config.fine_tuned_model_id)
                result = conn.execute(stmt).first()
                if result:
                    adapter_record = result._asdict()
                    logger.info(f"Found fine-tuned model record for ID: {agent_config.fine_tuned_model_id}")
                else:
                    logger.error(f"Fine-tuned model record not found in DB for ID: {agent_config.fine_tuned_model_id}.")
                    raise ValueError(f"Fine-tuned model '{agent_config.fine_tuned_model_id}' not found.")

            if adapter_record:
                adapter_path_from_db = adapter_record['adapter_path']
                base_model_name_from_db = adapter_record['base_model_name']

                loaded_ft_model_and_tokenizer = await asyncio.to_thread(
                    model_manager.load_adapter,
                    adapter_path_from_db,
                    base_model_name_from_db
                )

                if loaded_ft_model_and_tokenizer:
                    final_llm_model_raw_hf = loaded_ft_model_and_tokenizer

                    final_tokenizer_hf = AutoTokenizer.from_pretrained(adapter_path_from_db)
                    logger.info(
                        f"Successfully loaded fine-tuned model '{agent_config.fine_tuned_model_id}' and its tokenizer.")
                else:
                    logger.error(
                        f"Failed to load fine-tuned model '{agent_config.fine_tuned_model_id}'. Cannot proceed without it.")
                    raise RuntimeError(f"Failed to load mandatory fine-tuned model {agent_config.fine_tuned_model_id}.")

        if agent_config.fine_tuned_model_id:
            logger.info(f"Fine-tuned model ID specified: {agent_config.fine_tuned_model_id}. Loading adapter…")
            paths = get_path_settings()
            manager = FineTunedModelManager(Path(paths["FINE_TUNED_MODELS_BASE_DIR"]))

            from app.db.models import fine_tuned_models_table
            from sqlalchemy import select
            with get_sync_db_engine().connect() as conn:
                row = conn.execute(
                    select(fine_tuned_models_table).where(
                        fine_tuned_models_table.c.adapter_id == agent_config.fine_tuned_model_id
                    )
                ).first()
            if not row:
                raise ValueError(f"Fine-tuned model '{agent_config.fine_tuned_model_id}' not found in DB")

            adapter_path = row.adapter_path
            base_name = row.base_model_name

            loaded = await asyncio.to_thread(manager.load_adapter, adapter_path, base_name)
            if not loaded:
                raise RuntimeError(f"Failed to load fine-tuned model {agent_config.fine_tuned_model_id}")

            final_llm_model_raw_hf = loaded
            final_tokenizer_hf = AutoTokenizer.from_pretrained(adapter_path)
            logger.info("Adapter + tokenizer loaded, building pipeline…")

            hf_pipeline_instance = pipeline(
                "text-generation",
                model=final_llm_model_raw_hf,
                tokenizer=final_tokenizer_hf,
                max_new_tokens=model_kwargs.get("max_tokens", 512),
                temperature=model_kwargs.get("temperature", 0.7),
                return_full_text=False,
            )

            langchain_hf_llm_instance = LCHuggingFacePipeline(pipeline=hf_pipeline_instance)

            chat_hf_kwargs = model_kwargs.copy()
            chat_hf_kwargs["streaming"] = False

            final_llm_model = ChatHuggingFace(
                llm=langchain_hf_llm_instance,
                **chat_hf_kwargs
            )
            logger.info("Wrapped fine-tuned HF model with ChatHuggingFace (sync mode).")

        else:
            logger.info("No fine-tuned model; using base LLM")
            if model_name.startswith("ollama/"):
                final_llm_model = ChatOllama(model=model_name.split("/", 1)[1], **model_kwargs)
            else:
                provider = ("anthropic" if "claude" in model_name.lower()
                            else "google" if "gemini" in model_name.lower()
                else "openai")
                api_key = get_api_key(provider)
                if provider == "google":
                    final_llm_model = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, **model_kwargs)
                else:
                    openai_kwargs = model_kwargs.copy()
                    if provider == "openai" and openai_kwargs.get("streaming", True):
                        openai_kwargs["stream_usage"] = True
                        final_llm_model = init_chat_model(model_name, model_provider=provider,
                                                          streaming=True, api_key=api_key, **openai_kwargs)
                    else:
                        final_llm_model = init_chat_model(model_name, model_provider=provider,
                                                          streaming=True, api_key=api_key, **model_kwargs)

        model_with_tools = final_llm_model.bind_tools(tools)
        logger.info("Invoking the LLM (sync pipeline wrapped in thread)…")

        t1 = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                model_with_tools.invoke,
                prompt_with_history
            )
        except ResponseError as e:
            if "does not support tools" in str(e):
                logger.warning(f"Model '{model_name}' does not support tools. Retrying without tools.")
                response = await model_with_tools.ainvoke(prompt_with_history)
            else:
                raise e
        generation_time = time.perf_counter() - t1
        logger.info("Model invocation successful.")

        logger.debug(f"[call_model] Raw LLM response: {response}")
        logger.debug(f"[call_model] Raw LLM response metadata: {response.response_metadata}")
        logger.debug(f"[call_model] Raw LLM usage_metadata: {getattr(response, 'usage_metadata', {})}")

        prompt_tokens = 0
        completion_tokens = 0

        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = response.usage_metadata.get("input_tokens", 0)
            completion_tokens = response.usage_metadata.get("output_tokens", 0)
            logger.debug(
                f"[call_model] Retrieved tokens from usage_metadata: input={prompt_tokens}, output={completion_tokens}")

        if not prompt_tokens and not completion_tokens:
            token_usage_from_metadata = response.response_metadata.get("token_usage", {})
            if token_usage_from_metadata:
                prompt_tokens = token_usage_from_metadata.get("prompt_tokens", 0)
                completion_tokens = token_usage_from_metadata.get("completion_tokens", 0)
                logger.debug(
                    f"[call_model] Retrieved tokens from response_metadata.token_usage: input={prompt_tokens}, output={completion_tokens}")

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

        llm_cost_usd = calculate_cost(
            llm_model_name=agent_config.llm_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        total_estimated_cost_usd = llm_cost_usd + embedding_cost_usd

        state.update({
            "messages": [response],
            "retrieval_time_s": retrieval_time,
            "generation_time_s": generation_time,
            "total_duration_s": time.perf_counter() - start_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": total_estimated_cost_usd,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_documents_meta_for_citation": retrieved_documents_meta_for_citation
        })

        current_timestamp = datetime.utcnow()

        db_metrics_data = {
            "request_id": request_id,
            "session_id": session_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "retrieval_time_s": retrieval_time,
            "generation_time_s": generation_time,
            "estimated_cost_usd": llm_cost_usd,
            "embedding_cost_usd": embedding_cost_usd,
            "timestamp": current_timestamp,
            "llm_model": model_name,
            "fine_tuned_model_id": agent_config.fine_tuned_model_id

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
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": total_estimated_cost_usd,
            "embedding_cost_usd": embedding_cost_usd,
            "embedding_model_name": embedding_model_name,
            "retrieved_chunks": retrieved_chunk_ids,
            "timestamp": current_timestamp.isoformat(),
            "fine_tuned_model_id": agent_config.fine_tuned_model_id

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