# app/api/query.py
import logging
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update
from typing import Any, Dict, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4

from app.db import get_db
from app.core.security import get_http_api_key, PermissionChecker
from app.core.validation import sanitize_for_path
from app.db.models import agent_runs, chat_sessions_table, chat_messages_table, users_table
from app.agents.config_manager import load_agent_config
from app.agents.agent_graph import get_agent_workflow, AgentState
from app.tools.retriever_tool import get_retriever_tool
from app.tools.sql_tool import create_sql_toolkit
from app.tools.arxiv_tool import get_arxiv_tool
from app.tools.search_engine_tool import SearchTool
from langchain_core.messages import HumanMessage, AIMessage
from app.core.serialization import _serialize_for_db
from app.schemas.security import User
from app.db.dao import save_conversation_metrics
from app.tools.api_toolkit import APIToolkit

logger = logging.getLogger("ragnetic")

router = APIRouter(prefix="/api/v1/agents", tags=["Query API"])


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query to the agent.")
    thread_id: Optional[str] = Field(None, description="A unique identifier for the conversation thread.")
    files: List[Dict[str, str]] = Field(default_factory=list,
                                        description="List of successfully uploaded temporary files with their temp_doc_ids.")


class QueryResponse(BaseModel):
    response: str = Field(..., description="The agent's final response content.")
    run_id: str = Field(..., description="The unique ID for this specific run, for auditing.")
    final_state: Dict[str, Any] = Field(..., description="The final state of the agent graph, containing metrics.")
    citations: List[Dict[str, Any]] = Field(default_factory=list,
                                            description="Extracted citations from the AI's response.")


@router.post("/{agent_name}/query", response_model=QueryResponse)
async def query_agent(
        agent_name: str,
        request: QueryRequest = Body(...),
        current_user: User = Depends(PermissionChecker(["agent:query"])),
        db: AsyncSession = Depends(get_db),
):
    # Load agent configuration
    try:
        agent_config = load_agent_config(agent_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading agent config '{agent_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"An unexpected server error occurred while loading agent configuration for '{agent_name}'. Please check server logs.")

    # Prepare IDs
    safe_user_id = sanitize_for_path(current_user.username)
    safe_thread_id = (
        sanitize_for_path(request.thread_id) if request.thread_id else f"api-thread-{uuid4().hex[:8]}"
    )
    request_id = str(uuid4())

    user_db_id = current_user.id

    # Ensure session record
    session_id = (await db.execute(
        select(chat_sessions_table.c.id).where(
            (chat_sessions_table.c.thread_id == safe_thread_id) &
            (chat_sessions_table.c.agent_name == agent_name) &
            (chat_sessions_table.c.user_id == user_db_id)
        )
    )).scalar_one_or_none()
    if not session_id:
        session_id = (await db.execute(
            insert(chat_sessions_table)
            .values(thread_id=safe_thread_id, agent_name=agent_name, user_id=user_db_id)
            .returning(chat_sessions_table.c.id)
        )).scalar_one()
        await db.commit()  # Commit new session creation

    # Extract temporary document IDs for the current query
    temp_document_ids_for_this_query = [f['temp_doc_id'] for f in request.files]
    # Also capture the file names and sizes for storing in chat_messages_table meta
    quick_uploaded_files_meta = [
        {"file_name": f['file_name'], "file_size": f['file_size'], "temp_doc_id": f['temp_doc_id']} for f in
        request.files]

    # Save the human query with quick_uploaded_files meta
    human_message_meta = {"quick_uploaded_files": quick_uploaded_files_meta} if quick_uploaded_files_meta else None
    await db.execute(
        insert(chat_messages_table).values(
            session_id=session_id,
            sender="human",
            content=request.query,
            timestamp=datetime.utcnow(),
            meta=human_message_meta
        )
    )
    await db.commit()

    # Re-load full history (including meta)
    history_rows = (await db.execute(
        select(chat_messages_table.c.sender, chat_messages_table.c.content,
               chat_messages_table.c.meta)
        .where(chat_messages_table.c.session_id == session_id)
        .order_by(chat_messages_table.c.timestamp.asc())
    )).fetchall()
    history = [
        HumanMessage(content=row.content) if row.sender == "human" else AIMessage(content=row.content)
        for row in history_rows
    ]

    # Create audit run with serialized initial messages
    run_db_id = (await db.execute(
        insert(agent_runs).values(
            run_id=request_id,
            session_id=session_id,
            start_time=datetime.utcnow(),
            status="running",
            initial_messages=_serialize_for_db(history),
        )
        .returning(agent_runs.c.id)
    )).scalar_one()
    await db.commit()

    # Build and execute the agent
    tools = []
    if "retriever" in agent_config.tools:
        retriever_tool = await get_retriever_tool(agent_config, user_db_id, safe_thread_id)
        tools.append(retriever_tool)
    if "arxiv" in agent_config.tools:
        tools.extend(get_arxiv_tool())
    if "search_engine" in agent_config.tools:
        tools.append(SearchTool(agent_config=agent_config))
    if "sql_toolkit" in agent_config.tools:
        db_src = next((s for s in agent_config.sources if s.type == "db"), None)
        if db_src and db_src.db_connection:
            tools.extend(
                create_sql_toolkit(
                    db_connection_string=db_src.db_connection,
                    llm_model_name=agent_config.llm_model,
                )
            )
    if "api_toolkit" in agent_config.tools:
        tools.append(APIToolkit())

    agent = get_agent_workflow(tools).compile()
    initial_state: AgentState = {
        "messages": history,
        "request_id": request_id,
        "agent_config": agent_config,
        "temp_document_ids": temp_document_ids_for_this_query,
    }

    final_state = {}
    try:
        # Pass a custom `RunnableConfig` for the LangGraph agent
        # to inject the DB session, user_id, and thread_id.
        run_config = {
            "configurable": {
                "db_session": db,
                "user_id": user_db_id,
                "thread_id": safe_thread_id,
                "session_id": session_id,
                "tools": tools,
            }
        }

        # Ainvoke the agent with the initial state and config.
        final_state_output = await agent.ainvoke(initial_state, config=run_config)

        final_state = final_state_output
        extracted_citations = final_state.get("citations", [])
        ai_response_content_str = final_state.get('messages', [])[-1].content if final_state.get('messages') else ""

    except Exception as e:
        logger.error(f"Agent run {request_id} failed: {e}", exc_info=True)
        final_state = {"error": True, "errorMessage": str(e)}
        retrieved_documents_meta_for_citation = []
        ai_response_content_str = ""
        extracted_citations = []

    if extracted_citations:
        logger.info(f"Using {len(extracted_citations)} parsed citations for run {request_id}.")
    else:
        logger.info(f"No parsed citations for run {request_id}.")

    if ai_response_content_str and not final_state.get("error"):
        ai_message_meta = {"citations": extracted_citations} if extracted_citations else None
        await db.execute(
            insert(chat_messages_table).values(
                session_id=session_id,
                sender="ai",
                content=ai_response_content_str,
                timestamp=datetime.utcnow(),
                meta=ai_message_meta
            )
        )
        await db.commit()

    serialized = _serialize_for_db(final_state)
    await db.execute(
        update(agent_runs)
        .where(agent_runs.c.id == run_db_id)
        .values(
            end_time=datetime.utcnow(),
            status="completed" if not final_state.get("error") else "failed",
            final_state=serialized,
        )
    )
    await db.commit()
    logger.info(f"User '{current_user.username}' queried agent '{agent_name}' (Run ID: {request_id}).")

    return QueryResponse(response=ai_response_content_str, run_id=request_id, final_state=serialized,
                         citations=extracted_citations)
