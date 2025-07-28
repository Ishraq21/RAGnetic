# app/api/query.py
import logging
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update
from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4

from app.db import get_db
from app.core.security import get_http_api_key, PermissionChecker # Import PermissionChecker
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
from app.db.dao import save_conversation_metrics_sync
from app.schemas.security import User # Import User schema

logger = logging.getLogger("ragnetic")

router = APIRouter(prefix="/api/v1/agents", tags=["Query API"])


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query to the agent.")
    # user_id and thread_id will now be primarily derived from authenticated user/session
    # but kept as optional for backward compatibility or specific use cases.
    user_id: Optional[str] = Field(None, description="A unique identifier for the user (will be overridden by authenticated user_id if available).")
    thread_id: Optional[str] = Field(None, description="A unique identifier for the conversation thread.")


class QueryResponse(BaseModel):
    response: str = Field(..., description="The agent's final response content.")
    run_id: str = Field(..., description="The unique ID for this specific run, for auditing.")
    final_state: Dict[str, Any] = Field(..., description="The final state of the agent graph, containing metrics.")


@router.post("/{agent_name}/query", response_model=QueryResponse)
async def query_agent(
        agent_name: str,
        request: QueryRequest = Body(...),
        # Require 'agent:query' permission to query an agent
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
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred while loading agent configuration for '{agent_name}'. Please check server logs.")

    # Prepare IDs
    # Use the authenticated user's user_id. The request.user_id is now secondary/legacy.
    safe_user_id = sanitize_for_path(current_user.username) # Use username from authenticated user
    safe_thread_id = (
        sanitize_for_path(request.thread_id) if request.thread_id else f"api-thread-{uuid4().hex[:8]}"
    )
    request_id = str(uuid4())

    # Ensure user record exists based on authenticated user's ID
    # This part can be simplified since current_user is already from DB.
    # We need the DB ID (int) of the user, not just the user_id (string)
    user_db_id = current_user.id # Use the ID of the authenticated user

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

    # Save the human query
    await db.execute(
        insert(chat_messages_table).values(
            session_id=session_id,
            sender="human",
            content=request.query,
            timestamp=datetime.utcnow(),
        )
    )
    await db.commit()

    # Re-load full history
    history_rows = (await db.execute(
        select(chat_messages_table.c.sender, chat_messages_table.c.content)
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
        tools.append(get_retriever_tool(agent_config))
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

    agent = get_agent_workflow(tools).compile()
    initial_state: AgentState = {
        "messages": history,
        "request_id": request_id,
        "agent_config": agent_config,
    }

    try:
        final_state = await agent.ainvoke(initial_state, {"configurable": {"thread_id": safe_thread_id}})
    except Exception as e:
        logger.error(f"Agent run {request_id} failed: {e}", exc_info=True)
        final_state = {"error": True, "errorMessage": str(e)}

    # Extract AI reply
    ai_content = ""
    if final_state.get("messages"):
        ai_content = final_state["messages"][-1].content

    # Save AI reply if successful
    if ai_content and not final_state.get("error"):
        await db.execute(
            insert(chat_messages_table).values(
                session_id=session_id,
                sender="ai",
                content=ai_content,
                timestamp=datetime.utcnow(),
            )
        )

    # Save metrics to the database
    if final_state.get("total_tokens") is not None:
        metrics_data = {
            "session_id": session_id,
            "request_id": request_id,
            "prompt_tokens": final_state.get("prompt_tokens", 0),
            "completion_tokens": final_state.get("completion_tokens", 0),
            "total_tokens": final_state.get("total_tokens", 0),
            "retrieval_time_s": final_state.get("retrieval_time_s"),
            "generation_time_s": final_state.get("generation_time_s"),
            "estimated_cost_usd": final_state.get("estimated_cost_usd"),
            "timestamp": datetime.utcnow()
        }
        await save_conversation_metrics_sync(db, metrics_data)

    # Finalize audit run with serialized final_state
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

    return QueryResponse(response=ai_content, run_id=request_id, final_state=serialized)

