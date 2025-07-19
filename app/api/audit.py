# app/api/audit.py
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field

from app.db import get_db
from app.core.security import get_http_api_key
from app.db.models import agent_runs, agent_run_steps, chat_sessions_table, users_table


# --- Pydantic Models for API Responses ---

class AgentRunStepModel(BaseModel):
    id: int
    node_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    status: str

    class Config:
        from_attributes = True


class AgentRunSummaryModel(BaseModel):
    run_id: str
    status: str
    start_time: datetime
    agent_name: str
    topic_name: Optional[str] = "New Chat"

    class Config:
        from_attributes = True


class AgentRunDetailModel(AgentRunSummaryModel):
    user_identifier: str
    end_time: Optional[datetime] = None
    duration_s: Optional[float] = None
    initial_messages: Optional[List[Dict[str, Any]]] = None
    final_state: Optional[Dict[str, Any]] = None
    steps: List[AgentRunStepModel] = []

    class Config:
        from_attributes = True


# --- API Router for Auditing ---
router = APIRouter(prefix="/api/v1/audit", tags=["Audit API"])


@router.get("/runs", response_model=List[AgentRunSummaryModel])
async def list_agent_runs(
        agent_name: Optional[str] = Query(None, description="Filter runs by a specific agent name."),
        limit: int = Query(20, ge=1, le=100, description="Number of recent runs to display."),
        offset: int = Query(0, ge=0, description="Number of runs to skip for pagination."),
        api_key: str = Depends(get_http_api_key),
        db: AsyncSession = Depends(get_db)
):
    """
    Retrieves a list of recent agent runs, with optional filtering.
    """
    try:
        stmt = (
            select(
                agent_runs.c.run_id,
                agent_runs.c.status,
                agent_runs.c.start_time,
                chat_sessions_table.c.agent_name,
                chat_sessions_table.c.topic_name,
            )
            .join(chat_sessions_table, agent_runs.c.session_id == chat_sessions_table.c.id)
        )

        if agent_name:
            stmt = stmt.where(chat_sessions_table.c.agent_name == agent_name)

        stmt = stmt.order_by(agent_runs.c.start_time.desc()).limit(limit).offset(offset)

        result = await db.execute(stmt)
        runs = result.fetchall()

        return runs
    except Exception as e:
        logging.error(f"API: Error fetching agent runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching agent runs.")


@router.get("/runs/{run_id}", response_model=AgentRunDetailModel)
async def get_run_details(
        run_id: str,
        api_key: str = Depends(get_http_api_key),
        db: AsyncSession = Depends(get_db)
):
    """
    Fetches and displays the details for a single agent run and all of its steps.
    """
    try:
        # 1. Fetch the main run details
        run_stmt = (
            select(
                agent_runs,
                chat_sessions_table.c.agent_name,
                users_table.c.user_id.label("user_identifier")
            )
            .join(chat_sessions_table, agent_runs.c.session_id == chat_sessions_table.c.id)
            .join(users_table, chat_sessions_table.c.user_id == users_table.c.id)
            .where(agent_runs.c.run_id == run_id)
        )
        run_result = await db.execute(run_stmt)
        run = run_result.first()

        if not run:
            raise HTTPException(status_code=404, detail=f"Run with ID '{run_id}' not found.")

        # 2. Fetch the steps for that run
        steps_stmt = (
            select(agent_run_steps)
            .where(agent_run_steps.c.agent_run_id == run.id)
            .order_by(agent_run_steps.c.start_time.asc())
        )
        steps_result = await db.execute(steps_stmt)
        steps = steps_result.fetchall()

        # 3. Combine into a single response model
        run_data = dict(run._mapping)
        run_data['steps'] = steps
        if run.end_time and run.start_time:
            run_data['duration_s'] = (run.end_time - run.start_time).total_seconds()

        return run_data

    except HTTPException:
        raise  # Re-raise HTTPException to preserve status code and detail
    except Exception as e:
        logging.error(f"API: Error inspecting run '{run_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while inspecting run '{run_id}'.")
