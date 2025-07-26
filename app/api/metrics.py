# app/api/metrics.py
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, join

from app.db import get_db
from app.core.security import get_http_api_key, PermissionChecker # Import PermissionChecker
from app.db.models import ragnetic_logs_table, conversation_metrics_table, chat_sessions_table # Import chat_sessions_table
from app.schemas.security import User # Import User schema

logger = logging.getLogger("ragnetic")

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics API"])


# --- Pydantic Models for API Responses ---

class LogEntryModel(BaseModel):
    timestamp: datetime
    level: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line: Optional[int] = None
    exc_info: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class MetricEntryModel(BaseModel):
    id: int
    session_id: int
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    retrieval_time_s: Optional[float] = None
    generation_time_s: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    timestamp: datetime

    class Config:
        from_attributes = True


# --- API Endpoints ---

@router.get("/logs", response_model=List[LogEntryModel])
async def get_logs(
        level: Optional[str] = Query(None, description="Filter by log level (e.g., 'ERROR', 'CRITICAL')."),
        module: Optional[str] = Query(None, description="Filter by module name (e.g., 'ragnetic')."),
        start_time: Optional[datetime] = Query(None, description="Filter logs after this timestamp."),
        end_time: Optional[datetime] = Query(None, description="Filter logs before this timestamp."),
        limit: int = Query(100, ge=1, le=1000, description="Number of log entries to retrieve."),
        offset: int = Query(0, ge=0, description="Number of log entries to skip for pagination."),
        # Users need 'metrics:read_logs' permission to view logs
        current_user: User = Depends(PermissionChecker(["metrics:read_logs"])),
        db: AsyncSession = Depends(get_db),
):
    """
    Retrieves a paginated and filterable list of structured logs.
    Requires: 'metrics:read_logs' permission.
    """
    try:
        stmt = select(ragnetic_logs_table).order_by(desc(ragnetic_logs_table.c.timestamp))

        if level:
            stmt = stmt.where(ragnetic_logs_table.c.level == level.upper())
        if module:
            stmt = stmt.where(ragnetic_logs_table.c.module.contains(module))
        if start_time:
            stmt = stmt.where(ragnetic_logs_table.c.timestamp >= start_time)
        if end_time:
            stmt = stmt.where(ragnetic_logs_table.c.timestamp <= end_time)

        stmt = stmt.limit(limit).offset(offset)

        result = await db.execute(stmt)
        logs = result.fetchall()
        return logs
    except Exception as e:
        logger.error(f"API: Failed to fetch logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve logs.")


@router.get("/metrics", response_model=List[MetricEntryModel])
async def get_metrics(
        agent_name: Optional[str] = Query(None, description="Filter by agent name."),
        start_time: Optional[datetime] = Query(None, description="Filter metrics after this timestamp."),
        end_time: Optional[datetime] = Query(None, description="Filter metrics before this timestamp."),
        limit: int = Query(100, ge=1, le=1000, description="Number of metrics entries to retrieve."),
        offset: int = Query(0, ge=0, description="Number of metrics entries to skip for pagination."),
        # Users need 'metrics:read_conversation_metrics' permission to view conversation metrics
        current_user: User = Depends(PermissionChecker(["metrics:read_conversation_metrics"])),
        db: AsyncSession = Depends(get_db),
):
    """
    Retrieves a paginated and filterable list of conversation metrics.
    Requires: 'metrics:read_conversation_metrics' permission.
    """
    try:
        # Start with a join to chat_sessions_table to enable filtering by agent_name
        stmt = select(conversation_metrics_table).select_from(
            join(conversation_metrics_table, chat_sessions_table,
                 conversation_metrics_table.c.session_id == chat_sessions_table.c.id)
        ).order_by(desc(conversation_metrics_table.c.timestamp))

        if agent_name:
            # Now filtering by agent_name is possible
            stmt = stmt.where(chat_sessions_table.c.agent_name == agent_name)
        if start_time:
            stmt = stmt.where(conversation_metrics_table.c.timestamp >= start_time)
        if end_time:
            stmt = stmt.where(conversation_metrics_table.c.timestamp <= end_time)

        stmt = stmt.limit(limit).offset(offset)

        result = await db.execute(stmt)
        metrics = result.fetchall()
        return metrics
    except HTTPException: # Re-raise HTTPExceptions (e.g., from PermissionChecker)
        raise
    except Exception as e:
        logger.error(f"API: Failed to fetch metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve metrics.")

