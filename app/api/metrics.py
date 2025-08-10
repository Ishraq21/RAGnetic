# app/api/metrics.py
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, Query, status, Response, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, join, func, text, case, and_

from app.db import get_db
from app.core.security import get_http_api_key, PermissionChecker
from app.db.models import ragnetic_logs_table, conversation_metrics_table, chat_sessions_table
from app.schemas.security import User

logger = logging.getLogger("ragnetic")

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics API"])


# --- Pydantic Models for API Responses ---

class AggregatedMetricsSummary(BaseModel):
    """Aggregated summary of key application metrics."""
    total_runs: int = Field(..., description="Total number of agent runs.")
    total_requests: int = Field(..., description="Total number of requests with LLM interaction.")
    total_errors: int = Field(..., description="Total number of runs that ended in an error.")
    error_rate: float = Field(..., description="Percentage of runs that ended in an error.")
    avg_run_latency_s: float = Field(..., description="Average latency of a full agent run in seconds.")
    total_llm_cost_usd: float = Field(..., description="Total estimated LLM cost in USD.")
    total_embedding_cost_usd: float = Field(..., description="Total estimated embedding cost in USD.")

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
        current_user: User = Depends(PermissionChecker(["metrics:read_conversation_metrics"])),
        db: AsyncSession = Depends(get_db),
):
    """
    Retrieves a paginated and filterable list of conversation metrics.
    Requires: 'metrics:read_conversation_metrics' permission.
    """
    try:
        stmt = select(conversation_metrics_table).select_from(
            join(conversation_metrics_table, chat_sessions_table,
                 conversation_metrics_table.c.session_id == chat_sessions_table.c.id)
        ).order_by(desc(conversation_metrics_table.c.timestamp))

        if agent_name:
            stmt = stmt.where(chat_sessions_table.c.agent_name == agent_name)
        if start_time:
            stmt = stmt.where(conversation_metrics_table.c.timestamp >= start_time)
        if end_time:
            stmt = stmt.where(conversation_metrics_table.c.timestamp <= end_time)

        stmt = stmt.limit(limit).offset(offset)

        result = await db.execute(stmt)
        metrics = result.fetchall()
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API: Failed to fetch metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve metrics.")

@router.get("/summary", response_model=AggregatedMetricsSummary)
async def get_metrics_summary(
    current_user: User = Depends(PermissionChecker(["metrics:read_summary"])),
    db: AsyncSession = Depends(get_db)
):
    """
    Provides a high-level, aggregated summary of key application metrics.
    Requires: 'metrics:read_summary' permission.
    """
    logger.info(f"API: User '{current_user.username}' fetching a metrics summary.")
    try:
        runs_stmt = select(
            func.count().label("total_runs"),
            func.sum(case((conversation_metrics_table.c.estimated_cost_usd > 0, 1), else_=0)).label("total_requests"),
            func.sum(case((conversation_metrics_table.c.estimated_cost_usd < 0, 1), else_=0)).label("total_errors"),
            func.avg(conversation_metrics_table.c.generation_time_s).label("avg_latency"),
            func.sum(conversation_metrics_table.c.estimated_cost_usd).label("total_llm_cost"),
            func.sum(conversation_metrics_table.c.embedding_cost_usd).label("total_embedding_cost")
        ).select_from(conversation_metrics_table)

        runs_result = await db.execute(runs_stmt)
        runs_data = runs_result.first()

        total_runs = runs_data.total_runs if runs_data.total_runs else 0
        total_errors = runs_data.total_errors if runs_data.total_errors else 0

        error_rate = total_errors / total_runs if total_runs > 0 else 0.0

        return AggregatedMetricsSummary(
            total_runs=total_runs,
            total_requests=runs_data.total_requests if runs_data.total_requests else 0,
            total_errors=total_errors,
            error_rate=error_rate,
            avg_run_latency_s=runs_data.avg_latency if runs_data.avg_latency else 0.0,
            total_llm_cost_usd=runs_data.total_llm_cost if runs_data.total_llm_cost else 0.0,
            total_embedding_cost_usd=runs_data.total_embedding_cost if runs_data.total_embedding_cost else 0.0
        )

    except Exception as e:
        logger.error(f"API: Failed to fetch metrics summary: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Could not retrieve metrics summary.")

@router.get("/prometheus", tags=["Metrics API"], summary="Exposes Prometheus-compatible metrics")
async def get_prometheus_metrics(
    current_user: User = Depends(PermissionChecker(["metrics:read_prometheus"])),
    db: AsyncSession = Depends(get_db)
):
    """
    Exposes application metrics in a Prometheus-compatible format.
    Requires: 'metrics:read_prometheus' permission.
    """
    try:
        runs_stmt = select(
            func.count(conversation_metrics_table.c.request_id).label("total_requests"),
            func.sum(conversation_metrics_table.c.estimated_cost_usd).label("total_llm_cost"),
            func.sum(conversation_metrics_table.c.embedding_cost_usd).label("total_embedding_cost"),
            func.sum(conversation_metrics_table.c.prompt_tokens).label("total_prompt_tokens"),
            func.sum(conversation_metrics_table.c.completion_tokens).label("total_completion_tokens"),
            func.count(case((conversation_metrics_table.c.estimated_cost_usd < 0, 1), else_=None)).label("total_errors"),
            func.avg(conversation_metrics_table.c.generation_time_s).label("avg_generation_time_s")
        ).select_from(conversation_metrics_table)

        runs_result = await db.execute(runs_stmt)
        runs_data = runs_result.first()

        metrics_string = ""

        if runs_data:
            metrics_string += '# HELP ragnetic_requests_total Total number of agent requests.\n'
            metrics_string += '# TYPE ragnetic_requests_total counter\n'
            metrics_string += f'ragnetic_requests_total {runs_data.total_requests if runs_data.total_requests else 0}\n'

            metrics_string += '# HELP ragnetic_llm_cost_total Total estimated LLM cost in USD.\n'
            metrics_string += '# TYPE ragnetic_llm_cost_total counter\n'
            metrics_string += f'ragnetic_llm_cost_total {runs_data.total_llm_cost if runs_data.total_llm_cost else 0.0}\n'

            metrics_string += '# HELP ragnetic_average_generation_latency_seconds Average agent generation latency in seconds.\n'
            metrics_string += '# TYPE ragnetic_average_generation_latency_seconds gauge\n'
            metrics_string += f'ragnetic_average_generation_latency_seconds {runs_data.avg_generation_time_s if runs_data.avg_generation_time_s else 0.0}\n'

            metrics_string += '# HELP ragnetic_errors_total Total number of agent errors.\n'
            metrics_string += '# TYPE ragnetic_errors_total counter\n'
            metrics_string += f'ragnetic_errors_total {runs_data.total_errors if runs_data.total_errors else 0}\n'

            metrics_string += '# HELP ragnetic_prompt_tokens_total Total prompt tokens sent to LLM.\n'
            metrics_string += '# TYPE ragnetic_prompt_tokens_total counter\n'
            metrics_string += f'ragnetic_prompt_tokens_total {runs_data.total_prompt_tokens if runs_data.total_prompt_tokens else 0}\n'

            metrics_string += '# HELP ragnetic_completion_tokens_total Total completion tokens received from LLM.\n'
            metrics_string += '# TYPE ragnetic_completion_tokens_total counter\n'
            metrics_string += f'ragnetic_completion_tokens_total {runs_data.total_completion_tokens if runs_data.total_completion_tokens else 0}\n'

        return Response(content=metrics_string, media_type="text/plain")

    except Exception as e:
        logger.error(f"API: Failed to fetch Prometheus metrics: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Could not retrieve Prometheus metrics.")