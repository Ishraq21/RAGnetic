import logging
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, join, and_

from app.db import get_db
from app.db.models import conversation_metrics_table, chat_sessions_table, users_table
from app.core.security import PermissionChecker
from app.schemas.security import User
from pydantic import BaseModel, Field

logger = logging.getLogger("ragnetic")
router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics API"])


# Pydantic model for the aggregated usage summary response
class UsageSummaryEntry(BaseModel):
    agent_name: str = Field(..., description="Name of the agent.")
    llm_model: str = Field(..., description="Name of the LLM model used.")
    user_id: str = Field(..., description="User ID (username) who initiated the requests.")
    total_prompt_tokens: int = Field(..., description="Total prompt tokens sent to LLM.")
    total_completion_tokens: int = Field(..., description="Total completion tokens received from LLM.")
    total_tokens: int = Field(..., description="Total combined prompt and completion tokens.")
    total_llm_cost_usd: float = Field(..., description="Total estimated cost from LLM usage in USD.")
    total_embedding_cost_usd: float = Field(..., description="Total estimated cost from embedding usage in USD.")
    total_estimated_cost_usd: float = Field(..., description="Total estimated cost (LLM + Embedding) in USD.")
    avg_retrieval_time_s: float = Field(..., description="Average time taken for document retrieval in seconds.")
    avg_generation_time_s: float = Field(..., description="Average time taken for LLM generation in seconds.")
    total_requests: int = Field(..., description="Total number of requests in this aggregated group.")

    class Config:
        from_attributes = True


@router.get("/usage-summary", response_model=List[UsageSummaryEntry])
async def get_usage_summary(
        agent_name: Optional[str] = Query(None, description="Filter metrics by a specific agent name."),
        llm_model: Optional[str] = Query(None, description="Filter metrics by a specific LLM model name."),
        user_id: Optional[str] = Query(None, description="Filter metrics by a specific user ID (username)."),
        start_time: Optional[datetime] = Query(None,
                                               description="Filter metrics after this timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."),
        end_time: Optional[datetime] = Query(None,
                                             description="Filter metrics before this timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."),
        limit: int = Query(50, ge=1, le=100, description="Limit the number of aggregated results."),
        current_user: User = Depends(PermissionChecker(["analytics:read_usage"])),  # Require permission
        db: AsyncSession = Depends(get_db),
):
    """
    Retrieves aggregated LLM and embedding usage and cost metrics.
    Requires: 'analytics:read_usage' permission.
    """
    logger.info(
        f"API: User '{current_user.username}' fetching usage summary with filters: {agent_name=}, {llm_model=}, {user_id=}, {start_time=}, {end_time=}")

    try:
        # Initial join of conversation_metrics_table with chat_sessions_table and users_table
        stmt = select(
            func.sum(conversation_metrics_table.c.prompt_tokens).label("total_prompt_tokens"),
            func.sum(conversation_metrics_table.c.completion_tokens).label("total_completion_tokens"),
            func.sum(conversation_metrics_table.c.total_tokens).label("total_tokens"),
            func.sum(conversation_metrics_table.c.estimated_cost_usd).label("total_llm_cost_usd"),
            func.sum(conversation_metrics_table.c.embedding_cost_usd).label("total_embedding_cost_usd"),
            func.avg(conversation_metrics_table.c.retrieval_time_s).label("avg_retrieval_time_s"),
            func.avg(conversation_metrics_table.c.generation_time_s).label("avg_generation_time_s"),
            func.count(conversation_metrics_table.c.request_id).label("total_requests"),
            chat_sessions_table.c.agent_name,
            conversation_metrics_table.c.llm_model,
            users_table.c.user_id.label("user_id_alias")  # Alias to avoid conflict with `user_id` param
        ).outerjoin(
            chat_sessions_table, conversation_metrics_table.c.session_id == chat_sessions_table.c.id
        ).outerjoin(
            users_table, chat_sessions_table.c.user_id == users_table.c.id
        )

        filters = []
        if agent_name:
            filters.append(chat_sessions_table.c.agent_name == agent_name)
        if llm_model:
            filters.append(conversation_metrics_table.c.llm_model == llm_model)
        if user_id:
            filters.append(users_table.c.user_id == user_id)  # Filter by username
        if start_time:
            filters.append(conversation_metrics_table.c.timestamp >= start_time)
        if end_time:
            filters.append(conversation_metrics_table.c.timestamp <= end_time)

        if filters:
            stmt = stmt.where(and_(*filters))  # Apply all filters

        # Group by Agent Name, LLM Model, and User ID (username)
        stmt = stmt.group_by(
            chat_sessions_table.c.agent_name,
            conversation_metrics_table.c.llm_model,
            users_table.c.user_id
        )

        # Order by total LLM cost descending (can be changed to total_estimated_cost_usd)
        stmt = stmt.order_by(func.sum(conversation_metrics_table.c.estimated_cost_usd).desc())

        stmt = stmt.limit(limit)

        result = await db.execute(stmt)
        rows = result.fetchall()

        if not rows:
            return []  # Return empty list if no results

        # Manually construct list of dictionaries for Pydantic response model
        # Use column names directly as returned by SQLAlchemy, then map to Pydantic model
        response_data = []
        for row in rows:
            # Handle potential None from outer joins for grouping columns
            agent_name_val = row.agent_name if row.agent_name is not None else "N/A"
            llm_model_val = row.llm_model if row.llm_model is not None else "N/A"
            user_id_val = row.user_id_alias if row.user_id_alias is not None else "N/A (No User)"

            # Sum LLM and Embedding costs for total estimated cost
            total_llm_cost = row.total_llm_cost_usd if row.total_llm_cost_usd is not None else 0.0
            total_embedding_cost = row.total_embedding_cost_usd if row.total_embedding_cost_usd is not None else 0.0
            total_estimated_cost = total_llm_cost + total_embedding_cost

            response_data.append(UsageSummaryEntry(
                agent_name=agent_name_val,
                llm_model=llm_model_val,
                user_id=user_id_val,
                total_prompt_tokens=row.total_prompt_tokens,
                total_completion_tokens=row.total_completion_tokens,
                total_tokens=row.total_tokens,
                total_llm_cost_usd=total_llm_cost,
                total_embedding_cost_usd=total_embedding_cost,
                total_estimated_cost_usd=total_estimated_cost,
                avg_retrieval_time_s=row.avg_retrieval_time_s if row.avg_retrieval_time_s is not None else 0.0,
                avg_generation_time_s=row.avg_generation_time_s if row.avg_generation_time_s is not None else 0.0,
                total_requests=row.total_requests
            ))

        return response_data

    except HTTPException:  # Re-raise HTTPExceptions (e.g., from PermissionChecker)
        raise
    except Exception as e:
        logger.error(f"API: Failed to fetch usage summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve usage summary.")