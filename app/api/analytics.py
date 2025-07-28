import logging
from pathlib import Path

import pandas as pd
import glob
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.sql import expression

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, join, and_

from app.db import get_db
from app.db.models import conversation_metrics_table, chat_sessions_table, users_table
from app.core.security import PermissionChecker
from app.schemas.security import User
from pydantic import BaseModel, Field

from app.core.config import get_path_settings

logger = logging.getLogger("ragnetic")
router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics API"])

_APP_PATHS = get_path_settings()
_BENCHMARK_DIR = _APP_PATHS["BENCHMARK_DIR"]


class WorkflowRunSummaryEntry(BaseModel):
    workflow_name: str = Field(..., description="Name of the workflow.")
    total_runs: int = Field(..., description="Total number of times the workflow has run.")
    success_rate: float = Field(..., description="Percentage of successful workflow runs.")
    failure_rate: float = Field(..., description="Percentage of failed workflow runs.")
    paused_rate: float = Field(..., description="Percentage of paused workflow runs.")
    avg_duration_s: float = Field(..., description="Average duration of workflow runs in seconds.")
    completed_runs: int = Field(..., description="Total number of completed workflow runs.")
    failed_runs: int = Field(..., description="Total number of failed workflow runs.")
    paused_runs: int = Field(..., description="Total number of paused workflow runs.")

    class Config:
        from_attributes = True
        populate_by_name = True

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



class BenchmarkSummaryEntry(BaseModel):
    agent_name: str
    total_test_cases_evaluated: int = Field(alias="Total Test Cases Evaluated")
    avg_key_fact_recall: float = Field(alias="Avg Key Fact Recall")
    avg_faithfulness: float = Field(alias="Avg Faithfulness")
    avg_answer_relevance: float = Field(alias="Avg Answer Relevance")
    avg_retrieval_f1: float = Field(alias="Avg Retrieval F1")
    total_estimated_cost_usd: float = Field(alias="Total Estimated Cost (USD)")
    total_tokens: int = Field(alias="Total Tokens")
    avg_retrieval_time_s: float = Field(alias="Avg Retrieval Time (s)")
    avg_generation_time_s: float = Field(alias="Avg Generation Time (s)")

    # Agent Configuration Details (from benchmark CSV)
    agent_llm_model: Optional[str] = Field(None, alias="Agent LLM Model (Sample)")
    agent_embedding_model: Optional[str] = Field(None, alias="Agent Embedding Model (Sample)")
    chunking_mode: Optional[str] = Field(None, alias="Chunking Mode (Sample)")
    chunk_size: Optional[int] = Field(None, alias="Chunk Size (Sample)")
    chunk_overlap: Optional[int] = Field(None, alias="Chunk Overlap (Sample)")
    vector_store_type: Optional[str] = Field(None, alias="Vector Store Type (Sample)")
    retrieval_strategy: Optional[str] = Field(None, alias="Retrieval Strategy (Sample)")
    bm25_k: Optional[int] = Field(None, alias="BM25 K (Sample)")
    semantic_k: Optional[int] = Field(None, alias="Semantic K (Sample)")
    rerank_top_n: Optional[int] = Field(None, alias="Rerank Top N (Sample)")
    hit_rate_k_value: Optional[int] = Field(None, alias="Hit Rate K Value (Sample)")

    class Config:
        from_attributes = True
        populate_by_name = True


@router.get("/benchmarks", response_model=List[BenchmarkSummaryEntry])
async def get_benchmark_summary(
        agent_name: Optional[str] = Query(None, description="Filter benchmarks by a specific agent name."),
        latest: bool = Query(False,
                             description="If true, return only the latest benchmark run for the specified agent."),
        current_user: User = Depends(PermissionChecker(["analytics:read_benchmarks"])),  # Require permission
):
    """
    Retrieves aggregated benchmark results from CSV files.
    Requires: 'analytics:read_benchmarks' permission.
    """
    logger.info(
        f"API: User '{current_user.username}' fetching benchmark summary with filters: {agent_name=}, {latest=}")

    try:
        if not os.path.exists(_BENCHMARK_DIR) or not os.listdir(_BENCHMARK_DIR):
            raise HTTPException(status_code=404,
                                detail=f"No benchmark results found in '{_BENCHMARK_DIR}'. Run 'ragnetic evaluate benchmark' first.")

        all_benchmark_files = sorted(glob.glob(str(_BENCHMARK_DIR / "*.csv")), reverse=True)

        if not all_benchmark_files:
            raise HTTPException(status_code=404, detail=f"No .csv benchmark files found in '{_BENCHMARK_DIR}'.")

        filtered_files = []
        if agent_name:
            for f in all_benchmark_files:
                filename_parts = Path(f).name.split('_')
                if len(filename_parts) >= 2 and filename_parts[1] == agent_name:
                    filtered_files.append(f)
            if not filtered_files:
                raise HTTPException(status_code=404, detail=f"No benchmark results found for agent: '{agent_name}'.")
            all_benchmark_files = filtered_files

        if latest and all_benchmark_files:
            all_benchmark_files = [all_benchmark_files[0]]

        all_results_dfs = []
        for f_path in all_benchmark_files:
            try:
                df = pd.read_csv(f_path)
                df.rename(columns={
                    "Total Test Cases Evaluated": "total_test_cases_evaluated",
                    "Avg Key Fact Recall": "avg_key_fact_recall",
                    "Avg Faithfulness": "avg_faithfulness",
                    "Avg Answer Relevance": "avg_answer_relevance",
                    "Avg Retrieval F1": "avg_retrieval_f1",
                    "Total Estimated Cost (USD)": "total_estimated_cost_usd",
                    "Total Tokens": "total_tokens",
                    "Avg Retrieval Time (s)": "avg_retrieval_time_s",
                    "Avg Generation Time (s)": "avg_generation_time_s",
                    # New config columns from benchmark.py
                    "agent_llm_model": "agent_llm_model",
                    "agent_embedding_model": "agent_embedding_model",
                    "chunking_mode": "chunking_mode",
                    "chunk_size": "chunk_size",
                    "chunk_overlap": "chunk_overlap",
                    "vector_store_type": "vector_store_type",
                    "retrieval_strategy": "retrieval_strategy",
                    "bm25_k": "bm25_k",
                    "semantic_k": "semantic_k",
                    "rerank_top_n": "rerank_top_n",
                    "hit_rate_k_value": "hit_rate_k_value",
                }, inplace=True)

                # Add agent_name from filename for consistency
                df['agent_name'] = Path(f_path).name.split('_')[1]  # Use the top-level imported Path

                all_results_dfs.append(df)
            except Exception as e:
                logger.error(f"API: Failed to read benchmark file '{f_path}': {e}", exc_info=True)
                # Continue processing other files even if one fails

        if not all_results_dfs:
            return []

        combined_df = pd.concat(all_results_dfs, ignore_index=True)

        if combined_df.empty:
            return []  # Return empty if concat results in empty

        # Ensure columns exist before trying to aggregate or access them
        # Fill NaN for numeric columns before aggregation to avoid issues
        numeric_cols_to_fill = [
            "total_test_cases_evaluated", "avg_key_fact_recall", "avg_faithfulness",
            "avg_answer_relevance", "avg_retrieval_f1", "total_estimated_cost_usd",
            "total_tokens", "avg_retrieval_time_s", "avg_generation_time_s",
            "chunk_size", "chunk_overlap", "bm25_k", "semantic_k", "rerank_top_n", "hit_rate_k_value"
        ]
        for col in numeric_cols_to_fill:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)

        # Group by agent_name and calculate aggregates using named aggregations
        # This explicitly maps output column names to (source_column, aggregation_function)
        # For config fields, .iloc[0] is used to take the first unique non-NaN value,
        # assuming config is consistent within a single agent's benchmarks.
        actual_agg_dict = {
            'total_test_cases_evaluated': ('question', 'count'),
            'avg_key_fact_recall': ('key_fact_recalled', 'mean'),
            'avg_faithfulness': ('faithfulness', 'mean'),
            'avg_answer_relevance': ('answer_relevance', 'mean'),
            'avg_retrieval_f1': ('retrieval_f1', 'mean'),
            'total_estimated_cost_usd': ('estimated_cost_usd', 'sum'),
            'total_tokens': ('total_tokens', 'sum'),
            'avg_retrieval_time_s': ('retrieval_time_s', 'mean'),
            'avg_generation_time_s': ('generation_time_s', 'mean'),

            # For config fields, take the first non-NaN unique value within the group
            'agent_llm_model': ('agent_llm_model', lambda x: x.dropna().unique()[0] if not x.dropna().empty else None),
            'agent_embedding_model': ('agent_embedding_model',
                                      lambda x: x.dropna().unique()[0] if not x.dropna().empty else None),
            'chunking_mode': ('chunking_mode', lambda x: x.dropna().unique()[0] if not x.dropna().empty else None),
            'chunk_size': ('chunk_size', lambda x: x.dropna().unique()[0] if not x.dropna().empty else None),
            'chunk_overlap': ('chunk_overlap', lambda x: x.dropna().unique()[0] if not x.dropna().empty else None),
            'vector_store_type': ('vector_store_type',
                                  lambda x: x.dropna().unique()[0] if not x.dropna().empty else None),
            'retrieval_strategy': ('retrieval_strategy',
                                   lambda x: x.dropna().unique()[0] if not x.dropna().empty else None),
            'bm25_k': ('bm25_k', lambda x: x.dropna().unique()[0] if not x.dropna().empty else None),
            'semantic_k': ('semantic_k', lambda x: x.dropna().unique()[0] if not x.dropna().empty else None),
            'rerank_top_n': ('rerank_top_n', lambda x: x.dropna().unique()[0] if not x.dropna().empty else None),
            'hit_rate_k_value': ('hit_rate_k_value',
                                 lambda x: x.dropna().unique()[0] if not x.dropna().empty else None),
        }

        # Filter agg_dict to include only columns actually present in combined_df.columns
        # The issue is that some benchmark CSVs might not have all config columns from older runs
        # We need to explicitly handle this when preparing the dict for aggregation

        # Build the aggregation dictionary only with columns that exist in combined_df
        final_agg_dict_for_df = {}
        for k, v in actual_agg_dict.items():
            if isinstance(v, tuple) and v[0] in combined_df.columns:  # Named aggregation (column, func)
                final_agg_dict_for_df[k] = v
            elif k in combined_df.columns:  # Direct column name for simple aggregations
                final_agg_dict_for_df[k] = v

        # Perform aggregation
        aggregated_df = combined_df.groupby('agent_name').agg(**final_agg_dict_for_df).reset_index()

        # Fill None values for object types if they didn't have unique values
        for col in aggregated_df.columns:
            if aggregated_df[col].dtype == 'object':
                aggregated_df[col] = aggregated_df[col].fillna("N/A")
            elif pd.api.types.is_numeric_dtype(aggregated_df[col]):
                aggregated_df[col] = aggregated_df[col].fillna(0)  # Fill numeric NaNs with 0

        return [BenchmarkSummaryEntry.model_validate(row.to_dict()) for idx, row in aggregated_df.iterrows()]

    except HTTPException:  # Re-raise HTTPExceptions (e.g., from PermissionChecker)
        raise
    except Exception as e:
        logger.error(f"API: Failed to fetch benchmark summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve benchmark summary.")


@router.get("/workflow-runs", response_model=List[WorkflowRunSummaryEntry])
async def get_workflow_runs_summary(
        workflow_name: Optional[str] = Query(None, description="Filter metrics by a specific workflow name."),
        status: Optional[str] = Query(None,
                                      description="Filter by workflow status (running, completed, failed, paused)."),
        start_time: Optional[datetime] = Query(None,
                                               description="Filter metrics after this timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."),
        end_time: Optional[datetime] = Query(None,
                                             description="Filter metrics before this timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."),
        limit: int = Query(20, ge=1, le=100, description="Limit the number of aggregated workflow results."),
        current_user: User = Depends(PermissionChecker(["analytics:read_workflow_runs"])),  # Require permission
        db: AsyncSession = Depends(get_db),
):
    """
    Retrieves aggregated workflow run metrics.
    Requires: 'analytics:read_workflow_runs' permission.
    """
    logger.info(
        f"API: User '{current_user.username}' fetching workflow run summary with filters: {workflow_name=}, {status=}, {start_time=}, {end_time=}")

    try:
        from app.db.models import workflows_table, workflow_runs_table  # Import these models locally if not at top

        stmt = select(
            workflows_table.c.name.label("workflow_name"),
            func.count(workflow_runs_table.c.run_id).label("total_runs"),
            func.avg(
                func.julianday(workflow_runs_table.c.end_time) - func.julianday(workflow_runs_table.c.start_time)
            ).label("avg_duration_days"),
            func.sum(
                expression.case((workflow_runs_table.c.status == 'completed', 1), else_=0)
                # FIX: Removed square brackets
            ).label("completed_runs"),
            func.sum(
                expression.case((workflow_runs_table.c.status == 'failed', 1), else_=0)  # FIX: Removed square brackets
            ).label("failed_runs"),
            func.sum(
                expression.case((workflow_runs_table.c.status == 'paused', 1), else_=0)  # FIX: Removed square brackets
            ).label("paused_runs")
        ).join(
            workflows_table, workflow_runs_table.c.workflow_id == workflows_table.c.id
        )

        filters = []
        if workflow_name:
            filters.append(workflows_table.c.name == workflow_name)
        if status:
            filters.append(workflow_runs_table.c.status == status)
        if start_time:
            filters.append(workflow_runs_table.c.start_time >= start_time)
        if end_time:
            filters.append(workflow_runs_table.c.end_time <= end_time)

        if filters:
            stmt = stmt.where(and_(*filters))

        stmt = stmt.group_by(workflows_table.c.name)
        stmt = stmt.order_by(func.count(workflow_runs_table.c.run_id).desc())
        stmt = stmt.limit(limit)

        result = await db.execute(stmt)
        rows = result.fetchall()

        if not rows:
            return []

        response_data = []
        for row in rows:
            total_runs = row.total_runs if row.total_runs is not None else 0
            completed_runs = row.completed_runs if row.completed_runs is not None else 0
            failed_runs = row.failed_runs if row.failed_runs is not None else 0
            paused_runs = row.paused_runs if row.paused_runs is not None else 0

            success_rate = completed_runs / total_runs if total_runs > 0 else 0.0
            failure_rate = failed_runs / total_runs if total_runs > 0 else 0.0
            paused_rate = paused_runs / total_runs if total_runs > 0 else 0.0

            avg_duration_s = (row.avg_duration_days * 86400) if row.avg_duration_days is not None else 0.0

            response_data.append(WorkflowRunSummaryEntry(
                workflow_name=row.workflow_name,
                total_runs=total_runs,
                success_rate=success_rate,
                failure_rate=failure_rate,
                paused_rate=paused_rate,
                avg_duration_s=avg_duration_s,
                completed_runs=completed_runs,
                failed_runs=failed_runs,
                paused_runs=paused_runs
            ))

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API: Failed to fetch workflow run summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve workflow run summary.")


class AgentStepSummaryEntry(BaseModel):
    agent_name: str = Field(..., description="Name of the agent.")
    node_name: str = Field(..., description="Name of the agent step/node (e.g., 'agent', 'retriever').")
    total_calls: int = Field(..., description="Total number of times this node was called.")
    success_rate: float = Field(..., description="Percentage of successful calls to this node.")
    failure_rate: float = Field(..., description="Percentage of failed calls to this node.")
    avg_duration_s: float = Field(..., description="Average duration of calls to this node in seconds.")
    completed_calls: int = Field(..., description="Total number of completed calls to this node.")
    failed_calls: int = Field(..., description="Total number of failed calls to this node.")

    class Config:
        from_attributes = True
        populate_by_name = True


@router.get("/agent-steps", response_model=List[AgentStepSummaryEntry])
async def get_agent_steps_summary(
        agent_name: Optional[str] = Query(None, description="Filter metrics by a specific agent name."),
        node_name: Optional[str] = Query(None,
                                         description="Filter by a specific node name (e.g., 'agent', 'retriever')."),
        status: Optional[str] = Query(None, description="Filter by step status (running, completed, failed)."),
        start_time: Optional[datetime] = Query(None,
                                               description="Filter metrics after this timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."),
        end_time: Optional[datetime] = Query(None,
                                             description="Filter metrics before this timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."),
        limit: int = Query(20, ge=1, le=100, description="Limit the number of aggregated node results."),
        current_user: User = Depends(PermissionChecker(["analytics:read_agent_steps"])),  # Require permission
        db: AsyncSession = Depends(get_db),
):
    """
    Retrieves aggregated agent step metrics.
    Requires: 'analytics:read_agent_steps' permission.
    """
    logger.info(
        f"API: User '{current_user.username}' fetching agent step summary with filters: {agent_name=}, {node_name=}, {status=}, {start_time=}, {end_time=}")

    try:
        from app.db.models import agent_runs, agent_run_steps, chat_sessions_table  # Ensure these are imported

        stmt = select(
            chat_sessions_table.c.agent_name,
            agent_run_steps.c.node_name,
            func.count(agent_run_steps.c.id).label("total_calls"),
            func.avg(
                func.julianday(agent_run_steps.c.end_time) - func.julianday(agent_run_steps.c.start_time)
            ).label("avg_duration_days"),
            func.sum(
                expression.case((agent_run_steps.c.status == 'completed', 1), else_=0)
            ).label("completed_calls"),
            func.sum(
                expression.case((agent_run_steps.c.status == 'failed', 1), else_=0)
            ).label("failed_calls")
        ).join(
            agent_runs, agent_run_steps.c.agent_run_id == agent_runs.c.id
        ).outerjoin(  # Use outerjoin to safely get agent_name, as session_id can be null for agent_runs
            chat_sessions_table, agent_runs.c.session_id == chat_sessions_table.c.id
        )

        filters = []
        if agent_name:
            filters.append(chat_sessions_table.c.agent_name == agent_name)
        if node_name:
            filters.append(agent_run_steps.c.node_name == node_name)
        if status:
            filters.append(agent_run_steps.c.status == status)
        if start_time:
            filters.append(agent_run_steps.c.start_time >= start_time)
        if end_time:
            filters.append(agent_run_steps.c.end_time <= end_time)

        if filters:
            stmt = stmt.where(and_(*filters))

        stmt = stmt.group_by(
            chat_sessions_table.c.agent_name,  # Grouping by agent_name which can be null from outerjoin
            agent_run_steps.c.node_name
        )
        stmt = stmt.order_by(func.count(agent_run_steps.c.id).desc())
        stmt = stmt.limit(limit)

        result = await db.execute(stmt)
        rows = result.fetchall()

        if not rows:
            return []

        response_data = []
        for row in rows:
            total_calls = row.total_calls if row.total_calls is not None else 0
            completed_calls = row.completed_calls if row.completed_calls is not None else 0
            failed_calls = row.failed_calls if row.failed_calls is not None else 0

            success_rate = completed_calls / total_calls if total_calls > 0 else 0.0
            failure_rate = failed_calls / total_calls if total_calls > 0 else 0.0

            avg_duration_s = (row.avg_duration_days * 86400) if row.avg_duration_days is not None else 0.0

            response_data.append(AgentStepSummaryEntry(
                agent_name=row.agent_name if row.agent_name is not None else "N/A",  # Handle None agent_name
                node_name=row.node_name,
                total_calls=total_calls,
                success_rate=success_rate,
                failure_rate=failure_rate,
                avg_duration_s=avg_duration_s,
                completed_calls=completed_calls,
                failed_calls=failed_calls
            ))

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API: Failed to fetch agent step summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve agent step summary.")
