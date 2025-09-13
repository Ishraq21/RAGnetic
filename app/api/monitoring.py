# app/api/monitoring.py
import logging
import psutil
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, text

from app.db import get_db
from app.core.security import PermissionChecker
from app.db.models import (
    user_api_keys_table, users_table, ragnetic_logs_table, 
    fine_tuned_models_table, document_chunks_table, conversation_metrics_table,
    agent_runs, benchmark_runs_table, lambda_runs
)
from app.schemas.security import User
from app.core.config import get_path_settings

logger = logging.getLogger("ragnetic")
router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring API"])

_APP_PATHS = get_path_settings()

# --- Pydantic Models ---

class SecurityMetrics(BaseModel):
    active_api_keys: int = Field(..., description="Number of active API keys")
    failed_auth_24h: int = Field(..., description="Failed authentication attempts in last 24h")
    rate_limited_24h: int = Field(..., description="Rate limited requests in last 24h")
    active_sessions: int = Field(..., description="Currently active user sessions")

class ResourceMetrics(BaseModel):
    memory_used_mb: float = Field(..., description="Memory usage in MB")
    memory_total_mb: float = Field(..., description="Total memory in MB")
    memory_percent: float = Field(..., description="Memory usage percentage")
    cpu_percent: float = Field(..., description="CPU usage percentage")
    disk_used_gb: float = Field(..., description="Disk usage in GB")
    disk_total_gb: float = Field(..., description="Total disk space in GB")
    disk_percent: float = Field(..., description="Disk usage percentage")

class TrainingOverview(BaseModel):
    running_jobs: int = Field(..., description="Currently running training jobs")
    total_models: int = Field(..., description="Total fine-tuned models")
    gpu_hours_24h: float = Field(..., description="GPU hours consumed in last 24h")
    training_cost_24h: float = Field(..., description="Training cost in last 24h USD")

class DataPipelineMetrics(BaseModel):
    total_documents: int = Field(..., description="Total documents in vector store")
    embeddings_24h: int = Field(..., description="Embedding operations in last 24h")
    vector_store_size_mb: float = Field(..., description="Vector store size in MB")
    failed_ingests_24h: int = Field(..., description="Failed ingestion attempts in last 24h")

# --- API Endpoints ---

@router.get("/security", response_model=SecurityMetrics)
async def get_security_metrics(
    current_user: User = Depends(PermissionChecker(["monitoring:read_security"])),
    db: AsyncSession = Depends(get_db)
):
    """Get security-related metrics for monitoring dashboard."""
    try:
        # Active API keys (non-revoked)
        active_keys_stmt = select(func.count(user_api_keys_table.c.id)).where(
            user_api_keys_table.c.revoked == False
        )
        active_keys_result = await db.execute(active_keys_stmt)
        active_api_keys = active_keys_result.scalar() or 0

        # Failed auth attempts in last 24h (from logs)
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        failed_auth_stmt = select(func.count(ragnetic_logs_table.c.id)).where(
            and_(
                ragnetic_logs_table.c.timestamp >= twenty_four_hours_ago,
                ragnetic_logs_table.c.level == 'ERROR',
                ragnetic_logs_table.c.message.like('%authentication%')
            )
        )
        failed_auth_result = await db.execute(failed_auth_stmt)
        failed_auth_24h = failed_auth_result.scalar() or 0

        # Rate limited requests in last 24h (from logs)
        rate_limited_stmt = select(func.count(ragnetic_logs_table.c.id)).where(
            and_(
                ragnetic_logs_table.c.timestamp >= twenty_four_hours_ago,
                ragnetic_logs_table.c.message.like('%rate limit%')
            )
        )
        rate_limited_result = await db.execute(rate_limited_stmt)
        rate_limited_24h = rate_limited_result.scalar() or 0

        # Active sessions (users with recent activity)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        active_sessions_stmt = select(func.count(func.distinct(conversation_metrics_table.c.session_id))).where(
            conversation_metrics_table.c.timestamp >= one_hour_ago
        )
        active_sessions_result = await db.execute(active_sessions_stmt)
        active_sessions = active_sessions_result.scalar() or 0

        return SecurityMetrics(
            active_api_keys=active_api_keys,
            failed_auth_24h=failed_auth_24h,
            rate_limited_24h=rate_limited_24h,
            active_sessions=active_sessions
        )
    except Exception as e:
        logger.error(f"Failed to get security metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve security metrics")

@router.get("/resources", response_model=ResourceMetrics)
async def get_resource_metrics(
    current_user: User = Depends(PermissionChecker(["monitoring:read_resources"]))
):
    """Get system resource utilization metrics."""
    try:
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)
        memory_percent = memory.percent

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)

        # Disk metrics (for the app directory)
        disk_usage = psutil.disk_usage('/')
        disk_used_gb = disk_usage.used / (1024 * 1024 * 1024)
        disk_total_gb = disk_usage.total / (1024 * 1024 * 1024)
        disk_percent = (disk_usage.used / disk_usage.total) * 100

        return ResourceMetrics(
            memory_used_mb=round(memory_used_mb, 1),
            memory_total_mb=round(memory_total_mb, 1),
            memory_percent=round(memory_percent, 1),
            cpu_percent=round(cpu_percent, 1),
            disk_used_gb=round(disk_used_gb, 1),
            disk_total_gb=round(disk_total_gb, 1),
            disk_percent=round(disk_percent, 1)
        )
    except Exception as e:
        logger.error(f"Failed to get resource metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve resource metrics")

@router.get("/training-overview", response_model=TrainingOverview)
async def get_training_overview(
    current_user: User = Depends(PermissionChecker(["monitoring:read_training"])),
    db: AsyncSession = Depends(get_db)
):
    """Get training and model metrics overview."""
    try:
        # Running training jobs
        running_jobs_stmt = select(func.count(fine_tuned_models_table.c.id)).where(
            fine_tuned_models_table.c.training_status.in_(['pending', 'running'])
        )
        running_jobs_result = await db.execute(running_jobs_stmt)
        running_jobs = running_jobs_result.scalar() or 0

        # Total fine-tuned models
        total_models_stmt = select(func.count(fine_tuned_models_table.c.id)).where(
            fine_tuned_models_table.c.training_status == 'completed'
        )
        total_models_result = await db.execute(total_models_stmt)
        total_models = total_models_result.scalar() or 0

        # GPU hours in last 24h
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        gpu_hours_stmt = select(func.sum(fine_tuned_models_table.c.gpu_hours_consumed)).where(
            and_(
                fine_tuned_models_table.c.created_at >= twenty_four_hours_ago,
                fine_tuned_models_table.c.gpu_hours_consumed.isnot(None)
            )
        )
        gpu_hours_result = await db.execute(gpu_hours_stmt)
        gpu_hours_24h = gpu_hours_result.scalar() or 0.0

        # Training cost in last 24h
        training_cost_stmt = select(func.sum(fine_tuned_models_table.c.estimated_training_cost_usd)).where(
            and_(
                fine_tuned_models_table.c.created_at >= twenty_four_hours_ago,
                fine_tuned_models_table.c.estimated_training_cost_usd.isnot(None)
            )
        )
        training_cost_result = await db.execute(training_cost_stmt)
        training_cost_24h = training_cost_result.scalar() or 0.0

        return TrainingOverview(
            running_jobs=running_jobs,
            total_models=total_models,
            gpu_hours_24h=round(gpu_hours_24h, 2),
            training_cost_24h=round(training_cost_24h, 2)
        )
    except Exception as e:
        logger.error(f"Failed to get training overview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve training overview")

@router.get("/data-pipeline", response_model=DataPipelineMetrics)
async def get_data_pipeline_metrics(
    current_user: User = Depends(PermissionChecker(["monitoring:read_pipeline"])),
    db: AsyncSession = Depends(get_db)
):
    """Get data pipeline and vector store metrics."""
    try:
        # Total documents in vector store
        total_docs_stmt = select(func.count(document_chunks_table.c.id))
        total_docs_result = await db.execute(total_docs_stmt)
        total_documents = total_docs_result.scalar() or 0

        # Embedding operations in last 24h (approximate from conversation metrics)
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        embeddings_stmt = select(func.count(conversation_metrics_table.c.id)).where(
            conversation_metrics_table.c.timestamp >= twenty_four_hours_ago
        )
        embeddings_result = await db.execute(embeddings_stmt)
        embeddings_24h = embeddings_result.scalar() or 0

        # Vector store size (estimate from vectorstore directory)
        vector_store_size_mb = 0.0
        try:
            vectorstore_path = _APP_PATHS.get("VECTORSTORE_DIR")
            if vectorstore_path and os.path.exists(vectorstore_path):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(vectorstore_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except (OSError, IOError):
                            continue
                vector_store_size_mb = total_size / (1024 * 1024)
        except Exception:
            pass

        # Failed ingests in last 24h (from error logs)
        failed_ingests_stmt = select(func.count(ragnetic_logs_table.c.id)).where(
            and_(
                ragnetic_logs_table.c.timestamp >= twenty_four_hours_ago,
                ragnetic_logs_table.c.level == 'ERROR',
                ragnetic_logs_table.c.message.like('%ingest%')
            )
        )
        failed_ingests_result = await db.execute(failed_ingests_stmt)
        failed_ingests_24h = failed_ingests_result.scalar() or 0

        return DataPipelineMetrics(
            total_documents=total_documents,
            embeddings_24h=embeddings_24h,
            vector_store_size_mb=round(vector_store_size_mb, 1),
            failed_ingests_24h=failed_ingests_24h
        )
    except Exception as e:
        logger.error(f"Failed to get data pipeline metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve data pipeline metrics")
