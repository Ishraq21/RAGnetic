# app/api/training.py
import os
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, func
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime

from app.schemas.fine_tuning import FineTuningJobConfig, FineTuningStatus, FineTunedModel
from app.db import get_db
from app.db.models import fine_tuned_models_table

from app.training.trainer_tasks import fine_tune_llm_task

from app.core.security import PermissionChecker
from app.schemas.security import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/training", tags=["Fine-tuning API"])

@router.post("/apply", response_model=FineTunedModel, status_code=status.HTTP_202_ACCEPTED)
async def apply_fine_tuning_job_config(
    job_config: FineTuningJobConfig,
    current_user: User = Depends(PermissionChecker(["fine_tune:initiate"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Validates request and enqueues a fine-tuning job. Adds allowlists and safe path checks.
    """
    logger.info(f"API: User '{current_user.username}' submitted fine-tuning job '{job_config.job_name}'.")

    # --- Allowlist checks (configurable via env) ---
    allowed_models_env = os.getenv("RAGNETIC_ALLOWED_BASE_MODELS", "")
    allowed_models = {m.strip() for m in allowed_models_env.split(",") if m.strip()}
    if allowed_models and job_config.base_model_name not in allowed_models:
        raise HTTPException(
            status_code=400,
            detail=f"Base model '{job_config.base_model_name}' not allowed. Allowed: {sorted(allowed_models)}"
        )

    # --- Safe dataset path (default allow-root = ./data ) ---
    allowed_root = Path(os.getenv("RAGNETIC_DATASET_ROOT", "data")).resolve()
    try:
        ds_path = Path(job_config.dataset_path).resolve(strict=False)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid dataset_path.")

    if not str(ds_path).startswith(str(allowed_root)):
        raise HTTPException(
            status_code=400,
            detail=f"dataset_path must be under '{allowed_root}'. Got: '{ds_path}'"
        )

    try:
        generated_adapter_id = str(uuid4())
        out_dir = Path(job_config.output_base_dir) / job_config.job_name / generated_adapter_id
        logs_path = out_dir / "training_logs.txt"

        new_job_data = {
            "adapter_id": generated_adapter_id,
            "job_name": job_config.job_name,
            "base_model_name": job_config.base_model_name,
            "adapter_path": str(out_dir),
            "training_dataset_id": str(ds_path),
            "training_status": FineTuningStatus.PENDING.value,
            "training_logs_path": str(logs_path),
            "hyperparameters": job_config.hyperparameters.model_dump(),
            "created_by_user_id": current_user.id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        stmt = insert(fine_tuned_models_table).values(**new_job_data).returning(fine_tuned_models_table)
        result = await db.execute(stmt)
        created_job_row = result.mappings().first()
        await db.commit()

        job_cfg = job_config.model_dump()
        job_cfg["adapter_id"] = generated_adapter_id
        fine_tune_llm_task.delay(job_cfg, current_user.id)

        logger.info(f"API: Fine-tuning job '{job_config.job_name}' with ID '{generated_adapter_id}' dispatched.")
        return FineTunedModel.model_validate(created_job_row)

    except Exception as e:
        await db.rollback()
        logger.error(f"API: Failed to submit fine-tuning job '{job_config.job_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit fine-tuning job: {e}")


@router.get("/jobs/{adapter_id}", response_model=FineTunedModel)
async def get_fine_tune_job_status(
    adapter_id: str, # The unique adapter_id for the job
    current_user: User = Depends(PermissionChecker(["fine_tune:read_status"])), # Requires 'fine_tune:read_status' permission
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieves the current status and detailed metadata for a specific fine-tuning job.
    Requires: 'fine_tune:read_status' permission.
    """
    stmt = select(fine_tuned_models_table).where(fine_tuned_models_table.c.adapter_id == adapter_id)
    result = await db.execute(stmt)
    job_data = result.mappings().first() # Fetch a single row
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job with ID '{adapter_id}' not found.")
    return FineTunedModel.model_validate(job_data) # Convert SQLAlchemy row to Pydantic model

@router.get("/models", response_model=List[FineTunedModel])
async def list_fine_tuned_models(
    status_filter: Optional[FineTuningStatus] = Query(None, description="Filter models by training status (e.g., 'completed')."),
    base_model_name: Optional[str] = Query(None, description="Filter models by the base LLM model name (e.g., 'ollama/llama2')."),
    job_name: Optional[str] = Query(None, description="Filter models by the user-defined job name."),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of models to return."),
    offset: int = Query(0, ge=0, description="Number of models to skip (for pagination)."),
    current_user: User = Depends(PermissionChecker(["fine_tune:list_models"])), # Requires 'fine_tune:list_models' permission
    db: AsyncSession = Depends(get_db),
):
    """
    Lists available fine-tuned models and their metadata, with filtering and pagination options.
    Requires: 'fine_tune:list_models' permission.
    """
    stmt = select(fine_tuned_models_table).order_by(fine_tuned_models_table.c.created_at.desc())

    # Apply filters based on query parameters
    if status_filter:
        stmt = stmt.where(fine_tuned_models_table.c.training_status == status_filter.value)
    if base_model_name:
        stmt = stmt.where(fine_tuned_models_table.c.base_model_name.ilike(f"%{base_model_name}%")) # Case-insensitive search
    if job_name:
        stmt = stmt.where(fine_tuned_models_table.c.job_name.ilike(f"%{job_name}%"))

    # Apply pagination
    stmt = stmt.limit(limit).offset(offset)

    result = await db.execute(stmt)
    models_data = result.mappings().fetchall() # Fetch all matching rows
    return [FineTunedModel.model_validate(row) for row in models_data] # Convert to Pydantic models


@router.delete("/jobs/{adapter_id}")
async def delete_training_job(
    adapter_id: str,
    current_user: User = Depends(PermissionChecker(["fine_tune:delete"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a training job and its associated model files.
    Requires: 'fine_tune:delete' permission.
    """
    # First, get the job details
    stmt = select(fine_tuned_models_table).where(fine_tuned_models_table.c.adapter_id == adapter_id)
    result = await db.execute(stmt)
    job_data = result.mappings().first()
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Training job with ID '{adapter_id}' not found.")
    
    # Check if job is currently running
    if job_data["training_status"] == "running":
        raise HTTPException(
            status_code=400, 
            detail="Cannot delete a training job that is currently running. Please wait for it to complete or fail."
        )
    
    try:
        # Delete the database record
        delete_stmt = fine_tuned_models_table.delete().where(fine_tuned_models_table.c.adapter_id == adapter_id)
        await db.execute(delete_stmt)
        await db.commit()
        
        # TODO: Delete model files from filesystem
        # This would require implementing file cleanup logic
        
        logger.info(f"API: Training job '{adapter_id}' deleted by user '{current_user.username}'.")
        return {"message": f"Training job '{adapter_id}' deleted successfully."}
        
    except Exception as e:
        await db.rollback()
        logger.error(f"API: Failed to delete training job '{adapter_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete training job: {e}")


@router.get("/jobs/{adapter_id}/logs")
async def get_training_logs(
    adapter_id: str,
    current_user: User = Depends(PermissionChecker(["fine_tune:read_logs"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Get training logs for a specific job.
    Requires: 'fine_tune:read_logs' permission.
    """
    stmt = select(fine_tuned_models_table).where(fine_tuned_models_table.c.adapter_id == adapter_id)
    result = await db.execute(stmt)
    job_data = result.mappings().first()
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Training job with ID '{adapter_id}' not found.")
    
    logs_path = job_data.get("training_logs_path")
    if not logs_path or not os.path.exists(logs_path):
        return {"logs": "No logs available yet.", "path": logs_path}
    
    try:
        with open(logs_path, 'r') as f:
            logs_content = f.read()
        return {"logs": logs_content, "path": logs_path}
    except Exception as e:
        logger.error(f"Failed to read logs for job '{adapter_id}': {e}")
        return {"logs": f"Error reading logs: {str(e)}", "path": logs_path}


@router.get("/stats")
async def get_training_stats(
    current_user: User = Depends(PermissionChecker(["fine_tune:read_stats"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Get training statistics and metrics.
    Requires: 'fine_tune:read_stats' permission.
    """
    try:
        # Get total jobs count
        total_jobs_stmt = select(func.count(fine_tuned_models_table.c.id))
        total_jobs_result = await db.execute(total_jobs_stmt)
        total_jobs = total_jobs_result.scalar()
        
        # Get jobs by status
        status_counts_stmt = select(
            fine_tuned_models_table.c.training_status,
            func.count(fine_tuned_models_table.c.id)
        ).group_by(fine_tuned_models_table.c.training_status)
        status_counts_result = await db.execute(status_counts_stmt)
        status_counts = {row[0]: row[1] for row in status_counts_result.fetchall()}
        
        # Get total training cost
        cost_stmt = select(fine_tuned_models_table.c.estimated_training_cost_usd).where(
            fine_tuned_models_table.c.estimated_training_cost_usd.isnot(None)
        )
        cost_result = await db.execute(cost_stmt)
        total_cost = sum(row[0] for row in cost_result.fetchall() if row[0])
        
        # Get total GPU hours
        gpu_hours_stmt = select(fine_tuned_models_table.c.gpu_hours_consumed).where(
            fine_tuned_models_table.c.gpu_hours_consumed.isnot(None)
        )
        gpu_hours_result = await db.execute(gpu_hours_stmt)
        total_gpu_hours = sum(row[0] for row in gpu_hours_result.fetchall() if row[0])
        
        return {
            "total_jobs": total_jobs,
            "status_counts": status_counts,
            "total_cost_usd": total_cost,
            "total_gpu_hours": total_gpu_hours,
            "completed_jobs": status_counts.get("completed", 0),
            "running_jobs": status_counts.get("running", 0),
            "failed_jobs": status_counts.get("failed", 0),
            "pending_jobs": status_counts.get("pending", 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get training stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get training statistics: {e}")


@router.post("/jobs/{adapter_id}/cancel")
async def cancel_training_job(
    adapter_id: str,
    current_user: User = Depends(PermissionChecker(["fine_tune:cancel"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel a running training job.
    Requires: 'fine_tune:cancel' permission.
    """
    stmt = select(fine_tuned_models_table).where(fine_tuned_models_table.c.adapter_id == adapter_id)
    result = await db.execute(stmt)
    job_data = result.mappings().first()
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Training job with ID '{adapter_id}' not found.")
    
    if job_data["training_status"] != "running":
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel job with status '{job_data['training_status']}'. Only running jobs can be cancelled."
        )
    
    try:
        # Update job status to failed with cancellation note
        update_stmt = fine_tuned_models_table.update().where(
            fine_tuned_models_table.c.adapter_id == adapter_id
        ).values(
            training_status="failed",
            updated_at=datetime.utcnow()
        )
        await db.execute(update_stmt)
        await db.commit()
        
        # TODO: Implement actual job cancellation logic
        # This would require integration with the task queue system
        
        logger.info(f"API: Training job '{adapter_id}' cancelled by user '{current_user.username}'.")
        return {"message": f"Training job '{adapter_id}' cancellation requested."}
        
    except Exception as e:
        await db.rollback()
        logger.error(f"API: Failed to cancel training job '{adapter_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel training job: {e}")