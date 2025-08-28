# app/api/training.py
import os
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert
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