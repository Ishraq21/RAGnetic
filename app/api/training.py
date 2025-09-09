# app/api/training.py
import os
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, func
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
import os
import shutil

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

@router.get("/jobs", response_model=List[FineTunedModel])
async def list_training_jobs(
    status_filter: Optional[FineTuningStatus] = Query(None, description="Filter jobs by training status (e.g., 'running', 'pending')."),
    base_model_name: Optional[str] = Query(None, description="Filter jobs by the base LLM model name (e.g., 'ollama/llama2')."),
    job_name: Optional[str] = Query(None, description="Filter jobs by the user-defined job name."),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of jobs to return."),
    offset: int = Query(0, ge=0, description="Number of jobs to skip (for pagination)."),
    current_user: User = Depends(PermissionChecker(["fine_tune:list_models"])), # Requires 'fine_tune:list_models' permission
    db: AsyncSession = Depends(get_db),
):
    """
    Lists all training jobs (including pending, running, completed, and failed) with their metadata.
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
    jobs_data = result.mappings().fetchall() # Fetch all matching rows
    return [FineTunedModel.model_validate(row) for row in jobs_data] # Convert to Pydantic models


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


@router.get("/models/available", response_model=List[Dict[str, Any]])
async def get_available_models_for_agents(
    current_user: User = Depends(PermissionChecker(["fine_tune:list_models"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns a list of available models for agent configuration, including fine-tuned models.
    Format: [{"value": "model_id", "label": "display_name", "type": "fine_tuned|base"}]
    """
    # Get completed fine-tuned models
    stmt = select(fine_tuned_models_table).where(
        fine_tuned_models_table.c.training_status == FineTuningStatus.COMPLETED.value
    ).order_by(fine_tuned_models_table.c.created_at.desc())
    
    result = await db.execute(stmt)
    fine_tuned_models = result.mappings().fetchall()
    
    # Format fine-tuned models
    available_models = []
    for model in fine_tuned_models:
        model_data = FineTunedModel.model_validate(model)
        available_models.append({
            "value": f"fine_tuned:{model_data.adapter_id}",
            "label": f"{model_data.job_name} (Fine-tuned {model_data.base_model_name})",
            "type": "fine_tuned",
            "adapter_id": model_data.adapter_id,
            "base_model": model_data.base_model_name,
            "job_name": model_data.job_name
        })
    
    # Add common base models
    base_models = [
        {"value": "gpt-4o", "label": "GPT-4o", "type": "base"},
        {"value": "gpt-4o-mini", "label": "GPT-4o Mini", "type": "base"},
        {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo", "type": "base"},
        {"value": "claude-3-5-sonnet-20241022", "label": "Claude 3.5 Sonnet", "type": "base"},
        {"value": "claude-3-haiku-20240307", "label": "Claude 3 Haiku", "type": "base"},
        {"value": "gemini-1.5-pro", "label": "Gemini 1.5 Pro", "type": "base"},
        {"value": "gemini-1.5-flash", "label": "Gemini 1.5 Flash", "type": "base"},
        {"value": "ollama/llama3.1", "label": "Ollama Llama 3.1", "type": "base"},
        {"value": "ollama/mistral", "label": "Ollama Mistral", "type": "base"},
    ]
    
    # Combine fine-tuned models first, then base models
    return available_models + base_models


@router.get("/configs", response_model=List[Dict[str, Any]])
async def list_training_configs(
    current_user: User = Depends(PermissionChecker(["fine_tune:list_models"])),
):
    """
    Lists available training configuration files in the training_configs directory.
    """
    import yaml
    from pathlib import Path
    
    configs_dir = Path("training_configs")
    configs = []
    
    if configs_dir.exists():
        for config_file in configs_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    configs.append({
                        "filename": config_file.name,
                        "job_name": config_data.get("job_name", config_file.stem),
                        "base_model_name": config_data.get("base_model_name", "Unknown"),
                        "created_at": config_file.stat().st_mtime
                    })
            except Exception as e:
                logger.warning(f"Failed to read config file {config_file}: {e}")
    
    return sorted(configs, key=lambda x: x["created_at"], reverse=True)


@router.post("/save-config")
async def save_training_config(
    config_data: Dict[str, Any] = Body(...),
    current_user: User = Depends(PermissionChecker(["fine_tune:initiate"])),
):
    """
    Saves a training configuration to a YAML file in the training_configs directory.
    """
    import yaml
    from pathlib import Path
    
    configs_dir = Path("training_configs")
    configs_dir.mkdir(exist_ok=True)
    
    config = config_data.get("config", {})
    filename = config_data.get("filename", "training_config.yaml")
    
    # Ensure filename has .yaml extension
    if not filename.endswith('.yaml'):
        filename += '.yaml'
    
    config_path = configs_dir / filename
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return {"message": f"Configuration saved to {config_path}", "filename": filename}
    except Exception as e:
        logger.error(f"Failed to save config file {config_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {e}")


@router.get("/configs/{filename}")
async def get_training_config(
    filename: str,
    current_user: User = Depends(PermissionChecker(["fine_tune:list_models"])),
):
    """
    Retrieves a specific training configuration file.
    """
    import yaml
    from pathlib import Path
    
    configs_dir = Path("training_configs")
    config_path = configs_dir / filename
    
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Configuration file not found")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return config_data
    except Exception as e:
        logger.error(f"Failed to read config file {config_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read configuration: {e}")


@router.post("/upload-dataset")
async def upload_training_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(PermissionChecker(["fine_tune:initiate"])),
):
    """
    Upload a training dataset file (JSONL format).
    """
    # Validate file type
    if not file.filename.endswith(('.jsonl', '.json')):
        raise HTTPException(
            status_code=400, 
            detail="Only JSONL and JSON files are supported for training datasets"
        )
    
    # Create uploads directory
    uploads_dir = Path("data/uploaded_temp")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_id = str(uuid4())
    file_extension = Path(file.filename).suffix
    safe_filename = f"{file_id}{file_extension}"
    file_path = uploads_dir / safe_filename
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate JSONL format
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    # Try to parse first line as JSON
                    import json
                    json.loads(lines[0].strip())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # Clean up invalid file
            file_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSONL format: {str(e)}"
            )
        
        return {
            "message": "Dataset uploaded successfully",
            "file_path": str(file_path),
            "filename": file.filename,
            "file_id": file_id,
            "size": file_path.stat().st_size,
            "lines": len(lines) if 'lines' in locals() else 0
        }
        
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        logger.error(f"Failed to upload dataset file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {e}")


@router.get("/uploaded-datasets")
async def list_uploaded_datasets(
    current_user: User = Depends(PermissionChecker(["fine_tune:list_models"])),
):
    """
    List all uploaded dataset files.
    """
    uploads_dir = Path("data/uploaded_temp")
    datasets = []
    
    if uploads_dir.exists():
        for file_path in uploads_dir.glob("*.jsonl"):
            try:
                stat = file_path.stat()
                datasets.append({
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "created_at": stat.st_mtime,
                    "file_id": file_path.stem
                })
            except Exception as e:
                logger.warning(f"Failed to read file info for {file_path}: {e}")
    
    return sorted(datasets, key=lambda x: x["created_at"], reverse=True)


@router.delete("/uploaded-datasets/{file_id}")
async def delete_uploaded_dataset(
    file_id: str,
    current_user: User = Depends(PermissionChecker(["fine_tune:delete"])),
):
    """
    Delete an uploaded dataset file.
    """
    uploads_dir = Path("data/uploaded_temp")
    file_path = uploads_dir / f"{file_id}.jsonl"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    try:
        file_path.unlink()
        return {"message": "Dataset file deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete dataset file {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e}")


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