# app/training/trainer_tasks.py
import logging
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path # New: Import Path for file operations
import shutil # New: Import shutil for directory operations

# Import the celery_app instance from the new central tasks file
from app.core.tasks import celery_app # MODIFIED: Import celery_app from app.core.tasks

# Import necessary components from your project's core and database modules
from app.core.config import get_db_connection, get_memory_storage_config, get_log_storage_config, get_path_settings
from app.db import get_sync_db_engine
from app.schemas.fine_tuning import FineTuningJobConfig, FineTuningStatus
from app.training.trainer import LLMFineTuner
from app.db.models import fine_tuned_models_table
from app.executors.gpu_training_executor import GPUTrainingExecutor
from sqlalchemy import update, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


logger = logging.getLogger(__name__)


# This get_beat_db_uri is likely redundant now if celery_app is imported and configured centrally,
# but keeping it if it's used elsewhere for non-Celery DB connections.
def get_beat_db_uri():
    conn_name = (get_memory_storage_config().get("connection_name") or
                 get_log_storage_config().get("connection_name"))
    if not conn_name:
        return "sqlite:///celery_schedule.db"
    conn_str = get_db_connection(conn_name).replace('+aiosqlite', '').replace('+asyncpg', '')
    return conn_str


@celery_app.task(name="app.training.trainer_tasks.fine_tune_llm_task")
def fine_tune_llm_task(job_config_dict: Dict[str, Any], user_id: int):
    """
    Enforce idempotency: we atomically flip PENDING/PAUSED -> RUNNING.
    If the row isn't in a startable state, we no-op (a duplicate enqueue).
    Also persist worker/device metadata up front.
    """
    import socket
    import torch

    job_config = None
    try:
        job_config = FineTuningJobConfig(**job_config_dict)
        adapter_id = job_config.adapter_id
        logger.info(f"Received fine-tuning task for job '{job_config.job_name}' (user: {user_id}, adapter_id={adapter_id}).")

        db_engine = get_sync_db_engine()

        # Detect device + GPU name once (for metadata)
        if torch.cuda.is_available():
            device = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "cuda:unknown"
        elif torch.backends.mps.is_available():
            device = "mps"
            gpu_name = "apple-mps"
        else:
            device = "cpu"
            gpu_name = "cpu"

        mixed_precision = job_config.hyperparameters.mixed_precision_dtype
        # Only potentially true; trainer will decide final activation of 4bit
        bnb_possible = bool(torch.cuda.is_available())

        # Only start if row is PENDING or PAUSED
        with db_engine.begin() as conn:
            upd = (
                update(fine_tuned_models_table)
                .where(fine_tuned_models_table.c.adapter_id == adapter_id)
                .where(fine_tuned_models_table.c.training_status.in_(["pending", "paused"]))
                .values(
                    training_status="running",
                    worker_host=socket.gethostname(),
                    worker_pid=os.getpid(),
                    device=device,
                    gpu_name=gpu_name,
                    mixed_precision=mixed_precision,
                    bitsandbytes_4bit=bnb_possible,
                    updated_at=datetime.utcnow(),
                )
            )
            res = conn.execute(upd)
            if res.rowcount == 0:
                logger.warning(
                    f"Not starting job '{job_config.job_name}' (adapter_id={adapter_id}): "
                    f"row not in startable state (maybe duplicate or already running/finished)."
                )
                return  # no-op duplicate

        # Check if this is a GPU job
        if job_config.use_gpu and job_config.gpu_instance_id:
            # Execute on GPU infrastructure
            logger.info(f"Executing GPU training for job '{job_config.job_name}' on instance {job_config.gpu_instance_id}")
            import asyncio
            asyncio.run(execute_gpu_training(job_config_dict, user_id))
        else:
            # Execute locally
            logger.info(f"Executing local training for job '{job_config.job_name}'")
            fine_tuner = LLMFineTuner(db_engine)
            fine_tuner.fine_tune_llm(job_config, user_id)
        
        logger.info(f"Fine-tuning task '{job_config.job_name}' completed successfully.")

    except Exception as e:
        job_name_for_logging = job_config.job_name if job_config else job_config_dict.get('job_name', 'Unknown Job')
        logger.error(f"Fine-tuning task failed for job '{job_name_for_logging}': {e}", exc_info=True)
        raise


async def execute_gpu_training(job_config_dict: Dict[str, Any], user_id: int) -> None:
    """
    Execute training on GPU infrastructure using the GPU training executor.
    """
    try:
        # Get database connection
        db_uri = get_db_connection("default")
        if not db_uri:
            raise ValueError("No database connection configured")
        
        # Create async database session
        engine = create_async_engine(db_uri)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session() as db:
            # Create GPU training executor
            gpu_executor = GPUTrainingExecutor(db)
            
            # Execute the training job
            success = await gpu_executor.execute_training_job(job_config_dict, user_id)
            
            if not success:
                raise Exception("GPU training execution failed")
            
            logger.info(f"GPU training completed successfully for user {user_id}")
            
    except Exception as e:
        logger.error(f"GPU training execution failed: {e}", exc_info=True)
        raise
