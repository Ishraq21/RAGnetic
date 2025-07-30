# app/training/trainer_tasks.py
import logging
import os
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
from sqlalchemy import update, select


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
    Celery task that receives a fine-tuning job configuration (as a dictionary),
    validates it, and then initiates the LLM fine-tuning process via LLMFineTuner.
    """
    job_config = None
    try:
        job_config = FineTuningJobConfig(**job_config_dict)
        logger.info(f"Received fine-tuning task for job '{job_config.job_name}' (user: {user_id}).")

        db_engine = get_sync_db_engine()
        fine_tuner = LLMFineTuner(db_engine)

        fine_tuner.fine_tune_llm(job_config, user_id)
        logger.info(f"Fine-tuning task '{job_config.job_name}' completed successfully.")

    except Exception as e:
        job_name_for_logging = job_config.job_name if job_config else job_config_dict.get('job_name', 'Unknown Job')
        logger.error(f"Fine-tuning task failed for job '{job_name_for_logging}': {e}", exc_info=True)
        raise  # Re-raise the exception to allow Celery's backend to mark the task as failed