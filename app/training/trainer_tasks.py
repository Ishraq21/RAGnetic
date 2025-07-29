# app/training/trainer_tasks.py
import logging
import os
from celery import Celery
from typing import Dict, Any

# Import necessary components from your project's core and database modules
from app.core.config import get_db_connection, get_memory_storage_config, get_log_storage_config
from app.db import get_sync_db_engine
from app.schemas.fine_tuning import FineTuningJobConfig, FineTuningStatus
from app.training.trainer import LLMFineTuner

import app.workflows.tasks  # Simply importing the module registers its tasks with this Celery app instance

logger = logging.getLogger(__name__)

# Configure Celery using environment variables. Redis is a common choice for broker/backend.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("ragnetic_tasks", broker=REDIS_URL, backend=REDIS_URL)  # Renamed app for generality


# Helper function to get the database URI for Celery Beat scheduler.
def get_beat_db_uri():
    conn_name = (get_memory_storage_config().get("connection_name") or
                 get_log_storage_config().get("connection_name"))
    if not conn_name:
        return "sqlite:///celery_schedule.db"  # Fallback for Celery Beat if no other DB is explicitly configured
    conn_str = get_db_connection(conn_name).replace('+aiosqlite', '').replace('+asyncpg', '')
    return conn_str


# Apply Celery configuration settings
celery_app.conf.update(
    task_track_started=True,  # Allow tracking tasks in 'STARTED' state
    broker_connection_retry_on_startup=True,  # Retry connection to broker on startup
    beat_dburi=get_beat_db_uri(),  # Configure Celery Beat to use a persistent scheduler
    # Define task queues here. 'celery' is the default queue.
    task_queues={
        'celery': {},  # Default queue
        'ragnetic_fine_tuning_tasks': {},  # Dedicated queue for fine-tuning
    },
    task_routes={
        # Route fine-tuning tasks to their specific queue
        'app.training.trainer_tasks.fine_tune_llm_task': {'queue': 'ragnetic_fine_tuning_tasks'},
        # Route workflow tasks to the default queue (or a specific one if needed)
        'app.workflows.tasks.execute_workflow_task': {'queue': 'celery'},
    }
)


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

