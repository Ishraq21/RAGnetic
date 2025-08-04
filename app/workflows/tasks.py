# app/workflows/tasks.py
import asyncio
import logging
import os
import redis
from typing import Optional

# Import the celery_app instance from the new central tasks file
from app.core.tasks import celery_app # MODIFIED: Import celery_app from app.core.tasks

# Imports for WorkflowEngine and DB
from app.db import get_sync_db_engine
from app.workflows.engine import WorkflowEngine

# These imports are now redundant if celery_app is imported from core/tasks
# from celery import Celery
# from app.core.config import get_db_connection, get_memory_storage_config, get_log_storage_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
task_logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# The celery_app definition is now centralized in app/core/tasks.py
# So, remove the following lines:
# celery_app = Celery("ragnetic_workflows", broker=REDIS_URL, backend=REDIS_URL)

# The get_beat_db_uri function and celery_app.conf.update are also centralized.
# So, remove the following lines:
# def get_beat_db_uri():
#     conn_name = (get_memory_storage_config().get("connection_name") or
#                  get_log_storage_config().get("connection_name"))
#     if not conn_name:
#         return "sqlite:///schedule.db"
#     conn_str = get_db_connection(conn_name).replace('+aiosqlite', '').replace('+asyncpg', '')
#     return conn_str
#
# celery_app.conf.update(
#     task_track_started=True,
#     broker_connection_retry_on_startup=True,
#     beat_dburi=get_beat_db_uri()
# )


@celery_app.task(name="app.workflows.tasks.run_workflow_task")
def run_workflow_task(workflow_name: str, initial_input: dict = None, user_id: Optional[int] = None):
    """
    Celery task to run a workflow, using a Redis lock to ensure single execution.
    """
    redis_client = redis.from_url(REDIS_URL)
    lock_key = f"workflow_lock:{workflow_name}"
    if not redis_client.set(lock_key, "running", nx=True, ex=60):
        task_logger.warning(f"Workflow run for '{workflow_name}' skipped due to active Redis lock.")
        return

    try:
        db_engine = get_sync_db_engine()
        effective_user_id = user_id if user_id is not None else 0
        engine = WorkflowEngine(db_engine, user_id=effective_user_id)
        asyncio.run(engine.run_workflow(workflow_name, initial_input,  user_id=effective_user_id))

        task_logger.info(f"Successfully completed workflow: '{workflow_name}'")
    except Exception as e:
        task_logger.error(f"Error running workflow '{workflow_name}': {e}", exc_info=True)
        raise
    finally:
        redis_client.delete(lock_key)