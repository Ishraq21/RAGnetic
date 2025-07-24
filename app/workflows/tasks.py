import logging
import os
from celery import Celery

from app.core.config import get_log_storage_config # Assuming this is needed for config checks
from app.db import get_sync_db_engine
from app.workflows.engine import WorkflowEngine

# --- Worker-Specific Logging Configuration ---
# Celery workers run as separate processes and need their own logging setup.
# We configure it directly here for simplicity and robustness.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# ---

# Get the module-level logger after configuration
task_logger = logging.getLogger(__name__)

# Get the Redis URL from environment variables, defaulting for local development
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Initialize the Celery application
celery_app = Celery(
    "ragnetic_workflows",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)

@celery_app.task(name="app.workflows.tasks.run_workflow_task")
def run_workflow_task(workflow_name: str, initial_input: dict = None):
    """
    Celery task that initializes a WorkflowEngine and runs a workflow.
    This runs in a separate background worker process.
    """
    task_logger.info(f"Worker received task to run workflow: '{workflow_name}'")

    try:
        db_engine = get_sync_db_engine()
        engine = WorkflowEngine(db_engine)
        engine.run_workflow(workflow_name, initial_input)
        task_logger.info(f"Successfully completed workflow: '{workflow_name}'")
    except Exception as e:
        task_logger.error(f"Error running workflow '{workflow_name}': {e}", exc_info=True)
        # Re-raise the exception so Celery can mark the task as FAILED
        raise