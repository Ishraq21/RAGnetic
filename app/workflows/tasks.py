from celery import Celery
import os

from app.workflows.engine import WorkflowEngine
from app.db import get_sync_db_engine # We'll need a synchronous engine for the background worker

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
    db_engine = get_sync_db_engine()
    engine = WorkflowEngine(db_engine)
    engine.run_workflow(workflow_name, initial_input)