# app/core/tasks.py

import logging
import os
from datetime import datetime, timedelta
from typing import List

from celery import Celery

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.db.models import chat_sessions_table, temporary_documents_table
from app.db import get_sync_db_engine
from app.services.temporary_document_service import TemporaryDocumentService
from app.core.config import get_db_connection, get_memory_storage_config, get_log_storage_config, get_path_settings
from app.db.dao import delete_temp_document_data_sync, mark_temp_document_cleaned_sync

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
task_logger = logging.getLogger(__name__)

# --- Celery App Initialization (Centralized Here) ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("ragnetic", broker=REDIS_URL, backend=REDIS_URL)


def get_beat_db_uri():
    conn_name = (get_memory_storage_config().get("connection_name") or \
                 get_log_storage_config().get("connection_name"))
    if not conn_name:
        _APP_PATHS = get_path_settings()
        default_beat_db_path = _APP_PATHS["RAGNETIC_DIR"] / "celery_beat_schedule.db"
        logger.warning(
            f"No explicit database configured for memory/logging. Using default SQLite for Celery Beat at: {default_beat_db_path}")
        return f"sqlite:///{default_beat_db_path.resolve()}"
    conn_str = get_db_connection(conn_name).replace('+aiosqlite', '').replace('+asyncpg', '')
    return conn_str


celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    beat_dburi=get_beat_db_uri(),
    task_queues={
        'celery': {},
        'ragnetic_fine_tuning_tasks': {},
        'ragnetic_cleanup_tasks': {},
    },
    task_routes={
        'app.core.tasks.cleanup_temporary_documents': {'queue': 'ragnetic_cleanup_tasks'},
        'app.training.trainer_tasks.fine_tune_llm_task': {'queue': 'ragnetic_fine_tuning_tasks'},
    },
    beat_schedule={
        'cleanup-temporary-documents-every-hour': {
            'task': 'app.core.tasks.cleanup_temporary_documents',
            'schedule': timedelta(hours=1),
            'options': {'queue': 'ragnetic_cleanup_tasks'},
        },
    },
    include=[
        'app.core.tasks',
        'app.training.trainer_tasks',
        'app.executors.docker_executor',
    ]
)


@celery_app.task(name='app.core.tasks.cleanup_temporary_documents', queue='ragnetic_cleanup_tasks')
def cleanup_temporary_documents():
    """
    Celery task to clean up temporary documents that have expired.
    Idempotent, transactional, and robust.
    """
    task_logger.info("Starting robust cleanup task for expired temporary documents.")

    sync_engine = get_sync_db_engine()
    Session = sessionmaker(bind=sync_engine)
    db_session = Session()

    try:
        # Step 1: Find all expired temporary documents that are not yet cleaned up
        expired_docs = db_session.execute(
            select(temporary_documents_table)
            .where(
                temporary_documents_table.c.expires_at <= datetime.utcnow(),
                temporary_documents_table.c.cleaned_up == False,
            )
        ).mappings().all()

        if not expired_docs:
            task_logger.info("No expired temporary documents found. Task finished.")
            return

        task_logger.info(f"Found {len(expired_docs)} expired temporary documents to clean up.")

        # End the implicit transaction started by the SELECT so we can open clean per-doc txns
        db_session.rollback()

        # Step 2: Process each expired document
        for row in expired_docs:
            temp_doc_id = row["temp_doc_id"]

            try:
                # Start a new transaction for each cleanup operation
                with db_session.begin():
                    # 1) Filesystem cleanup
                    TemporaryDocumentService.cleanup_fs(row)

                    # 2) Database cleanup (sync helpers; no commits inside)
                    delete_temp_document_data_sync(db_session, temp_doc_id)

                    # 3) Mark the record as cleaned up (sync helper; no commit)
                    mark_temp_document_cleaned_sync(db_session, row["id"])

                task_logger.info(f"Cleanup for temp doc '{temp_doc_id}' completed successfully.")

            except Exception:
                # Context manager rolls back automatically on exception
                task_logger.exception(
                    f"A critical error occurred during cleanup of temp_doc_id '{temp_doc_id}'. "
                    f"Transaction rolled back."
                )
    except Exception as e:
        task_logger.error(f"A critical error occurred during the overall cleanup task: {e}", exc_info=True)
    finally:
        db_session.close()


@celery_app.task
def sync_agents_from_filesystem():
    """Sync agents from file system to database."""
    try:
        from app.services.agent_sync_scheduler import agent_sync_scheduler
        
        result = agent_sync_scheduler.sync_agents_from_filesystem()
        
        if result['success']:
            task_logger.info(f"Agent sync completed: {result['synced_count']} new agents synced, {result['total_count']} total agents")
            if result['errors']:
                for error in result['errors']:
                    task_logger.error(error)
        else:
            task_logger.error(f"Agent sync failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        task_logger.error(f"Failed to sync agents: {e}", exc_info=True)
