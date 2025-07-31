# app/core/tasks.py

import logging
import os
from datetime import datetime, timedelta
from typing import List

from celery import Celery
from celery.schedules import crontab

from sqlalchemy import select, delete
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import chat_sessions_table, chat_messages_table, document_chunks_table, citations_table
from app.db import get_sync_db_engine
from app.services.temporary_document_service import TemporaryDocumentService
from app.core.config import get_db_connection, get_memory_storage_config, get_log_storage_config, get_path_settings, _get_config_parser
from app.agents.config_manager import load_agent_config

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
task_logger = logging.getLogger(__name__)

# --- Celery App Initialization (Centralized Here) ---
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("ragnetic_workflows", broker=REDIS_URL, backend=REDIS_URL)

def get_beat_db_uri():
    conn_name = (get_memory_storage_config().get("connection_name") or \
                 get_log_storage_config().get("connection_name"))
    if not conn_name:
        _APP_PATHS = get_path_settings()
        default_beat_db_path = _APP_PATHS["RAGNETIC_DIR"] / "celery_beat_schedule.db"
        logger.warning(f"No explicit database configured for memory/logging. Using default SQLite for Celery Beat at: {default_beat_db_path}")
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
        'app.workflows.tasks.run_workflow_task': {'queue': 'celery'},
    },
    beat_schedule={
        'cleanup-temporary-documents-every-hour': {
            'task': 'app.core.tasks.cleanup_temporary_documents',
            'schedule': timedelta(hours=1),
            'args': (24,),
            'options': {'queue': 'ragnetic_cleanup_tasks'},
        },
    },
    include=[
        'app.core.tasks',
        'app.training.trainer_tasks',
        'app.workflows.tasks'
    ]
)


@celery_app.task(name='app.core.tasks.cleanup_temporary_documents', queue='ragnetic_cleanup_tasks')
def cleanup_temporary_documents(inactive_hours: int = 24):
    """
    Celery task to clean up temporary files and their associated vector stores
    for chat sessions that have been inactive for a specified duration.
    This task DOES NOT delete chat messages, sessions, or core document chunks.
    """
    task_logger.info(f"Starting cleanup task for temporary files associated with sessions inactive for > {inactive_hours} hours.")
    sync_engine = get_sync_db_engine()
    Session = sessionmaker(bind=sync_engine)
    db_session = Session()

    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=inactive_hours)

        # Step 1: Find all inactive sessions and gather all necessary info upfront.
        # We need the session_id to find associated messages, but not necessarily thread_id or agent_name for file cleanup itself.
        inactive_sessions_stmt = select(
            chat_sessions_table.c.id
        ).where(chat_sessions_table.c.updated_at < cutoff_time)

        inactive_session_ids = [s.id for s in db_session.execute(inactive_sessions_stmt).fetchall()]
        task_logger.info(f"Found {len(inactive_session_ids)} inactive chat sessions with associated temporary files to potentially clean up.")

        if not inactive_session_ids:
            task_logger.info("No inactive sessions with temporary files to clean up. Task finished.")
            return

        # Step 2: Collect all temp_doc_ids linked to messages within these inactive sessions.
        temp_doc_ids_to_clean = set()
        for session_id in inactive_session_ids:
            messages_meta_stmt = select(chat_messages_table.c.meta).where(
                chat_messages_table.c.session_id == session_id
            )
            messages_with_meta = db_session.execute(messages_meta_stmt).fetchall()

            for msg_meta_row in messages_with_meta:
                if msg_meta_row.meta and 'quick_uploaded_files' in msg_meta_row.meta:
                    for file_info in msg_meta_row.meta['quick_uploaded_files']:
                        if 'temp_doc_id' in file_info:
                            temp_doc_ids_to_clean.add(file_info['temp_doc_id'])

        if not temp_doc_ids_to_clean:
            task_logger.info("No temporary documents found for cleanup in inactive sessions. Task finished.")
            return

        task_logger.info(f"Cleaning up {len(temp_doc_ids_to_clean)} unique temporary document(s) from filesystem.")
        for temp_doc_id in list(temp_doc_ids_to_clean): # Convert to list to iterate safely if set changes
            try:
                # Call the static cleanup method directly. It doesn't need an agent_config instance.
                TemporaryDocumentService.cleanup_temp_document(temp_doc_id)
            except Exception as e:
                task_logger.error(
                    f"Failed during temporary file cleanup for temp_doc_id '{temp_doc_id}'. Error: {e}",
                    exc_info=True)
                # Continue to the next temp_doc_id even if one fails.

        task_logger.info("Finished temporary document filesystem cleanup.")

        # IMPORTANT: Removed all database deletions for chat_messages_table, document_chunks_table, citations_table, chat_sessions_table.
        # This preserves chat history indefinitely by default, and only cleans up the physical temporary files.

        # No database commit needed here since we are not modifying DB records in this revised task.
        # If any future changes require DB commits, ensure they are within a transaction.

    except Exception as e:
        task_logger.error(f"A critical error occurred during the temporary file cleanup task: {e}", exc_info=True)
        # No rollback needed as no DB modifications are performed in this revised task.
    finally:
        db_session.close()