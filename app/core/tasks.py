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
    Celery task to clean up temporary files, vector stores, and their
    associated database records for chat sessions inactive for a specified duration.
    """
    task_logger.info(f"Starting cleanup task for sessions inactive for > {inactive_hours} hours.")
    sync_engine = get_sync_db_engine()
    Session = sessionmaker(bind=sync_engine)
    db_session = Session()

    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=inactive_hours)

        # Step 1: Find all inactive sessions and gather all necessary info upfront.
        inactive_sessions_stmt = select(
            chat_sessions_table.c.id,
            chat_sessions_table.c.thread_id,
            chat_sessions_table.c.agent_name
        ).where(chat_sessions_table.c.updated_at < cutoff_time)

        inactive_sessions = db_session.execute(inactive_sessions_stmt).fetchall()
        task_logger.info(f"Found {len(inactive_sessions)} inactive chat sessions to process.")

        if not inactive_sessions:
            task_logger.info("No inactive sessions to clean up. Task finished.")
            return

        inactive_session_ids = [s.id for s in inactive_sessions]

        # Step 2: Perform Filesystem Cleanup
        # Loop through each session to clean its associated temp files from disk.
        for session_info in inactive_sessions:
            try:
                # Load agent config to get correct paths for the temporary document service.
                agent_config = load_agent_config(session_info.agent_name)
                temp_doc_service = TemporaryDocumentService(agent_config=agent_config)

                # Find all temp_doc_ids associated with this specific session.
                messages_meta_stmt = select(chat_messages_table.c.meta).where(
                    chat_messages_table.c.session_id == session_info.id
                )
                messages_with_meta = db_session.execute(messages_meta_stmt).fetchall()

                temp_doc_ids_to_clean = set()
                for msg_meta_row in messages_with_meta:
                    if msg_meta_row.meta and 'quick_uploaded_files' in msg_meta_row.meta:
                        for file_info in msg_meta_row.meta['quick_uploaded_files']:
                            if 'temp_doc_id' in file_info:
                                temp_doc_ids_to_clean.add(file_info['temp_doc_id'])

                if temp_doc_ids_to_clean:
                    task_logger.info(
                        f"Cleaning up {len(temp_doc_ids_to_clean)} temporary document(s) from filesystem for session: {session_info.thread_id}")
                    for temp_doc_id in temp_doc_ids_to_clean:
                        temp_doc_service.cleanup_temp_document(temp_doc_id)

            except Exception as e:
                task_logger.error(
                    f"Failed during file cleanup for agent '{session_info.agent_name}' / session '{session_info.thread_id}'. Error: {e}",
                    exc_info=True)
                # We continue to the next session even if one fails.

        # Step 3: Perform Database Cleanup in a single transaction.
        task_logger.info(f"Starting database cleanup for {len(inactive_session_ids)} inactive sessions.")

        # Get all message IDs from the inactive sessions.
        messages_to_delete_stmt = select(chat_messages_table.c.id).where(
            chat_messages_table.c.session_id.in_(inactive_session_ids))
        message_ids_to_delete = [row.id for row in db_session.execute(messages_to_delete_stmt).fetchall()]

        if message_ids_to_delete:
            # Find all unique chunk IDs cited in those messages.
            chunks_to_delete_stmt = select(citations_table.c.chunk_id).distinct().where(
                citations_table.c.message_id.in_(message_ids_to_delete))
            chunk_ids_to_delete = [row.chunk_id for row in db_session.execute(chunks_to_delete_stmt).fetchall()]

            # Delete citations linked to the messages.
            db_session.execute(delete(citations_table).where(citations_table.c.message_id.in_(message_ids_to_delete)))
            task_logger.info(f"Deleted citations for {len(message_ids_to_delete)} messages.")

            # Delete the orphaned document chunks.
            if chunk_ids_to_delete:
                db_session.execute(
                    delete(document_chunks_table).where(document_chunks_table.c.id.in_(chunk_ids_to_delete)))
                task_logger.info(f"Deleted {len(chunk_ids_to_delete)} orphaned document chunks from the database.")

        # Delete all messages from the inactive sessions.
        db_session.execute(
            delete(chat_messages_table).where(chat_messages_table.c.session_id.in_(inactive_session_ids)))
        task_logger.info(f"Deleted messages for {len(inactive_session_ids)} inactive sessions.")

        # Finally, delete the inactive session records themselves.
        db_session.execute(delete(chat_sessions_table).where(chat_sessions_table.c.id.in_(inactive_session_ids)))
        task_logger.info(f"Deleted {len(inactive_session_ids)} inactive session records.")

        # Commit all database deletions.
        db_session.commit()

    except Exception as e:
        task_logger.error(f"A critical error occurred during the cleanup task: {e}", exc_info=True)
        db_session.rollback()  # Rollback database changes on any failure.
    finally:
        db_session.close()  # Ensure the database session is always closed.
