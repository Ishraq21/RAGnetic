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

from app.db.models import chat_sessions_table, chat_messages_table
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
    Celery task to clean up temporary documents (files and vector stores)
    associated with chat sessions that have been inactive for a specified duration.
    """
    logger.info(f"Running scheduled temporary document cleanup for sessions inactive for > {inactive_hours} hours.")
    sync_engine = get_sync_db_engine()
    Session = sessionmaker(bind=sync_engine)
    session = Session()

    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=inactive_hours)

        inactive_sessions_stmt = select(
            chat_sessions_table.c.thread_id,
            chat_sessions_table.c.user_id,
            chat_sessions_table.c.id,
            chat_sessions_table.c.agent_name
        ).where(
            chat_sessions_table.c.updated_at < cutoff_time
        )
        inactive_sessions = session.execute(inactive_sessions_stmt).fetchall()
        logger.info(f"Found {len(inactive_sessions)} inactive chat sessions to process for cleanup.")


        for s in inactive_sessions:
            thread_id = s.thread_id
            user_id = s.user_id
            session_db_id = s.id
            agent_name = s.agent_name # Get the agent_name for this session

            logger.info(f"Cleaning up session: User={user_id}, Thread={thread_id}, Agent={agent_name}")

            # ❗ **Change**: Load the agent config and instantiate the service inside the loop
            try:
                agent_config = load_agent_config(agent_name)
                temp_doc_service = TemporaryDocumentService(agent_config=agent_config)
            except Exception as e:
                logger.error(f"Could not load agent config for '{agent_name}' for cleanup. Skipping session {thread_id}. Error: {e}", exc_info=True)
                continue # Skip to the next session if agent config cannot be loaded

            temp_doc_ids_to_clean: List[str] = []

            messages_stmt = select(chat_messages_table.c.meta).where(
                chat_messages_table.c.session_id == session_db_id
            )
            messages_with_meta = session.execute(messages_stmt).fetchall()

            for msg_meta_row in messages_with_meta:
                if msg_meta_row.meta and 'quick_uploaded_files' in msg_meta_row.meta:
                    for file_info in msg_meta_row.meta['quick_uploaded_files']:
                        if 'temp_doc_id' in file_info:
                            temp_doc_ids_to_clean.append(file_info['temp_doc_id'])

            temp_doc_ids_to_clean = list(set(temp_doc_ids_to_clean))

            if temp_doc_ids_to_clean:
                logger.info(f"Found {len(temp_doc_ids_to_clean)} unique temporary documents in session {thread_id} to clean.")
                for temp_doc_id in temp_doc_ids_to_clean:
                    try:
                        # This now correctly uses the service instance for the specific agent
                        temp_doc_service.cleanup_temp_document(temp_doc_id)
                    except Exception as e:
                        logger.error(f"Failed to clean up temp_doc_id '{temp_doc_id}' for session {thread_id}: {e}", exc_info=True)
            else:
                logger.info(f"No temporary documents found in chat history for session {thread_id}.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during temporary document cleanup: {e}", exc_info=True)
    finally:
        session.close()
    logger.info("Finished temporary document cleanup task.")