# app/main.py
import asyncio
import json
import uuid

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
import os
import logging
import logging.config
from queue import Queue
from logging.handlers import QueueListener, QueueHandler
from uuid import uuid4
from typing import Optional, List, Dict, Any, Tuple, Union
from multiprocessing import Process
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import Form

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Depends, status, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from app.api.webhooks import setup_dynamic_webhooks
# Database & Models
from app.db.models import (chat_sessions_table,
                           chat_messages_table,
                           users_table,
                           agent_runs,
                           agent_run_steps)

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, insert, update, text, delete
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# Core Components
from app.core.config import get_path_settings, get_log_storage_config, \
    get_memory_storage_config, get_db_connection_config, get_db_connection, get_cors_settings, _get_config_parser
from app.db import initialize_db_connections, get_db
from app.core.validation import sanitize_for_path
from app.core.security import get_http_api_key, get_websocket_api_key, get_current_user_from_api_key
from app.agents.config_manager import get_agent_configs, load_agent_config
from app.core.serialization import _serialize_for_db
from app.workflows.sync import sync_workflows_from_files, is_db_configured_sync
from app.api.security import router as security_api_router

from app.schemas.security import User

# Agents & Pipelines
from app.agents.agent_graph import get_agent_workflow, AgentState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler
from app.core.structured_logging import get_logging_config, DatabaseLogHandler, ragnetic_logs_table

# Tools
from app.tools.sql_tool import create_sql_toolkit
from app.tools.retriever_tool import get_retriever_tool
from app.tools.arxiv_tool import get_arxiv_tool
from app.tools.search_engine_tool import SearchTool

# System
from app.watcher import start_watcher
from sqlalchemy import create_engine as create_sync_engine
import alembic.config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

# API Routers
from app.api.agents import router as agents_api_router
from app.api.audit import router as audit_api_router
from app.api.query import router as query_api_router
from app.api.analytics import router as analytics_api_router
from app.api.evaluation import router as evaluation_api_router
from app.api.metrics import router as metrics_api_router
from app.api.training import router as training_api_router
from app.api.citations import router as citations_api_router

from app.api import workflows
from app.api import webhooks

from app.services.temporary_document_service import TemporaryDocumentService, TemporaryDocumentUploadResult
from app.core.citation_parser import extract_citations_from_text

# Get the main application logger after configuration is applied
logger = logging.getLogger("ragnetic")
load_dotenv()

# --- Global Settings & App Initialization ---
_APP_PATHS = get_path_settings()
_PROJECT_ROOT = _APP_PATHS["PROJECT_ROOT"]
_DATA_DIR = _APP_PATHS["DATA_DIR"]
_WORKFLOWS_DIR = _APP_PATHS["WORKFLOWS_DIR"]

config = _get_config_parser()
WEBSOCKET_MODE = config.get('SERVER', 'websocket_mode', fallback='memory')
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost")
allowed_origins = get_cors_settings()

app = FastAPI(title="RAGnetic API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

log_listener: Optional[QueueListener] = None


# --- WebSocket Connection Managers (Dual Mode) ---
class MemoryConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str):
        if channel not in self.active_connections: self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)

    def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.active_connections and websocket in self.active_connections[channel]:
            self.active_connections[channel].remove(websocket)

    async def broadcast(self, channel: str, message: str):
        if channel in self.active_connections:
            for connection in self.active_connections[channel]:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_text(message)


class RedisConnectionManager:
    def __init__(self, url: str):
        self.redis_url = url
        self.connection: Optional[redis.Redis] = None
        self.pubsub = None

    async def connect(self, websocket=None, channel=None):
        if self.connection is None:
            self.connection = redis.from_url(self.redis_url, decode_responses=True)
            await self.connection.ping()
            self.pubsub = self.connection.pubsub()

    def disconnect(self, websocket=None, channel=None):
        pass

    async def broadcast(self, channel: str, message: str):
        if not self.connection:
            logger.error("Cannot broadcast: Redis connection is not available.")
            return
        max_retries = 5
        backoff_factor = 0.5
        for attempt in range(max_retries):
            try:
                await self.connection.publish(channel, message)
                return
            except RedisConnectionError as e:
                logger.warning(f"Redis publish failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff_factor * (2 ** attempt))
        logger.error("Giving up on Redis publish after retries; message may be lost.")


if WEBSOCKET_MODE == 'redis':
    logger.info("Using Redis for WebSocket connections (scalable mode).")
    manager = RedisConnectionManager(REDIS_URL)
else:
    logger.info("Using in-memory WebSocket manager (single-process mode).")
    manager = MemoryConnectionManager()

_watcher_process: Optional[Process] = None


@app.on_event("startup")
async def startup_event():
    global _watcher_process, log_listener

    # --- Logging Setup ---
    logging.config.dictConfig(get_logging_config())

    log_storage_cfg = get_log_storage_config()
    if log_storage_cfg.get("type") == "db":
        log_queue = Queue(-1)
        db_handler = DatabaseLogHandler(
            connection_name=log_storage_cfg.get("connection_name"),
            table=ragnetic_logs_table
        )
        queue_handler = QueueHandler(log_queue)
        log_listener = QueueListener(log_queue, db_handler)
        log_listener.start()

        db_logger_names = ["ragnetic", "app.workflows", "ragnetic.metrics"]
        for name in db_logger_names:
            logging.getLogger(name).addHandler(queue_handler)

        logging.getLogger("ragnetic").info("Database logging initialized with QueueListener.")

    logger.info("Application startup: Initializing components.")
    if isinstance(manager, RedisConnectionManager):
        try:
            await manager.connect()
            logger.info(f"Successfully connected to Redis at {REDIS_URL}")
        except RedisConnectionError as e:
            logger.critical(
                f"CRITICAL: Could not connect to Redis at {REDIS_URL}. Please ensure it is running. Error: {e}")
            raise RuntimeError("Failed to connect to Redis") from e

    if is_db_configured_sync():
        db_config = get_db_connection_config()
        if not db_config:
            raise RuntimeError("Database is configured but connection details could not be loaded.")
        try:
            logger.info("Verifying database schema version...")
            sync_dialect = db_config.get('dialect', '').replace('+aiosqlite', '').replace('+asyncpg', '')
            if 'sqlite' in sync_dialect:
                db_path = Path(db_config.get('database_path', ''))
                if not db_path.is_absolute():
                    db_path = _PROJECT_ROOT / db_path
                sync_conn_str = f"sqlite:///{db_path.resolve()}"
            else:
                conn_name = get_memory_storage_config().get("connection_name") or get_log_storage_config().get(
                    "connection_name")
                if not conn_name:
                    raise RuntimeError("Database connection name not found for migration check.")
                sync_conn_str = get_db_connection(conn_name).replace("+aiomysql", "").replace("+asyncpg", "")

            engine = create_sync_engine(sync_conn_str)
            with engine.connect() as connection:
                alembic_cfg = alembic.config.Config(str(_PROJECT_ROOT / "alembic.ini"))
                script = ScriptDirectory.from_config(alembic_cfg)
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
                head_rev = script.get_current_head()
                if head_rev is not None and current_rev != head_rev:
                    logger.critical(f"CRITICAL: DB schema mismatch (Current: {current_rev}, Latest: {head_rev}).")
                    raise RuntimeError("Please run 'ragnetic migrate' to update the database.")
                else:
                    logger.info("Database schema is up-to-date.")
            engine.dispose()

            sync_workflows_from_files()
            setup_dynamic_webhooks(app)
            for route in app.router.routes:
                if "Webhooks" in getattr(route, "tags", []):
                    logger.info(f"Registered webhook: {route.path} → {route.methods}")

            conn_name = get_memory_storage_config().get("connection_name") or get_log_storage_config().get(
                "connection_name")
            if not conn_name:
                raise RuntimeError("Database connection name not found for async initialization.")
            initialize_db_connections(conn_name)
        except Exception as e:
            logger.critical(f"CRITICAL: Failed during database startup. Error: {e}", exc_info=True)
            raise
    else:
        logger.warning("No database configured. DB-dependent features will be disabled.")

    if not os.path.exists(_DATA_DIR):
        logger.error(f"'{_DATA_DIR}' not found. Please run 'ragnetic init'.")
        return

    if not os.path.exists(_WORKFLOWS_DIR):
        logger.warning(f"'{_WORKFLOWS_DIR}' not found. Workflow auto-sync may not work correctly.")

    monitored_dirs = [str(_DATA_DIR), str(_WORKFLOWS_DIR)]
    _watcher_process = Process(target=start_watcher, args=(monitored_dirs,))
    _watcher_process.daemon = False
    _watcher_process.start()
    logger.info(f"Automated file watcher started for directories: {', '.join(monitored_dirs)}.")


@app.on_event("shutdown")
async def shutdown_event():
    global _watcher_process, log_listener
    if isinstance(manager, RedisConnectionManager) and manager.connection:
        await manager.connection.close()
        logger.info("Redis connection closed.")

    if _watcher_process and _watcher_process.is_alive():
        _watcher_process.terminate()
        _watcher_process.join(timeout=5)
        logger.info("File watcher process stopped.")

    if log_listener:
        log_listener.stop()
        logger.info("Logging queue listener stopped.")


# --- INCLUDE THE API ROUTERS ---
app.include_router(agents_api_router)
app.include_router(audit_api_router)
app.include_router(query_api_router)
app.include_router(metrics_api_router)
app.include_router(analytics_api_router)
app.include_router(security_api_router)
app.include_router(training_api_router)
app.include_router(workflows.router, prefix="/api/v1")
app.include_router(webhooks.router, prefix="/webhooks/v1")
app.include_router(citations_api_router)



class RenameRequest(BaseModel):
    new_name: str


# --- NEW Pydantic Models for Quick Upload ---
class QuickUploadFileItem(BaseModel):
    file_name: str = Field(..., description="Name of the uploaded file.")
    file_size: int = Field(..., description="Size of the uploaded file in bytes.")
    temp_doc_id: str = Field(..., description="Unique ID assigned to the temporary document.")

class WebSocketUploadedFileItem(BaseModel):
    file_name: str
    file_size: int
    temp_doc_id: str

# This model is conceptual, based on what frontend sends to WS.
# Your actual chat message model (if it exists) might need updating.
class ChatMessagePayloadWithFiles(BaseModel):
    agent: str
    user_id: int
    thread_id: Optional[str] = None
    query: str
    # List of dictionaries, each with 'file_name' and 'temp_doc_id'
    files: List[WebSocketUploadedFileItem] = Field(default_factory=list,
                                                   description="List of successfully uploaded temporary files with their temp_doc_ids and sizes.")


@app.post("/api/v1/chat/upload-temp-document", response_model=QuickUploadFileItem, status_code=status.HTTP_200_OK,
          tags=["Chat"])
async def upload_temp_document(
        file: UploadFile = File(...),
        thread_id: str = Form(...),
        current_user: User = Depends(get_current_user_from_api_key),
        db: AsyncSession = Depends(get_db) #
):
    """
    Uploads a temporary document for use within a specific chat session.
    The document will be processed, embedded, and made available for retrieval
    by the agent for a limited time.
    """
    if not thread_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="thread_id is required for temporary document uploads."
        )

    try:
        session_query = select(chat_sessions_table.c.agent_name).where(chat_sessions_table.c.thread_id == thread_id)
        agent_name = (await db.execute(session_query)).scalar_one_or_none()
        if not agent_name:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent for this thread not found.")
        agent_config = load_agent_config(agent_name) # Load the config

        temp_doc_service = TemporaryDocumentService(agent_config=agent_config)

        upload_result = await temp_doc_service.process_and_store_temp_document(
            file=file,
            user_id=current_user.id,
            thread_id=thread_id
        )

        return QuickUploadFileItem(
            file_name=upload_result.file_name,
            file_size=upload_result.file_size,
            temp_doc_id=upload_result.temp_doc_id
        )
    except ValueError as ve:
        logger.error(f"Validation error during temporary file upload: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Failed to upload temporary document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload temporary document: An unexpected error occurred."
        )

async def _get_or_create_user(db: AsyncSession, user_id_string: str) -> int:
    """
    Retrieves a user's database ID based on a unique user_id string (e.g., username),
    or creates a new user if not found. Returns the integer user DB ID.
    """
    stmt = select(users_table.c.id).where(users_table.c.user_id == user_id_string)
    existing_user_id = (await db.execute(stmt)).scalar_one_or_none()

    if existing_user_id:
        return existing_user_id
    else:
        insert_stmt = insert(users_table).values(
            user_id=user_id_string,
            is_active=True,
            is_superuser=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            hashed_password="",
        ).returning(users_table.c.id)
        user_db_id = (await db.execute(insert_stmt)).scalar_one()
        await db.commit()
        logger.info(f"Created new user with DB ID: {user_db_id} (user_id_string: {user_id_string})")
        return user_db_id


async def _get_or_create_session(db: AsyncSession, thread_id_from_frontend: Optional[str], agent_name: str,
                                 user_db_id: int) -> Tuple[int, Optional[str], str]:
    """
    Retrieves an existing chat session or creates a new one.
    Returns (session_db_id, topic_name, canonical_thread_id).
    """
    if thread_id_from_frontend:
        stmt = select(
            chat_sessions_table.c.id, chat_sessions_table.c.topic_name, chat_sessions_table.c.thread_id
        ).where(
            (chat_sessions_table.c.thread_id == thread_id_from_frontend) &
            (chat_sessions_table.c.agent_name == agent_name) &
            (chat_sessions_table.c.user_id == user_db_id)
        )
        existing_session = (await db.execute(stmt)).first()

        if existing_session:
            return existing_session.id, existing_session.topic_name, existing_session.thread_id

    new_thread_id = str(uuid.uuid4())

    insert_stmt = insert(chat_sessions_table).values(
        thread_id=new_thread_id,
        agent_name=agent_name,
        user_id=user_db_id,
        topic_name=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    ).returning(chat_sessions_table.c.id, chat_sessions_table.c.topic_name, chat_sessions_table.c.thread_id)

    try:
        new_session = (await db.execute(insert_stmt)).one()
        await db.commit()
        logger.info(
            f"Created new chat session: ID={new_session.id}, Thread={new_session.thread_id}, Agent={agent_name}, User_DB_ID={user_db_id}")
        return new_session.id, new_session.topic_name, new_session.thread_id
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"IntegrityError creating session for thread '{new_thread_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Could not create unique session. Please try again.")
    except Exception as e:
        await db.rollback()
        logger.error(f"Unexpected error creating session for thread '{new_thread_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to create new chat session.")


async def _save_message_and_update_session(db: AsyncSession, session_id: int, sender: str, content: str,
                                           meta: Optional[Dict] = None):
    now = datetime.utcnow()
    msg_stmt = insert(chat_messages_table).values(session_id=session_id, sender=sender, content=content, timestamp=now,
                                                  meta=meta)
    session_stmt = update(chat_sessions_table).where(chat_sessions_table.c.id == session_id).values(updated_at=now)
    await db.execute(msg_stmt)
    await db.execute(session_stmt)
    await db.commit()


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket, api_key: str = Depends(get_websocket_api_key),
                         db: AsyncSession = Depends(get_db)):
    await ws.accept()
    canonical_thread_id: str = "uninitialized_thread"
    pubsub_task = None
    channel: str = f"chat:{canonical_thread_id}"

    # This list will persist for the duration of the WebSocket connection.
    all_temp_doc_ids_in_session: List[str] = []

    retrieved_documents_meta_for_citation: List[Dict[str, Any]] = []

    try:
        message_data = await ws.receive_json()
        if not (message_data.get("type") == "query" and "payload" in message_data):
            await ws.close(code=status.WS_1003_UNSUPPORTED_DATA,
                           reason="Protocol violation: Expected initial query payload.")
            return

        payload_raw = message_data["payload"]
        try:
            payload = ChatMessagePayloadWithFiles(**payload_raw)
        except Exception as e:
            logger.error(f"WebSocket received invalid payload: {e}, Raw: {payload_raw}")
            await ws.close(code=status.WS_1003_UNSUPPORTED_DATA, reason=f"Invalid payload format: {e}")
            return

        agent_name = payload.agent
        user_db_id = payload.user_id
        if not isinstance(user_db_id, int):
            logger.error(f"WebSocket received non-integer user_id: {user_db_id}. Closing connection.")
            await ws.close(code=status.WS_1003_UNSUPPORTED_DATA, reason="Invalid 'user_id' type. Expected integer.")
            return

        thread_id_from_frontend = sanitize_for_path(payload.thread_id)

        session_id, session_topic, returned_thread_id = await _get_or_create_session(db, thread_id_from_frontend,
                                                                                     agent_name, user_db_id)
        logger.debug(
            f"[main.py:websocket_chat] Session ID obtained: {session_id}, User DB ID: {user_db_id}, Agent Name: {agent_name}")

        canonical_thread_id = returned_thread_id
        channel = f"chat:{canonical_thread_id}"

        await manager.connect(ws, channel)
        if isinstance(manager, RedisConnectionManager):
            if pubsub_task and not pubsub_task.done():
                pubsub_task.cancel()
                await asyncio.sleep(0.01)
            pubsub_task = asyncio.create_task(redis_listen(ws, channel))

        is_first_message_in_session = (session_topic is None)

        agent_config = load_agent_config(agent_name)
        all_tools = []
        if "arxiv" in agent_config.tools: all_tools.extend(get_arxiv_tool())
        if "search_engine" in agent_config.tools: all_tools.append(SearchTool(agent_config=agent_config))
        if "sql_toolkit" in agent_config.tools:
            db_source = next((s for s in agent_config.sources if s.type == 'db'), None)
            if db_source and db_source.db_connection:
                all_tools.extend(create_sql_toolkit(db_connection_string=db_source.db_connection,
                                                    llm_model_name=agent_config.llm_model))

        langgraph_agent = get_agent_workflow(all_tools).compile()

        while True:
            # Get files and query from the current payload
            newly_uploaded_files = payload.files
            query = payload.query

            # If new files were uploaded, add their IDs to the session-long list
            if newly_uploaded_files:
                new_ids = [f.temp_doc_id for f in newly_uploaded_files]
                all_temp_doc_ids_in_session.extend(new_ids)
                all_temp_doc_ids_in_session = list(dict.fromkeys(all_temp_doc_ids_in_session))

            # Case 1: Handle "file-only" uploads.
            if not query and newly_uploaded_files:
                logger.info(
                    f"[{canonical_thread_id}] Acknowledged {len(newly_uploaded_files)} files. Waiting for user query.")

                # Send a "done" message back to the frontend immediately.
                # This tells the UI to stop the "thinking" indicator and wait for user input.
                ack_done_message = {
                    "done": True,
                    "error": False,
                    "errorMessage": None,
                    "request_id": str(uuid4()), # Use a dummy request ID for this ack
                    "user_id": user_db_id,
                    "thread_id": canonical_thread_id,
                    "topic_name": session_topic,
                    "citations": []
                }
                # Use ws.send_text for a direct reply instead of manager.broadcast
                await ws.send_text(json.dumps(ack_done_message))

                # Now, continue waiting for the next message from the client
                message_data = await ws.receive_json()
                try:
                    payload = ChatMessagePayloadWithFiles(**message_data["payload"])
                except Exception as e:
                    logger.error(
                        f"WebSocket received invalid payload on subsequent message: {e}, Raw: {message_data.get('payload')}")
                    await ws.close(code=status.WS_1003_UNSUPPORTED_DATA, reason=f"Invalid payload format: {e}")
                    return
                continue # Restart the loop with the new payload


            request_id = str(uuid4())
            logger.info(
                f"[{canonical_thread_id}] Processing request {request_id} for query: '{query[:50]}...' with {len(all_temp_doc_ids_in_session)} temporary files.")

            # Prepare metadata using only the files from this specific message
            quick_uploaded_files_meta = [
                {"file_name": f.file_name, "file_size": f.file_size, "temp_doc_id": f.temp_doc_id} for f in
                newly_uploaded_files]
            human_message_meta = {
                "quick_uploaded_files": quick_uploaded_files_meta} if quick_uploaded_files_meta else None

            await _save_message_and_update_session(db, session_id, 'human', query, meta=human_message_meta)

            if is_first_message_in_session:
                title = query.strip()[:100] if query else f"Chat with {len(all_temp_doc_ids_in_session)} files"
                await db.execute(
                    update(chat_sessions_table).where(chat_sessions_table.c.id == session_id).values(topic_name=title))
                await db.commit()
                is_first_message_in_session = False

            history_stmt = select(chat_messages_table.c.sender, chat_messages_table.c.content,
                                  chat_messages_table.c.meta).where(
                chat_messages_table.c.session_id == session_id).order_by(chat_messages_table.c.timestamp.asc())
            history_result = (await db.execute(history_stmt)).fetchall()
            history = [HumanMessage(content=msg.content) if msg.sender == 'human' else AIMessage(content=msg.content)
                       for msg in history_result]

            run_start_time = datetime.utcnow()
            insert_run_stmt = insert(agent_runs).values(
                run_id=request_id,
                session_id=session_id,
                start_time=run_start_time,
                status='running',
                initial_messages=[msg.dict() for msg in history]
            ).returning(agent_runs.c.id)
            run_db_id = (await db.execute(insert_run_stmt)).scalar_one()
            await db.commit()

            initial_state: AgentState = {
                "messages": history,
                "request_id": request_id,
                "agent_config": agent_config,
                # FIX: Pass the complete list of session documents to the agent.
                "temp_document_ids": all_temp_doc_ids_in_session
            }

            token_callback = UsageMetadataCallbackHandler()

            run_config = {
                "configurable": {
                    "thread_id": canonical_thread_id,
                    "session_id": session_id,
                    "user_id": user_db_id,
                    "agent_name": agent_name,
                    "callbacks": [token_callback]
                }
            }
            final_state, ai_response_content = await handle_query_streaming(
                initial_state,
                run_config,
                langgraph_agent,
                canonical_thread_id,
                run_db_id,
                db
            )

            retrieved_documents_meta_for_citation = final_state.get('retrieved_documents_meta_for_citation', [])
            ai_response_content_str = final_state.get('accumulated_content', '')

            extracted_citations = []
            if ai_response_content_str:
                extracted_citations = extract_citations_from_text(
                    llm_response_text=ai_response_content_str,
                    retrieved_documents_metadata=retrieved_documents_meta_for_citation
                )
                if extracted_citations:
                    logger.info(f"Extracted {len(extracted_citations)} citations from AI response.")
                else:
                    logger.info("No citations extracted from AI response.")

            done_message = {
                "done": True,
                "error": final_state.get("error", False),
                "errorMessage": final_state.get("errorMessage") if final_state and final_state.get("error") else None,
                "request_id": request_id,
                "user_id": user_db_id,
                "thread_id": canonical_thread_id,
                "topic_name": session_topic,
                "citations": extracted_citations
            }
            await manager.broadcast(channel, json.dumps({"token": final_state.get("errorMessage", ""), **done_message}))

            if ai_response_content_str and not (final_state and final_state.get("error")):
                ai_message_meta = {"citations": extracted_citations} if extracted_citations else None
                await _save_message_and_update_session(db, session_id, 'ai', ai_response_content_str,
                                                       meta=ai_message_meta)

            run_end_time = datetime.utcnow()
            update_run_stmt = update(agent_runs).where(agent_runs.c.id == run_db_id).values(
                end_time=run_end_time,
                status='completed' if not (final_state and final_state.get("error")) else 'failed',
                final_state=_serialize_for_db(final_state)
            )
            await db.execute(update_run_stmt)
            await db.commit()

            message_data = await ws.receive_json()
            try:
                payload = ChatMessagePayloadWithFiles(**message_data["payload"])
            except Exception as e:
                logger.error(
                    f"WebSocket received invalid payload on subsequent message: {e}, Raw: {message_data.get('payload')}")
                await ws.close(code=status.WS_1003_UNSUPPORTED_DATA, reason=f"Invalid payload format: {e}")
                return


    except WebSocketDisconnect:
        logger.info(f"[{canonical_thread_id}] Client disconnected.")
    except Exception as e:
        logger.error(f"[{canonical_thread_id}] Unhandled WebSocket Error: {e}", exc_info=True)
        try:
            await ws.send_text(json.dumps({
                "done": True,
                "error": True,
                "errorMessage": f"An unexpected server error occurred: {e}. Please try again.",
                "user_id": user_db_id if 'user_db_id' in locals() else None,
                "thread_id": canonical_thread_id,
            }))
        except Exception as send_error:
            logger.error(f"Failed to send error message to client: {send_error}")
    finally:
        if pubsub_task and not pubsub_task.done():
            pubsub_task.cancel()
        manager.disconnect(ws, channel)


async def redis_listen(ws: WebSocket, channel: str):
    if not isinstance(manager, RedisConnectionManager):
        return

    while True:
        try:
            if manager.pubsub is None:
                await manager.connect()
            await manager.pubsub.subscribe(channel)
            break
        except RedisConnectionError as e:
            logger.warning(f"Could not subscribe to {channel}, retrying in 1s: {e}")
            await asyncio.sleep(1)
    try:
        while True:
            try:
                message = await manager.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            except RedisConnectionError as e:
                logger.warning(f"Redis pubsub lost for {channel}; reconnecting: {e}")
                manager.pubsub = None
                return await redis_listen(ws, channel)
            if message and ws.client_state == WebSocketState.CONNECTED:
                await ws.send_text(message["data"])
    except asyncio.CancelledError:
        logger.info(f"Redis listener for channel {channel} cancelled.")
    finally:
        try:
            if manager.pubsub:
                await manager.pubsub.unsubscribe(channel)
        except RedisConnectionError:
            pass


async def handle_query_streaming(initial_state: AgentState, cfg: dict, langgraph_agent: Any, thread_id: str,
                                 run_db_id: int, db: AsyncSession) -> Tuple[Dict, str]:
    final_state = {}
    accumulated_content = ""
    channel = f"chat:{thread_id}"
    running_step_ids: Dict[str, int] = {}
    retrieved_documents_meta_for_citation: List[Dict[str, Any]] = []

    try:
        async for event in langgraph_agent.astream_events(initial_state, cfg):
            kind = event.get("event")
            name = event.get("name")
            run_id = event.get("run_id")
            if kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    accumulated_content += token
                    await manager.broadcast(channel, json.dumps({"token": token}))
            elif kind in ("on_chain_start", "on_tool_start"):
                if name in ["agent", "retriever", "sql_toolkit", "search_engine", "arxiv"]:
                    try:
                        serialized_input = _serialize_for_db(event["data"].get("input"))
                        step_start_time = datetime.utcnow()
                        insert_step_stmt = insert(agent_run_steps).values(
                            agent_run_id=run_db_id,
                            node_name=name,
                            start_time=step_start_time,
                            inputs=serialized_input
                        ).returning(agent_run_steps.c.id)
                        step_db_id = (await db.execute(insert_step_stmt)).scalar_one()
                        await db.commit()
                        if run_id:
                            running_step_ids[run_id] = step_db_id
                    except Exception as node_start_error:
                        logger.error(f"Failed to process start event for node {name}: {node_start_error}",
                                     exc_info=True)
            elif kind in ("on_chain_end", "on_tool_end"):
                if name == "retriever":  # Capture retrieved documents from the retriever tool
                    retriever_output = event["data"].get("output")
                    if isinstance(retriever_output, list):
                        for doc in retriever_output:
                            if isinstance(doc, dict) and 'metadata' in doc:  # Check for dict structure
                                retrieved_documents_meta_for_citation.append(doc['metadata'])
                            elif hasattr(doc, 'metadata'):  # Check for Langchain Document object
                                retrieved_documents_meta_for_citation.append(doc.metadata)
                    logger.info(
                        f"Captured {len(retrieved_documents_meta_for_citation)} retrieved document metadata for citation.")

                if name in ["agent", "retriever", "sql_toolkit", "search_engine", "arxiv"]:
                    try:
                        step_db_id = running_step_ids.pop(run_id, None)
                        if step_db_id:
                            serialized_output = _serialize_for_db(event["data"].get("output"))
                            step_end_time = datetime.utcnow()
                            update_step_stmt = update(agent_run_steps).where(
                                agent_run_steps.c.id == step_db_id
                            ).values(
                                end_time=step_end_time,
                                status='completed',
                                outputs=serialized_output
                            )
                            await db.execute(update_step_stmt)
                            await db.commit()
                    except Exception as node_end_error:
                        logger.error(f"Failed to process end event for node {name}: {node_end_error}", exc_info=True)
            elif kind == "on_graph_end":
                final_state = event['data']['output']

        # Add captured retrieved documents metadata to the final state for access in websocket_chat
        final_state['retrieved_documents_meta_for_citation'] = retrieved_documents_meta_for_citation
        final_state['accumulated_content'] = accumulated_content  # Ensure accumulated content is also passed back

        return _serialize_for_db(
            final_state), accumulated_content  # Keep original return for now, but main will use final_state dict


    except asyncio.CancelledError:
        logger.info(f"[{thread_id}] Generation task cancelled.")
        return {"error": True, "errorMessage": "Generation cancelled."}, accumulated_content
    except Exception as e:
        error_message = f"An error occurred during generation: {e}"
        logger.error(f"[{thread_id}] {error_message}", exc_info=True)
        await manager.broadcast(channel, json.dumps({"token": f"\n\n{error_message}"}))
        return {"error": True, "errorMessage": error_message}, ""


@app.get("/", tags=["Application"])
async def home(request: Request):
    agents_list = []
    try:
        agent_configs = get_agent_configs()
        agents_list = [{"name": c.name, "display_name": c.display_name or c.name} for c in agent_configs]
    except Exception as e:
        logger.error(f"Could not load agent configs: {e}")
    default_agent = agents_list[0]['name'] if agents_list else ""
    return templates.TemplateResponse("agent_interface.html", {
        "request": request, "agents": agents_list, "agent": default_agent
    })


@app.get("/login", tags=["Application"])
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/health", tags=["System"])
async def health_check():
    if not is_db_configured_sync():
        return JSONResponse({"status": "ok", "db_check": "skipped (no database configured)"})
    db_config = get_db_connection_config()
    if not db_config:
        raise HTTPException(status_code=500, detail="Database connection details missing or invalid in configuration.")
    try:
        sync_dialect = db_config.get('dialect', '').replace('+aiosqlite', '')
        if 'sqlite' in sync_dialect:
            db_path = Path(db_config.get('database_path', ''))
            if not db_path.is_absolute():
                db_path = _PROJECT_ROOT / db_path
            sync_conn_str = f"sqlite:///{db_path.resolve()}"
        else:
            conn_name = get_memory_storage_config().get("connection_name") or get_log_storage_config().get(
                "connection_name")
            if not conn_name:
                raise RuntimeError("Database connection name not found for health check.")
            sync_conn_str = get_db_connection(conn_name).replace("+aiomysql", "").replace("+asyncpg", "")
        engine = create_sync_engine(sync_conn_str)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return JSONResponse({"status": "ok", "db_check": "connected"})
    except Exception as e:
        logger.error(f"Health check failed to connect to database: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Database connection failed: {e}")


@app.get("/history/{thread_id}", tags=["Memory"])
async def get_history(thread_id: str, agent_name: str, user_id: int, api_key: str = Depends(get_http_api_key),
                      db: AsyncSession = Depends(get_db)):
    if not is_db_configured_sync():
        raise HTTPException(status_code=501, detail="Chat history not supported without a database.")
    safe_agent_name = sanitize_for_path(agent_name)
    safe_thread_id = sanitize_for_path(thread_id)
    try:
        session_id = (await db.execute(select(chat_sessions_table.c.id).where(
            (chat_sessions_table.c.thread_id == safe_thread_id) &
            (chat_sessions_table.c.agent_name == safe_agent_name) &
            (chat_sessions_table.c.user_id == user_id)
        ))).scalar_one_or_none()
        if not session_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Chat session not found for the given criteria.")
        messages_stmt = select(chat_messages_table.c.sender, chat_messages_table.c.content,
                               chat_messages_table.c.meta).where(
            chat_messages_table.c.session_id == session_id).order_by(chat_messages_table.c.timestamp.asc())
        messages_result = (await db.execute(messages_stmt)).fetchall()
        history = [{"type": row.sender, "content": row.content, "meta": row.meta} for row in messages_result]
        return JSONResponse(content=history)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading chat history for thread {safe_thread_id}: {e}",
                     exc_info=True)
        raise HTTPException(status_code=500,
                            detail="An unexpected error occurred while loading chat history. Please check server logs for details.")


@app.get("/sessions", tags=["Memory"])
async def list_sessions(agent_name: str, user_id: int, api_key: str = Depends(get_http_api_key),
                        db: AsyncSession = Depends(get_db)):
    if not is_db_configured_sync():
        raise HTTPException(status_code=501, detail="Chat history not supported without a database.")
    safe_agent_name = sanitize_for_path(agent_name)
    try:
        sessions_stmt = select(chat_sessions_table.c.thread_id, chat_sessions_table.c.topic_name).where(
            (chat_sessions_table.c.agent_name == safe_agent_name) &
            (chat_sessions_table.c.user_id == user_id)
        ).order_by(desc(chat_sessions_table.c.updated_at))
        sessions_result = (await db.execute(sessions_stmt)).fetchall()
        sessions = [{"thread_id": row.thread_id, "topic_name": row.topic_name or "New Chat"} for row in sessions_result]
        return JSONResponse(content=sessions)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading chat sessions for agent {safe_agent_name}: {e}",
                     exc_info=True)
        raise HTTPException(status_code=500,
                            detail="An unexpected error occurred while loading chat sessions. Please check server logs for details.")


@app.put("/sessions/{thread_id}/rename", tags=["Memory"], status_code=status.HTTP_200_OK)
async def rename_session(thread_id: str, request: RenameRequest, agent_name: str, user_id: int,
                         api_key: str = Depends(get_http_api_key), db: AsyncSession = Depends(get_db)):
    if not is_db_configured_sync():
        raise HTTPException(status_code=501, detail="Functionality not supported without a database.")
    safe_thread_id = sanitize_for_path(thread_id)
    safe_agent_name = sanitize_for_path(agent_name)
    new_name = request.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New name cannot be empty.")
    try:
        update_stmt = update(chat_sessions_table).where(
            (chat_sessions_table.c.thread_id == safe_thread_id) &
            (chat_sessions_table.c.agent_name == safe_agent_name) &
            (chat_sessions_table.c.user_id == user_id)
        ).values(topic_name=new_name, updated_at=datetime.utcnow())
        result = await db.execute(update_stmt)
        await db.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Chat session not found or permission denied.")
        return JSONResponse({"status": "ok", "message": "Chat session renamed successfully."})
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"A database error occurred while renaming session {safe_thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail="A database error occurred while renaming the session. Please try again or create a GitHub issue.")


@app.delete("/sessions/{thread_id}", tags=["Memory"], status_code=status.HTTP_200_OK)
async def delete_session(thread_id: str, agent_name: str, user_id: int, api_key: str = Depends(get_http_api_key),
                         db: AsyncSession = Depends(get_db)):
    if not is_db_configured_sync():
        raise HTTPException(status_code=501, detail="Functionality not supported without a database.")
    safe_thread_id = sanitize_for_path(thread_id)
    safe_agent_name = sanitize_for_path(agent_name)
    try:
        session_id = (await db.execute(select(chat_sessions_table.c.id).where(
            (chat_sessions_table.c.thread_id == safe_thread_id) &
            (chat_sessions_table.c.agent_name == safe_agent_name) &
            (chat_sessions_table.c.user_id == user_id)
        ))).scalar_one_or_none()
        if not session_id:
            raise HTTPException(status_code=404, detail="Chat session not found or permission denied.")
        await db.execute(delete(chat_messages_table).where(chat_messages_table.c.session_id == session_id))
        await db.execute(delete(chat_sessions_table).where(chat_sessions_table.c.id == session_id))
        await db.commit()
        return JSONResponse({"status": "ok", "message": "Chat session deleted successfully."})
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"A database error occurred while deleting session {safe_thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail="A database error occurred while deleting the session. Please try again or create a GitHub issue.")


#22