# app/main.py
import asyncio
import json
import uuid
import time

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
from pydantic import BaseModel, Field, ValidationError
from fastapi import Form
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.core.rate_limit import rate_limiter
from app.db.dao import create_chat_message
from app.api.lambda_tool import router as lambda_tool_router
from app.db.dao import create_lambda_run, get_lambda_run


from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Depends, status, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

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
    get_memory_storage_config, get_db_connection_config, get_db_connection, get_cors_settings, _get_config_parser, \
    get_allowed_hosts
from app.db import initialize_db_connections, get_db
import app.db as db_mod
from app.core.validation import sanitize_for_path
from app.core.security import get_http_api_key, get_websocket_api_key, get_current_user_from_api_key, \
    get_current_user_from_websocket, PermissionChecker
from app.agents.config_manager import get_agent_configs, load_agent_config
from app.core.serialization import _serialize_for_db
from app.tools.api_toolkit import APIToolkit
from app.api.security import router as security_api_router
from app.tools.lambda_tool import LambdaTool


from app.schemas.security import User

# Agents & Pipelines
from app.agents.agent_graph import get_agent_graph, AgentState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler
from app.core.structured_logging import get_logging_config, DatabaseLogHandler
from app.db.models import ragnetic_logs_table



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
from app.api.documents import router as documents_api_router
from app.api.monitoring import router as monitoring_api_router


from app.services.temporary_document_service import TemporaryDocumentService, TemporaryDocumentUploadResult
from app.core.config import get_debug_mode, get_memory_storage_config, get_log_storage_config

# Get the main application logger after configuration is applied
load_dotenv()
logging.config.dictConfig(get_logging_config())
logger = logging.getLogger("ragnetic")

def is_db_configured_sync() -> bool:
    """Checks if a database is configured for either memory or logging."""
    mem_config = get_memory_storage_config()
    log_config = get_log_storage_config()
    return (mem_config.get("type") in ["db", "sqlite"] and mem_config.get("connection_name")) or \
        (log_config.get("type") == "db" and log_config.get("connection_name"))

# --- Global Settings & App Initialization ---
_APP_PATHS = get_path_settings()
_PROJECT_ROOT = _APP_PATHS["PROJECT_ROOT"]
_DATA_DIR = _APP_PATHS["DATA_DIR"]
# _WORKFLOWS_DIR = _APP_PATHS["WORKFLOWS_DIR"]  # Removed workflow functionality

config = _get_config_parser()
WEBSOCKET_MODE = config.get('SERVER', 'websocket_mode', fallback='memory')
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
allowed_origins = get_cors_settings()


debug_on = get_debug_mode()
paths = get_path_settings()
uvicorn_log_cfg_path = paths["PROJECT_ROOT"] / (
    "logging.uvicorn.debug.json" if debug_on else "logging.uvicorn.json"
)


def _infer_server_url(websocket: WebSocket) -> str:
    headers = dict(websocket.headers)
    xf_proto = headers.get("x-forwarded-proto")
    xf_host  = headers.get("x-forwarded-host")

    scheme = xf_proto or ("https" if websocket.url.scheme == "wss" else "http")
    host   = xf_host or websocket.url.hostname
    port   = websocket.url.port

    # If proxy provided host:port already, don’t re-append a port
    if xf_host and ":" in xf_host:
        return f"{scheme}://{host}"

    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        return f"{scheme}://{host}:{port}"
    return f"{scheme}://{host}"

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id

        process_time = (time.time() - start_time) * 1000
        client_host = request.client.host if request.client else "unknown"
        log_message = f"{client_host} \"{request.method} {request.url.path}\" {response.status_code} - {process_time:.2f}ms"

        # We use a structured logger here
        logger.info(
            log_message,
            extra={
                "extra_data": {
                    "request_id": correlation_id,  # use as request_id
                    "route": request.url.path,
                    "user_id": getattr(request.state, "user_id", None),
                    "service": "api_access",
                    "latency_ms": int(process_time)  # store as number
                }
            },
        )

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        # HSTS (enable only when served over HTTPS)
        resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        # CSP (allow inline because pages use inline scripts; tighten with nonces later if desired)
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' ws: wss: https://cdn.jsdelivr.net; "
            "frame-ancestors 'none';"
        )
        return resp

app = FastAPI(title="RAGnetic API", version="0.1.0")
app.add_middleware(CorrelationIdMiddleware)

# CORS: wildcard → no credentials; explicit origins → credentials allowed
cors_kwargs = {"allow_methods": ["*"], "allow_headers": ["*"]}
if allowed_origins == ["*"]:
    cors_kwargs.update({"allow_origin_regex": ".*", "allow_credentials": False})
else:
    cors_kwargs.update({"allow_origins": allowed_origins, "allow_credentials": True})
app.add_middleware(CORSMiddleware, **cors_kwargs)

# Hardening + perf
app.add_middleware(TrustedHostMiddleware, allowed_hosts=get_allowed_hosts())
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(SecurityHeadersMiddleware)


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

    async def connect(self, websocket=None, channel=None):
        if self.connection is None:
            self.connection = redis.from_url(self.redis_url, decode_responses=True)
            await self.connection.ping()

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

        db_logger_names = ["ragnetic", "app"]
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

            conn_name = get_memory_storage_config().get("connection_name") or get_log_storage_config().get(
                "connection_name")
            if not conn_name:
                raise RuntimeError("Database connection name not found for async initialization.")
            initialize_db_connections(conn_name)

            from app.db.dao import create_default_roles_and_permissions
            if db_mod.AsyncSessionLocal is None:
                raise RuntimeError("AsyncSessionLocal is not initialized. Did initialize_db_connections() run?")

            async with db_mod.AsyncSessionLocal() as db_session:
                await create_default_roles_and_permissions(db_session)

            # Webhook registration code removed - no longer needed
        except Exception as e:
            logger.critical(f"CRITICAL: Failed during database startup. Error: {e}", exc_info=True)
            raise
    else:
        logger.warning("No database configured. DB-dependent features will be disabled.")

    if not os.path.exists(_DATA_DIR):
        logger.error(f"'{_DATA_DIR}' not found. Please run 'ragnetic init'.")
        return

    monitored_dirs = [str(_DATA_DIR), str(_APP_PATHS["AGENTS_DIR"])]
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
app.include_router(citations_api_router)
app.include_router(evaluation_api_router)
app.include_router(lambda_tool_router)
app.include_router(documents_api_router)
app.include_router(monitoring_api_router)



class RenameRequest(BaseModel):
    new_name: str


class QuickUploadFileItem(BaseModel):
    file_name: str = Field(..., description="Name of the uploaded file.")
    file_size: int = Field(..., description="Size of the uploaded file in bytes.")
    temp_doc_id: str = Field(..., description="Unique ID assigned to the temporary document.")

class WebSocketUploadedFileItem(BaseModel):
    file_name: str
    file_size: int
    temp_doc_id: str

class ChatMessagePayloadWithFiles(BaseModel):
    agent: str
    thread_id: Optional[str] = None
    query: str
    # List of dictionaries, each with 'file_name' and 'temp_doc_id'
    files: List[WebSocketUploadedFileItem] = Field(default_factory=list,
                                                   description="List of successfully uploaded temporary files with their temp_doc_ids and sizes.")

class CreateSessionRequest(BaseModel):
    agent_name: str
    user_id: int

@app.post(
    "/api/v1/sessions/create",
    status_code=status.HTTP_201_CREATED,
    tags=["Memory"],
    dependencies=[Depends(rate_limiter("session_create", 10, 60))]
)
async def create_new_session(
    request: CreateSessionRequest,
    current_user: User = Depends(PermissionChecker(["session:create"])),
    db: AsyncSession = Depends(get_db)
):
    """
    Explicitly creates a new chat session and returns its details.
    Used by the frontend to get a thread_id before the first message.
    """
    session_id, topic_name, thread_id = await _get_or_create_session(
        db,
        thread_id_from_frontend=None,
        agent_name=request.agent_name,
        user_db_id=request.user_id
    )
    return {"thread_id": thread_id, "session_id": session_id, "topic_name": topic_name}


@app.post(
    "/api/v1/chat/upload-temp-document",
    response_model=QuickUploadFileItem,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    dependencies=[Depends(rate_limiter("upload", 20, 60))]  # <-- add
)
async def upload_temp_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
    current_user: User = Depends(PermissionChecker(["document:upload"])),
    db: AsyncSession = Depends(get_db)
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
            thread_id=thread_id,
            db = db,
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
            "session_created",
            extra={"extra_data": {
                "session_id": new_session.id,
                "thread_id": new_session.thread_id,
                "agent_id": agent_name,
                "user_id": user_db_id
            }},
        )
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


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket,
                         current_user: User = Depends(get_current_user_from_websocket),
                         db: AsyncSession = Depends(get_db)):
    await ws.accept()
    canonical_thread_id: str = "uninitialized_thread"
    pubsub_task = None
    channel: str = f"chat:{canonical_thread_id}"
    all_temp_doc_ids_in_session: List[str] = []

    try:
        # Initial connection and payload validation
        message_data = await ws.receive_json()
        if not (message_data.get("type") == "query" and "payload" in message_data):
            await ws.close(code=status.WS_1003_UNSUPPORTED_DATA,
                           reason="Protocol violation: Expected initial query payload.")
            return

        payload_raw = message_data["payload"]
        try:
            payload = ChatMessagePayloadWithFiles(**payload_raw)
        except ValidationError as e:
            logger.error("ws_invalid_payload",
                         extra={"extra_data": {"error": str(e)}})
            await ws.close(code=status.WS_1003_UNSUPPORTED_DATA, reason="Invalid payload format.")
            return

        agent_name = payload.agent
        user_db_id = current_user.id
        if not isinstance(user_db_id, int):
            logger.error(f"WebSocket received non-integer user_id: {user_db_id}. Closing connection.")
            await ws.close(code=status.WS_1003_UNSUPPORTED_DATA, reason="Invalid 'user_id' type. Expected integer.")
            return

        # Session creation and tool setup
        thread_id_from_frontend = sanitize_for_path(payload.thread_id)
        session_id, session_topic, returned_thread_id = await _get_or_create_session(db, thread_id_from_frontend,
                                                                                     agent_name, user_db_id)
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

        if "retriever" in agent_config.tools:
            retriever_tool = await get_retriever_tool(agent_config, user_db_id, canonical_thread_id)
            all_tools.append(retriever_tool)

        if "api_toolkit" in agent_config.tools:
            all_tools.append(APIToolkit())

        if "lambda_tool" in agent_config.tools:
            all_tools.append(LambdaTool(server_url=_infer_server_url(ws)))

        langgraph_agent = get_agent_graph(all_tools).compile()


        # Main message loop
        while True:
            # 1. Process incoming payload
            newly_uploaded_files = payload.files
            query = payload.query
            if newly_uploaded_files:
                new_ids = [f.temp_doc_id for f in newly_uploaded_files]
                all_temp_doc_ids_in_session.extend(new_ids)
                all_temp_doc_ids_in_session = list(dict.fromkeys(all_temp_doc_ids_in_session))

            # 2. Handle file-only uploads or empty messages
            if not query:
                if newly_uploaded_files:
                    logger.info("files_acknowledged",
                                extra={"extra_data": {"thread_id": canonical_thread_id,
                                                      "files": len(newly_uploaded_files)}})
                    ack_message = {"done": True, "error": False, "request_id": str(uuid4()), "user_id": user_db_id,
                                   "thread_id": canonical_thread_id, "topic_name": session_topic, "citations": []}
                    await ws.send_text(json.dumps(ack_message))
                else:
                    logger.warning("empty_query_skipped",
                                   extra={"extra_data": {"thread_id": canonical_thread_id}})

                message_data = await ws.receive_json()
                payload = ChatMessagePayloadWithFiles(**message_data["payload"])
                continue

            # 3. Save HUMAN message and prepare for agent run
            request_id = str(uuid4())

            # Use the DAO function to save the user's message and update the session timestamp
            await create_chat_message(
                db, session_id, 'human', query, meta={
                "quick_uploaded_files": [f.dict() for f in newly_uploaded_files]
            } if newly_uploaded_files else None)

            if is_first_message_in_session:
                title = query.strip()[:100]
                await db.execute(
                    update(chat_sessions_table).where(chat_sessions_table.c.id == session_id).values(topic_name=title)
                )
                await db.commit()
                is_first_message_in_session = False

            # 4. Fetch history and configure the agent run
            history_stmt = select(chat_messages_table.c.sender, chat_messages_table.c.content).where(
                chat_messages_table.c.session_id == session_id).order_by(chat_messages_table.c.timestamp.asc())
            history = [HumanMessage(content=msg.content) if msg.sender == 'human' else AIMessage(content=msg.content)
                       for msg in (await db.execute(history_stmt)).fetchall()]

            run_db_id = (await db.execute(insert(agent_runs).values(run_id=request_id, session_id=session_id,
                                                                    start_time=datetime.utcnow()).returning(
                agent_runs.c.id))).scalar_one()
            await db.commit()

            initial_state = {
                "messages": history,
                "request_id": request_id,
                "agent_config": agent_config,
                "temp_document_ids": all_temp_doc_ids_in_session
            }

            # Inject the database session into the agent graph's config
            run_config = {
                "configurable": {
                    "thread_id": canonical_thread_id,
                    "session_id": session_id,
                    "user_id": user_db_id,
                    "agent_name": agent_name,
                    "db_session": db,
                    "callbacks": [UsageMetadataCallbackHandler()],
                    "tools": all_tools
                }
            }

            # 5. Run agent and handle cancellation
            generation_task = asyncio.create_task(
                handle_query_streaming(initial_state, run_config, langgraph_agent, canonical_thread_id, run_db_id, db))
            listener_task = asyncio.create_task(ws.receive_json())

            done, pending = await asyncio.wait({generation_task, listener_task}, return_when=asyncio.FIRST_COMPLETED)

            if listener_task in done:
                message = listener_task.result()
                if message.get("type") == "interrupt":
                    logger.info("generation_interrupt_received",
                                extra={"extra_data": {"thread_id": canonical_thread_id}})
                    generation_task.cancel()
            else:
                listener_task.cancel()

            final_state_dict, ai_response_content_str = await generation_task

            # 6. Process Final Result (Simplified)
            # The agent graph has already saved the AI message and citations.
            # We just send the final "done" message with the citation data from the agent.
            final_citations = final_state_dict.get('citations', [])

            done_message = {
                "done": True,
                "error": final_state_dict.get("error", False),
                "errorMessage": final_state_dict.get("errorMessage"),
                "request_id": request_id,
                "user_id": user_db_id,
                "thread_id": canonical_thread_id,
                "topic_name": session_topic,
                "citations": final_citations
            }
            await manager.broadcast(channel, json.dumps(done_message))

            # Update the agent_runs record with the final state
            serialized_final_state = _serialize_for_db(final_state_dict)
            await db.execute(update(agent_runs).where(agent_runs.c.id == run_db_id).values(
                end_time=datetime.utcnow(),
                status='completed' if not final_state_dict.get("error") else 'failed',
                final_state=serialized_final_state
            ))
            await db.commit()

            # 7. Wait for the next user message
            if listener_task.done() and not listener_task.cancelled() and listener_task.result().get(
                    "type") != "interrupt":
                message_data = listener_task.result()
            else:
                message_data = await ws.receive_json()

            payload = ChatMessagePayloadWithFiles(**message_data["payload"])

    except WebSocketDisconnect:
        logger.info("ws_client_disconnected", extra={"extra_data": {"thread_id": canonical_thread_id}})
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning("ws_bad_payload",
                       extra={"extra_data": {"thread_id": canonical_thread_id, "error": str(e)}})
        try:
            await ws.send_text(json.dumps({
                "done": True,
                "error": True,
                "errorMessage": "Invalid payload format.",
                "user_id": user_db_id if 'user_db_id' in locals() else None,
                "thread_id": canonical_thread_id,
            }))
        finally:
            await ws.close(code=status.WS_1003_UNSUPPORTED_DATA, reason="Invalid payload format.")
    except SQLAlchemyError as e:
        logger.error("ws_db_error", exc_info=True,
                     extra={"extra_data": {"thread_id": canonical_thread_id}})
        try:
            await ws.send_text(json.dumps({
                "done": True,
                "error": True,
                "errorMessage": "A database error occurred.",
                "user_id": user_db_id if 'user_db_id' in locals() else None,
                "thread_id": canonical_thread_id,
            }))
        finally:
            await ws.close(code=status.WS_1011_INTERNAL_ERROR, reason="Database error.")
    except asyncio.CancelledError:
        logger.info("ws_task_cancelled", extra={"extra_data": {"thread_id": canonical_thread_id}})
        raise

    except Exception as e:
        logger.error("ws_unhandled_error", exc_info=True,
                     extra={"extra_data": {"thread_id": canonical_thread_id}})
        try:
            await ws.send_text(json.dumps({
                "done": True,
                "error": True,
                "errorMessage": "An unexpected server error occurred. Please try again.",
                "user_id": user_db_id if 'user_db_id' in locals() else None,
                "thread_id": canonical_thread_id,
            }))
        finally:
            await ws.close(code=status.WS_1011_INTERNAL_ERROR, reason="Unhandled error.")
    finally:
        if pubsub_task and not pubsub_task.done():
            pubsub_task.cancel()
        manager.disconnect(ws, channel)



async def redis_listen(ws: WebSocket, channel: str):
    if not isinstance(manager, RedisConnectionManager):
        return

    # Create a new, dedicated Redis client for this specific listener task.
    listener_client = redis.from_url(manager.redis_url, decode_responses=True)
    pubsub = listener_client.pubsub()

    try:
        await pubsub.subscribe(channel)
        logger.info(f"Redis listener subscribed to channel {channel}.")
        while True:
            # Periodically check if the WebSocket client is still connected.
            if ws.client_state != WebSocketState.CONNECTED:
                logger.info(f"Client disconnected from channel {channel}. Stopping listener.")
                break

            try:
                # Wait for a message with a timeout.
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    await ws.send_text(message["data"])
            except asyncio.TimeoutError:
                # Timeout is expected, just continue the loop to check client state again.
                continue
            except RedisConnectionError as e:
                logger.warning(f"Redis connection lost for listener on {channel}: {e}")
                break # Exit the loop if the connection fails.

    except asyncio.CancelledError:
        logger.info(f"Redis listener for channel {channel} was cancelled.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in Redis listener for {channel}: {e}", exc_info=True)
    finally:
        # Crucially, ensure the dedicated connection for this task is always closed.
        logger.info(f"Closing Redis listener connection for channel {channel}.")
        try:
            try:
                await pubsub.unsubscribe(channel)
            except Exception:
                pass
            await pubsub.close()
        finally:
            await listener_client.close()



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
                if name in ["agent", "retriever", "sql_toolkit", "search_engine", "arxiv", "api_toolkit","lambda_tool"]:
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
                if name == "retriever":
                    retriever_output = event["data"].get("output")
                    if isinstance(retriever_output, list):
                        for doc in retriever_output:
                            if isinstance(doc, dict) and 'metadata' in doc:
                                retrieved_documents_meta_for_citation.append(doc['metadata'])
                            elif hasattr(doc, 'metadata'):
                                retrieved_documents_meta_for_citation.append(doc.metadata)
                    logger.info(
                        "retriever_metadata_captured",
                        extra={"extra_data": {
                            "thread_id": thread_id,
                            "count": len(retrieved_documents_meta_for_citation)
                        }},
                    )

                if name in ["agent", "retriever", "sql_toolkit", "search_engine", "arxiv", "api_toolkit","lambda_tool"]:
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

        final_state['retrieved_documents_meta_for_citation'] = retrieved_documents_meta_for_citation
        final_state['accumulated_content'] = accumulated_content

        # *** CORRECTED: Return the dictionary, not the serialized string. ***
        return final_state, accumulated_content

    except asyncio.CancelledError:
        logger.info("generation_cancelled", extra={"extra_data": {"thread_id": thread_id}})

        return {"error": True, "errorMessage": "Generation cancelled."}, accumulated_content
    except Exception as e:
        error_message = f"An error occurred during generation: {e}"
        logger.error("generation_error", exc_info=True,
                     extra={"extra_data": {"thread_id": thread_id, "error": str(e)}})
        await manager.broadcast(channel, json.dumps({"token": f"\n\n{error_message}"}))
        return {"error": True, "errorMessage": error_message}, ""


@app.get("/", tags=["Application"])
async def home(request: Request, db: AsyncSession = Depends(get_db)):
    agents_list = []
    try:
        agent_configs = get_agent_configs()
        # Filter out stopped agents for chat interface
        from app.db.models import agents_table
        from sqlalchemy import select
        
        # Get agent statuses from database
        result = await db.execute(select(agents_table.c.name, agents_table.c.status))
        agent_statuses = {row.name: row.status for row in result.fetchall()}
        
        # Only include agents that are not stopped
        agents_list = [
            {"name": c.name, "display_name": c.display_name or c.name} 
            for c in agent_configs 
            if agent_statuses.get(c.name, "created") != "stopped"
        ]
    except Exception as e:
        logger.error(f"Could not load agent configs: {e}")
    default_agent = agents_list[0]['name'] if agents_list else ""
    return templates.TemplateResponse("agent_interface.html", {
        "request": request, "agents": agents_list, "agent": default_agent
    })


@app.get("/dashboard", tags=["Application"])
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/login", tags=["Application"])
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/health", tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    health_status = {"status": "ok", "checks": {}}
    has_critical_failure = False

    # 1. Database Check (async, non-blocking)
    db_status = {"status": "ok"}
    try:
        if is_db_configured_sync():
            await db.execute(text("SELECT 1"))
            db_status["status"] = "connected"
        else:
            db_status["status"] = "skipped (no database configured)"
    except SQLAlchemyError as e:
        logger.error(f"Health check DB error: {e}", exc_info=True)
        db_status["status"] = "failed"
        db_status["error"] = str(e)
        has_critical_failure = True
    health_status["checks"]["database"] = db_status

    # 2. Redis/Queue Check (unchanged)
    redis_status = {"status": "ok"}
    if WEBSOCKET_MODE == 'redis':
        try:
            r = redis.from_url(REDIS_URL, decode_responses=True)
            await r.ping()
            redis_status["status"] = "connected"
        except Exception as e:
            logger.error(f"Health check failed to connect to Redis: {e}", exc_info=True)
            redis_status["status"] = "failed"
            redis_status["error"] = str(e)
            has_critical_failure = True
    else:
        redis_status["status"] = "skipped (in-memory mode)"
    health_status["checks"]["redis"] = redis_status

    # 3. Vector Store Check (unchanged)
    vs_status = {"status": "ok"}
    vs_dir = _APP_PATHS["VECTORSTORE_DIR"]
    if not os.path.exists(vs_dir):
        vs_status["status"] = "failed"
        vs_status["error"] = "vector store directory not found"
        has_critical_failure = True
    health_status["checks"]["vector_store_dir"] = vs_status

    # 4. Provider Credentials Check (unchanged)
    provider_status = {"status": "ok", "providers": {}}
    providers_to_check = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "pinecone": "PINECONE_API_KEY",
        "qdrant": "QDRANT_API_KEY"
    }
    for provider, env_key in providers_to_check.items():
        provider_status["providers"][provider] = "ok" if os.environ.get(env_key) else "missing"
    health_status["checks"]["api_keys"] = provider_status

    if has_critical_failure:
        health_status["status"] = "critical_failure"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=health_status)

    return JSONResponse(health_status)


# Corrected endpoint for get_history
@app.get(
    "/history/{thread_id}",
    tags=["Memory"],
    dependencies=[Depends(rate_limiter("history", 120, 60))]
)
async def get_history(thread_id: str, agent_name: str, user_id: int,
                      current_user: User = Depends(PermissionChecker(["history:read"])),
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
            # Return empty history instead of 404 error for better UX
            logger.info(f"No chat session found for thread {safe_thread_id}, agent {safe_agent_name}, user {user_id}. Returning empty history.")
            return JSONResponse(content=[])
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


# Corrected endpoint for list_sessions
@app.get(
    "/sessions",
    tags=["Memory"],
    dependencies=[Depends(rate_limiter("sessions", 60, 60))]
)
async def list_sessions(agent_name: str, user_id: int,
                        current_user: User = Depends(PermissionChecker(["sessions:read"])),
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


@app.put(
    "/sessions/{thread_id}/rename",
    tags=["Memory"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limiter("sessions_update", 20, 60))]
)
async def rename_session(thread_id: str, request: RenameRequest, agent_name: str, user_id: int,
                         current_user: User = Depends(PermissionChecker(["sessions:update"])),
                         db: AsyncSession = Depends(get_db)):
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


@app.delete(
    "/sessions/{thread_id}",
    tags=["Memory"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(rate_limiter("sessions_delete", 10, 60))]
)
async def delete_session(
        thread_id: str,
        agent_name: str,
        user_id: int,
        current_user: User = Depends(PermissionChecker(["sessions:delete"])),
        db: AsyncSession = Depends(get_db)
):
    if not is_db_configured_sync():
        raise HTTPException(status_code=501, detail="Functionality not supported without a database.")

    safe_thread_id = sanitize_for_path(thread_id)
    safe_agent_name = sanitize_for_path(agent_name)

    try:
        # First, retrieve the agent config to initialize the TemporaryDocumentService
        agent_config = load_agent_config(safe_agent_name)
        temp_doc_service = TemporaryDocumentService(agent_config=agent_config)

        # Before deleting the session record, perform a comprehensive cleanup of temporary files
        await temp_doc_service.cleanup_user_thread_temp_documents(
            user_id=user_id,
            thread_id=safe_thread_id,
            db=db
        )

        # Now, proceed with deleting the chat history and session record
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

        return JSONResponse(
            {"status": "ok", "message": "Chat session and all temporary documents deleted successfully."})

    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"A database error occurred while deleting session {safe_thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail="A database error occurred while deleting the session. Please try again or create a GitHub issue.")
    except Exception as e:
        # If any other part of the cleanup or deletion fails, log it and return an error.
        logger.error(f"An unexpected error occurred during session deletion for {safe_thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail="An unexpected error occurred during session deletion. Please check server logs for details.")