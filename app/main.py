# app/main.py
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
import os
import logging
import logging.config
import json
import asyncio
from uuid import uuid4
from typing import Optional, List, Dict, Any, Tuple
from multiprocessing import Process
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
import yaml

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, status
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# --- RAGnetic Imports ---

# Database & Models
from app.db.models import chat_sessions_table, chat_messages_table, users_table, agent_runs, agent_run_steps, workflows_table
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, insert, update, text, delete
from sqlalchemy.exc import SQLAlchemyError, IntegrityError # <-- ADD IntegrityError

# Core Components
from app.core.config import get_path_settings, get_server_api_keys, get_log_storage_config, \
    get_memory_storage_config, get_db_connection_config, get_db_connection, get_cors_settings, _get_config_parser
from app.db import initialize_db_connections, get_db
from app.core.structured_logging import JSONFormatter
from app.core.validation import sanitize_for_path
from app.core.security import get_http_api_key, get_websocket_api_key
from app.schemas.agent import AgentConfig
from app.core.serialization import _serialize_for_db
from app.schemas.workflow import WorkflowCreate

# Agents & Pipelines
from app.agents.config_manager import get_agent_configs, load_agent_config
from app.pipelines.embed import embed_agent_data
from app.agents.agent_graph import get_agent_workflow, AgentState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document  # <-- CORRECTED IMPORT

# Tools
from app.tools.sql_tool import create_sql_toolkit
from app.tools.retriever_tool import get_retriever_tool
from app.tools.arxiv_tool import get_arxiv_tool
from app.tools.search_engine_tool import SearchTool

# System
from app.watcher import start_watcher
from app.workflows.scheduler import start_scheduler_process
from sqlalchemy import create_engine as create_sync_engine
import alembic.config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

# --- API Routers ---
from app.api.agents import router as agents_api_router
from app.api.audit import router as audit_api_router
from app.api.query import router as query_api_router
from app.api.evaluation import router as evaluation_api_router
from app.api.metrics import router as metrics_api_router
from app.api.schedules import router as schedules_api_router
from app.api.webhooks import setup_dynamic_webhooks

from app.api import workflows
from app.db.models import schedules_table
import re

# --- Base Logging Configuration ---
_APP_PATHS = get_path_settings()
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "json": {"()": "app.core.structured_logging.JSONFormatter"},
        "metrics_json": {
            "()": "app.core.structured_logging.JSONFormatter",
            "format": "%(message)s"
        }
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default"},
        "metrics_file_json": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "metrics_json",
            "filename": _APP_PATHS["LOGS_DIR"] / "ragnetic_metrics.json",
            "when": "midnight",
            "backupCount": 7,
            "encoding": "utf-8"
        },
        "database_handler": {
            "()": "app.core.structured_logging.DatabaseLogHandler",
            "connection_name": "ragnetic_logs",
            "table_name": "ragnetic_logs"
        }
    },
    "loggers": {
        "ragnetic": {"handlers": ["console", "database_handler"], "level": "INFO", "propagate": False},
        "ragnetic.metrics": {
            "handlers": ["database_handler"],
            "level": "INFO",
            "propagate": False
        },
        "uvicorn.error": {"handlers": ["console"], "level": "INFO"},
        "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "sqlalchemy.engine": {"handlers": ["console"], "level": "WARNING"},
    },
    "root": {"handlers": ["console"], "level": "INFO"},
}
log_storage_config = get_log_storage_config()
if log_storage_config.get("type") == "file":
    log_dir = _APP_PATHS["LOGS_DIR"]
    log_dir.mkdir(exist_ok=True)
    LOGGING_CONFIG["handlers"]["app_file"] = {
        "class": "logging.handlers.TimedRotatingFileHandler",
        "formatter": "default",
        "filename": log_dir / "ragnetic_app.log",
        "when": "midnight", "backupCount": 7, "encoding": "utf-8"
    }
    LOGGING_CONFIG["loggers"]["ragnetic"]["handlers"].append("app_file")
    LOGGING_CONFIG["loggers"]["ragnetic.metrics"]["handlers"].append("metrics_file_json")
elif log_storage_config.get("type") == "db":
    conn_name = log_storage_config.get("connection_name")
    table_name = log_storage_config.get("log_table_name", "ragnetic_logs")
    if conn_name:
        LOGGING_CONFIG["handlers"]["database"] = {
            "()": "app.core.structured_logging.DatabaseLogHandler",
            "connection_name": conn_name,
            "table_name": table_name,
        }
        LOGGING_CONFIG["loggers"]["ragnetic"]["handlers"].append("database")
        LOGGING_CONFIG["loggers"]["ragnetic.metrics"]["handlers"].append("database")
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("ragnetic")
load_dotenv()

# --- Global Settings & App Initialization ---
_PROJECT_ROOT = _APP_PATHS["PROJECT_ROOT"]
_DATA_DIR = _APP_PATHS["DATA_DIR"]

config = _get_config_parser()
WEBSOCKET_MODE = config.get('SERVER', 'websocket_mode', fallback='memory')
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost")
allowed_origins = get_cors_settings()

app = FastAPI(title="RAGnetic API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")




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
_scheduler_process: Optional[Process] = None


def _parse_schedule_time(time_str: str) -> Dict[str, int]:
    """Parses time strings like '9:00am' or '16:30' into hour and minute."""
    if not isinstance(time_str, str):
        return {}

    time_str = time_str.lower()
    is_pm = 'pm' in time_str

    # Remove am/pm for parsing and split
    time_part = re.sub(r'[ap]m', '', time_str).strip()

    try:
        hour, minute = map(int, time_part.split(':'))

        if is_pm and hour < 12:
            hour += 12
        elif not is_pm and hour == 12:  # Handle '12:00am' case
            hour = 0

        return {"hour": hour, "minute": minute}
    except (ValueError, TypeError):
        logger.warning(f"Could not parse time string '{time_str}' in schedule. Skipping.")
        return {}


def _sync_workflows_from_files():
    """
    Scans the workflows directory, syncs YAML definitions with the database,
    and automatically manages associated schedules based on the 'trigger' section.
    This version correctly handles both frequency/time and direct cron configurations.
    """
    if not is_db_configured():
        logger.warning("Skipping workflow sync: No database is configured.")
        return

    workflows_dir = _APP_PATHS.get("WORKFLOWS_DIR")
    if not workflows_dir or not workflows_dir.is_dir():
        return

    logger.info("Starting sync of workflow and schedule files with the database...")

    conn_name = (get_memory_storage_config().get("connection_name") or
                 get_log_storage_config().get("connection_name"))
    if not conn_name:
        logger.error("Could not determine DB connection name for workflow sync.")
        return

    sync_conn_str = get_db_connection(conn_name).replace('+aiosqlite', '').replace('+asyncpg', '')
    engine = create_sync_engine(sync_conn_str)

    with engine.connect() as conn:
        synced_workflow_names = set()
        for filepath in workflows_dir.glob("*.yaml"):
            try:
                with open(filepath, 'r') as f:
                    payload = yaml.safe_load(f)
                    wf_in = WorkflowCreate(**payload)
                    synced_workflow_names.add(wf_in.name)

                existing_workflow = conn.execute(
                    select(workflows_table).where(workflows_table.c.name == wf_in.name)).first()
                wf_values = {"agent_name": wf_in.agent_name, "description": wf_in.description, "definition": payload,
                             "updated_at": datetime.utcnow()}
                if existing_workflow:
                    stmt = update(workflows_table).where(workflows_table.c.id == existing_workflow.id).values(
                        **wf_values)
                else:
                    stmt = insert(workflows_table).values(name=wf_in.name, **wf_values)
                conn.execute(stmt)

                trigger = payload.get("trigger", {})
                schedule_info = trigger.get("schedule")
                is_scheduled = trigger.get("type") == "schedule" and isinstance(schedule_info, dict)

                existing_schedule = conn.execute(
                    select(schedules_table).where(schedules_table.c.workflow_name == wf_in.name)).first()

                if is_scheduled:
                    cron_config = {}
                    # Handle frequency/time format
                    if "frequency" in schedule_info:
                        cron_config = _parse_schedule_time(schedule_info.get("time", ""))
                        if schedule_info.get("frequency") == "weekly":
                            days = schedule_info.get("day_of_week", [])
                            cron_config["day_of_week"] = ",".join(day[:3].lower() for day in days)
                    # Handle direct cron format
                    else:
                        cron_config = {k: v for k, v in schedule_info.items() if
                                       k in ['year', 'month', 'day', 'week', 'day_of_week', 'hour', 'minute', 'second']}

                    if not cron_config:
                        logger.warning(
                            f"Could not determine a valid cron configuration for workflow '{wf_in.name}'. Skipping schedule.")
                        continue

                    schedule_values = {"cron_schedule": cron_config, "is_enabled": True}

                    if existing_schedule:
                        update_stmt = update(schedules_table).where(
                            schedules_table.c.id == existing_schedule.id).values(**schedule_values)
                        conn.execute(update_stmt)
                        logger.info(f"Updated schedule for workflow '{wf_in.name}' with rule: {cron_config}")
                    else:
                        insert_stmt = insert(schedules_table).values(name=f"schedule_for_{wf_in.name}",
                                                                     workflow_name=wf_in.name, **schedule_values)
                        conn.execute(insert_stmt)
                        logger.info(f"Created new schedule for workflow '{wf_in.name}' with rule: {cron_config}")

                elif existing_schedule and existing_schedule.is_enabled:
                    # Disable schedule if trigger is removed from YAML
                    disable_stmt = update(schedules_table).where(schedules_table.c.id == existing_schedule.id).values(
                        is_enabled=False)
                    conn.execute(disable_stmt)
                    logger.info(f"Disabled schedule for '{wf_in.name}' (trigger removed from YAML).")

                conn.commit()

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to sync workflow file '{filepath.name}': {e}", exc_info=True)

        # Disable schedules for deleted workflow files
        stale_schedules = conn.execute(
            select(schedules_table).where(schedules_table.c.workflow_name.not_in(synced_workflow_names))).fetchall()
        for stale in stale_schedules:
            if stale.is_enabled:
                conn.execute(update(schedules_table).where(schedules_table.c.id == stale.id).values(is_enabled=False))
                logger.info(f"Disabled schedule for deleted workflow YAML: '{stale.workflow_name}'.")
        conn.commit()

    engine.dispose()
    logger.info("Workflow and schedule sync complete.")

@app.on_event("startup")
async def startup_event():
    global _watcher_process, _scheduler_process
    logger.info("Application startup: Initializing components.")
    if isinstance(manager, RedisConnectionManager):
        try:
            await manager.connect()
            logger.info(f"Successfully connected to Redis at {REDIS_URL}")
        except RedisConnectionError as e:
            logger.critical(
                f"CRITICAL: Could not connect to Redis at {REDIS_URL}. Please ensure it is running. Error: {e}")
            raise RuntimeError("Failed to connect to Redis") from e

    if is_db_configured():
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

            _sync_workflows_from_files()
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

    _watcher_process = Process(target=start_watcher, args=(_DATA_DIR,))
    _watcher_process.daemon = False  # <-- ADDED
    _watcher_process.start()
    logger.info("Automated file watcher started.")

    # --- START: New scheduler process (Task 3) ---
    schedule_config = {
        "workflows": [
            {"workflow_name": "hourly_report_workflow"},  # Placeholder for a real config
        ]
    }
    _scheduler_process = Process(target=start_scheduler_process, args=())
    _scheduler_process.daemon = False
    _scheduler_process.start()
    logger.info("Automated workflow scheduler started.")


@app.on_event("shutdown")
async def shutdown_event():
    global _watcher_process, _scheduler_process
    if isinstance(manager, RedisConnectionManager) and manager.connection:
        await manager.connection.close()
        logger.info("Redis connection closed.")

    if _watcher_process and _watcher_process.is_alive():
        _watcher_process.terminate()
        _watcher_process.join(timeout=5)
        logger.info("File watcher process stopped.")

    # --- START: Stop the scheduler process (Task 3) ---
    if _scheduler_process and _scheduler_process.is_alive():
        _scheduler_process.terminate()
        _scheduler_process.join(timeout=5)
        logger.info("Workflow scheduler process stopped.")
    # --- END: Stop the scheduler process (Task 3) ---



# --- INCLUDE THE API ROUTERS ---
app.include_router(agents_api_router)
app.include_router(audit_api_router)
app.include_router(query_api_router)
app.include_router(evaluation_api_router)
app.include_router(metrics_api_router)

app.include_router(workflows.router, prefix="/api/v1")
app.include_router(schedules_api_router, prefix="/api/v1")


class RenameRequest(BaseModel):
    new_name: str


def is_db_configured() -> bool:
    mem_config = get_memory_storage_config()
    log_config = get_log_storage_config()
    return (mem_config.get("type") in ["db", "sqlite"]) or (log_config.get("type") == "db")


async def _get_or_create_user(db: AsyncSession, user_id: str) -> int:
    stmt = select(users_table.c.id).where(users_table.c.user_id == user_id)
    user_db_id = (await db.execute(stmt)).scalar_one_or_none()
    if not user_db_id:
        stmt = insert(users_table).values(user_id=user_id).returning(users_table.c.id)
        user_db_id = (await db.execute(stmt)).scalar_one()
        await db.commit()
    return user_db_id


async def _get_or_create_session(db: AsyncSession, thread_id: str, agent_name: str, user_db_id: int) -> Tuple[
    int, Optional[str]]:
    stmt = select(chat_sessions_table.c.id, chat_sessions_table.c.topic_name).where(
        (chat_sessions_table.c.thread_id == thread_id) &
        (chat_sessions_table.c.agent_name == agent_name) &
        (chat_sessions_table.c.user_id == user_db_id)
    )
    session_record = (await db.execute(stmt)).fetchone()
    if session_record:
        return session_record.id, session_record.topic_name
    stmt = insert(chat_sessions_table).values(
        thread_id=thread_id,
        agent_name=agent_name,
        user_id=user_db_id
    ).returning(chat_sessions_table.c.id)
    session_id = (await db.execute(stmt)).scalar_one()
    await db.commit()
    return session_id, None


async def _save_message_and_update_session(db: AsyncSession, session_id: int, sender: str, content: str):
    now = datetime.utcnow()
    msg_stmt = insert(chat_messages_table).values(session_id=session_id, sender=sender, content=content, timestamp=now)
    session_stmt = update(chat_sessions_table).where(chat_sessions_table.c.id == session_id).values(updated_at=now)
    await db.execute(msg_stmt)
    await db.execute(session_stmt)
    await db.commit()


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket, api_key: str = Depends(get_websocket_api_key),
                         db: AsyncSession = Depends(get_db)):
    await ws.accept()
    thread_id = "uninitialized"
    pubsub_task = None
    channel = ""

    try:
        message_data = await ws.receive_json()
        if not (message_data.get("type") == "query" and "payload" in message_data):
            await ws.close(code=1003, reason="Protocol violation")
            return

        payload = message_data["payload"]
        agent_name = payload.get("agent", "unknown_agent")
        user_id = sanitize_for_path(payload.get("user_id")) or f"user-{uuid4().hex[:8]}"
        thread_id = sanitize_for_path(payload.get("thread_id")) or f"thread-{uuid4().hex[:8]}"
        channel = f"chat:{thread_id}"

        await manager.connect(ws, channel)

        if isinstance(manager, RedisConnectionManager):
            pubsub_task = asyncio.create_task(redis_listen(ws, channel))

        user_db_id = await _get_or_create_user(db, user_id)
        session_id, session_topic = await _get_or_create_session(db, thread_id, agent_name, user_db_id)
        is_first_message_in_session = session_topic is None

        agent_config = load_agent_config(agent_name)
        all_tools = []
        if "retriever" in agent_config.tools: all_tools.append(get_retriever_tool(agent_config))
        if "arxiv" in agent_config.tools: all_tools.extend(get_arxiv_tool())
        if "search_engine" in agent_config.tools: all_tools.append(SearchTool(agent_config=agent_config))
        if "sql_toolkit" in agent_config.tools:
            db_source = next((s for s in agent_config.sources if s.type == 'db'), None)
            if db_source and db_source.db_connection:
                all_tools.extend(create_sql_toolkit(db_connection_string=db_source.db_connection,
                                                    llm_model_name=agent_config.llm_model))

        langgraph_agent = get_agent_workflow(all_tools).compile()

        while True:
            query = message_data.get("payload", {}).get("query")
            request_id = str(uuid4())
            logger.info(f"[{thread_id}] Processing request {request_id} for query: '{query[:50]}...'")

            await _save_message_and_update_session(db, session_id, 'human', query)

            if is_first_message_in_session:
                title = query.strip()[:100]
                await db.execute(
                    update(chat_sessions_table).where(chat_sessions_table.c.id == session_id).values(topic_name=title))
                await db.commit()
                is_first_message_in_session = False

            history_stmt = select(chat_messages_table.c.sender, chat_messages_table.c.content).where(
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

            initial_state: AgentState = {"messages": history, "request_id": request_id, "agent_config": agent_config}

            final_state, ai_response_content = await handle_query_streaming(
                initial_state,
                {"configurable": {"thread_id": thread_id}},
                langgraph_agent,
                thread_id,
                run_db_id,
                db
            )

            done_message = {
                "done": True, "error": final_state.get("error", False) if final_state else True,
                "errorMessage": final_state.get("errorMessage") if final_state else "Agent returned no output.",
                "request_id": request_id, "user_id": user_id, "thread_id": thread_id,
            }
            await manager.broadcast(channel, json.dumps(done_message))

            if ai_response_content and not (final_state and final_state.get("error")):
                await _save_message_and_update_session(db, session_id, 'ai', ai_response_content)

            run_end_time = datetime.utcnow()
            update_run_stmt = update(agent_runs).where(agent_runs.c.id == run_db_id).values(
                end_time=run_end_time,
                status='completed' if not (final_state and final_state.get("error")) else 'failed',
                final_state=_serialize_for_db(final_state)
            )
            await db.execute(update_run_stmt)
            await db.commit()

            message_data = await ws.receive_json()

    except WebSocketDisconnect:
        logger.info(f"[{thread_id}] Client disconnected.")
    except Exception as e:
        logger.error(f"[{thread_id}] Unhandled WebSocket Error: {e}", exc_info=True)
    finally:
        if pubsub_task and not pubsub_task.done():
            pubsub_task.cancel()
        manager.disconnect(ws, channel)


async def redis_listen(ws: WebSocket, channel: str):
    if not isinstance(manager, RedisConnectionManager):
        return

    while True:
        # ensure we have a live pubsub
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
                return await redis_listen(ws, channel)  # restart subscription loop
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
        return _serialize_for_db(final_state), accumulated_content
    except asyncio.CancelledError:
        logger.info(f"[{thread_id}] Generation task cancelled.")
        return {}, accumulated_content
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
    server_api_keys = get_server_api_keys()
    frontend_api_key = server_api_keys[0] if server_api_keys else ""
    return templates.TemplateResponse("agent_interface.html", {
        "request": request, "agents": agents_list, "agent": default_agent, "api_key": frontend_api_key
    })


@app.get("/health", tags=["System"])
async def health_check():
    if not is_db_configured():
        return JSONResponse({"status": "ok", "db_check": "skipped (no database configured)"})
    db_config = get_db_connection_config()
    if not db_config:
        raise HTTPException(status_code=500, detail="DB configured but connection details not found.")
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
async def get_history(thread_id: str, agent_name: str, user_id: str, api_key: str = Depends(get_http_api_key),
                      db: AsyncSession = Depends(get_db)):
    if not is_db_configured():
        raise HTTPException(status_code=501, detail="Chat history not supported without a database.")
    safe_agent_name = sanitize_for_path(agent_name)
    safe_user_id = sanitize_for_path(user_id)
    safe_thread_id = sanitize_for_path(thread_id)
    try:
        user_db_id = (await db.execute(
            select(users_table.c.id).where(users_table.c.user_id == safe_user_id))).scalar_one_or_none()
        if not user_db_id:
            return JSONResponse(content=[])
        session_id = (await db.execute(select(chat_sessions_table.c.id).where(
            (chat_sessions_table.c.thread_id == safe_thread_id) &
            (chat_sessions_table.c.agent_name == safe_agent_name) &
            (chat_sessions_table.c.user_id == user_db_id)
        ))).scalar_one_or_none()
        if not session_id:
            return JSONResponse(content=[])
        messages_stmt = select(chat_messages_table.c.sender, chat_messages_table.c.content).where(
            chat_messages_table.c.session_id == session_id).order_by(chat_messages_table.c.timestamp.asc())
        messages_result = (await db.execute(messages_stmt)).fetchall()
        history = [{"type": row.sender, "content": row.content} for row in messages_result]
        return JSONResponse(content=history)
    except Exception as e:
        logger.error(f"Error loading chat history for thread {safe_thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load chat history.")


@app.get("/sessions", tags=["Memory"])
async def list_sessions(agent_name: str, user_id: str, api_key: str = Depends(get_http_api_key),
                        db: AsyncSession = Depends(get_db)):
    if not is_db_configured():
        raise HTTPException(status_code=501, detail="Chat history not supported without a database.")
    safe_agent_name = sanitize_for_path(agent_name)
    safe_user_id = sanitize_for_path(user_id)
    try:
        user_db_id = (await db.execute(
            select(users_table.c.id).where(users_table.c.user_id == safe_user_id))).scalar_one_or_none()
        if not user_db_id:
            return JSONResponse(content=[])
        sessions_stmt = select(chat_sessions_table.c.thread_id, chat_sessions_table.c.topic_name).where(
            (chat_sessions_table.c.agent_name == safe_agent_name) &
            (chat_sessions_table.c.user_id == user_db_id)
        ).order_by(desc(chat_sessions_table.c.updated_at))
        sessions_result = (await db.execute(sessions_stmt)).fetchall()
        sessions = [{"thread_id": row.thread_id, "topic_name": row.topic_name or "New Chat"} for row in sessions_result]
        return JSONResponse(content=sessions)
    except Exception as e:
        logger.error(f"Error loading chat sessions for agent {safe_agent_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load chat sessions.")


@app.put("/sessions/{thread_id}/rename", tags=["Memory"], status_code=status.HTTP_200_OK)
async def rename_session(thread_id: str, request: RenameRequest, agent_name: str, user_id: str,
                         api_key: str = Depends(get_http_api_key), db: AsyncSession = Depends(get_db)):
    if not is_db_configured():
        raise HTTPException(status_code=501, detail="Functionality not supported without a database.")
    safe_thread_id = sanitize_for_path(thread_id)
    safe_agent_name = sanitize_for_path(agent_name)
    safe_user_id = sanitize_for_path(user_id)
    new_name = request.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New name cannot be empty.")
    try:
        user_db_id = (await db.execute(
            select(users_table.c.id).where(users_table.c.user_id == safe_user_id))).scalar_one_or_none()
        if not user_db_id:
            raise HTTPException(status_code=404, detail="User not found.")
        update_stmt = update(chat_sessions_table).where(
            (chat_sessions_table.c.thread_id == safe_thread_id) &
            (chat_sessions_table.c.agent_name == safe_agent_name) &
            (chat_sessions_table.c.user_id == user_db_id)
        ).values(topic_name=new_name, updated_at=datetime.utcnow())
        result = await db.execute(update_stmt)
        await db.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Chat session not found or permission denied.")
        return JSONResponse({"status": "ok", "message": "Chat session renamed successfully."})
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Database error renaming session {safe_thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error while renaming session.")


@app.delete("/sessions/{thread_id}", tags=["Memory"], status_code=status.HTTP_200_OK)
async def delete_session(thread_id: str, agent_name: str, user_id: str, api_key: str = Depends(get_http_api_key),
                         db: AsyncSession = Depends(get_db)):
    if not is_db_configured():
        raise HTTPException(status_code=501, detail="Functionality not supported without a database.")
    safe_thread_id = sanitize_for_path(thread_id)
    safe_agent_name = sanitize_for_path(agent_name)
    safe_user_id = sanitize_for_path(user_id)
    try:
        user_db_id = (await db.execute(
            select(users_table.c.id).where(users_table.c.user_id == safe_user_id))).scalar_one_or_none()
        if not user_db_id:
            raise HTTPException(status_code=404, detail="User not found.")
        session_id = (await db.execute(select(chat_sessions_table.c.id).where(
            (chat_sessions_table.c.thread_id == safe_thread_id) &
            (chat_sessions_table.c.agent_name == safe_agent_name) &
            (chat_sessions_table.c.user_id == user_db_id)
        ))).scalar_one_or_none()
        if not session_id:
            raise HTTPException(status_code=404, detail="Chat session not found or permission denied.")
        await db.execute(delete(chat_messages_table).where(chat_messages_table.c.session_id == session_id))
        await db.execute(delete(chat_sessions_table).where(chat_sessions_table.c.id == session_id))
        await db.commit()
        return JSONResponse({"status": "ok", "message": "Chat session deleted successfully."})
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Database error deleting session {safe_thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error while deleting session.")