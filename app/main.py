# app/main.py
import os
import logging
import json
import asyncio
from uuid import uuid4
from typing import Optional, List, Dict, Any, Tuple
from multiprocessing import Process
import configparser
from urllib.parse import urlparse, urlunparse
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, status
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
from datetime import datetime
from pydantic import BaseModel

# Import database-related modules
from app.db.models import chat_sessions_table, chat_messages_table, users_table, metadata
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

# Corrected Imports: Importing the functions and variables from app.db and app.core.config
from app.core.config import get_path_settings, get_server_api_keys, get_log_storage_config, \
    get_memory_storage_config, get_db_connection_config, get_db_connection, get_cors_settings
from app.db import initialize_db_connections, get_db

from app.core.validation import validate_agent_name, sanitize_for_path
from app.core.security import get_http_api_key, get_websocket_api_key
from app.schemas.agent import AgentConfig
from app.agents.config_manager import save_agent_config, load_agent_config, get_agent_configs
from app.pipelines.embed import embed_agent_data
from app.agents.agent_graph import get_agent_workflow, AgentState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.tools.sql_tool import create_sql_toolkit
from app.tools.retriever_tool import get_retriever_tool
from app.tools.arxiv_tool import get_arxiv_tool
from app.tools.search_engine_tool import SearchTool

from app.watcher import start_watcher
from sqlalchemy import create_engine as create_sync_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Imports for Alembic integration
import alembic.config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
TURN_TIMEOUT = 300.0

load_dotenv()

_APP_PATHS = get_path_settings()
_PROJECT_ROOT = _APP_PATHS["PROJECT_ROOT"]
_DATA_DIR = _APP_PATHS["DATA_DIR"]
_AGENTS_DIR = _APP_PATHS["AGENTS_DIR"]
_MEMORY_DIR = _APP_PATHS["MEMORY_DIR"]
_CONFIG_FILE = _APP_PATHS["CONFIG_FILE_PATH"]
_BENCHMARK_DIR = _APP_PATHS["BENCHMARK_DIR"]

allowed_origins = get_cors_settings()

app = FastAPI(title="RAGnetic API", version="0.1.0",
              description="API for managing and interacting with RAGnetic agents.")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class RenameRequest(BaseModel):
    new_name: str


# --- Connection Manager for WebSockets ---
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def send(self, msg: Dict, ws: WebSocket):
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json(msg)


manager = ConnectionManager()
_watcher_process: Optional[Process] = None


def is_db_configured() -> bool:
    mem_config = get_memory_storage_config()
    log_config = get_log_storage_config()
    return (mem_config.get("type") in ["db", "sqlite"]) or (log_config.get("type") == "db")


@app.on_event("startup")
async def startup_event():
    global _watcher_process
    logger.info("Application startup: Initializing RAGnetic components.")
    if is_db_configured():
        db_config = get_db_connection_config()
        if not db_config:
            raise RuntimeError("Database is configured but connection details could not be loaded.")
        try:
            logger.info("Verifying database schema version...")
            # --- Build a purely synchronous URL for the migration check ---
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
        except Exception as e:
            logger.critical(f"CRITICAL: Failed during database startup. Error: {e}", exc_info=True)
            raise
    else:
        logger.warning("No database configured. DB-dependent features will be disabled.")
    if not os.path.exists(_DATA_DIR):
        logger.error(f"'{_DATA_DIR}' not found. Please run 'ragnetic init'.")
        return
    _watcher_process = Process(target=start_watcher, args=(_DATA_DIR,), daemon=True)
    _watcher_process.start()
    logger.info("Automated file watcher started.")


@app.on_event("shutdown")
async def shutdown_event():
    global _watcher_process
    if _watcher_process and _watcher_process.is_alive():
        _watcher_process.terminate()
        _watcher_process.join(timeout=5)
        logger.info("File watcher process stopped.")


# --- Endpoints ---
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
async def get_history(
        thread_id: str,
        agent_name: str,
        user_id: str,
        api_key: str = Depends(get_http_api_key),
        db: AsyncSession = Depends(get_db)
):
    if not is_db_configured():
        raise HTTPException(status_code=501, detail="Chat history not supported without a database.")
    safe_agent_name = sanitize_for_path(agent_name)
    safe_user_id = sanitize_for_path(user_id)
    safe_thread_id = sanitize_for_path(thread_id)
    try:
        select_user_stmt = text(f"SELECT id FROM {users_table.name} WHERE user_id = :user_id")
        user_db_id = (await db.execute(select_user_stmt, {"user_id": safe_user_id})).scalar_one_or_none()
        if not user_db_id:
            return JSONResponse(content=[])
        select_session_stmt = text(
            f"SELECT id FROM {chat_sessions_table.name} WHERE thread_id = :thread_id AND agent_name = :agent_name AND user_id = :user_id"
        )
        session_id_result = (await db.execute(select_session_stmt, {
            "thread_id": safe_thread_id, "agent_name": safe_agent_name, "user_id": user_db_id
        })).scalar_one_or_none()
        if not session_id_result:
            return JSONResponse(content=[])
        select_messages_stmt = text(
            f"SELECT sender, content FROM {chat_messages_table.name} WHERE session_id = :session_id ORDER BY timestamp ASC"
        )
        messages_result = (await db.execute(select_messages_stmt, {"session_id": session_id_result})).fetchall()
        history = [{"type": row.sender, "content": row.content} for row in messages_result]
        return JSONResponse(content=history)
    except Exception as e:
        logger.error(f"Error loading chat history for thread {safe_thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load chat history.")


@app.get("/sessions", tags=["Memory"])
async def list_sessions(
        agent_name: str,
        user_id: str,
        api_key: str = Depends(get_http_api_key),
        db: AsyncSession = Depends(get_db),
):
    if not is_db_configured():
        raise HTTPException(status_code=501, detail="Chat history not supported without a database.")

    safe_agent_name = sanitize_for_path(agent_name)
    safe_user_id = sanitize_for_path(user_id)

    try:
        select_user_stmt = select(users_table.c.id).where(users_table.c.user_id == safe_user_id)
        user_db_id = (await db.execute(select_user_stmt)).scalar_one_or_none()

        if not user_db_id:
            return JSONResponse(content=[])

        select_sessions_stmt = (
            select(chat_sessions_table.c.thread_id, chat_sessions_table.c.topic_name, chat_sessions_table.c.updated_at)
            .where(
                (chat_sessions_table.c.agent_name == safe_agent_name) &
                (chat_sessions_table.c.user_id == user_db_id)
            )
            .order_by(desc(chat_sessions_table.c.updated_at))
        )

        sessions_result = (await db.execute(select_sessions_stmt)).fetchall()
        sessions = [{"thread_id": row.thread_id, "topic_name": row.topic_name or "New Chat"} for row in sessions_result]
        return JSONResponse(content=sessions)

    except Exception as e:
        logger.error(f"Error loading chat sessions for agent {safe_agent_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load chat sessions.")


@app.post("/create-agent", tags=["Agents"])
async def create_agent(config: AgentConfig, bg: BackgroundTasks, api_key: str = Depends(get_http_api_key)):
    try:
        save_agent_config(config)
        bg.add_task(embed_agent_data, config)
        return JSONResponse(content={"status": "Agent config saved; embedding started.", "agent": config.name})
    except Exception as e:
        logger.error(f"Error creating agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def handle_query_streaming(initial_state: AgentState, cfg: dict, langgraph_agent: Any, ws: WebSocket,
                                 thread_id: str) -> Tuple[Optional[Dict], str]:
    final_state = None
    accumulated_content = ""
    try:
        async for event in langgraph_agent.astream_events(initial_state, cfg, version="v2"):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    accumulated_content += token
                    await manager.send({"token": token}, ws)
            elif event["event"] == "on_graph_end":
                final_state = event['data']['output']
        return final_state, accumulated_content
    except asyncio.CancelledError:
        logger.info(f"[{thread_id}] Generation task cancelled.")
        return None, accumulated_content
    except Exception as e:
        error_message = f"An error occurred during generation: {e}"
        logger.error(f"[{thread_id}] {error_message}", exc_info=True)
        await manager.send({"token": f"\n\n{error_message}"}, ws)
        return {"error": True, "errorMessage": error_message}, ""


@app.websocket("/ws")
async def websocket_chat(
        ws: WebSocket,
        api_key: str = Depends(get_websocket_api_key),
        db_session: AsyncSession = Depends(get_db)
):
    await manager.connect(ws)
    thread_id = "uninitialized"
    is_db_enabled = is_db_configured()
    try:
        try:
            message_data = await ws.receive_json()
        except ValueError:
            logger.warning(f"[{thread_id}] Received invalid JSON. Closing connection.")
            await ws.close(code=1003, reason="Invalid JSON received.")
            return
        if not (isinstance(message_data, dict) and message_data.get("type") == "query" and "payload" in message_data):
            await ws.close(code=1003, reason="Protocol violation: First message must be a valid query.")
            return
        payload = message_data.get("payload", {})
        agent_name = payload.get("agent", "unknown_agent")
        user_id = sanitize_for_path(payload.get("user_id")) or f"user-{uuid4().hex[:8]}"
        thread_id = sanitize_for_path(payload.get("thread_id")) or f"thread-{uuid4().hex[:8]}"
        session_id = None
        session_topic = None

        if is_db_enabled and db_session:
            select_user_stmt = text(f"SELECT id FROM {users_table.name} WHERE user_id = :user_id")
            user_result = await db_session.execute(select_user_stmt, {"user_id": user_id})
            user_db_id = user_result.scalar_one_or_none()
            if not user_db_id:
                insert_user_stmt = text(
                    f"INSERT INTO {users_table.name} (user_id, created_at, updated_at) VALUES (:user_id, :created_at, :updated_at) RETURNING id"
                )
                now = datetime.utcnow()
                user_result = await db_session.execute(
                    insert_user_stmt,
                    {"user_id": user_id, "created_at": now, "updated_at": now}
                )
                user_db_id = user_result.scalar_one()
                await db_session.commit()

            select_session_stmt = text(
                f"SELECT id, topic_name FROM {chat_sessions_table.name} WHERE thread_id = :thread_id AND agent_name = :agent_name AND user_id = :user_id"
            )
            session_result = await db_session.execute(select_session_stmt, {
                "thread_id": thread_id, "agent_name": agent_name, "user_id": user_db_id
            })
            session_record = session_result.fetchone()
            if session_record:
                session_id = session_record.id
                session_topic = session_record.topic_name

            if not session_id:
                insert_session_stmt = text(
                    f"INSERT INTO {chat_sessions_table.name} (thread_id, agent_name, user_id, created_at, updated_at) VALUES (:thread_id, :agent_name, :user_id, :created_at, :updated_at) RETURNING id"
                )
                now = datetime.utcnow()
                session_result = await db_session.execute(
                    insert_session_stmt,
                    {
                        "thread_id": thread_id,
                        "agent_name": agent_name,
                        "user_id": user_db_id,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                session_id = session_result.scalar_one()
                await db_session.commit()

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
        is_first_message_in_session = (session_topic is None)

        while True:
            query = message_data.get("payload", {}).get("query")
            request_id = str(uuid4())
            logger.info(f"[{thread_id}] Processing request {request_id} for query: '{query[:50]}...'")
            history: List[BaseMessage] = []
            if is_db_enabled and db_session and session_id:
                messages_result = await db_session.execute(
                    text(
                        f"SELECT sender, content FROM {chat_messages_table.name} WHERE session_id = :session_id ORDER BY timestamp ASC"),
                    {"session_id": session_id}
                )
                for msg in messages_result:
                    history.append(
                        HumanMessage(content=msg.content) if msg.sender == 'human' else AIMessage(content=msg.content))

                now = datetime.utcnow()
                await db_session.execute(
                    text(
                        f"INSERT INTO {chat_messages_table.name} (session_id, sender, content, timestamp) VALUES (:session_id, 'human', :content, :timestamp)"),
                    {"session_id": session_id, "content": query, "timestamp": now}
                )
                await db_session.execute(
                    text(f"UPDATE {chat_sessions_table.name} SET updated_at = :now WHERE id = :session_id"),
                    {"now": now, "session_id": session_id}
                )
                await db_session.commit()

                if is_first_message_in_session:
                    title = query.strip()
                    if len(title) > 100:
                        title = title[:97] + "..."
                    await db_session.execute(
                        text(
                            f"UPDATE {chat_sessions_table.name} "
                            "SET topic_name = :topic WHERE id = :sid"
                        ),
                        {"topic": title, "sid": session_id}
                    )
                    await db_session.commit()
                    is_first_message_in_session = False

            initial_state: AgentState = {"messages": history + [HumanMessage(content=query)], "request_id": request_id}
            gen_task = asyncio.create_task(
                handle_query_streaming(
                    initial_state,
                    {"configurable": {"agent_config": agent_config}},
                    langgraph_agent,
                    ws,
                    thread_id
                )
            )
            try:
                final_state, ai_response_content = await gen_task
            except asyncio.CancelledError:
                logger.info(f"[{thread_id}] Generation task cancelled.")
                final_state, ai_response_content = {}, ""
            except Exception as e:
                logger.error(f"[{thread_id}] Generation error: {e}", exc_info=True)
                final_state, ai_response_content = {"error": True, "errorMessage": str(e)}, ""
            await manager.send(
                {
                    "done": True,
                    "error": final_state.get("error", False) if final_state else True,
                    "errorMessage": final_state.get("errorMessage") if final_state else "Agent returned no output.",
                    "request_id": request_id,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "is_first_message": is_first_message_in_session,
                },
                ws
            )

            if ai_response_content and not (
                    final_state and final_state.get("error")) and is_db_enabled and db_session and session_id:
                now = datetime.utcnow()
                await db_session.execute(
                    text(
                        "INSERT INTO chat_messages (session_id, sender, content, timestamp) VALUES (:session_id, 'ai', :content, :timestamp)"
                    ),
                    {"session_id": session_id, "content": ai_response_content, "timestamp": now}
                )
                await db_session.execute(
                    text(f"UPDATE {chat_sessions_table.name} SET updated_at = :now WHERE id = :session_id"),
                    {"now": now, "session_id": session_id}
                )
                await db_session.commit()
            try:
                message_data = await ws.receive_json()
            except WebSocketDisconnect:
                logger.info(f"[{thread_id}] Client disconnected.")
                break
    except WebSocketDisconnect:
        logger.info(f"[{thread_id}] Client disconnected.")
    except Exception as e:
        logger.error(f"[{thread_id}] Unhandled WebSocket Error: {e}", exc_info=True)
    finally:
        pass


@app.put("/sessions/{thread_id}/rename", tags=["Memory"], status_code=status.HTTP_200_OK)
async def rename_session(
        thread_id: str,
        request: RenameRequest,
        agent_name: str,
        user_id: str,
        api_key: str = Depends(get_http_api_key),
        db: AsyncSession = Depends(get_db),
):
    if not is_db_configured():
        raise HTTPException(status_code=501, detail="Functionality not supported without a database.")

    safe_thread_id = sanitize_for_path(thread_id)
    safe_agent_name = sanitize_for_path(agent_name)
    safe_user_id = sanitize_for_path(user_id)
    new_name = request.new_name.strip()

    if not new_name:
        raise HTTPException(status_code=400, detail="New name cannot be empty.")

    try:
        # First, get the internal user ID
        user_db_id = (await db.execute(
            select(users_table.c.id).where(users_table.c.user_id == safe_user_id)
        )).scalar_one_or_none()

        if not user_db_id:
            raise HTTPException(status_code=404, detail="User not found.")

        # Find the session and verify ownership
        update_stmt = (
            chat_sessions_table.update()
            .where(
                (chat_sessions_table.c.thread_id == safe_thread_id) &
                (chat_sessions_table.c.agent_name == safe_agent_name) &
                (chat_sessions_table.c.user_id == user_db_id)
            )
            .values(topic_name=new_name, updated_at=datetime.utcnow())
        )
        result = await db.execute(update_stmt)
        await db.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Chat session not found or permission denied.")

        return JSONResponse({"status": "ok", "message": "Chat session renamed successfully."})

    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Database error renaming session {safe_thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error while renaming session.")


# --- NEW: Endpoint to delete a chat session ---
@app.delete("/sessions/{thread_id}", tags=["Memory"], status_code=status.HTTP_200_OK)
async def delete_session(
        thread_id: str,
        agent_name: str,
        user_id: str,
        api_key: str = Depends(get_http_api_key),
        db: AsyncSession = Depends(get_db),
):
    if not is_db_configured():
        raise HTTPException(status_code=501, detail="Functionality not supported without a database.")

    safe_thread_id = sanitize_for_path(thread_id)
    safe_agent_name = sanitize_for_path(agent_name)
    safe_user_id = sanitize_for_path(user_id)

    try:
        # Get internal user and session IDs to ensure ownership before deleting
        user_db_id = (await db.execute(
            select(users_table.c.id).where(users_table.c.user_id == safe_user_id)
        )).scalar_one_or_none()

        if not user_db_id:
            raise HTTPException(status_code=404, detail="User not found.")

        session_id = (await db.execute(
            select(chat_sessions_table.c.id).where(
                (chat_sessions_table.c.thread_id == safe_thread_id) &
                (chat_sessions_table.c.agent_name == safe_agent_name) &
                (chat_sessions_table.c.user_id == user_db_id)
            )
        )).scalar_one_or_none()

        if not session_id:
            raise HTTPException(status_code=404, detail="Chat session not found or permission denied.")

        # Delete associated messages first
        await db.execute(
            chat_messages_table.delete().where(chat_messages_table.c.session_id == session_id)
        )
        # Then delete the session
        await db.execute(
            chat_sessions_table.delete().where(chat_sessions_table.c.id == session_id)
        )
        await db.commit()

        return JSONResponse({"status": "ok", "message": "Chat session deleted successfully."})

    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Database error deleting session {safe_thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error while deleting session.")