import os
import logging
import json
import asyncio
from uuid import uuid4
from typing import Optional, List, Dict, Any, Tuple
from multiprocessing import Process
import configparser
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, status
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# Import database-related modules
from app.db.models import chat_sessions_table, chat_messages_table, users_table, memory_entries_table, \
    ragnetic_logs_table, metadata
from app.db import initialize_db_connections, get_db
from sqlalchemy.ext.asyncio import AsyncSession  # For type hinting in Depends
from app.core.structured_logging import DatabaseLogHandler

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
from app.core.config import get_path_settings, get_server_api_keys, get_log_storage_config, \
    get_memory_storage_config, get_db_connection

from sqlalchemy import create_engine, text
from sqlalchemy.sql import func
from sqlalchemy.exc import SQLAlchemyError

# Imports for Alembic integration
import alembic.config
import alembic.command
from pathlib import Path

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
allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="RAGnetic API", version="0.1.0",
              description="API for managing and interacting with RAGnetic agents.")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


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
        else:
            logger.warning("Attempted to send message on a closed or disconnected websocket.")


manager = ConnectionManager()
_watcher_process: Optional[Process] = None


def ensure_db_configured():
    """Helper function to centralize DB memory configuration check."""
    mem_config = get_memory_storage_config()
    log_config = get_log_storage_config()

    if mem_config.get("type") == "db" or mem_config.get("type") == "sqlite":
        return
    if log_config.get("type") == "db":
        return

    raise HTTPException(status_code=501,
                        detail="Chat history is not supported. Please configure memory storage via a database.")


@app.on_event("startup")
async def startup_event():
    """Initializes the app, starts the file watcher, and sets up DB connections."""
    global _watcher_process

    logger.info("Application startup: Initializing RAGnetic components.")

    log_config = get_log_storage_config()
    memory_config = get_memory_storage_config()
    db_conn_name = None

    if memory_config.get("type") == "db":
        db_conn_name = memory_config.get("connection_name")
    if log_config.get("type") == "db" and not db_conn_name:
        db_conn_name = log_config.get("connection_name")

    logger.info(f"Determined DB connection name: {db_conn_name}")

    if db_conn_name:
        try:
            # --- CORRECTED STARTUP ORDER ---
            # 1. Run migrations FIRST to ensure all tables exist.
            logger.info("Running database migrations on startup...")
            try:
                alembic_cfg = alembic.config.Config(str(_PROJECT_ROOT / "alembic.ini"))
                alembic.command.upgrade(alembic_cfg, "head")
                logger.info("Database migrations applied successfully.")
            except Exception as alembic_e:
                logger.critical(f"CRITICAL: RAGnetic migrations failed during startup. Error: {alembic_e}",
                                exc_info=True)
                # Re-raise to prevent the app from starting with an inconsistent DB
                raise alembic_e

            # 2. Initialize database connections.
            initialize_db_connections(db_conn_name)
            logger.info("Database connections initialized.")

            # 3. Attach DatabaseLogHandler AFTER tables are created and connections are up.
            if log_config.get("type") == "db":
                logger.warning("Database logging is configured but has been disabled for stability.")

        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize database connections or schema. Error: {e}", exc_info=True)
            raise

    if not db_conn_name:
        logger.warning(
            "No explicit database connection name determined. Chat history will rely on SQLite fallback if configured.")

    if not os.path.exists(_DATA_DIR):
        logger.error(f"Error: The '{_DATA_DIR}' directory does not exist. Please run 'ragnetic init' first.")
        return
    _watcher_process = Process(target=start_watcher, args=(_DATA_DIR,), daemon=True)
    _watcher_process.start()
    logger.info("Automated file watcher started in the background.")


@app.on_event("shutdown")
async def shutdown_event():
    global _watcher_process
    logger.info("Application shutdown: Terminating RAGnetic components.")
    if _watcher_process and _watcher_process.is_alive():
        _watcher_process.terminate()
        _watcher_process.join(timeout=5)
        logger.info("File watcher process stopped.")


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


@app.get("/healthz", tags=["System"])
async def health_check():
    """A simple health check endpoint to test database connectivity."""
    mem_config = get_memory_storage_config()
    log_config = get_log_storage_config()

    if not (mem_config.get("type") == "db" or mem_config.get("type") == "sqlite" or log_config.get("type") == "db"):
        return JSONResponse({"status": "ok", "db_check": "skipped (no database configured for memory or logs)"})

    db_conn_name = None
    if mem_config.get("type") == "db":
        db_conn_name = mem_config.get("connection_name")
    elif log_config.get("type") == "db":
        db_conn_name = log_config.get("connection_name")

    if mem_config.get("type") == "sqlite":
        _APP_PATHS = get_path_settings()
        db_url = f"sqlite:///{_APP_PATHS['MEMORY_DIR'] / 'ragnetic.db'}"
    elif db_conn_name:
        db_url = get_db_connection(db_conn_name)
    else:
        return JSONResponse({"status": "ok", "db_check": "skipped (no valid database URL determined)"})

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return JSONResponse({"status": "ok", "db_check": "connected"})
    except Exception as e:
        logger.error(f"Health check failed to connect to database: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database connection failed.")


@app.get("/history/{thread_id}", tags=["Memory"])
async def get_history(
        thread_id: str,
        agent_name: str,
        user_id: str,
        api_key: str = Depends(get_http_api_key),
        db: AsyncSession = Depends(get_db)
):
    ensure_db_configured()

    safe_agent_name = sanitize_for_path(agent_name)
    safe_user_id = sanitize_for_path(user_id)
    safe_thread_id = sanitize_for_path(thread_id)
    if not all([safe_agent_name, safe_user_id, safe_thread_id]):
        raise HTTPException(status_code=400, detail="Invalid characters in request.")

    try:
        select_user_stmt = text(f"SELECT id FROM {users_table.name} WHERE user_id = :user_id")
        user_db_id = (await db.execute(select_user_stmt, {"user_id": safe_user_id})).scalar_one_or_none()

        if not user_db_id:
            return JSONResponse(content=[])

        select_session_stmt = text(
            f"SELECT id FROM {chat_sessions_table.name} WHERE thread_id = :thread_id AND agent_name = :agent_name AND user_id = :user_id"
        )
        session_id_result = (await db.execute(select_session_stmt, {
            "thread_id": safe_thread_id,
            "agent_name": safe_agent_name,
            "user_id": user_db_id
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
    """
    Handles a streaming query to the agent, sending tokens as they are generated.
    Returns the final state and the full accumulated message content.
    """
    final_state = None
    accumulated_content = ""
    try:
        # Use .astream_events() for a streaming call.
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
        logger.info(f"[{thread_id}] Generation task was cancelled.")
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
        db: AsyncSession = Depends(get_db)
):
    await manager.connect(ws)
    thread_id: str = "uninitialized"
    session_id: Optional[int] = None
    is_db_session: bool = False

    try:
        message_data = await ws.receive_json()
        if not (isinstance(message_data, dict) and message_data.get("type") == "query" and "payload" in message_data):
            await ws.close(code=1003, reason="Protocol violation: First message must be valid.")
            return

        session_init_payload = message_data["payload"]
        agent_name = session_init_payload.get("agent", "unknown_agent")
        user_id = sanitize_for_path(session_init_payload.get("user_id")) or f"user-{uuid4().hex[:8]}"
        thread_id = sanitize_for_path(session_init_payload.get("thread_id")) or f"thread-{uuid4().hex[:8]}"

        try:
            ensure_db_configured()
            is_db_session = True
        except HTTPException as e:
            logger.warning(f"Skipping DB-backed chat session. Reason: {e.detail}")
            is_db_session = False

        if is_db_session:
            # Database session and user management...
            select_user_stmt = text(f"SELECT id FROM {users_table.name} WHERE user_id = :user_id")
            user_db_id = (await db.execute(select_user_stmt, {"user_id": user_id})).scalar_one_or_none()
            if not user_db_id:
                insert_user_stmt = text(f"INSERT INTO {users_table.name} (user_id) VALUES (:user_id) RETURNING id")
                user_db_id = (await db.execute(insert_user_stmt, {"user_id": user_id})).scalar_one()
                await db.commit()

            select_session_stmt = text(
                f"SELECT id FROM {chat_sessions_table.name} WHERE thread_id = :thread_id AND agent_name = :agent_name AND user_id = :user_id"
            )
            session_id = (await db.execute(select_session_stmt, {
                "thread_id": thread_id, "agent_name": agent_name, "user_id": user_db_id
            })).scalar_one_or_none()

            if not session_id:
                insert_session_stmt = text(
                    f"INSERT INTO {chat_sessions_table.name} (thread_id, agent_name, user_id) VALUES (:thread_id, :agent_name, :user_id) RETURNING id"
                )
                session_id = (await db.execute(insert_session_stmt, {
                    "thread_id": thread_id, "agent_name": agent_name, "user_id": user_db_id
                })).scalar_one()
                await db.commit()

        agent_config = load_agent_config(agent_name)
        all_tools = []
        if "retriever" in agent_config.tools:
            all_tools.append(get_retriever_tool(agent_config))
        if "sql_toolkit" in agent_config.tools:
            db_source = next((s for s in agent_config.sources if s.type == 'db'), None)
            if db_source:
                all_tools.extend(create_sql_toolkit(db_connection_string=db_source.db_connection,
                                                    llm_model_name=agent_config.llm_model))
        if "arxiv" in agent_config.tools:
            all_tools.extend(get_arxiv_tool())
        if "search_engine" in agent_config.tools:
            all_tools.append(SearchTool(agent_config=agent_config))

        langgraph_agent = get_agent_workflow(all_tools).compile()

        is_first_message = True
        while True:
            query = message_data["payload"]["query"]
            request_id = str(uuid4())
            logger.info(f"[{thread_id}] Processing request {request_id} for query: '{query[:50]}...'")

            # Load message history
            history: List[BaseMessage] = []
            if is_db_session and session_id:
                select_messages_stmt = text(
                    f"SELECT sender, content FROM {chat_messages_table.name} WHERE session_id = :session_id ORDER BY timestamp ASC"
                )
                messages_result = (await db.execute(select_messages_stmt, {"session_id": session_id})).fetchall()
                for msg in messages_result:
                    if msg.sender == 'human':
                        history.append(HumanMessage(content=msg.content))
                    elif msg.sender == 'ai':
                        history.append(AIMessage(content=msg.content))

            if is_db_session and session_id:
                insert_message_stmt = text(
                    f"INSERT INTO {chat_messages_table.name} (session_id, sender, content) VALUES (:session_id, :sender, :content)")
                await db.execute(insert_message_stmt,
                                 {"session_id": session_id, "sender": "human", "content": query})
                await db.commit()

            # Include history in the initial state
            initial_state: AgentState = {"messages": history + [HumanMessage(content=query)],
                                         "request_id": request_id}

            # Create tasks for agent generation and listening for client messages
            gen_task = asyncio.create_task(handle_query_streaming(initial_state, {
                "configurable": {"agent_config": agent_config, "tools": all_tools}}, langgraph_agent, ws, thread_id))
            listen_task = asyncio.create_task(ws.receive_json())

            done, pending = set(), {gen_task, listen_task}
            final_state, next_message = {"request_id": request_id}, None
            ai_response_content = ""

            try:
                done, pending = await asyncio.wait_for(
                    asyncio.wait({gen_task, listen_task}, return_when=asyncio.FIRST_COMPLETED), timeout=TURN_TIMEOUT)
                if listen_task in done:
                    next_message = listen_task.result()
                    logger.info(f"[{thread_id}] Interrupt received, cancelling generation for request {request_id}.")
                if gen_task in done:
                    task_result, accumulated_content = gen_task.result()
                    final_state.update(task_result or {})
                    ai_response_content = accumulated_content
            except asyncio.TimeoutError:
                logger.warning(f"[{thread_id}] Turn timed out after {TURN_TIMEOUT} seconds for request {request_id}.")
                final_state.update({"error": True, "errorMessage": "The request timed out."})
            except Exception as e:
                logger.error(f"[{thread_id}] An unexpected error occurred: {e}", exc_info=True)
                final_state.update({"error": True, "errorMessage": "An unexpected server error occurred."})
            finally:
                for task in pending:
                    if not task.done():
                        task.cancel()
                try:
                    task_result, accumulated_content = await gen_task
                    if not final_state:
                        final_state = task_result or {}
                    ai_response_content = accumulated_content
                except asyncio.CancelledError:
                    pass  # Expected on interrupt
                await asyncio.gather(*pending, return_exceptions=True)

            has_error = final_state is not None and final_state.get("error", False)
            await manager.send({"done": True, "error": has_error, "errorMessage": final_state.get("errorMessage"),
                                "request_id": final_state.get("request_id"), "user_id": user_id,
                                "thread_id": thread_id, "is_first_message": is_first_message}, ws)
            is_first_message = False

            if ai_response_content and not has_error:
                if is_db_session and session_id:
                    insert_message_stmt = text(
                        f"INSERT INTO {chat_messages_table.name} (session_id, sender, content) VALUES (:session_id, 'ai', :content)"
                    )
                    await db.execute(insert_message_stmt,
                                     {"session_id": session_id, "content": ai_response_content})
                    await db.commit()

                # Log metrics...
                if final_state:
                    log_payload = {"request_id": final_state.get("request_id", "N/A"),
                                   "agent_name": agent_name, "thread_id": thread_id,
                                   "metrics": {"retrieval_time_s": round(final_state.get("retrieval_time_s", 0), 4),
                                               "generation_time_s": round(final_state.get("generation_time_s", 0), 4),
                                               "total_duration_s": round(final_state.get("total_duration_s", 0), 4),
                                               "prompt_tokens": final_state.get("prompt_tokens", 0),
                                               "completion_tokens": final_state.get("completion_tokens", 0),
                                               "total_tokens": final_state.get("total_tokens", 0),
                                               "estimated_cost_usd": final_state.get("estimated_cost_usd", 0.0), },
                                   "retrieved_chunk_ids": final_state.get("retrieved_chunk_ids", []),
                                   "user_question_preview": f"{query[:80]}..."}
                    logger.info(f"[{thread_id}] " + json.dumps({"RAGneticTrace": log_payload}))

            # Wait for the next message from the client
            message_data = next_message or await ws.receive_json()
            while not (isinstance(message_data, dict) and message_data.get("type") == "query" and message_data.get(
                    "payload", {}).get("query")):
                logger.info(
                    f"[{thread_id}] Received non-query message: {message_data}. Waiting for next valid query.")
                message_data = await ws.receive_json()

    except WebSocketDisconnect:
        logger.info(f"[{thread_id}] Client disconnected.")
    except Exception as e:
        logger.error(f"[{thread_id}] Unhandled WebSocket Error: {e}", exc_info=True)
    finally:
        manager.disconnect(ws)