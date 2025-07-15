import os
import logging
import json
import asyncio
from uuid import uuid4
from typing import Optional, List, Dict, Any, Tuple
from multiprocessing import Process

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, status
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from app.core.validation import validate_agent_name, sanitize_for_path
from app.core.security import get_http_api_key, get_websocket_api_key
from app.schemas.agent import AgentConfig, SearchEngineToolInput
from app.agents.config_manager import save_agent_config, load_agent_config, get_agent_configs
from app.pipelines.embed import embed_agent_data
from app.agents.agent_graph import get_agent_workflow, AgentState
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.tools.sql_tool import create_sql_toolkit
from app.tools.retriever_tool import get_retriever_tool
from app.tools.arxiv_tool import get_arxiv_tool
from app.tools.search_engine_tool import SearchTool
from langchain_core.tools import Tool

from app.watcher import start_watcher
from app.core.config import get_path_settings, get_server_api_keys # Import get_server_api_keys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
TURN_TIMEOUT = 300.0  # 5-minute master timeout for an entire turn

load_dotenv()

_APP_PATHS = get_path_settings()
_DATA_DIR = _APP_PATHS["DATA_DIR"]
_AGENTS_DIR = _APP_PATHS["AGENTS_DIR"]
_MEMORY_DIR = _APP_PATHS["MEMORY_DIR"]
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
        if ws.client_state is WebSocketState.CONNECTED:
            await ws.send_json(msg)
        else:
            logger.warning("Attempted to send message on a closed or disconnected websocket.")


manager = ConnectionManager()
_watcher_process: Optional[Process] = None


@app.on_event("startup")
async def startup_event():
    global _watcher_process
    logger.info("Application startup event: Initializing RAGnetic components.")
    if not os.path.exists(_DATA_DIR):
        logger.error(f"Error: The '{_DATA_DIR}' directory does not exist. Please run 'ragnetic init' first.")
        return
    _watcher_process = Process(target=start_watcher, args=(_DATA_DIR,), daemon=True)
    _watcher_process.start()
    logger.info("Automated file watcher started in the background (via startup event).")


@app.on_event("shutdown")
async def shutdown_event():
    global _watcher_process
    logger.info("Application shutdown event: Terminating RAGnetic components.")
    if _watcher_process and _watcher_process.is_alive():
        _watcher_process.terminate()
        _watcher_process.join(timeout=5)
        if _watcher_process.is_alive():
            _watcher_process.kill()
        logger.info("File watcher process stopped (via shutdown event).")


@app.get("/", tags=["Application"])
async def home(request: Request):
    agents_list = []
    try:
        agent_configs = get_agent_configs()
        for config in agent_configs:
            agents_list.append({"name": config.name, "display_name": config.display_name or config.name})
    except Exception as e:
        logger.error(f"Could not load agent configs: {e}")
    default_agent = agents_list[0]['name'] if agents_list else ""

    # --- Get server API keys and pass one to the template ---
    server_api_keys = get_server_api_keys()
    # If keys exist, take the first one; otherwise, pass an empty string.
    # This key will be used by the frontend to authenticate WebSocket connections.
    frontend_api_key = server_api_keys[0] if server_api_keys else ""

    return templates.TemplateResponse("agent_interface.html",
                                      {"request": request,
                                       "agents": agents_list,
                                       "agent": default_agent,
                                       "api_key": frontend_api_key}) # Pass the key here


@app.get("/history/{thread_id}", tags=["Memory"])
async def get_history(
    thread_id: str,
    agent_name: str,
    user_id: str,
    api_key: str = Depends(get_http_api_key)
):
    safe_agent_name = sanitize_for_path(agent_name)
    safe_user_id = sanitize_for_path(user_id)
    safe_thread_id = sanitize_for_path(thread_id)

    if not all([safe_agent_name, safe_user_id, safe_thread_id]):
        raise HTTPException(status_code=400, detail="Invalid characters in agent_name, user_id, or thread_id.")

    db_path = _MEMORY_DIR / f"{safe_agent_name}_{safe_user_id}_{safe_thread_id}.db"
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="History not found.")
    try:
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
            config = {"configurable": {"thread_id": safe_thread_id}}
            checkpoint_tuple = await saver.aget_tuple(config)
            if not checkpoint_tuple:
                return JSONResponse(content=[])
            messages = checkpoint_tuple.checkpoint["channel_values"]["messages"]
            history = [{"type": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content} for msg in
                       messages]
            return JSONResponse(content=history)
    except Exception as e:
        logger.error(f"Error loading history from {db_path}: {e}")
        raise HTTPException(status_code=500, detail="Could not load chat history.")


@app.post("/create-agent", tags=["Agents"])
async def create_agent(
    config: AgentConfig,
    bg: BackgroundTasks,
    api_key: str = Depends(get_http_api_key)
):
    try:
        save_agent_config(config)
        bg.add_task(embed_agent_data, config)
        return JSONResponse(
            content={"status": "Agent configuration saved and embedding started in background.", "agent": config.name})
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_query_streaming(
        initial_state: AgentState, cfg: dict, langgraph_agent: Any, ws: WebSocket, thread_id: str
) -> Optional[Dict]:
    """Streams a single response. Returns final state on success, or an error dict on failure."""
    final_state = None
    try:
        async for event in langgraph_agent.astream_events(initial_state, cfg, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token: await manager.send({"token": token}, ws)
            elif kind == "on_graph_end":
                final_state = event['data']['output']
        return final_state
    except asyncio.CancelledError:
        logger.info(f"[{thread_id}] Generation task was cancelled.")
        return None
    except Exception as e:
        error_message = f"An error occurred during generation: {e}"
        logger.error(f"[{thread_id}] {error_message}", exc_info=True)
        await manager.send({"token": f"\n\n{error_message}"}, ws)
        return {"error": True, "errorMessage": error_message}


async def _run_one_turn(
        ws: WebSocket, langgraph_agent, cfg: dict, query_payload: dict, thread_id: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Runs a single turn with a timeout, handling generation and interruption.
    Returns the final state and the next message to process.
    """
    query = query_payload["query"]
    request_id = str(uuid4())
    logger.info(f"[{thread_id}] Processing request. RequestID: {request_id}")
    initial_state: AgentState = {"messages": [HumanMessage(content=query)], "request_id": request_id,
                                 "agent_name": cfg["configurable"]["agent_config"].name, "tool_calls": [],
                                 "retrieval_time_s": 0.0, "generation_time_s": 0.0, "total_duration_s": 0.0,
                                 "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                                 "estimated_cost_usd": 0.0, "retrieved_chunk_ids": []}

    gen_task = asyncio.create_task(handle_query_streaming(initial_state, cfg, langgraph_agent, ws, thread_id))
    listen_task = asyncio.create_task(ws.receive_json())

    done, pending = set(), {gen_task, listen_task}
    final_state, next_message = {"request_id": request_id}, None

    try:
        done, pending = await asyncio.wait_for(
            asyncio.wait({gen_task, listen_task}, return_when=asyncio.FIRST_COMPLETED),
            timeout=TURN_TIMEOUT
        )

        if listen_task in done:
            next_message = listen_task.result()
            logger.info(
                f"[{thread_id}] Interrupt or new query received, cancelling generation for request {request_id}.")
        if gen_task in done:
            final_state.update(gen_task.result() or {})

    except asyncio.TimeoutError:
        logger.warning(f"[{thread_id}] Turn timed out after {TURN_TIMEOUT} seconds for request {request_id}.")
        final_state.update({"error": True, "errorMessage": "The request timed out as it took too long to process."})

    except Exception as e:
        logger.error(f"[{thread_id}] An unexpected error occurred in _run_one_turn: {e}", exc_info=True)
        final_state.update({"error": True, "errorMessage": "An unexpected server error occurred."})

    finally:
        all_tasks = pending.union(done)
        for task in all_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)

    return final_state, next_message



async def _initialize_agent_session(initial_payload: dict) -> Optional[Dict]:
    """Loads agent config, tools, and workflow based on the first message."""
    try:
        agent_name = initial_payload.get("agent", "unknown_agent")
        validate_agent_name(agent_name)

        user_id = sanitize_for_path(initial_payload.get("user_id")) or f"user-{uuid4().hex[:8]}"
        thread_id = sanitize_for_path(initial_payload.get("thread_id")) or f"thread-{uuid4().hex[:8]}"

        agent_config = load_agent_config(agent_name)
        all_tools = []
        if "retriever" in agent_config.tools: all_tools.append(get_retriever_tool(agent_config))
        if "sql_toolkit" in agent_config.tools:
            db_source = next((s for s in agent_config.sources if s.type == 'db'), None)
            if db_source: all_tools.extend(
                create_sql_toolkit(db_connection_string=db_source.db_connection, llm_model_name=agent_config.llm_model,
                                   llm_model_params=agent_config.model_params))
        if "arxiv" in agent_config.tools: all_tools.extend(get_arxiv_tool())

        # NEW: Register the Search Engine Tool using the custom SearchTool class
        if "search_engine" in agent_config.tools:
            # Instantiate the custom SearchTool class. It inherently knows its name, description, and args_schema.
            search_tool_instance = SearchTool()
            all_tools.append(search_tool_instance)

        workflow = get_agent_workflow(all_tools)

        return {"workflow": workflow, "agent_config": agent_config, "all_tools": all_tools, "user_id": user_id,
                "thread_id": thread_id}
    except Exception as e:
        logger.error(f"Failed to initialize agent session: {e}", exc_info=True)
        return None


@app.websocket("/ws")
async def websocket_chat(
    ws: WebSocket,
    api_key: str = Depends(get_websocket_api_key)
):
    if not api_key:
        # The get_websocket_api_key function returns "" if no valid key is found.
        # If no server keys are configured, it returns "development_mode_unsecured".
        # We need to distinguish between 'no keys configured' and 'invalid key provided'.
        # Assuming that if server_api_keys() returns an empty list, it's unsecured dev mode.
        # Otherwise, if api_key is empty here, it's an invalid key.
        configured_keys = get_server_api_keys()
        if configured_keys: # Server has keys configured, but the client didn't provide a valid one
            await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid or missing API Key.")
            logger.warning(f"WebSocket connection from {ws.client.host}:{ws.client.port} rejected due to missing/invalid API key.")
            return
        # Else: configured_keys is empty, meaning it's development_mode_unsecured. Allow connection.


    await manager.connect(ws)
    thread_id = "uninitialized"
    try:
        message_data = await ws.receive_json()
        if not (isinstance(message_data, dict) and message_data.get("type") == "query" and "payload" in message_data):
            reason = f"Protocol violation: First message must be a valid query object. Got: {message_data}"
            logger.warning(reason)
            await ws.close(code=1003, reason=reason)
            return

        session = await _initialize_agent_session(message_data["payload"])
        if not session:
            await ws.close(code=1011, reason="Agent initialization failed.")
            return

        thread_id = session["thread_id"]
        memory_path = _MEMORY_DIR / f"{session['agent_config'].name}_{session['user_id']}_{thread_id}.db"
        os.makedirs(_MEMORY_DIR, exist_ok=True)

        async with AsyncSqliteSaver.from_conn_string(str(memory_path)) as memory:
            langgraph_agent = session["workflow"].compile(checkpointer=memory)
            is_first_message = True

            while True:
                cfg = {"configurable": {"thread_id": thread_id, "agent_config": session["agent_config"],
                                        "tools": session["all_tools"]}}

                final_state, next_message = await _run_one_turn(ws, langgraph_agent, cfg, message_data["payload"],
                                                                thread_id)

                has_error = final_state is not None and final_state.get("error", False)
                error_message = final_state.get("errorMessage") if has_error else None
                await manager.send({"done": True, "error": has_error, "errorMessage": error_message,
                                    "request_id": final_state.get("request_id"), "user_id": session["user_id"],
                                    "thread_id": thread_id, "is_first_message": is_first_message}, ws)
                is_first_message = False

                if final_state and not has_error:
                    query = message_data["payload"]["query"]
                    log_payload = {"request_id": final_state.get("request_id", "N/A"),
                                   "agent_name": session["agent_config"].name, "thread_id": thread_id,
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

                message_data = next_message

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