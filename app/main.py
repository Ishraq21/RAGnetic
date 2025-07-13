import os
import logging
import json
from uuid import uuid4
from typing import Optional, List, Dict, Any
from multiprocessing import Process  # NEW: Import Process for watcher
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.core.validation import validate_agent_name
# AGENTS_DIR will come from centralized paths
from app.schemas.agent import AgentConfig
from app.agents.config_manager import save_agent_config, load_agent_config, get_agent_configs
from app.pipelines.embed import embed_agent_data
from app.agents.agent_graph import get_agent_workflow, AgentState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.tools.sql_tool import create_sql_toolkit
from app.tools.retriever_tool import get_retriever_tool
from app.tools.arxiv_tool import get_arxiv_tool
from app.tools.extraction_tool import get_extraction_tool
from app.watcher import start_watcher
from app.core.config import get_path_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Centralized Path Configuration ---
_APP_PATHS = get_path_settings()
_DATA_DIR = _APP_PATHS["DATA_DIR"]
_AGENTS_DIR = _APP_PATHS["AGENTS_DIR"]
_MEMORY_DIR = _APP_PATHS["MEMORY_DIR"]  # For memory_path in websocket_chat
# --- End Centralized Configuration ---

allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(
    title="RAGnetic API",
    version="0.1.0",
    description="API for managing and interacting with RAGnetic agents."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if "*" in allowed_origins:
    logger.warning("=" * 80)
    logger.warning("!!! SECURITY WARNING: CORS is configured to allow all origins ('*').")
    logger.warning("!!! This is convenient for local development but is INSECURE for production.")
    logger.warning("!!! For production, set the 'CORS_ALLOWED_ORIGINS' environment variable.")
    logger.warning("!!! Example: export CORS_ALLOWED_ORIGINS='https://your.domain.com,https://another.domain'")
    logger.warning("=" * 80)

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
        await ws.send_json(msg)


manager = ConnectionManager()

_watcher_process: Optional[Process] = None


@app.on_event("startup")
async def startup_event():  # MODIFIED: Changed to async def for app.on_event
    global _watcher_process
    logger.info("Application startup event: Initializing RAGnetic components.")

    # Start the file watcher process
    # Use centralized _DATA_DIR
    if not os.path.exists(_DATA_DIR):  # MODIFIED
        logger.error(
            f"Error: The '{_DATA_DIR}' directory does not exist. Please run 'ragnetic init' first.")  # MODIFIED
        # Consider raising HTTPException if this is critical for startup
        return  # Do not start watcher if data dir is missing

    # Only start watcher if not in reload mode (Uvicorn handles reloads)
    # The --reload flag is handled by Uvicorn externally, so this startup only runs once per app instance.
    _watcher_process = Process(target=start_watcher, args=(_DATA_DIR,), daemon=True)  # MODIFIED
    _watcher_process.start()
    logger.info("Automated file watcher started in the background (via startup event).")


@app.on_event("shutdown")
async def shutdown_event():  # MODIFIED: Changed to async def for app.on_event
    global _watcher_process
    logger.info("Application shutdown event: Terminating RAGnetic components.")
    # Gracefully terminate the watcher process
    if _watcher_process and _watcher_process.is_alive():
        _watcher_process.terminate()
        _watcher_process.join(timeout=5)  # Give watcher a bit of time to shut down
        if _watcher_process.is_alive():
            logger.warning("Watcher process did not terminate gracefully. Forcing kill.")
            _watcher_process.kill()
        logger.info("File watcher process stopped (via shutdown event).")
    else:
        logger.info("No active watcher process to stop.")


# --- API Endpoints ---
@app.get("/health", tags=["Application"])
def health_check():
    # Use centralized _AGENTS_DIR for readiness check
    if not os.path.exists(_AGENTS_DIR):  # MODIFIED
        raise HTTPException(
            status_code=503,
            detail=f"Service Unavailable: Agents directory not found at '{_AGENTS_DIR}'."  # MODIFIED
        )
    return JSONResponse(content={"status": "ok", "message": "Service is healthy."})


@app.get("/", tags=["Application"])
async def home(request: Request):
    agents_list = []
    seen_agent_names = set()
    try:
        # Use centralized _AGENTS_DIR for get_agent_configs
        # get_agent_configs implicitly uses AGENTS_DIR, so no change needed here directly
        agent_configs = get_agent_configs()
        for config in agent_configs:
            if config.name not in seen_agent_names:
                agents_list.append({"name": config.name, "display_name": config.display_name or config.name})
                seen_agent_names.add(config.name)
    except Exception as e:
        logger.error(f"Could not load agent configs: {e}")
    default_agent = agents_list[0]['name'] if agents_list else ""
    return templates.TemplateResponse("agent_interface.html",
                                      {"request": request, "agents": agents_list, "agent": default_agent})


@app.get("/history/{thread_id}", tags=["Memory"])
async def get_history(thread_id: str, agent_name: str, user_id: str):
    validate_agent_name(agent_name)
    # Use centralized _MEMORY_DIR
    db_path = _MEMORY_DIR / f"{agent_name}_{user_id}_{thread_id}.db"  # MODIFIED
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="History not found.")
    try:
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:  # MODIFIED
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint_tuple = await saver.aget_tuple(config)
            if not checkpoint_tuple:
                return JSONResponse(content=[])
            messages = checkpoint_tuple.checkpoint["channel_values"]["messages"]
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({"type": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"type": "ai", "content": msg.content})
            return JSONResponse(content=history)
    except Exception as e:
        logger.error(f"Error loading history from {db_path}: {e}")
        raise HTTPException(status_code=500, detail="Could not load chat history.")


@app.post("/create-agent", tags=["Agents"])
# Note: create_agent is currently synchronous. If embed_agent_data becomes async,
# this needs to change to async def and await embed_agent_data.
def create_agent(config: AgentConfig):
    try:
        save_agent_config(config)
        # Assuming embed_agent_data will be called in an asyncio.run context in cli.py if needed.
        # If this endpoint is used directly, embed_agent_data would need to be async or run in thread.
        # For now, it's typically called by cli.py's deploy command.
        embed_agent_data(config)
        return JSONResponse(content={"status": "Agent configuration saved successfully.", "agent": config.name})
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_query(
        initial_state: AgentState,
        cfg: dict,
        langgraph_agent: Any,
        user_id: str,
        thread_id: str,
        ws: WebSocket
) -> Optional[AgentState]:
    """
    Handles a single user query, streams the response, and returns the final state with metrics.
    """
    final_state = None
    async for event in langgraph_agent.astream_events(initial_state, cfg, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token: await manager.send({"token": token}, ws)
        elif kind == "on_graph_end":
            final_state = event['data']['output']

    await manager.send({"done": True, "user_id": user_id, "thread_id": thread_id, "citations": []}, ws)
    return final_state


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    await manager.connect(ws)
    thread_id = "uninitialized"
    user_id = "uninitialized"
    agent_name = "uninitialized"

    try:
        initial_data = await ws.receive_json()
        agent_name = initial_data["agent"]
        validate_agent_name(agent_name)
        query = initial_data["query"]
        user_id = initial_data.get("user_id") or f"user-{uuid4().hex[:8]}"
        thread_id = initial_data.get("thread_id") or f"thread-{uuid4().hex[:8]}"

        agent_config = load_agent_config(agent_name)

        all_tools = []
        if "retriever" in agent_config.tools: all_tools.append(get_retriever_tool(agent_config))
        if "sql_toolkit" in agent_config.tools:
            db_source = next((s for s in agent_config.sources if s.type == 'db'), None)
            if db_source:
                all_tools.extend(create_sql_toolkit(
                    db_connection_string=db_source.db_connection,
                    llm_model_name=agent_config.llm_model,
                    llm_model_params=agent_config.model_params
                ))
        if "arxiv" in agent_config.tools: all_tools.extend(get_arxiv_tool())
        if "extractor" in agent_config.tools: all_tools.append(get_extraction_tool(agent_config))

        workflow = get_agent_workflow(all_tools)
        # Use centralized _MEMORY_DIR
        memory_path = _MEMORY_DIR / f"{agent_name}_{user_id}_{thread_id}.db"  # MODIFIED
        os.makedirs(_MEMORY_DIR, exist_ok=True)  # MODIFIED

        async with AsyncSqliteSaver.from_conn_string(str(memory_path)) as saver:  # MODIFIED
            langgraph_agent = workflow.compile(checkpointer=saver)

            # --- Initial Query ---
            request_id = str(uuid4())
            logger.info(
                f"New chat session started. RequestID: {request_id}, Agent: {agent_name}, ThreadID: {thread_id}")

            initial_state: AgentState = {
                "messages": [HumanMessage(content=query)], "tool_calls": [], "request_id": request_id,
                "agent_name": agent_name, "retrieval_time_s": 0.0, "generation_time_s": 0.0,
                "total_duration_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "estimated_cost_usd": 0.0, "retrieved_chunk_ids": []
            }

            final_state = await handle_query(initial_state, {
                "configurable": {"thread_id": thread_id, "agent_config": agent_config, "tools": all_tools}},
                                             langgraph_agent, user_id, thread_id, ws)

            if final_state:
                log_payload = {
                    "request_id": request_id, "agent_name": agent_name, "thread_id": thread_id,
                    "metrics": {
                        "retrieval_time_s": round(final_state.get("retrieval_time_s", 0), 4),
                        "generation_time_s": round(final_state.get("generation_time_s", 0), 4),
                        "total_duration_s": round(final_state.get("total_duration_s", 0), 4),
                        "prompt_tokens": final_state.get("prompt_tokens", 0),
                        "completion_tokens": final_state.get("completion_tokens", 0),
                        "total_tokens": final_state.get("total_tokens", 0),
                        "estimated_cost_usd": final_state.get("estimated_cost_usd", 0.0),
                    },
                    "retrieved_chunk_ids": final_state.get("retrieved_chunk_ids", []),
                    "user_question_preview": f"{query[:80]}..."
                }
                logger.info(json.dumps({"RAGneticTrace": log_payload}))

            # --- Loop for subsequent queries in the same session ---
            while True:
                data = await ws.receive_json()
                query = data["query"]
                request_id = str(uuid4())
                logger.info(f"Continuing chat. RequestID: {request_id}, ThreadID: {thread_id}")

                initial_state: AgentState = {
                    "messages": [HumanMessage(content=query)], "tool_calls": [], "request_id": request_id,
                    "agent_name": agent_name, "retrieval_time_s": 0.0, "generation_time_s": 0.0,
                    "total_duration_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "estimated_cost_usd": 0.0, "retrieved_chunk_ids": []
                }

                final_state = await handle_query(initial_state, {
                    "configurable": {"thread_id": thread_id, "agent_config": agent_config, "tools": all_tools}},
                                                 langgraph_agent, user_id, thread_id, ws)

                if final_state:
                    log_payload = {
                        "request_id": request_id, "agent_name": agent_name, "thread_id": thread_id,
                        "metrics": {
                            "retrieval_time_s": round(final_state.get("retrieval_time_s", 0), 4),
                            "generation_time_s": round(final_state.get("generation_time_s", 0), 4),
                            "total_duration_s": round(final_state.get("total_duration_s", 0), 4),
                            "prompt_tokens": final_state.get("prompt_tokens", 0),
                            "completion_tokens": final_state.get("completion_tokens", 0),
                            "total_tokens": final_state.get("total_tokens", 0),
                            "estimated_cost_usd": final_state.get("estimated_cost_usd", 0.0),
                        },
                        "retrieved_chunk_ids": final_state.get("retrieved_chunk_ids", []),
                        "user_question_preview": f"{query[:80]}..."
                    }
                    logger.info(json.dumps({"RAGneticTrace": log_payload}))

    except WebSocketDisconnect:
        logger.info(f"Client disconnected for thread_id: {thread_id}")
    except Exception as e:
        logger.error(f"WebSocket Error for thread_id {thread_id}: {e}", exc_info=True)
    finally:
        manager.disconnect(ws)