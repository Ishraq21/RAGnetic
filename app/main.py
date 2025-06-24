import os
import pickle
import aiosqlite
import logging
from uuid import uuid4
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.schemas.agent import AgentConfig
from app.agents.config_manager import save_agent_config, load_agent_config
from app.pipelines.embed import embed_agent_data
from app.agents.agent_graph import get_agent_workflow
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.tools.sql_tool import create_sql_toolkit
from app.tools.retriever_tool import get_retriever_tool

# --- NEW: Production-Grade Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
# --- NEW: Added versioning to the API ---
app = FastAPI(title="RAGnetic API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            logger.info("Client connection removed.")

    async def send(self, msg: Dict, ws: WebSocket):
        await ws.send_json(msg)


manager = ConnectionManager()


# Health Check Endpoint
@app.get("/health", tags=["Application"])
def health_check():
    """A simple endpoint to confirm that the server is running."""
    return JSONResponse(content={"status": "ok"})


@app.get("/", tags=["Application"])
async def home(request: Request):
    """Serves the main chat interface."""
    agents_dir = "agents_data"
    if not os.path.exists(agents_dir):
        os.makedirs(agents_dir)
    agents = [f.split(".")[0] for f in os.listdir(agents_dir) if f.endswith(".yaml")]
    default_agent = agents[0] if agents else ""
    return templates.TemplateResponse(
        "agent_interface.html",
        {"request": request, "agents": agents, "agent": default_agent},
    )


@app.get("/history/{thread_id}", tags=["Memory"])
async def get_history(thread_id: str, agent_name: str, user_id: str):
    """Retrieves message history from the LangGraph checkpointer database."""
    db_path = f"memory/{agent_name}_{user_id}_{thread_id}.db"
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="History not found.")
    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
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
def create_agent(config: AgentConfig):
    """Creates an agent configuration and triggers the embedding pipeline."""
    try:
        save_agent_config(config)
        embed_agent_data(config)
        return JSONResponse(content={"status": "Agent configuration saved successfully.", "agent": config.name})
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_query(
        q: str,
        cfg: dict,
        langgraph_agent: Any,
        user_id: str,
        thread_id: str,
        ws: WebSocket
):
    final_state = None
    input_dict = {"messages": [HumanMessage(content=q)]}

    async for event in langgraph_agent.astream_events(input_dict, cfg, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token: await manager.send({"token": token}, ws)
        elif kind == "on_graph_end":
            final_state = event['data']['output']

    await manager.send({
        "done": True,
        "user_id": user_id,
        "thread_id": thread_id,
        "citations": []
    }, ws)


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    await manager.connect(ws)
    thread_id = "uninitialized"
    user_id = "uninitialized"
    try:
        initial_data = await ws.receive_json()
        agent_name = initial_data["agent"]
        query = initial_data["query"]
        user_id = initial_data.get("user_id") or f"user-{uuid4().hex[:8]}"
        thread_id = initial_data.get("thread_id") or f"thread-{uuid4().hex[:8]}"

        agent_config = load_agent_config(agent_name)

        all_tools = []
        if "retriever" in agent_config.tools:
            all_tools.append(get_retriever_tool(agent_name))

        if "sql_toolkit" in agent_config.tools:
            db_source = next((s for s in agent_config.sources if s.type == 'db'), None)
            if db_source and db_source.db_connection:
                all_tools.extend(create_sql_toolkit(db_source.db_connection))

        workflow = get_agent_workflow(all_tools)

        memory_path = f"memory/{agent_name}_{user_id}_{thread_id}.db"
        os.makedirs("memory", exist_ok=True)

        session_config = {
            "configurable": {
                "thread_id": thread_id,
                "agent_config": agent_config,
            }
        }

        async with AsyncSqliteSaver.from_conn_string(memory_path) as saver:
            langgraph_agent = workflow.compile(checkpointer=saver)

            await handle_query(query, session_config, langgraph_agent, user_id, thread_id, ws)

            while True:
                data = await ws.receive_json()
                await handle_query(data["query"], session_config, langgraph_agent, user_id, thread_id, ws)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected for thread_id: {thread_id}")
    except Exception as e:
        logger.error(f"WebSocket Error for thread_id {thread_id}: {e}")
        await manager.send({
            "error": str(e),
            "user_id": user_id,
            "thread_id": thread_id
        }, ws)
    finally:
        manager.disconnect(ws)
