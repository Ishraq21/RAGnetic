import os
from uuid import uuid4
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from app.schemas.agent import AgentConfig
from app.agents.config_manager import save_agent_config
from app.pipelines.embed import embed_agent_data
from app.agents.agent_graph import get_agent_workflow
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAGnetic API")

# This is critical for allowing browser-based clients (like the default FastAPI docs)
# to interact with your API without running into security errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

templates = Jinja2Templates(directory="templates")


class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
            print("Client connection removed.")

    async def send(self, msg: Dict, ws: WebSocket):
        await ws.send_json(msg)

manager = ConnectionManager()


@app.get("/")
async def home(request: Request):
    """Serves the main chat interface."""
    agents_dir = "agents_data"
    if not os.path.exists(agents_dir):
        os.makedirs(agents_dir)
    agents = [f.split(".")[0] for f in os.listdir(agents_dir) if f.endswith(".json")]
    default_agent = agents[0] if agents else ""
    return templates.TemplateResponse(
        "agent_interface.html",
        {
            "request": request,
            "agents": agents,
            "agent": default_agent,
            "user_id": "",
            "thread_id": "",
        },
    )


@app.post("/create-agent")
def create_agent(config: AgentConfig, openai_api_key: Optional[str] = None):
    """Creates a new agent, embeds its data, and makes it available for use."""
    try:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key missing.")
        save_agent_config(config)
        embed_agent_data(config, openai_api_key=api_key)
        return {"status": "Agent created", "agent": config.name}
    except Exception as e:
        # It's good practice to log the actual error on the server
        print(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    """Handles the real-time, stateful chat conversations using WebSockets."""
    await manager.connect(ws)
    try:
        session_user_id = None
        session_thread_id = None

        while True:
            data = await ws.receive_json()
            agent_name = data["agent"]
            query = data["query"]

            user_id = data.get("user_id") or session_user_id or f"user-{uuid4().hex[:8]}"
            thread_id = data.get("thread_id") or session_thread_id or f"thread-{uuid4().hex[:8]}"

            session_user_id = user_id
            session_thread_id = thread_id

            workflow = get_agent_workflow(agent_name)
            memory_path = f"memory/{agent_name}_{user_id}_{thread_id}.db"
            os.makedirs("memory", exist_ok=True)

            async with AsyncSqliteSaver.from_conn_string(memory_path) as saver:
                langgraph_agent = workflow.compile(checkpointer=saver)

                config = {
                    "configurable": {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "new_message": HumanMessage(content=query),
                    }
                }

                async for event in langgraph_agent.astream_events({}, config=config, version="v2"):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        token = event["data"]["chunk"].content
                        if token:
                            await manager.send({"token": token}, ws)

                await manager.send({
                    "done": True,
                    "user_id": user_id,
                    "thread_id": thread_id,
                }, ws)

    except WebSocketDisconnect:
        print(f"Client disconnected.")
        manager.disconnect(ws)
    except Exception as e:
        print(f"WebSocket Error: {e}")
        # It's good practice to notify the client that an error occurred
        await manager.send({"error": str(e)}, ws)
    finally:
        manager.disconnect(ws)