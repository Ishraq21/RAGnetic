# app/main.py

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict
from uuid import uuid4
import os

from app.schemas.agent import AgentConfig
from app.agents.loader import save_agent_config
from app.pipelines.embed import embed_agent_data
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Updated import
from app.agents.agent_graph import get_agent_workflow

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory=".")


class AskRequest(BaseModel):
    query: str
    agent: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None


class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def send(self, msg: Dict, ws: WebSocket):
        await ws.send_json(msg)


manager = ConnectionManager()


@app.get("/")
async def home(request: Request):
    agents_dir = "agents_data"
    if not os.path.exists(agents_dir):
        os.makedirs(agents_dir)
    agents = [f.split(".")[0] for f in os.listdir(agents_dir) if f.endswith(".json")]
    default_agent = agents[0] if agents else ""
    return templates.TemplateResponse(
        "templates/agent_interface.html",
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
    try:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key missing.")
        save_agent_config(config)
        embed_agent_data(config, openai_api_key=api_key)
        return {"status": "Agent created", "agent": config.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
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

            # Get the uncompiled graph workflow
            workflow = get_agent_workflow(agent_name)

            memory_path = f"memory/{agent_name}_{user_id}_{thread_id}.db"
            os.makedirs("memory", exist_ok=True)

            # Use async with to correctly manage the checkpointer lifecycle
            async with AsyncSqliteSaver.from_conn_string(memory_path) as saver:
                # Compile the graph inside the async context with the checkpointer
                langgraph_agent = workflow.compile(checkpointer=saver)

                config = {
                    "configurable": {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "new_message": HumanMessage(content=query),
                    }
                }

                # Stream events using the compiled agent
                async for event in langgraph_agent.astream_events({}, config=config, version="v2"):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        token = event["data"]["chunk"].content
                        if token:
                            await manager.send({"token": token}, ws)

                # Signal completion
                await manager.send({
                    "done": True,
                    "user_id": user_id,
                    "thread_id": thread_id,
                }, ws)

    except WebSocketDisconnect:
        print("Client disconnected.")
        manager.disconnect(ws)
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await manager.send({"error": str(e)}, ws)
    finally:
        manager.disconnect(ws)