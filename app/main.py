import os
import pickle
import aiosqlite
from uuid import uuid4
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.schemas.agent import AgentConfig
from app.agents.config_manager import save_agent_config, load_agent_config
from app.pipelines.embed import embed_agent_data
from app.agents.agent_graph import get_agent_workflow
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()
app = FastAPI(title="RAGnetic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        },
    )


@app.get("/history/{thread_id}")
async def get_history(thread_id: str, agent_name: str, user_id: str):
    db_path = f"memory/{agent_name}_{user_id}_{thread_id}.db"
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="History not found.")
    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint_tuple = await saver.aget_tuple(config)
            if not checkpoint_tuple:
                return []
            messages = checkpoint_tuple.checkpoint["channel_values"]["messages"]
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({"type": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"type": "ai", "content": msg.content})
            return history
    except Exception as e:
        print(f"Error loading history from {db_path}: {e}")
        raise HTTPException(status_code=500, detail="Could not load chat history.")


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
        print(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    await manager.connect(ws)
    thread_id = "uninitialized"
    try:
        initial_data = await ws.receive_json()
        agent_name = initial_data["agent"]
        query = initial_data["query"]
        user_id = initial_data.get("user_id") or f"user-{uuid4().hex[:8]}"
        thread_id = initial_data.get("thread_id") or f"thread-{uuid4().hex[:8]}"

        workflow = get_agent_workflow(agent_name)
        memory_path = f"memory/{agent_name}_{user_id}_{thread_id}.db"
        os.makedirs("memory", exist_ok=True)

        async with AsyncSqliteSaver.from_conn_string(memory_path) as saver:
            langgraph_agent = workflow.compile(checkpointer=saver)

            async def handle_query(q: str):
                config = {"configurable": {"thread_id": thread_id}}
                final_state = None

                # Pass an explicit dictionary as input
                input_dict = {"messages": [HumanMessage(content=q)]}

                async for event in langgraph_agent.astream_events(
                        input_dict, config, version="v2"
                ):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        token = event["data"]["chunk"].content
                        if token:
                            await manager.send({"token": token}, ws)
                    elif kind == "on_graph_end":
                        final_state = event['data']['output']

                citations = final_state.get("citations", []) if final_state else []
                await manager.send({
                    "done": True, "user_id": user_id, "thread_id": thread_id, "citations": citations
                }, ws)

            await handle_query(query)

            while True:
                data = await ws.receive_json()
                await handle_query(data["query"])
    except WebSocketDisconnect:
        print(f"Client disconnected for thread_id: {thread_id}")
    except Exception as e:
        print(f"WebSocket Error for thread_id {thread_id}: {e}")
        await manager.send({"error": str(e)}, ws)
    finally:
        manager.disconnect(ws)