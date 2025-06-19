from fastapi import FastAPI, HTTPException
from app.schemas.agent import AgentConfig
from app.agents.loader import save_agent_config
from app.pipelines.embed import embed_agent_data
from app.agents.base_agent import build_agent
from typing import Optional, List, Dict
from app.agents.agent_graph import build_langgraph_agent
from fastapi.responses import StreamingResponse
from uuid import uuid4
from langchain_core.messages import HumanMessage
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi import WebSocket, WebSocketDisconnect
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.base import AsyncCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()

templates = Jinja2Templates(directory="templates")


app = FastAPI()
agents_cache = {}

from pydantic import BaseModel

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
    import os
    # discover available agents
    agents = [
        f.split(".")[0]
        for f in os.listdir("agents_data")
        if f.endswith(".json")
    ]
    # pick a default agent if any
    default_agent = agents[0] if agents else ""
    # Render template, including agent/user/thread in context
    return templates.TemplateResponse(
        "agent_interface.html",
        {
            "request":    request,
            "agents":     agents,
            "agent":      default_agent,  # so {{agent}} is never undefined
            "user_id":    "",             # start blank
            "thread_id":  "",             # start blank
        }
    )


@app.post("/create-agent")
def create_agent(config: AgentConfig, openai_api_key: Optional[str] = None):
    try:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key missing.")

        save_agent_config(config)
        embed_agent_data(config, openai_api_key=api_key)
        agents_cache[config.name] = build_agent(config.name)

        return {"status": "Agent created", "agent": config.name}
    except HTTPException:
        raise  # Let FastAPI handle known HTTPExceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(request: AskRequest):
    try:
        user_id = request.user_id or f"user-{uuid4().hex[:8]}"
        thread_id = request.thread_id or f"thread-{uuid4().hex[:8]}"

        # Build a fresh agent with unique memory path per user+thread
        langgraph_agent = build_langgraph_agent(request.agent, user_id, thread_id)

        result = langgraph_agent.invoke(
            input={},
            config={
                "configurable": {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "new_message": HumanMessage(content=request.query),
                }
            },
        )

        return {
            "response": result["messages"][-1].content,
            "user_id": user_id,
            "thread_id": thread_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            agent = data["agent"]
            query = data["query"]
            user_id = data.get("user_id")
            thread_id = data.get("thread_id")

            # build a streaming RetrievalQA
            vectordb = FAISS.load_local(
                f"vectorstore/{agent}",
                OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )
            retriever = vectordb.as_retriever()
            llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

            class WSHandler(AsyncCallbackHandler):
                async def on_llm_new_token(self, token: str, **kwargs):
                    await manager.send({"token": token}, ws)

            llm.callbacks = [WSHandler()]
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=False
            )
            # kick off streaming
            await chain.ainvoke({"query": query})
            # signal end
            await manager.send({"done": True}, ws)
    except WebSocketDisconnect:
        manager.disconnect(ws)