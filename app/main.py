from fastapi import FastAPI, HTTPException
from app.schemas.agent import AgentConfig
from app.agents.loader import save_agent_config
from app.pipelines.embed import embed_agent_data
from app.agents.base_agent import build_agent
from typing import Optional, List
from app.agents.agent_graph import build_langgraph_agent
from fastapi.responses import StreamingResponse
from uuid import uuid4
from langchain_core.messages import HumanMessage


import os
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
agents_cache = {}

from pydantic import BaseModel

class AskRequest(BaseModel):
    query: str
    agent: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None


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
