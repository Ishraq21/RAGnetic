from fastapi import FastAPI, HTTPException
from app.schemas.agent import AgentConfig
from app.agents.loader import save_agent_config
from app.pipelines.embed import embed_agent_data
from app.agents.base_agent import build_agent
from typing import Optional, List
from app.agents.agent_graph import build_langgraph_agent
from fastapi.responses import StreamingResponse


import os
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
agents_cache = {}

from pydantic import BaseModel

class AskRequest(BaseModel):
    query: str
    agent: str
    thread_id: str = "default"


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
        if request.agent not in agents_cache:
            agents_cache[request.agent] = build_langgraph_agent(request.agent)

        langgraph_agent = agents_cache[request.agent]
        print(langgraph_agent.input_keys)

        result = langgraph_agent.invoke(
            {"query": request.query},
            config={"configurable": {"thread_id": request.thread_id}},
        )

        if isinstance(result, dict) and "output" in result:
            return {"response": result["output"]}
        return {"response": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


