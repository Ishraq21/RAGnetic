from fastapi import FastAPI, HTTPException
from app.schemas.agent import AgentConfig
from app.agents.loader import save_agent_config
from app.pipelines.embed import embed_agent_data
from app.agents.base_agent import build_agent
from typing import Optional

import os
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
agents_cache = {}


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
def ask(query: str, agent: str):
    try:
        if agent not in agents_cache:
            agents_cache[agent] = build_agent(agent)

        qa = agents_cache[agent]
        return {"response": qa.invoke(query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
