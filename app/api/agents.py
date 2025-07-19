# app/api/agents.py
import os
import shutil
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks

from app.core.config import get_path_settings
from app.core.security import get_http_api_key
from app.schemas.agent import AgentConfig
from app.agents.config_manager import get_agent_configs, load_agent_config, save_agent_config
from app.pipelines.embed import embed_agent_data
import logging

logger = logging.getLogger("ragnetic")

# --- Path Settings ---
_APP_PATHS = get_path_settings()
_AGENTS_DIR = _APP_PATHS["AGENTS_DIR"]
_VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]

# --- API v1 Router for Agents ---
# Note: The prefix is now part of the router definition
router = APIRouter(prefix="/api/v1/agents", tags=["Agents API"])


@router.get("", response_model=List[AgentConfig])
async def get_all_agents(api_key: str = Depends(get_http_api_key)):
    """
    Retrieves the full configuration for all available agents.
    """
    try:
        agent_configs = get_agent_configs()
        return agent_configs
    except Exception as e:
        logger.error(f"API: Failed to get agent configs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load agent configurations.")


@router.get("/{agent_name}", response_model=AgentConfig)
async def get_agent_by_name(agent_name: str, api_key: str = Depends(get_http_api_key)):
    """
    Retrieves the full configuration for a single agent.
    """
    try:
        agent_config = load_agent_config(agent_name)
        return agent_config
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    except Exception as e:
        logger.error(f"API: Failed to get agent config for '{agent_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not load configuration for agent '{agent_name}'.")


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_new_agent(config: AgentConfig = Body(...), bg: BackgroundTasks = BackgroundTasks(),
                           api_key: str = Depends(get_http_api_key)):
    """
    Creates a new agent from a configuration payload and starts the data embedding process.
    """
    try:
        load_agent_config(config.name)
        raise HTTPException(
            status_code=409,
            detail=f"Agent '{config.name}' already exists. Use PUT to update."
        )
    except FileNotFoundError:
        pass  # This is the expected case for a new agent

    try:
        save_agent_config(config)
        bg.add_task(embed_agent_data, config)
        return {"status": "Agent config saved; embedding started.", "agent": config.name}
    except Exception as e:
        logger.error(f"API: Error creating agent '{config.name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{agent_name}")
async def update_agent_by_name(agent_name: str, config: AgentConfig = Body(...), bg: BackgroundTasks = BackgroundTasks(),
                               api_key: str = Depends(get_http_api_key)):
    """
    Updates an existing agent's configuration and triggers a re-embedding of its data.
    """
    if agent_name != config.name:
        raise HTTPException(status_code=400, detail="Agent name in path does not match agent name in body.")
    try:
        load_agent_config(agent_name)  # Ensure the agent exists

        save_agent_config(config)
        bg.add_task(embed_agent_data, config)
        return {"status": "Agent config updated; re-embedding started.", "agent": config.name}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    except Exception as e:
        logger.error(f"API: Error updating agent '{agent_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{agent_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_by_name(agent_name: str, api_key: str = Depends(get_http_api_key)):
    """
    Deletes an agent, its configuration, and all associated data.
    """
    try:
        load_agent_config(agent_name)  # Ensure the agent exists

        config_path = _AGENTS_DIR / f"{agent_name}.yaml"
        if os.path.exists(config_path):
            os.remove(config_path)

        vectorstore_path = _VECTORSTORE_DIR / f"{agent_name}"
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)

        return
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    except Exception as e:
        logger.error(f"API: Error deleting agent '{agent_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))