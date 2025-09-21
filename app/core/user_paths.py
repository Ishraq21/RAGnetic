"""
User-specific path utilities for RAGnetic.
Handles user-specific directories for agents, vector stores, data uploads, and sources.
"""

import os
from pathlib import Path
from typing import Optional
from app.core.config import get_path_settings


def get_user_agents_dir(user_id: int) -> Path:
    """Get the agents directory for a specific user."""
    agents_dir = get_path_settings()["AGENTS_DIR"]
    return agents_dir / "users" / str(user_id)


def get_user_vectorstore_dir(user_id: int) -> Path:
    """Get the vectorstore directory for a specific user."""
    vectorstore_dir = get_path_settings()["VECTORSTORE_DIR"]
    return vectorstore_dir / "users" / str(user_id)


def get_user_data_dir(user_id: int) -> Path:
    """Get the data directory for a specific user."""
    data_dir = get_path_settings()["DATA_DIR"]
    return data_dir / "uploads" / "users" / str(user_id)


def get_user_sources_dir(user_id: int) -> Path:
    """Get the sources directory for a specific user."""
    data_dir = get_path_settings()["DATA_DIR"]
    return data_dir / "sources" / "users" / str(user_id)


def get_user_agent_path(user_id: int, agent_name: str) -> Path:
    """Get the path to a specific agent's YAML file for a user."""
    return get_user_agents_dir(user_id) / f"{agent_name}.yaml"


def get_user_vectorstore_path(user_id: int, agent_name: str) -> Path:
    """Get the path to a specific agent's vector store for a user."""
    return get_user_vectorstore_dir(user_id) / agent_name


def ensure_user_directories(user_id: int) -> None:
    """Ensure all user-specific directories exist."""
    directories = [
        get_user_agents_dir(user_id),
        get_user_vectorstore_dir(user_id),
        get_user_data_dir(user_id),
        get_user_sources_dir(user_id)
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_agent_vectorstore_path(agent_name: str, user_id: Optional[int] = None) -> Path:
    """Get the vectorstore path for an agent, with user-specific path if user_id is provided."""
    if user_id:
        return get_user_vectorstore_path(user_id, agent_name)
    else:
        # Fall back to global vectorstore directory
        vectorstore_dir = get_path_settings()["VECTORSTORE_DIR"]
        return vectorstore_dir / agent_name


def get_agent_config_path(agent_name: str, user_id: Optional[int] = None) -> Path:
    """Get the config path for an agent, with user-specific path if user_id is provided."""
    if user_id:
        return get_user_agent_path(user_id, agent_name)
    else:
        # Fall back to global agents directory
        agents_dir = get_path_settings()["AGENTS_DIR"]
        return agents_dir / f"{agent_name}.yaml"
