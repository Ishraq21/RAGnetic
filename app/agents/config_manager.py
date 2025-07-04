import yaml
import os
from typing import List
from app.schemas.agent import AgentConfig

AGENTS_DIR = "agents_data"


def save_agent_config(config: AgentConfig):
    """Saves an agent's configuration to a YAML file."""
    if not os.path.exists(AGENTS_DIR):
        os.makedirs(AGENTS_DIR)

    file_path = os.path.join(AGENTS_DIR, f"{config.name}.yaml")
    # Use model_dump() instead of dict() for modern Pydantic versions
    with open(file_path, 'w') as f:
        yaml.dump(config.model_dump(), f, sort_keys=False, default_flow_style=False)


def load_agent_config(agent_name: str) -> AgentConfig:
    """Loads a specific agent's configuration from its YAML file."""
    file_path = os.path.join(AGENTS_DIR, f"{agent_name}.yaml")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Agent config not found: {file_path}")

    with open(file_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return AgentConfig(**config_dict)


def load_agent_from_yaml_file(path: str) -> AgentConfig:
    """Loads an agent's configuration directly from a specified YAML file path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent config file not found at: {path}")

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return AgentConfig(**config_dict)

def get_agent_configs() -> List[AgentConfig]:
    """
    Scans the agents directory and loads all agent configurations.

    This is used by the file watcher to determine which agents are affected
    by a data source change.
    """
    configs = []
    if not os.path.exists(AGENTS_DIR):
        return []

    for filename in os.listdir(AGENTS_DIR):
        if filename.endswith((".yaml", ".yml")):
            agent_name = os.path.splitext(filename)[0]
            try:
                config = load_agent_config(agent_name)
                configs.append(config)
            except Exception as e:
                # Log an error if a specific config fails to load, but continue
                # processing the others.
                print(f"Warning: Could not load agent configuration '{agent_name}': {e}")

    return configs