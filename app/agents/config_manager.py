import yaml
import os
from typing import List
from app.schemas.agent import AgentConfig

def represent_str(dumper, data):
    """Custom string representer that quotes strings with spaces."""
    if ' ' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

# Register the custom representer
yaml.add_representer(str, represent_str)

AGENTS_DIR = "agents"


def save_agent_config(config: AgentConfig, user_id: int = None):
    """Saves an agent's configuration to a YAML file."""
    if user_id and user_id > 0:  # Ensure user_id is valid
        # Use user-specific directory
        user_agents_dir = os.path.join(AGENTS_DIR, "users", str(user_id))
        if not os.path.exists(user_agents_dir):
            os.makedirs(user_agents_dir)
        file_path = os.path.join(user_agents_dir, f"{config.name}.yaml")
    else:
        # Use global agents directory (backward compatibility)
        if not os.path.exists(AGENTS_DIR):
            os.makedirs(AGENTS_DIR)
        file_path = os.path.join(AGENTS_DIR, f"{config.name}.yaml")
    
    # Convert the config to dict
    config_dict = config.model_dump()
    
    with open(file_path, 'w') as f:
        yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False, allow_unicode=True, width=float('inf'))


def load_agent_config(agent_name: str, user_id: int = None) -> AgentConfig:
    """Loads a specific agent's configuration from its YAML file."""
    if user_id:
        # Try user-specific directory first
        user_agents_dir = os.path.join(AGENTS_DIR, "users", str(user_id))
        file_path = os.path.join(user_agents_dir, f"{agent_name}.yaml")
        if not os.path.exists(file_path):
            # Fall back to global directory
            file_path = os.path.join(AGENTS_DIR, f"{agent_name}.yaml")
    else:
        # Use global agents directory
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