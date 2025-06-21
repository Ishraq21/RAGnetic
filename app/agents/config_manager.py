import json
from pathlib import Path
from app.schemas.agent import AgentConfig

AGENT_CONFIG_DIR = Path("agents_data")


def load_agent_config(name: str) -> AgentConfig:
    config_path = AGENT_CONFIG_DIR / f"{name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found for agent: {name}")

    with open(config_path, "r") as f:
        data = json.load(f)
    return AgentConfig(**data)


def save_agent_config(config: AgentConfig):
    config_path = AGENT_CONFIG_DIR / f"{config.name}.json"
    with open(config_path, "w") as f:
        json.dump(config.dict(), f, indent=2)
