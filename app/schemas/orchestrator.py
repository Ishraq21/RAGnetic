# app/schemas/orchestrator.py
from typing import List, Optional

from pydantic import Field

from app.schemas.agent import AgentConfig

class OrchestratorConfig(AgentConfig):
    """
    Configuration for an orchestrator agent that manages a roster of specialized sub-agents.

    This model extends the standard AgentConfig by adding a 'roster', which is a list of
    agent names that the orchestrator can call as tools.
    """
    roster: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of agent names that this orchestrator can call as tools."
    )

    class Config:
        title = "OrchestratorConfig"