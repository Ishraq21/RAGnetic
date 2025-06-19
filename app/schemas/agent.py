from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DataSource(BaseModel):
    type: str
    path: Optional[str] = None
    url: Optional[str] = None

class AgentConfig(BaseModel):
    name: str
    description: Optional[str]
    persona_prompt: str
    sources: List[DataSource]
    tools: Optional[List[str]] = []
