from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal

class DataSource(BaseModel):
    type: Literal['local', 'url', 'code_repository']
    path: Optional[str] = None
    url: Optional[str] = None

class AgentConfig(BaseModel):
    name: str
    description: Optional[str]
    persona_prompt: str
    sources: List[DataSource]
    tools: Optional[List[str]] = []
