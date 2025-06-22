from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal

class DataSource(BaseModel):
    type: Literal['local', 'url', 'code_repository','db','gdoc','web_crawler']
    path: Optional[str] = None
    url: Optional[str] = None
    db_connection: Optional[str] = None

    # Google Drive Variables
    folder_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    file_types: Optional[List[str]] = None  # e.g., ["document", "sheet", "pdf"]

    # Web Crawler Field
    max_depth: Optional[int] = 2  # Defaults to 2 levels deep

class AgentConfig(BaseModel):
    name: str
    description: Optional[str] = None
    persona_prompt: str
    sources: List[DataSource]
    tools: Optional[List[str]] = []
