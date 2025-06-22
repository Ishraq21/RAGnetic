from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal

class DataSource(BaseModel):
    type: Literal['local', 'url', 'code_repository','db','gdoc','web_crawler','api']
    path: Optional[str] = None
    url: Optional[str] = None
    db_connection: Optional[str] = None

    # Google Drive Variables
    folder_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    file_types: Optional[List[str]] = None  # e.g., ["document", "sheet", "pdf"]

    # Web Crawler Field
    max_depth: Optional[int] = 2  # Defaults to 2 levels deep

    # Fields for API Sources
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None      # For GET request query strings
    method: Optional[Literal['GET', 'POST']] = 'GET' # Add method, default to GET
    payload: Optional[Dict[str, Any]] = None     # Add payload for POST requests
    json_pointer: Optional[str] = None           # Pointer to a list of records in the JSON



class AgentConfig(BaseModel):
    name: str
    description: Optional[str] = None
    persona_prompt: str
    sources: List[DataSource]
    tools: Optional[List[str]] = []
