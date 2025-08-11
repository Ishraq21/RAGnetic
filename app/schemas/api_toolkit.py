from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

class APIRequestToolInput(BaseModel):
    """
    Input schema for a general-purpose API request tool.
    """
    url: str = Field(..., description="The full URL for the API endpoint.")
    method: Literal['GET', 'POST', 'PUT', 'DELETE'] = Field('GET', description="The HTTP method to use.")
    headers: Optional[Dict[str, str]] = Field(None, description="A dictionary of custom headers to include.")
    params: Optional[Dict[str, str]] = Field(None, description="A dictionary of query parameters for GET requests.")
    payload: Optional[Dict[str, Any]] = Field(None, description="The JSON payload for POST or PUT requests.")