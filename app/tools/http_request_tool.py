# app/tools/http_request_tool.py

from typing import Dict, Any, Optional, Literal
import requests
from pydantic.v1 import BaseModel, Field

class ToolInput(BaseModel):
    method: Literal['GET', 'POST', 'PUT', 'DELETE'] = Field(..., description="The HTTP method to use.")
    url: str = Field(..., description="The URL to send the request to.")
    params: Optional[Dict[str, Any]] = Field(None, description="URL parameters for GET requests.")
    headers: Optional[Dict[str, Any]] = Field(None, description="Request headers.")
    json_payload: Optional[Dict[str, Any]] = Field(None, description="JSON body for POST/PUT requests.")

class HTTPRequestTool:
    """A tool for making HTTP requests."""

    def run(self, method: str, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, Any]] = None, json_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes an HTTP request and returns the JSON response.
        """
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json=json_payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"HTTP request failed: {e}"}
        except ValueError: # Catches JSON decoding errors
            return {"error": "Failed to decode JSON response.", "content": response.text}

    # This method allows the engine to discover the tool's input schema
    def get_input_schema(self) -> Dict[str, Any]:
        return ToolInput.schema()