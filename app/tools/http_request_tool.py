# app/tools/http_request_tool.py

import requests
from typing import Dict, Any, Optional, Literal, Type
from pydantic.v1 import BaseModel, Field

class ToolInput(BaseModel):
    method: Literal['GET', 'POST', 'PUT', 'DELETE'] = Field(..., description="The HTTP method to use.")
    url: str = Field(..., description="The URL to send the request to.")
    params: Optional[Dict[str, Any]] = Field(None, description="URL parameters for GET requests.")
    headers: Optional[Dict[str, Any]] = Field(None, description="Request headers.")
    json_payload: Optional[Dict[str, Any]] = Field(None, description="JSON body for POST/PUT requests.")

class HTTPRequestTool:
    """A tool for making HTTP requests."""
    name: str = "http_request_tool"
    description: str = "A tool for making HTTP requests (GET, POST, PUT, DELETE) to a specified URL."
    args_schema: Type[BaseModel] = ToolInput

    def get_input_schema(self) -> Dict[str, Any]:
        return ToolInput.schema()

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes an HTTP request and returns the JSON response.
        """
        try:
            # Parse kwargs into the ToolInput model for validation and clear access
            parsed_input = ToolInput(**kwargs)
        except Exception as e:
            # Handle validation errors explicitly
            return {"error": f"Invalid input for HTTPRequestTool: {e}"}

        method = parsed_input.method
        url = parsed_input.url
        params = parsed_input.params
        headers = parsed_input.headers
        json_payload = parsed_input.json_payload

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
            # Ensure response text is included for debugging
            return {"error": "Failed to decode JSON response.", "content": response.text if response else "No response content."}