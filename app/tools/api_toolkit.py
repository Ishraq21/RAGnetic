# app/tools/api_toolkit.py

import logging
import requests
import asyncio
from typing import Any, Dict, Type

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from app.schemas.api_toolkit import APIRequestToolInput

logger = logging.getLogger(__name__)


# Synchronous helper function for blocking network calls
def _make_request_blocking(url: str, method: str, headers: Dict, params: Dict, json: Dict):
    if method.upper() == 'POST' or method.upper() == 'PUT':
        response = requests.request(method, url, headers=headers, json=json, timeout=30)
    else:
        response = requests.request(method, url, headers=headers, params=params, timeout=30)
    response.raise_for_status()  # Raises an exception for bad status codes
    return response.json()


class APIToolkit(BaseTool):
    name: str = "api_toolkit"
    description: str = (
        "A powerful tool for making authenticated HTTP requests to a specified URL. "
        "It supports GET, POST, and PUT methods with headers, parameters, and a JSON payload. "
        "Returns the JSON response from the API. Raises an exception on non-200 status codes."
    )
    args_schema: Type[BaseModel] = APIRequestToolInput

    async def _arun(self, **kwargs: Any) -> Any:
        try:
            tool_input = APIRequestToolInput(**kwargs)

            # Use asyncio.to_thread to run the blocking request function
            response_data = await asyncio.to_thread(
                _make_request_blocking,
                url=tool_input.url,
                method=tool_input.method,
                headers=tool_input.headers or {},
                params=tool_input.params or {},
                json=tool_input.payload or {}
            )

            return response_data

        except requests.exceptions.RequestException as e:
            error_msg = f"HTTP request failed: {e}"
            logger.error(error_msg, exc_info=True)
            raise ToolException(error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred during API call: {e}"
            logger.error(error_msg, exc_info=True)
            raise ToolException(error_msg)

    def _run(self, **kwargs: Any) -> Any:
        # A synchronous fallback for environments that don't support async.
        return asyncio.run(self._arun(**kwargs))