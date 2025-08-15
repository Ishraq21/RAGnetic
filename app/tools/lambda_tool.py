# app/tools/lambda_tool.py
import logging
import requests
import json
import asyncio
from typing import List, Dict, Any, Optional

from langchain_core.tools import BaseTool
from app.schemas.lambda_tool import LambdaRequestPayload
from app.core.config import get_server_api_keys, get_llm_model

logger = logging.getLogger(__name__)

class LambdaTool(BaseTool):
    """
    Tool for executing code and data workflows in a secure, isolated sandbox.
    """
    # CORRECTED: args_schema is now the Pydantic class itself, not the JSON schema dictionary.
    name: str = "lambda_tool"
    description: str = (
        "Useful for running Python code, performing data analysis, executing functions, "
        "or generating plots and reports. Input should be a JSON object conforming "
        "to the LambdaRequestPayload schema."
    )
    args_schema: type[LambdaRequestPayload] = LambdaRequestPayload

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.server_url = "http://localhost:8000"
        self.api_keys = get_server_api_keys()
        if not self.api_keys:
            logger.error("LambdaTool cannot be initialized without a server API key.")
            raise ValueError("Server API key not configured for LambdaTool.")

    async def _arun(self, **tool_input: Any) -> str:
        payload_dict = tool_input
        try:
            payload = LambdaRequestPayload(**payload_dict)
        except Exception as e:
            logger.error(f"Invalid payload for LambdaTool: {e}")
            return f"Error: Invalid payload format. Please ensure your input matches the schema: {e}"

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_keys[0],
        }

        url = f"{self.server_url}/lambda/execute"

        try:
            response = requests.post(url, headers=headers, data=payload.model_dump_json(), timeout=10)
            response.raise_for_status()
            response_data = response.json()
            run_id = response_data.get("run_id")
            return f"LambdaTool job submitted successfully. The run ID is '{run_id}'. You can check the status and results with the API."

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error calling LambdaTool API: {e}")
            return f"An error occurred while submitting the job: {e}"

    def _run(self, **kwargs: Any) -> str:
        return asyncio.run(self._arun(**kwargs))