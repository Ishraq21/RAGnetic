import os
import time
import json
import logging
from typing import List, Dict, Any, ClassVar, Type
import requests
from pydantic import Field
from langchain_core.tools import BaseTool

from app.schemas.lambda_tool import LambdaRequestPayload
from app.core.config import get_server_api_keys
from pydantic import BaseModel, Field
logger = logging.getLogger(__name__)

class LambdaTool(BaseTool):
    """
    Run code in the sandbox and return the final output to the chat.
    """
    name: str = "lambda_tool"
    description: str = (
        "Run short Python code in an isolated sandbox. "
        "Use it when the user asks to execute code or compute something. "
        "Input must match the LambdaRequestPayload schema: at minimum, set mode='code' and provide payload.code."
    )

    return_direct: bool = True

    # Configurable fields (Pydantic-managed)
    server_url: str = Field(default_factory=lambda: os.getenv("RAGNETIC_SERVER_URL", "http://127.0.0.1:8000"))
    api_keys: List[str] = Field(default_factory=get_server_api_keys)

    # LangChain expects args_schema if you want schema validation at tool-call time
    args_schema: ClassVar[Type[BaseModel]] = LambdaRequestPayload

    def _run(self, **tool_input: Any) -> str:
        # 1) Validate payload
        try:
            payload = LambdaRequestPayload(**tool_input)
        except Exception as e:
            logger.error(f"Invalid LambdaTool payload: {e}")
            return f"LambdaTool: invalid payload. {e}"

        if not self.api_keys:
            return "LambdaTool: no server API key configured."

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_keys[0],
        }

        # 2) Submit job
        submit_url = f"{self.server_url}/api/v1/lambda/execute"
        try:
            resp = requests.post(submit_url, headers=headers, json=payload.model_dump(), timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"LambdaTool submit error: {e}")
            return f"LambdaTool: failed to submit job. {e}"

        run_id = (resp.json() or {}).get("run_id")
        if not run_id:
            return "LambdaTool: submission succeeded but no run_id was returned."

        # 3) Poll for completion (quick, bounded wait)
        WAIT_SECS = int(os.getenv("LAMBDA_TOOL_WAIT_SECONDS", "30"))  # NEW: env-configurable
        get_url = f"{self.server_url}/api/v1/lambda/runs/{run_id}"
        deadline = time.time() + WAIT_SECS
        while time.time() < deadline:
            try:
                time.sleep(0.5)
                r = requests.get(get_url, headers=headers, timeout=10)
                r.raise_for_status()
                data = r.json() or {}
            except requests.RequestException as e:
                logger.warning(f"Polling error for run {run_id}: {e}")
                continue

            status = data.get("status")
            if status in ("completed", "failed"):
                final_state = data.get("final_state") or {}
                artifacts = data.get("artifacts") or []
                error_msg = data.get("error_message")

                if status == "failed":
                    structured_msg = (
                        final_state.get("message")
                        or (final_state.get("error_type") and final_state.get("traceback"))
                    )
                    msg = structured_msg or error_msg or "Unknown error."
                    return f"Sandbox run **failed** (run_id={run_id}).\n\n{msg}"

                # completed
                output_str = None
                if isinstance(final_state, dict):
                    output_str = final_state.get("output") or final_state.get("result")
                if isinstance(output_str, dict):
                    output_str = json.dumps(output_str, indent=2)

                lines = [f"Sandbox run **completed** (run_id={run_id})."]
                if output_str and output_str.strip():
                    lines.append("\n**Output**:\n")
                    lines.append("```text\n" + output_str.rstrip() + "\n```")
                else:
                    lines.append("\n**Final State**:\n")
                    lines.append("```json\n" + json.dumps(final_state, indent=2) + "\n```")

                if artifacts:
                    lines.append("\n**Artifacts**:")
                    for a in artifacts:
                        name = a.get("file_name", "artifact")
                        size = a.get("size_bytes", "?")
                        url  = a.get("signed_url", "")
                        lines.append(f"- {name} ({size} bytes) {url}")

                return "\n".join(lines)

        # 4) Timed out waiting → fall back to “submitted”
        return (
            f"Job submitted (run_id={run_id}) and still running after {WAIT_SECS}s. "
            f"Check status later or increase LAMBDA_TOOL_WAIT_SECONDS."
        )
