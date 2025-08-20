import os
import re
import time
import json
import logging
import uuid
from typing import List, Dict, Any, ClassVar, Type

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from app.services.file_service import FileService
from app.services.temporary_document_service import TemporaryDocumentService
from app.schemas.lambda_tool import LambdaRequestPayload
from app.core.config import get_server_api_keys
from app.agents.config_manager import AgentConfig

logger = logging.getLogger(__name__)

def _guess_filenames_from_code(code: str) -> List[str]:
    """Infer candidate filenames from code when 'inputs' aren't provided."""
    if not code:
        return []
    candidates = set()
    # quoted filenames like "invoice_sample.txt" or 'data.csv'
    for m in re.finditer(r"""['"]([^'"/\\\s]+\.[A-Za-z0-9]{1,8})['"]""", code):
        candidates.add(m.group(1))
    # explicit /work/<name.ext>
    for m in re.finditer(r"""/work/([^\s'"]+\.[A-Za-z0-9]{1,8})""", code):
        candidates.add(m.group(1))
    return list(candidates)

def _normalize_code_for_payload(code: str) -> str:
    if not code:
        return code
    s = code.strip()
    # strip markdown fences
    if s.startswith("```") and s.endswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # decode literal escapes if needed
    if "\\n" in s and "\n" not in s:
        try:
            s = bytes(s, "utf-8").decode("unicode_escape")
        except Exception:
            s = s.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    return s

class LambdaTool(BaseTool):
    """
    Run code in the sandbox and return the final output to the chat.
    """

    name: str = "lambda_tool"
    description: str = (
        "Run short Python code in an isolated sandbox. "
        "Use it when the user asks to execute code or compute something. "
        "Input must match the LambdaRequestPayload schema: "
        "at minimum, set mode='code' and provide payload.code."
    )

    return_direct: bool = True

    # Configurable fields (Pydantic-managed)
    server_url: str = Field(
        default_factory=lambda: os.getenv(
            "RAGNETIC_SERVER_URL", "http://127.0.0.1:8000"
        )
    )
    api_keys: List[str] = Field(default_factory=get_server_api_keys)

    # LangChain expects args_schema if you want schema validation at tool-call time
    args_schema: ClassVar[Type[BaseModel]] = LambdaRequestPayload

    def _run(self, **tool_input: Any) -> str:
        """Validate payload, stage input files (declared or inferred), submit job, and poll for results."""
        try:
            raw_payload = dict(tool_input)
            file_service = FileService()
            run_id = str(uuid.uuid4())
            thread_id = raw_payload.get("thread_id") or str(uuid.uuid4())
            raw_payload["thread_id"] = thread_id
            raw_payload["run_id"] = run_id

            svc = TemporaryDocumentService(agent_config=AgentConfig(name="lambda_tool", description="Internal agent"))

            staged_inputs: List[Dict[str, Any]] = []
            name_map: Dict[str, str] = {}  # {original_name or /work/original_name -> sandbox_path}

            declared_inputs = raw_payload.get("inputs") or []
            if (raw_payload.get("mode") == "code") and raw_payload.get("code") and not declared_inputs:
                inferred = _guess_filenames_from_code(raw_payload["code"])
                declared_inputs = [{"file_name": fn} for fn in inferred]

            missing: List[str] = []
            for f in declared_inputs:
                fname = f.get("file_name") if isinstance(f, dict) else getattr(f, "file_name", None)
                if not fname:
                    continue

                record = svc.get_latest_by_filename(fname)
                if not record:
                    missing.append(fname)
                    continue

                temp_id = record["temp_doc_id"]
                user_id = record["user_id"]
                th_id = record["thread_id"]
                original_file_name = record["original_name"]

                staged_info = file_service.stage_input_file(
                    temp_doc_id=temp_id,
                    user_id=user_id,
                    thread_id=th_id,
                    run_id=run_id,
                    file_name=fname,
                )
                sandbox_path = staged_info["sandbox_path"]  # e.g., /work/inputs/<uuid>_<name>

                # Map both bare filename and /work/<filename> to the staged path
                name_map[original_file_name] = sandbox_path
                name_map[f"/work/{original_file_name}"] = sandbox_path
                name_map[f"/mnt/work/{original_file_name}"] = sandbox_path

                staged_inputs.append({
                    "temp_doc_id": temp_id,
                    "file_name": fname,
                    "path_in_sandbox": sandbox_path,
                    "original_name": original_file_name,
                    "user_id": user_id,
                    "thread_id": th_id,
                })

            if missing and not staged_inputs:
                return (
                        "LambdaTool: no matching temporary documents were found for: "
                        + ", ".join(sorted(set(missing)))
                )

            raw_payload["inputs"] = staged_inputs

            if raw_payload.get("mode") == "code" and raw_payload.get("code") and name_map:
                code_string = raw_payload["code"]
                for orig in sorted(name_map.keys(), key=len, reverse=True):
                    code_string = code_string.replace(orig, name_map[orig])
                # Normalize code to turn literal '\n' into real newlines and strip ``` fences
                code_string = _normalize_code_for_payload(code_string)
                raw_payload["code"] = code_string
                for orig, path in name_map.items():
                    if not orig.startswith("/work/"):
                        logger.info(f"Rewrote file reference: {orig} -> {path}")

            # Also normalize if there was no name_map rewrite but mode=code
            if raw_payload.get("mode") == "code" and raw_payload.get("code"):
                raw_payload["code"] = _normalize_code_for_payload(raw_payload["code"])

            payload = LambdaRequestPayload(**raw_payload)

        except Exception as e:
            logger.error(f"Invalid LambdaTool payload: {e}", exc_info=True)
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
            resp = requests.post(
                submit_url,
                headers=headers,
                json=payload.model_dump(),
                timeout=15,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"LambdaTool submit error: {e}", exc_info=True)
            return f"LambdaTool: failed to submit job. {e}"

        run_id = (resp.json() or {}).get("run_id")
        if not run_id:
            return "LambdaTool: submission succeeded but no run_id was returned."

        # 3) Poll for completion...
        wait_secs = int(os.getenv("LAMBDA_TOOL_WAIT_SECONDS", "30"))
        get_url = f"{self.server_url}/api/v1/lambda/runs/{run_id}"
        deadline = time.time() + wait_secs

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
                error_msg = data.get("error_message")

                if status == "failed":
                    structured_msg = (
                            final_state.get("message")
                            or (final_state.get("error_type") and final_state.get("traceback"))
                    )
                    msg = structured_msg or error_msg or "Unknown error."
                    return f"Sandbox run **failed** (run_id={run_id}).\n\n{msg}"

                output_str = None
                if isinstance(final_state, dict):
                    output_str = final_state.get("output") or final_state.get("result")
                if isinstance(output_str, dict):
                    output_str = json.dumps(output_str, indent=2)

                lines = [f"Sandbox run **completed** (run_id={run_id})."]

                if output_str and output_str.strip():
                    lines.append("\n**Output**:\n")
                    lines.append(f"```text\n{output_str.rstrip()}\n```")
                else:
                    lines.append("\n**Final State**:\n")
                    lines.append("```json\n" + json.dumps(final_state, indent=2) + "\n```")

                # Show files if present (supports both keys)
                if isinstance(final_state, dict):
                    artifacts = final_state.get("artifacts") or final_state.get("result_files")
                    if artifacts:
                        lines.append("\n**Files**:\n")
                        lines.append("```json\n" + json.dumps(artifacts, indent=2) + "\n```")

                return "\n".join(lines)

        return (
            f"Job submitted (run_id={run_id}) and still running after {wait_secs}s. "
            f"Check status later or increase LAMBDA_TOOL_WAIT_SECONDS."
        )
