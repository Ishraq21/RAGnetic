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
    """Infer candidate INPUT filenames from code when 'inputs' aren't provided.
    
    Only looks for files that are being READ, not written to.
    """
    if not code:
        return []
    candidates = set()
    
    # Look for file READ operations specifically - simplified patterns
    read_patterns = [
        # with open() for reading - default mode or 'r'
        r"""with\s+open\s*\(\s*['"]([^'"/\\]+\.[a-zA-Z0-9]{1,8})['"](?:\s*,\s*['"]r['"])?""",
        # open() calls without 'w' or 'a' mode
        r"""(?<!\.)\bopen\s*\(\s*['"]([^'"/\\]+\.[a-zA-Z0-9]{1,8})['"](?:\s*,\s*['"]r['"])?""",
        # pandas and other read functions - including /work/ paths
        r"""pd\.read_\w+\s*\(\s*['"](?:/work/(.+?\.[a-zA-Z0-9]{1,8})|([^'"/\\]+\.[a-zA-Z0-9]{1,8}))['"]""",
        r"""read_\w+\s*\(\s*['"](?:/work/(.+?\.[a-zA-Z0-9]{1,8})|([^'"/\\]+\.[a-zA-Z0-9]{1,8}))['"]""",
    ]
    
    for pattern in read_patterns:
        for m in re.finditer(pattern, code):
            # Handle multiple capture groups (for /work/ paths vs regular paths)
            filename = None
            for i in range(1, len(m.groups()) + 1):
                if m.group(i):
                    filename = m.group(i)
                    break
            
            if filename:
                # Skip if it looks like an output file based on context
                if not _looks_like_output_file(code, filename):
                    candidates.add(filename)
    
    return list(candidates)

def _looks_like_output_file(code: str, filename: str) -> bool:
    """Check if a filename appears to be used for writing/output based on context."""
    # Look for write operations on this file - must have 'w' or 'a' mode
    write_patterns = [
        rf"""open\s*\(\s*['"](?:/work/)?{re.escape(filename)}['"](?:\s*,\s*['"][wa]['"])""",
        rf"""with\s+open\s*\(\s*['"](?:/work/)?{re.escape(filename)}['"](?:\s*,\s*['"][wa]['"])""",
        rf"""\.dump\s*\([^,]+,\s*\w+\).*{re.escape(filename)}""",
    ]
    
    for pattern in write_patterns:
        if re.search(pattern, code):
            return True
    return False

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

                try:
                    record = svc.get_latest_by_filename(fname)
                    if not record:
                        missing.append(fname)
                        continue
                except (RuntimeError, Exception) as e:
                    logger.warning(f"Database access failed for file {fname}: {e}. Skipping file input.")
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

            if missing and not staged_inputs and declared_inputs:
                # Only fail if inputs were explicitly declared but none found
                missing_files = ", ".join(sorted(set(missing)))
                return (
                    f"ERROR: **Lambda Tool Error**: No matching files found\n\n"
                    f"**Missing files**: {missing_files}\n\n"
                    f"**Solutions**:\n"
                    f"• Upload files using `/api/v1/documents/upload`\n"
                    f"• Use `temp_doc_id` in inputs instead of `file_name`\n"
                    f"• Check file names match exactly (case-sensitive)\n"
                    f"• Remove file references from code to run without inputs"
                )
            elif missing:
                logger.warning(f"Some files not found but execution will continue: {missing}")

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
            error_type = type(e).__name__
            return (
                f"X **Lambda Tool Validation Error**\n\n"
                f"**Error**: {error_type}: {str(e)}\n\n"
                f"**Expected format**:\n"
                f"• `mode`: \"code\"\n"
                f"• `code`: \"your python code here\"\n"
                f"• `inputs`: [{{\"temp_doc_id\": \"...\"}}, ...] (optional)\n\n"
                f"**Example**:\n"
                f"```json\n"
                f"{{\"mode\": \"code\", \"code\": \"print('Hello World!')\"}}\n"
                f"```"
            )

        if not self.api_keys:
            return (
                f"ERROR: **Lambda Tool Configuration Error**\n\n"
                f"**Issue**: No API key configured\n\n"
                f"**Solution**: Set `RAGNETIC_API_KEYS` in environment variables"
            )
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
            error_msg = str(e)
            if "ConnectionError" in error_msg:
                return (
                    f"ERROR: **Lambda Tool Connection Error**\n\n"
                    f"**Issue**: Cannot connect to lambda service\n"
                    f"**Server URL**: {self.server_url}\n\n"
                    f"**Solutions**:\n"
                    f"• Check if RAGnetic server is running\n"
                    f"• Verify RAGNETIC_SERVER_URL is correct\n"
                    f"• Check network connectivity"
                )
            elif "401" in error_msg or "403" in error_msg:
                return (
                    f"ERROR: **Lambda Tool Authentication Error**\n\n"
                    f"**Issue**: Invalid API key or unauthorized access\n\n"
                    f"**Solutions**:\n"
                    f"• Check RAGNETIC_API_KEYS environment variable\n"
                    f"• Verify API key is valid and not expired"
                )
            else:
                return (
                    f"ERROR: **Lambda Tool Submission Error**\n\n"
                    f"**Error**: {error_msg}\n\n"
                    f"Check server logs for more details"
                )

        run_id = (resp.json() or {}).get("run_id")
        if not run_id:
            return (
                f"ERROR: **Lambda Tool Response Error**\n\n"
                f"**Issue**: Server accepted job but returned no run_id\n\n"
                f"This indicates a server-side issue. Check server logs."
            )

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
                    error_type = final_state.get("error_type", "Unknown")
                    traceback = final_state.get("traceback", "")
                    message = final_state.get("message") or error_msg or "Unknown error"
                    
                    # Provide specific help for common errors
                    if "ImportError" in error_type and "not allowed" in message:
                        help_text = (
                            f"\n**SOLUTION**: This module is restricted in the sandbox.\n"
                            f"Try using allowed modules: pandas, numpy, json, csv, requests, etc."
                        )
                    elif "FileNotFoundError" in error_type:
                        help_text = (
                            f"\n**SOLUTION**: File not found in sandbox.\n"
                            f"• Upload files using `/api/v1/documents/upload`\n"
                            f"• Check file paths are correct\n"
                            f"• Files must be in `/work/` directory"
                        )
                    elif "PermissionError" in error_type:
                        help_text = (
                            f"\n**SOLUTION**: Permission denied.\n"
                            f"• Files must be within `/work/` directory\n"
                            f"• Some operations are restricted for security"
                        )
                    elif "SyntaxError" in error_type:
                        help_text = (
                            f"\n**SOLUTION**: Fix the Python syntax error.\n"
                            f"• Check indentation and brackets\n"
                            f"• Verify quotes and parentheses match"
                        )
                    else:
                        help_text = ""
                    
                    return (
                        f"ERROR: **Sandbox Execution Failed** (run_id={run_id})\n\n"
                        f"**Error Type**: {error_type}\n"
                        f"**Message**: {message}\n"
                        f"{help_text}\n\n"
                        f"**Traceback**:\n```\n{traceback.strip()}\n```" if traceback else ""
                    )

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
            f"TIMEOUT: **Lambda Tool Timeout** (run_id={run_id})\n\n"
            f"**Status**: Job is still running after {wait_secs} seconds\n\n"
            f"**Solutions**:\n"
            f"• Check job status later using run_id: {run_id}\n"
            f"• Increase timeout with LAMBDA_TOOL_WAIT_SECONDS environment variable\n"
            f"• Optimize code to run faster\n"
            f"• Check server logs for potential issues"
        )
