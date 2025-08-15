# sandbox/runner.py
import json
import logging
import os
import sys
import traceback
import io
import papermill as pm
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, Any, List

# --- Existing imports ---
from app.executors import function_registry
from app.executors.function_registry import basic_utilities
from app.executors.function_registry import data_analysis
from app.executors.function_registry import file_utilities

# Define the whitelist of allowed environment variables
# This list is intentionally empty because the LambdaTool will not be
# configured to use any external APIs or require host environment variables.

# Configure a basic logger for structured output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "mode": "%(mode)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define sandbox paths
WORK_DIR = Path("/work")
INPUTS_DIR = WORK_DIR / "inputs"
OUTPUTS_DIR = WORK_DIR / "outputs"
REQUEST_FILE = WORK_DIR / "request.json"

SANDBOX_ROOT = WORK_DIR

def _is_within_sandbox(path: Path) -> bool:
    """Check if a given path is inside the sandbox root."""
    return path.resolve().is_relative_to(SANDBOX_ROOT)


def save_output(output_data: Any):
    """Saves output data to result.json."""
    try:
        with open(OUTPUTS_DIR / "result.json", "w") as f:
            json.dump(output_data, f)
    except Exception as e:
        logger.error(f"Failed to save output to result.json: {e}")
        raise


def main():
    """Main execution entrypoint for the sandbox runner."""
    # Ensure all function modules are imported to register the functions
    try:
        logger.info("Sandbox runner started.")

        # Ensure output directory exists
        OUTPUTS_DIR.mkdir(exist_ok=True)

        if not REQUEST_FILE.exists():
            raise FileNotFoundError(f"Request file not found: {REQUEST_FILE}")

        with open(REQUEST_FILE, 'r') as f:
            request_data = json.load(f)

        mode = request_data.get("mode")
        payload = request_data.get("payload", {})
        output_artifacts = payload.get("output_artifacts", [])

        if mode == "code":
            execute_code_mode(payload.get("code", ""))
        elif mode == "function":
            execute_function_mode(payload.get("function_name"), payload.get("function_args", {}))
        elif mode == "notebook":
            execute_notebook_mode(payload.get("notebook_file_path"), payload.get("parameters", {}),
                                  payload.get("output_file_name"))
        else:
            raise ValueError(f"Unsupported execution mode: {mode}")

        logger.info("Execution completed successfully.")

        # Collect and log output artifacts
        artifacts = collect_artifacts(output_artifacts)
        with open(OUTPUTS_DIR / "artifacts.json", "w") as f:
            json.dump(artifacts, f)

    except Exception as e:
        error_payload = {
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed"
        }
        logger.error("Execution failed", extra={"details": error_payload})
        # Save a structured error file for the host to collect
        with open(OUTPUTS_DIR / "error.json", "w") as f:
            json.dump(error_payload, f)
        sys.exit(1)


# --- Existing functions (execute_code_mode, execute_function_mode, etc.) ---

def execute_function_mode(function_name: str, args: Dict[str, Any]):
    """Calls a pre-vetted function from the regislogging.basicConfig(level=logging.INFOtry."""
    logger.info(f"Executing in 'function' mode: {function_name}")

    function_data = function_registry.FunctionRegistration.get_function(function_name)
    if not function_data:
        raise ValueError(f"Function '{function_name}' is not registered.")

    func = function_data["function"]

    try:
        result = func(**args)
        logger.info(f"Function call result: {result}")
        with open(OUTPUTS_DIR / "result.json", "w") as f:
            json.dump(result, f)

    except Exception as e:
        logger.error(f"Error executing function '{function_name}': {e}")
        raise


def execute_code_mode(code: str):
    """Executes raw Python code."""
    logger.info("Executing in 'code' mode.", extra={"mode": "code"})

    safe_globals = {
        "__builtins__": {
            "print": print, "len": len, "range": range, "dict": dict, "list": list,
            "str": str, "int": int, "float": float, "bool": bool, "enumerate": enumerate,
            "zip": zip,
        },
    }
    local_vars = {}

    # Capture stdout/stderr
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        try:
            exec(code, safe_globals, local_vars)
            output = buf.getvalue()
            if output:
                logger.info(f"Execution Output:\n{output}", extra={"mode": "code"})
                save_output({"output": output})
            else:
                save_output({"output": "Code executed successfully with no output."})

        except Exception as e:
            logger.error(
                f"Code execution error: {e}",
                extra={
                    "mode": "code",
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            raise


def execute_notebook_mode(notebook_file_path: str, parameters: Dict[str, Any], output_file_name: str):
    """Executes a notebook, passing parameters, and saves the output."""
    logger.info(f"Executing notebook mode for file: {notebook_file_path}", extra={"mode": "notebook"})

    input_path = WORK_DIR / notebook_file_path
    if not _is_within_sandbox(input_path):
        raise PermissionError(f"Access outside sandbox denied: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Notebook file not found at: {input_path}")

    output_path = OUTPUTS_DIR / output_file_name

    try:
        # Run the notebook with papermill
        pm.execute_notebook(
            input_path=str(input_path),
            output_path=str(output_path),
            parameters=parameters,
            log_level="INFO"
        )
        logger.info(f"Notebook executed successfully. Output saved to {output_path}", extra={"mode": "notebook"})

    except Exception as e:
        logger.error(f"Notebook execution failed: {e}", extra={"mode": "notebook"})
        raise


def collect_artifacts(output_paths: List[str]) -> List[Dict[str, Any]]:
    """Gathers metadata for output files from the sandbox."""
    artifacts = []
    for p in output_paths:
        file_path = WORK_DIR / p
        if not _is_within_sandbox(file_path):
            logger.warning(f"Skipping artifact outside sandbox: {file_path}")
            continue
        if file_path.exists() and file_path.is_file():
            artifacts.append({
                "file_name": file_path.name,
                "relative_path": p,
                "size_bytes": file_path.stat().st_size
            })
    return artifacts


if __name__ == "__main__":
    main()