import json
import logging
import os
import sys
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, Any, List

# Configure a basic logger for structured output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define sandbox paths
WORK_DIR = Path("/work")
INPUTS_DIR = WORK_DIR / "inputs"
OUTPUTS_DIR = WORK_DIR / "outputs"
REQUEST_FILE = WORK_DIR / "request.json"

SANDBOX_ROOT = WORK_DIR


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    allowed = {"math", "statistics", "json", "re", "datetime" "sympy"}
    root = name.split(".")[0]
    if root not in allowed:
        raise ImportError(f"Import of '{name}' is not allowed")
    return __import__(name, globals, locals, fromlist, level)


def _is_within_sandbox(path: Path) -> bool:
    """Check if a given path is inside the sandbox root."""
    return path.resolve().is_relative_to(SANDBOX_ROOT)


def main():
    """Main execution entrypoint for the sandbox runner."""
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

        # We'll explicitly define the output artifacts here, as the code
        # execution mode does not produce them on its own.
        output_artifacts = ["outputs/test_output.txt"]

        if mode == "code":
            code_str = request_data.get("code") or payload.get("code", "")
            execute_code_mode(code_str)

        elif mode == "function":
            fn_name = request_data.get("function_name") or payload.get("function_name")
            fn_args = request_data.get("function_args") or payload.get("function_args", {}) or {}
            execute_function_mode(fn_name, fn_args)

        elif mode == "notebook":
            nb_path = request_data.get("notebook_file_path") or payload.get("notebook_file_path")
            nb_params = request_data.get("parameters") or payload.get("parameters", {}) or {}
            nb_out = request_data.get("output_file_name") or payload.get("output_file_name") or "executed.ipynb"
            execute_notebook_mode(nb_path, nb_params, nb_out)

        else:
            raise ValueError(f"Unsupported execution mode: {mode}")

        logger.info("Execution completed successfully.")

        result_path = OUTPUTS_DIR / "result.json"
        if not result_path.exists():
            with open(result_path, "w") as f:
                json.dump({"status": "ok", "mode": mode}, f)

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


def execute_function_mode(function_name: str, args: Dict[str, Any]):
    """Calls a pre-vetted function from a registry baked into the sandbox image."""
    logger.info(f"Executing in 'function' mode: {function_name}", extra={"mode": "function"})

    try:
        # Try the in-image registry path
        from app.executors import function_registry  # type: ignore
    except Exception:
        try:
            # Fallback: allow a local file named function_registry.py in /work or PYTHONPATH
            import function_registry  # type: ignore
        except Exception as e:
            raise ImportError(
                "Function mode is unavailable in this sandbox: no 'function_registry' module found. "
                "Either run in 'code' mode or bake your registry into the Docker image."
            ) from e

    function_data = function_registry.FunctionRegistration.get_function(function_name)
    if not function_data:
        raise ValueError(f"Function '{function_name}' is not registered.")

    func = function_data["function"]

    try:
        result = func(**(args or {}))
        logger.info(f"Function call result: {result}", extra={"mode": "function"})
        with open(OUTPUTS_DIR / "result.json", "w") as f:
            json.dump({"status": "ok", "function": function_name, "result": result}, f)
    except Exception as e:
        logger.error(f"Error executing function '{function_name}': {e}", extra={"mode": "function"})
        raise


def execute_code_mode(code: str):
    """
    Executes raw Python code in a restricted sandbox, captures its output,
    and saves the result to a structured JSON file.
    """
    logger.info("Executing code in 'code' mode.")

    safe_globals = {
        "__builtins__": {
            # imports
            "__import__": _safe_import,

            # io & introspection
            "print": print, "len": len, "enumerate": enumerate, "range": range, "zip": zip,

            # core types / conversions
            "dict": dict, "list": list, "tuple": tuple, "set": set,
            "str": str, "int": int, "float": float, "bool": bool, "complex": complex,
            "chr": chr, "ord": ord, "bin": bin, "oct": oct, "hex": hex,

            # numbers & math-y helpers
            "abs": abs, "round": round, "pow": pow, "min": min, "max": max, "sum": sum, "divmod": divmod,

            # iteration & ordering
            "sorted": sorted, "reversed": reversed, "next": next, "iter": iter, "slice": slice,

            # predicates & higher-order
            "any": any, "all": all, "map": map, "filter": filter,

            # type checks / formatting
            "isinstance": isinstance, "issubclass": issubclass,
            "format": format, "repr": repr,
        },

        # Optional convenience: let users use these without an explicit import
        "math": __import__("math"),
        "json": __import__("json"),
        "re": __import__("re"),
    }

    local_vars = {}

    # Capture stdout and stderr
    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            exec(code, safe_globals, local_vars)

        # Capture the output from the buffer
        stdout_output = output_buffer.getvalue()

        logger.info(f"Execution output captured (bytes={len(stdout_output)})", extra={"mode": "code"})


        # Write the final state to the result.json file
        final_state = {"output": stdout_output, "status": "completed"}
        with open(OUTPUTS_DIR / "result.json", "w") as f:
            json.dump(final_state, f)

        # Write the stdout to a plain text file as an artifact
        with open(OUTPUTS_DIR / "test_output.txt", "w") as f:
            f.write(stdout_output)

    except Exception as e:
        error_payload = {
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed"
        }
        with open(OUTPUTS_DIR / "error.json", "w") as f:
            json.dump(error_payload, f)
        raise


def execute_notebook_mode(notebook_file_path: str, parameters: Dict[str, Any], output_file_name: str):
    """Executes a notebook, passing parameters, and saves the output."""
    logger.info(f"Executing notebook mode for file: {notebook_file_path}", extra={"mode": "notebook"})

    import papermill as pm

    input_path = WORK_DIR / notebook_file_path
    if not _is_within_sandbox(input_path):
        raise PermissionError(f"Access outside sandbox denied: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Notebook file not found at: {input_path}")

    output_path = OUTPUTS_DIR / output_file_name

    try:
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