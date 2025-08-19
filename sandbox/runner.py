import json
import logging
import os
import sys
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, Any
import ast

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

ALLOWED_MODULES = {
    "math", "statistics", "random", "numpy", "pandas", "matplotlib",
    "textblob", "csv", "re", "seaborn"
}

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(WORK_DIR / ".mplconfig"))
(Path(os.environ["MPLCONFIGDIR"])).mkdir(parents=True, exist_ok=True)

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
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

    except Exception as e:
        error_payload = {
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed"
        }
        logger.error("Execution failed", extra={"details": error_payload})
        with open(OUTPUTS_DIR / "error.json", "w") as f:
            json.dump(error_payload, f)
        sys.exit(1)


def execute_function_mode(function_name: str, args: Dict[str, Any]):
    """Calls a pre-vetted function from a registry baked into the sandbox image."""
    logger.info(f"Executing in 'function' mode: {function_name}", extra={"mode": "function"})

    try:
        from app.executors import function_registry  # type: ignore
    except Exception:
        try:
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

        final_state = {
            "status": "completed",
            "function": function_name,
            "result": result,
            "output": str(result) if result is not None else None,
            "result_files": _collect_result_files()
        }
        with open(OUTPUTS_DIR / "result.json", "w") as f:
            json.dump(final_state, f, default=str)
    except Exception as e:
        logger.error(f"Error executing function '{function_name}': {e}", extra={"mode": "function"})
        raise


def _safe_open(path, mode="r", *args, **kwargs):
    p = Path(path)
    p = (WORK_DIR / p).resolve() if not p.is_absolute() else p.resolve()

    if "r" in mode and not p.is_relative_to(INPUTS_DIR):
        raise PermissionError(f"Read denied outside inputs: {p}")
    if any(flag in mode for flag in ("w", "a", "x", "+")) and not p.is_relative_to(OUTPUTS_DIR):
        raise PermissionError(f"Write denied outside outputs: {p}")

    return open(p, mode, *args, **kwargs)


def execute_code_mode(code: str):
    """Executes raw Python code in a restricted sandbox and saves the result."""
    logger.info("Executing code in 'code' mode.")

    safe_globals = {
        "__builtins__": {
            "__import__": _safe_import,
            "print": print, "len": len, "enumerate": enumerate, "range": range, "zip": zip,
            "dict": dict, "list": list, "tuple": tuple, "set": set,
            "str": str, "int": int, "float": float, "bool": bool, "complex": complex,
            "chr": chr, "ord": ord, "bin": bin, "oct": oct, "hex": hex,
            "abs": abs, "round": round, "pow": pow, "min": min, "max": max, "sum": sum, "divmod": divmod,
            "sorted": sorted, "reversed": reversed, "next": next, "iter": iter, "slice": slice,
            "any": any, "all": all, "map": map, "filter": filter,
            "isinstance": isinstance, "issubclass": issubclass,
            "format": format, "repr": repr,
            "open": _safe_open,
        },
        "math": __import__("math"),
        "json": __import__("json"),
        "re": __import__("re"),
    }

    local_vars = {}
    output_buffer = io.StringIO()

    try:
        tree = ast.parse(code, mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            expression_source = ast.unparse(tree.body[-1].value)
            code_to_execute = code + f"\nprint({expression_source})"
        else:
            code_to_execute = code
    except Exception as e:
        logger.warning(f"Failed to parse code for auto-printing. Executing as-is. Error: {e}")
        code_to_execute = code

    try:
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            exec(code_to_execute, safe_globals, local_vars)

        stdout_output = output_buffer.getvalue().strip()

        final_state = {
            "output": stdout_output,
            "status": "completed",
            "result_files": _collect_result_files()
        }
        with open(OUTPUTS_DIR / "result.json", "w") as f:
            json.dump(final_state, f, default=str)

    except Exception:
        logger.error("Code execution failed", exc_info=True)
        raise

def execute_notebook_mode(notebook_file_path: str, parameters: Dict[str, Any], output_file_name: str):
    """Executes a notebook, passing parameters, and saves the output notebook."""
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

        final_state = {
            "status": "completed",
            "notebook": str(output_path.name),
            "output": None,
            "result_files": _collect_result_files()
        }
        with open(OUTPUTS_DIR / "result.json", "w") as f:
            json.dump(final_state, f, default=str)
        logger.info(f"Notebook executed successfully. Output saved to {output_path}", extra={"mode": "notebook"})
    except Exception as e:
        logger.error(f"Notebook execution failed: {e}", extra={"mode": "notebook"})
        raise


def _collect_result_files():
    """Auto-discover all files in /outputs and return structured metadata."""
    result_files = []
    for p in OUTPUTS_DIR.glob("*"):
        if p.is_file() and p.name not in ("result.json", "error.json"):
            result_files.append({
                "file_name": p.name,
                "relative_path": f"outputs/{p.name}",
                "size_bytes": p.stat().st_size,
            })
    return result_files


if __name__ == "__main__":
    main()
