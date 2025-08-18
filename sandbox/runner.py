import json
import logging
import os
import sys
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, Any, List
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
    "math", "statistics", "random", "numpy", "pandas", "matplotlib", "textblob", "csv", "re"
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
        requested_artifacts = payload.get("output_artifacts") or []

        # We'll explicitly define the output artifacts here, as the code
        # execution mode does not produce them on its own.
        # Auto-discover all files saved under /work/outputs
        # Predefine artifacts: honor user-specified list if present
        if requested_artifacts:
            output_artifacts = []
            for rel_path in requested_artifacts:
                p = WORK_DIR / rel_path
                if _is_within_sandbox(p) and p.exists() and p.is_file():
                    output_artifacts.append(rel_path)
                else:
                    logger.warning(f"Requested artifact not found or invalid: {rel_path}")
        else:
            # fallback: auto-discover everything under outputs
            output_artifacts = [
                f"outputs/{p.name}" for p in OUTPUTS_DIR.glob("*") if p.is_file()
            ]

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
        artifacts = []
        if requested_artifacts:
            for rel_path in output_artifacts:
                p = WORK_DIR / rel_path
                if p.exists() and p.is_file():
                    artifacts.append({
                        "file_name": Path(rel_path).name,
                        "relative_path": rel_path,
                        "size_bytes": p.stat().st_size,
                    })
        else:
            artifacts = collect_artifacts()
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



def _safe_open(path, mode="r", *args, **kwargs):
    p = Path(path)
    p = (WORK_DIR / p).resolve() if not p.is_absolute() else p.resolve()

    # Only allow reads from /work/inputs
    if "r" in mode and not p.is_relative_to(INPUTS_DIR):
        raise PermissionError(f"Read denied outside inputs: {p}")

    # Only allow writes to /work/outputs
    if any(flag in mode for flag in ("w", "a", "x", "+")) and not p.is_relative_to(OUTPUTS_DIR):
        raise PermissionError(f"Write denied outside outputs: {p}")

    return open(p, mode, *args, **kwargs)

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

            # scalars
            "str": str, "int": int, "float": float, "bool": bool,

            "open": _safe_open,
        },

        # convenience
        "math": __import__("math"),
        "json": __import__("json"),
        "re": __import__("re"),
    }

    local_vars = {}

    output_buffer = io.StringIO()
    last_value_holder = {"value": None}

    def custom_displayhook(value):
        if value is not None:
            last_value_holder["value"] = value
            print(value)

    try:
        tree = ast.parse(code, mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            # The last statement is an expression, not an assignment or function call
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
            "status": "completed"
        }
        with open(OUTPUTS_DIR / "result.json", "w") as f:
            json.dump(final_state, f, default=str)

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


def collect_artifacts() -> List[Dict[str, Any]]:
    artifacts = []
    for p in OUTPUTS_DIR.glob("*"):
        if p.is_file():
            artifacts.append({
                "file_name": p.name,
                "relative_path": f"outputs/{p.name}",
                "size_bytes": p.stat().st_size,
            })
    return artifacts


if __name__ == "__main__":
    main()



#22