import base64
import enum
import json
import logging
import math
import os
import re
import sys
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, date

import ast
from uuid import UUID

import numpy as np

# Configure a basic logger for structured output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger(__name__)


try:
    import pandas as pd
except Exception:
    pd = None

# Tunables
MAX_DEPTH     = int(os.getenv("JSONIFY_MAX_DEPTH", "6"))
MAX_ITEMS     = int(os.getenv("JSONIFY_MAX_ITEMS", "2000"))
PREVIEW_ROWS  = int(os.getenv("JSONIFY_PREVIEW_ROWS", "20"))
MAX_STRING    = int(os.getenv("JSONIFY_MAX_STRING", "20000"))
MAX_BYTES_B64 = int(os.getenv("JSONIFY_MAX_BYTES_B64", "32768"))

def _finite_number(x):
    # Replace NaN/Inf with None to keep strict JSON (allow_nan=False)
    return x if (isinstance(x, (int,)) or (isinstance(x, float) and math.isfinite(x))) else None

def _jsonify(obj, _depth=0, _seen=None):
    if _seen is None:
        _seen = set()
    if _depth > MAX_DEPTH:
        return {"__type__": "truncated", "reason": "max_depth"}

    # ---- Fast path: primitives & immutable scalars (no cycle tracking) ----
    if obj is None or isinstance(obj, (bool, int)):
        return obj
    if isinstance(obj, float):
        return _finite_number(obj)
    if isinstance(obj, str):
        if len(obj) > MAX_STRING:
            return {"__type__": "string", "value": obj[:MAX_STRING], "truncated": True, "length": len(obj)}
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        if len(b) > MAX_BYTES_B64:
            b = b[:MAX_BYTES_B64]
            return {"__type__": "bytes", "b64": base64.b64encode(b).decode("ascii"), "truncated": True}
        return {"__type__": "bytes", "b64": base64.b64encode(b).decode("ascii")}
    # numpy scalar/dtype (still treat as scalars)
    if isinstance(obj, np.dtype):
        return str(obj)
    if isinstance(obj, np.generic):
        # Preserve non-float numpy scalars too (e.g., integers)
        val = obj.item()
        if isinstance(val, float):
            return _finite_number(val)
        return val

    # ---- From here on, track cycles (containers / complex objects) ----
    oid = id(obj)
    if oid in _seen:
        return {"__type__": "cycle_ref"}
    _seen.add(oid)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        data = obj
        info = {"__type__": "ndarray", "shape": list(data.shape), "dtype": str(data.dtype)}
        flat = data.ravel()
        n = flat.size
        if n > MAX_ITEMS:
            # Preview as simple scalars
            preview = []
            for v in flat[:MAX_ITEMS]:
                if isinstance(v, np.generic):
                    v = v.item()
                if isinstance(v, float):
                    v = _finite_number(v)
                preview.append(v)
            info["preview"] = preview
            info["truncated"] = True
        else:
            # Safe full value
            def _to_scalar(x):
                if isinstance(x, np.generic):
                    x = x.item()
                if isinstance(x, float):
                    x = _finite_number(x)
                return x
            info["value"] = [[_to_scalar(x) for x in row] for row in data.tolist()]
        return info

    # pandas
    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            rows, cols = obj.shape
            out = {
                "__type__": "dataframe",
                "shape": [int(rows), int(cols)],
                "dtypes": {c: str(obj.dtypes[c]) for c in obj.columns},
            }
            preview = obj.head(PREVIEW_ROWS).to_dict(orient="records")
            out["preview"] = _jsonify(preview, _depth=_depth + 1, _seen=_seen)
            if rows > PREVIEW_ROWS:
                out["truncated"] = True
            return out
        if isinstance(obj, pd.Series):
            s = obj
            out = {
                "__type__": "series",
                "name": getattr(s, "name", None),
                "dtype": str(s.dtype),
            }
            vals = s.head(PREVIEW_ROWS).to_dict()
            out["preview"] = _jsonify(vals, _depth=_depth + 1, _seen=_seen)
            if len(s) > PREVIEW_ROWS:
                out["truncated"] = True
            return out
        if isinstance(obj, pd.Index):
            vals = obj[:PREVIEW_ROWS].tolist()
            return {
                "__type__": "index",
                "dtype": str(obj.dtype),
                "preview": _jsonify(vals, _depth=_depth + 1, _seen=_seen),
                "truncated": len(obj) > PREVIEW_ROWS,
            }

    # mappings
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= MAX_ITEMS:
                out["__truncated__"] = True
                break
            out[str(k)] = _jsonify(v, _depth=_depth + 1, _seen=_seen)
        return out

    # iterables
    if isinstance(obj, (list, set, frozenset)) or hasattr(obj, "__iter__"):
        out_list = []
        for i, v in enumerate(obj):
            if i >= MAX_ITEMS:
                out_list.append({"__type__": "truncated", "reason": "max_items"})
                break
            out_list.append(_jsonify(v, _depth=_depth + 1, _seen=_seen))
        if isinstance(obj, tuple):
            return {"__type__": "tuple", "value": out_list}
        return out_list

    # fallback
    try:
        json.dumps(obj, allow_nan=False)
        return obj
    except Exception:
        return {"__type__": "repr", "value": repr(obj)}


# Define sandbox paths
WORK_DIR = Path("/work")
INPUTS_DIR = WORK_DIR / "inputs"
OUTPUTS_DIR = WORK_DIR / "outputs"
REQUEST_FILE = WORK_DIR / "request.json"

SANDBOX_ROOT = WORK_DIR

ALLOWED_MODULES = {
    # core math/data viz
    "math", "statistics", "random", "numpy", "pandas", "matplotlib", "seaborn",

    # ML / scientific stack
    "sklearn", "scipy",

    # stdlib commonly used in EDA/scripts
    "json", "re", "io", "itertools", "collections", "datetime", "time", "pathlib", "base64","decimal", "uuid", "enum", "typing",

    # keep if you still use it anywhere
    "textblob", "csv",
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
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


        if not REQUEST_FILE.exists():
            raise FileNotFoundError(f"Request file not found: {REQUEST_FILE}")

        with open(REQUEST_FILE, 'r') as f:
            request_data = json.load(f)

        mode = request_data.get("mode")
        payload = request_data.get("payload", {})

        if mode == "code":
            code_str = request_data.get("code") or payload.get("code", "")
            execute_code_mode(code_str)

        # elif mode == "function":
            # fn_name = request_data.get("function_name") or payload.get("function_name")
            # fn_args = request_data.get("function_args") or payload.get("function_args", {}) or {}

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

"""

def execute_function_mode(function_name: str, args: Dict[str, Any]):
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
"""

def _safe_open(path, mode="r", *args, **kwargs):
    """Allow read/write anywhere under /work."""
    p = Path(path)
    p = (WORK_DIR / p).resolve() if not p.is_absolute() else p.resolve()

    if not p.is_relative_to(WORK_DIR):
        raise PermissionError(f"Access denied outside sandbox root: {p}")

    return open(p, mode, *args, **kwargs)


def _normalize_code(code: str) -> str:
    """
    Make LLM-provided code executable:
    - Strip ``` fences (optionally with language tag).
    - Convert literal escape sequences like '\\n', '\\t', '\\r' to real characters
      if the string appears to be single-line or dominated by escaped newlines.
    """
    if not code:
        return code

    s = code.strip()

    # Strip Markdown code fences if present
    if s.startswith("```") and s.endswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)  # remove leading fence
        s = re.sub(r"\s*```$", "", s)                # remove trailing fence

    # If we see literal "\n" but very few real newlines, decode escapes
    has_literal_newlines = "\\n" in s
    real_newline_count = s.count("\n")
    if has_literal_newlines and real_newline_count <= 1:
        try:
            s = bytes(s, "utf-8").decode("unicode_escape")
        except Exception:
            s = s.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

    return s



def execute_code_mode(code: str):
    """Executes raw Python code in a restricted sandbox and saves the result."""
    logger.info("Executing code in 'code' mode.")

    # 1) Normalize LLM-provided code so escaped newlines don’t break parsing
    code = _normalize_code(code)

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

    env = dict(safe_globals)
    env["_jsonify"] = _jsonify
    env["np"] = np
    if pd is not None:
        env["pd"] = pd

    output_buffer = io.StringIO()

    try:
        tree = ast.parse(code, mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body[-1].value
            if not (isinstance(last_expr, ast.Call) and getattr(last_expr.func, "id", None) == "print"):
                expression_source = ast.unparse(last_expr)
                code_to_execute = code + (
                        "\nimport json\n"
                        "print(json.dumps(_jsonify(" + expression_source + "), indent=2, allow_nan=False))"
                )
            else:
                code_to_execute = code
        else:
            code_to_execute = code

    except Exception as e:
        logger.warning(f"Failed to parse code for auto-printing. Executing as-is. Error: {e}")
        code_to_execute = code

    try:
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            exec(code_to_execute, env, env)

        stdout_output = output_buffer.getvalue()

        # cap very large outputs
        MAX_STDOUT = int(os.getenv("MAX_STDOUT_CHARS", "200000"))
        if len(stdout_output) > MAX_STDOUT:
            stdout_output = stdout_output[:MAX_STDOUT] + "\n…[truncated]"
        stdout_output = stdout_output.strip()

        # try to parse machine-readable JSON, keep None if not JSON
        output_json = None
        try:
            output_json = json.loads(stdout_output)
        except Exception:
            pass

        final_state = {
            "output": stdout_output,
            "output_json": output_json,
            "status": "completed",
            "result_files": _collect_result_files()
        }
        with open(OUTPUTS_DIR / "result.json", "w") as f:
            json.dump(final_state, f, default=str)

    except Exception:
        tb = traceback.format_exc()
        logger.error("Code execution failed", extra={"details": {"traceback": tb}})
        raise



def _collect_result_files():
    """Auto-discover all files in /outputs and return structured metadata."""
    result_files = []
    max_bytes = int(os.getenv("MAX_RESULT_FILE_BYTES", str(50 * 1024 * 1024)))  # 50MB default
    for p in OUTPUTS_DIR.glob("*"):
        if not p.is_file() or p.name in ("result.json", "error.json"):
            continue
        size = p.stat().st_size
        if size > max_bytes:
            logger.warning(f"Skipping large artifact {p.name} ({size} bytes > {max_bytes})")
            continue
        result_files.append({
            "file_name": p.name,
            "relative_path": f"outputs/{p.name}",
            "size_bytes": size,
        })
    return result_files


if __name__ == "__main__":
    main()
