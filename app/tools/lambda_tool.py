import base64
import logging
import os
import queue
import re as _re
import tempfile
import time
import traceback
from dataclasses import dataclass
from multiprocessing import  Queue
from typing import Any, Dict, List, Optional, Type
import multiprocessing as mp
import sys


from pydantic.v1 import BaseModel, Field, conint
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import full_write_guard



import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Prints, but never reads 'printed' variable.*",
    category=SyntaxWarning,
    module="RestrictedPython.compile"
)


try:
    from langchain_core.tools import BaseTool as _LCBaseTool
except Exception:
    _LCBaseTool = object  # fallback so class can still be defined/imported

logger = logging.getLogger(__name__)

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import pandas as _pd
except Exception:
    _pd = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
except Exception:
    _plt = None

try:
    import requests as _requests
except Exception:
    _requests = None

try:
    from bs4 import BeautifulSoup as _BeautifulSoup  # noqa: F401
    import bs4 as _bs4
except Exception:
    _bs4 = None

try:
    import torch as _torch
except Exception:
    _torch = None


# -------------------- Whitelist modules exposed to sandbox --------------------
_ALLOWED_MODULES: Dict[str, Any] = {
    "re": __import__("re"),
    "json": __import__("json"),
    "math": __import__("math"),
}

_SAFE_EXTRA_BUILTINS = {
    "sum": sum,
    "range": range,
    "len": len,
    "min": min,
    "max": max,
    "abs": abs,
    "all": all,
    "any": any,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "map": map,
    "filter": filter,}


if _np is not None:
    _ALLOWED_MODULES["numpy"] = _np
if _pd is not None:
    _ALLOWED_MODULES["pandas"] = _pd
if _plt is not None:
    _ALLOWED_MODULES["matplotlib"] = _plt

_FUNCTIONS: Dict[str, Any] = {}


def register_lambda_function(name: str, fn: Any) -> None:
    """Register a pre-audited function callable from mode='function'."""
    if not callable(fn):
        raise TypeError("fn must be callable")
    _FUNCTIONS[name] = fn

def _safe_getattr(obj, name, default=None):
    # block dunder/private attrs
    if name.startswith("__") or name.endswith("__") or name.startswith("_"):
        raise AttributeError(name)
    return getattr(obj, name, default)


# -------------------- Input schema --------------------
class PythonScriptToolInput(BaseModel):
    # script optional to allow mode='function' without providing code
    script: Optional[str] = Field(
        "",
        description="Python code to execute (used when mode='code'). If mode='function', this can be empty."
    )

    # execution modes + call interface
    mode: str = Field("code", description="One of: 'code' or 'function'.")
    function: Optional[str] = Field(None, description="Registered function name (for mode='function').")
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    # Resource & behavior knobs (defaults are conservative)
    timeout_s: conint(ge=1, le=600) = Field(10, description="Max execution time in seconds.")
    memory_mb: conint(ge=64, le=4096) = Field(512, description="Address space cap (best-effort).")
    allow_network: bool = Field(False, description="Enable fetch() network access.")
    max_stdout_chars: conint(ge=0, le=200_000) = Field(20_000, description="Truncate printed output.")
    capture_plots: bool = Field(True, description="Capture matplotlib figures into artifacts.")
    callable_fn: Optional[Any] = Field(None, description="Direct callable to execute in function mode.")


    @staticmethod
    def _validate_mode(v: str) -> str:
        if v not in {"code", "function"}:
            raise ValueError("mode must be 'code' or 'function'")
        return v


@dataclass
class _JobConfig:
    code: str
    mode: str
    function: Optional[str]
    args: List[Any]
    kwargs: Dict[str, Any]
    timeout_s: int
    memory_mb: int
    allow_network: bool
    max_stdout_chars: int
    capture_plots: bool
    callable_fn: Optional[Any] = None

def _strip_imports(code: str) -> str:
    """Remove 'import x' and 'from x import y' lines to force sandbox importer."""
    code = _re.sub(
        r'^\s*import\s+[a-zA-Z0-9_.,\s]+(?:\s+as\s+\w+)?\s*$',
        '# stripped import',
        code,
        flags=_re.MULTILINE,
    )
    code = _re.sub(
        r'^\s*from\s+[a-zA-Z0-9_.]+\s+import\s+[a-zA-Z0-9_*,\s]+(?:\s+as\s+\w+)?\s*$',
        '# stripped from-import',
        code,
        flags=_re.MULTILINE,
    )
    return code


def _make_globals(cfg: _JobConfig, scratch_dir: str) -> Dict[str, Any]:
    """Build RestrictedPython globals with hardened importer + helpers."""
    rb = safe_builtins.copy()
    rb.update(_SAFE_EXTRA_BUILTINS)


    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = (name or "").split(".")[0]
        if root in _ALLOWED_MODULES:
            return _ALLOWED_MODULES[root]
        raise ImportError(f"Module '{name}' is not allowed in sandbox")

    rb["__import__"] = _restricted_import

    def fetch(url: str, as_what: str = "text", timeout: int = 15):
        if not cfg.allow_network:
            raise PermissionError("Network access is disabled.")
        if _requests is None:
            raise RuntimeError("requests is unavailable")
        resp = _requests.get(url, timeout=timeout, headers={"User-Agent": "RAGneticLambda/1.0"})
        resp.raise_for_status()
        if as_what == "json":
            return resp.json()
        if as_what == "html":
            if _bs4 is None:
                raise RuntimeError("bs4 is unavailable")
            parser = "lxml" if _bs4 and getattr(_bs4, "__name__", None) else "html.parser"
            try:
                return _bs4.BeautifulSoup(resp.text, parser)
            except Exception:
                return _bs4.BeautifulSoup(resp.text, "html.parser")
        return resp.text

    def device() -> str:
        if _torch is not None and _torch.cuda.is_available():
            return "cuda"
        return "cpu"

    g: Dict[str, Any] = {
        "__builtins__": rb,
        "_print_": PrintCollector,  # factory for RestrictedPython's collector

        "_getattr_": _safe_getattr,
        "_getitem_": default_guarded_getitem,  # safe item access (x[y])
        "_getiter_": default_guarded_getiter,  # safe iteration (for x in ...)
        "_write_": full_write_guard,  # prevent arbitrary writes

        # preload whitelisted modules…
        **_ALLOWED_MODULES,

        # helpers
        "fetch": fetch,
        "device": device,
    }

    if _np is not None:
        g["np"] = _np
    if _pd is not None:
        g["pd"] = _pd
    if _plt is not None:
        g["plt"] = _plt

    return g


def _capture_plots(scratch_dir: str) -> List[Dict[str, Any]]:
    if _plt is None:
        return []
    artifacts: List[Dict[str, Any]] = []
    for i, num in enumerate(_plt.get_fignums()):
        fig = _plt.figure(num)
        out = os.path.join(scratch_dir, f"plot_{i+1}.png")
        fig.savefig(out, bbox_inches="tight")
        with open(out, "rb") as f:
            data_b64 = base64.b64encode(f.read()).decode()
        artifacts.append(
            {"kind": "plot", "name": os.path.basename(out), "data_b64": data_b64, "path": out}
        )
    _plt.close("all")
    return artifacts


def _summarize(rv: Any, stdout: str, artifacts: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    if stdout:
        parts.append(f"Printed output ({len(stdout)} chars).")
    if rv is not None:
        t = type(rv).__name__
        parts.append(f"Return value of type {t}.")
        if _pd is not None and hasattr(rv, "head"):
            try:
                parts.append("Preview:\n" + rv.head(5).to_string())
            except Exception:
                pass
    if artifacts:
        parts.append(f"Generated {len(artifacts)} artifact(s).")
    return " ".join(parts) or "Execution completed."


def _worker(cfg: _JobConfig, result_q: Queue):
    """Executes RestrictedPython code or registered function in a subprocess."""
    scratch_dir = tempfile.mkdtemp(prefix="lambda_", dir=tempfile.gettempdir())
    t0 = time.time()
    stdout = ""
    return_value = None
    used_gpu = False
    artifacts: List[Dict[str, Any]] = []

    try:
        # Best-effort memory cap (Unix only)
        try:
            import resource
            limit = cfg.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass

        safe_globals = _make_globals(cfg, scratch_dir)
        local_ns: Dict[str, Any] = {}

        if cfg.mode == "function":
            # Prefer direct callable if provided
            if cfg.callable_fn is not None:
                if not callable(cfg.callable_fn):
                    raise TypeError("callable_fn must be callable")
                return_value = cfg.callable_fn(*cfg.args, **cfg.kwargs)
            else:
                fname = cfg.function or ""
                fn = _FUNCTIONS.get(fname)
                if fn is None:
                    raise ValueError(f"Function '{fname}' is not registered")
                return_value = fn(*cfg.args, **cfg.kwargs)

            if _torch is not None and _torch.cuda.is_available():
                used_gpu = True



        elif cfg.mode == "code":
            # ensure print capturing is enabled
            safe_globals["_print_"] = PrintCollector
            cleaned = _strip_imports(cfg.code or "")
            byte_code = compile_restricted(cleaned, filename="<sandbox>", mode="exec")
            exec(byte_code, safe_globals, local_ns)
            printed_text = ""
            # 1) most common: 'printed' variable holding a collector with .getvalue()
            obj = local_ns.get("printed", None)
            if obj is not None:
                try:
                    if hasattr(obj, "getvalue"):
                        printed_text = obj.getvalue()
                    else:
                        printed_text = str(obj)
                except Exception:
                    pass
            # 2) sometimes there's a callable '_print' that returns the buffer when called
            if not printed_text:
                maybe_fn = local_ns.get("_print", None)
                try:
                    if callable(maybe_fn):
                        printed_text = maybe_fn()
                except Exception:
                    pass
            # 3) last resort: call a local '_print_' if it exists and is callable
            if not printed_text:
                maybe_fn2 = local_ns.get("_print_", None)
                try:
                    if callable(maybe_fn2):
                        printed_text = maybe_fn2()
                except Exception:
                    pass
            stdout = (printed_text or "")[: cfg.max_stdout_chars]


            # If a main() exists, call it with args/kwargs
            if "main" in local_ns and callable(local_ns["main"]):
                return_value = local_ns["main"](*cfg.args, **cfg.kwargs)
                if _torch is not None and _torch.cuda.is_available():
                    used_gpu = True
        else:
            raise ValueError("Unsupported mode (expected 'code' or 'function')")

        if cfg.capture_plots:
            artifacts = _capture_plots(scratch_dir)

        runtime_ms = int((time.time() - t0) * 1000)

        payload = {
            "ok": True,
            "summary": _summarize(return_value, stdout, artifacts),
            "stdout": stdout,
            "return_value": return_value,
            "artifacts": artifacts,
            "runtime_ms": runtime_ms,
            "used_gpu": used_gpu,
            "error_type": None,
            "error_message": None,
            "traceback": None,
        }
        try:
            result_q.put(payload)
        except Exception:
            payload["return_value"] = f"<unserializable:{type(return_value).__name__}>"
            result_q.put(payload)


    except Exception:
        tb = traceback.format_exc()
        runtime_ms = int((time.time() - t0) * 1000)
        result_q.put(
            {
                "ok": False,
                "summary": "Execution failed.",
                "stdout": "",
                "return_value": None,
                "artifacts": artifacts,
                "runtime_ms": runtime_ms,
                "used_gpu": used_gpu,
                "error_type": "ExecutionError",
                "error_message": "See traceback.",
                "traceback": tb,
            }
        )


class LambdaTool(_LCBaseTool):
    """
    Single class usable everywhere:
      • As a LangChain tool (inherits BaseTool when available)
      • Directly via .run(script=..., **kwargs) for workflows/orchestrations
    """

    # Annotate these to satisfy Pydantic v2's field override rules
    name: str = "lambda_tool"
    description: str = (
        "Executes Python code (RestrictedPython) or calls registered functions in a sandboxed subprocess. "
        "Whitelist modules: re, json, math (+ numpy/pandas/matplotlib if installed). "
        "Use print() for stdout. Optional helpers: fetch(url, as_what='text'|'html'|'json'), device(). "
        "Returns structured result with stdout, artifacts, and summary."
    )
    args_schema: Type[BaseModel] = PythonScriptToolInput

    # LangChain entrypoints delegate to .run()
    def _run(self, **kwargs) -> Dict[str, Any]:
        return self.run(**kwargs)

    async def _arun(self, **kwargs) -> Dict[str, Any]:
        return self.run(**kwargs)

    # Direct entrypoint for workflows/engine; script may be omitted in function mode
    def run(self, script: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        allowed_keys = {
            "mode",
            "function",
            "args",
            "kwargs",
            "timeout_s",
            "memory_mb",
            "allow_network",
            "max_stdout_chars",
            "capture_plots",
            "callable_fn",
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed_keys}

        # auto-pick function mode if user passed function or callable
        if (filtered.get("callable_fn") or filtered.get("function")) and not filtered.get("mode"):
            filtered["mode"] = "function"

        # If script is None, allow schema default "" (valid for function mode)
        inp = PythonScriptToolInput(script=script if script is not None else "", **filtered)

        cfg = _JobConfig(
            code=inp.script or "",
            mode=inp.mode,
            function=inp.function,
            args=inp.args,
            kwargs=inp.kwargs,
            timeout_s=inp.timeout_s,
            memory_mb=inp.memory_mb,
            allow_network=inp.allow_network,
            max_stdout_chars=inp.max_stdout_chars,
            capture_plots=inp.capture_plots,
            callable_fn=inp.callable_fn,

        )
        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        result_q = ctx.Queue(maxsize=1)
        p = ctx.Process(target=_worker, args=(cfg, result_q), daemon=True)
        p.start()
        p.join(timeout=cfg.timeout_s)

        if p.is_alive():
            try:
                p.terminate()
            finally:
                p.join(1)
            logger.warning("Script timed out after %ss", cfg.timeout_s)
            return {
                "ok": False,
                "summary": "Timed out",
                "stdout": "",
                "return_value": None,
                "artifacts": [],
                "runtime_ms": cfg.timeout_s * 1000,
                "used_gpu": False,
                "error_type": "TimeoutError",
                "error_message": f"Exceeded {cfg.timeout_s}s",
                "traceback": None,
            }

        try:
            res = result_q.get_nowait()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Sandbox result: %s", res)
            else:
                logger.info("Sandbox completed (ok=%s, runtime_ms=%s)", res.get("ok"), res.get("runtime_ms"))
            return res
        except queue.Empty:
            logger.error("No result from sandbox process")
            return {
                "ok": False,
                "summary": "Execution crashed",
                "stdout": "",
                "return_value": None,
                "artifacts": [],
                "runtime_ms": inp.timeout_s * 1000,
                "used_gpu": False,
                "error_type": "Crash",
                "error_message": "No result from worker",
                "traceback": None,
            }