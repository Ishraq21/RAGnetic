# lambda_tool.py

import os
import base64
import hashlib
import json as _json
import logging
import os
import pathlib
import queue
import re as _re
import tempfile
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Queue
from typing import Any, Dict, List, Optional, Type
import multiprocessing as mp
import sys
import os
import json as _json
import hashlib
import base64
import time
from typing import Dict, Any
import textwrap

from langchain_core.runnables import RunnableConfig
from pydantic.v1 import BaseModel, Field, conint
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import full_write_guard
from RestrictedPython import utility_builtins


import warnings

from app.services.lambda_file_helper import get_temp_files_as_payload

warnings.filterwarnings(
    "ignore",
    message=r".*Prints, but never reads 'printed' variable.*",
    category=SyntaxWarning,
    module="RestrictedPython.compile"
)

try:
    # RP >= 6.0 or alt naming
    from RestrictedPython.Eval import default_guarded_unpack_sequence as _guarded_unpack
except Exception:
    try:
        from RestrictedPython.Eval import default_guarded_iter_unpack_sequence as _guarded_unpack
    except Exception:
        _guarded_unpack = None

if _guarded_unpack is None:
    def _iter_unpack_fallback(it, *args, **kwargs):
        """
        Compatible with RP call shapes:
          - (it, count)
          - (it, guards, count)
          - (it, guards)  # no fixed count (UNPACK_EX paths) → return an iterator
        """
        # find the last int among args as the count (if any)
        count = next((a for a in reversed(args) if isinstance(a, int)), None)

        # No explicit count → hand back an iterator (used by UNPACK_EX/starred cases)
        if count is None:
            return iter(it)

        it = iter(it)
        items = []
        for _ in range(count):
            try:
                items.append(next(it))
            except StopIteration:
                raise ValueError(f"not enough values to unpack (expected {count})")
        # ensure there are no extras for fixed-size unpack
        try:
            next(it)
            raise ValueError(f"too many values to unpack (expected {count})")
        except StopIteration:
            pass
        return tuple(items)
else:
    _iter_unpack_fallback = _guarded_unpack


try:
    from langchain_core.tools import BaseTool as _LCBaseTool, BaseTool
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

_ALLOWED_MODULES: Dict[str, Any] = {
    "re": __import__("re"),
    "json": __import__("json"),
    "math": __import__("math"),
    "io": __import__("io"),
    "datetime": __import__("datetime"),
    "random": __import__("random"),
    "statistics": __import__("statistics"),
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
    "filter": filter,
}

if _np is not None:
    _ALLOWED_MODULES["numpy"] = _np
if _pd is not None:
    _ALLOWED_MODULES["pandas"] = _pd
if _plt is not None:
    _ALLOWED_MODULES["matplotlib"] = _plt

_FUNCTIONS: Dict[str, Any] = {}

UNICODE_SPACES = "\u00A0\u1680\u180E\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u202F\u205F\u3000"

def _normalize_script(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _extract_fenced_code(s)

    if "\\n" in s and "\n" not in s:
        s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ")


    s = s.lstrip("\ufeff")  # strip BOM if present
    # normalize all odd spaces to ASCII space
    s = s.translate({ord(c): " " for c in UNICODE_SPACES})
    s = s.replace("\t", "    ")
    s = textwrap.dedent(s)
    if "\n" not in s and "\\n" in s:
        try:
            # robustly unescape (handles \n, \t, and \uXXXX safely)
            s = s.encode("utf-8").decode("unicode_escape")
        except Exception:
            s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ")

    # If the whole thing is wrapped in a single pair of quotes (JSON-ish), unwrap once.
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        inner = s[1:-1]
        if "\n" not in inner and "\\n" in inner:
            try:
                inner = inner.encode("utf-8").decode("unicode_escape")
            except Exception:
                inner = inner.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ")
            s = inner

    return s.strip("\n")


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


def _safe_open_factory(base_dir: str, upload_roots: List[str]):
    """
    Return a sandboxed open() confined to base_dir.
    On read: if the file isn't in base_dir, try to auto-stage it by basename
    from any discovered upload_roots.
    """
    def _copy_from_uploads_if_present(target_full: str, requested_name: str) -> None:
        base = os.path.basename(requested_name)
        if not base or not upload_roots:
            return
        candidates: List[str] = []
        # accept exact 'base' or '<uuid>_base'
        import fnmatch
        uu_prefixed = f"*_{base}"
        for root in upload_roots:
            try:
                for dirpath, _dirs, files in os.walk(root):
                    for fn in files:
                        if fn == base or fnmatch.fnmatch(fn, uu_prefixed):
                            candidates.append(os.path.join(dirpath, fn))
            except Exception:
                pass
        if not candidates:
            return
        # newest by mtime wins
        src = max(candidates, key=lambda p: os.path.getmtime(p))
        os.makedirs(os.path.dirname(target_full), exist_ok=True)
        with open(src, "rb") as s, open(target_full, "wb") as d:
            d.write(s.read())
        try:
            logger.info("LambdaTool auto-staged '%s' from uploads: %s", base, src)
        except Exception:
            pass

    def _safe_open(path, mode="r", *args, **kwargs):
        if any(c in mode for c in ("a", "+")):
            raise PermissionError("append/update modes are not allowed")

        full = os.path.realpath(os.path.join(base_dir, path))
        base_abs = os.path.realpath(base_dir) + os.sep
        if not full.startswith(base_abs):
            raise PermissionError("path escapes sandbox")
        if mode not in ("r", "rb", "w", "wb"):
            raise PermissionError("unsupported mode")

        if mode in ("r", "rb") and not os.path.exists(full):
            _copy_from_uploads_if_present(full, path)

        return open(full, mode, *args, **kwargs)

    return _safe_open



_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".ragnetic_lambda_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _stable_dumps(obj):
    return _json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _cache_key(cfg: "_JobConfig"):
    basis = {
        "mode": cfg.mode,
        "code": cfg.code,
        "function": cfg.function,
        "args": cfg.args,
        "kwargs": cfg.kwargs,
        "allow_network": cfg.allow_network,
        "libs": sorted(list(_ALLOWED_MODULES.keys())),
        "files": [
            {"name": f.get("name"),
             "sha256": hashlib.sha256(base64.b64decode(f.get("data_b64", ""))).hexdigest()
             } for f in cfg.files
        ],
        "capture_plots": cfg.capture_plots,
        "max_stdout_chars": cfg.max_stdout_chars,
        "strip_imports": cfg.strip_imports,
    }
    h = hashlib.sha256(_stable_dumps(basis).encode("utf-8")).hexdigest()
    return os.path.join(_CACHE_DIR, f"{h}.json")


def _cache_get(cfg: "_JobConfig"):
    if not cfg.enable_cache:
        return None
    path = _cache_key(cfg)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = _json.load(fh)
        if cfg.cache_ttl_s and (time.time() - obj.get("_ts", 0) > cfg.cache_ttl_s):
            return None
        return obj.get("payload")
    except Exception:
        return None


def _cache_put(cfg: "_JobConfig", payload: Dict[str, Any]):
    if not cfg.enable_cache:
        return
    path = _cache_key(cfg)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump({"_ts": time.time(), "payload": payload}, fh, ensure_ascii=False)
    except Exception:
        pass


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

    # Resource & behavior knobs
    timeout_s: conint(ge=1, le=600) = Field(10, description="Max execution time in seconds.")
    memory_mb: conint(ge=64, le=4096) = Field(512, description="Address space cap (best-effort).")
    allow_network: bool = Field(False, description="Enable fetch() network access.")
    max_stdout_chars: conint(ge=0, le=200_000) = Field(20_000, description="Truncate printed output.")
    capture_plots: bool = Field(True, description="Capture matplotlib figures into artifacts.")
    callable_fn: Optional[Any] = Field(None, description="Direct callable to execute in function mode.")

    files: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List like [{'name': 'data.csv', 'data_b64': '...'}] to stage into sandbox workdir."
    )
    preview_bytes: conint(ge=256, le=200_000) = Field(
        4096, description="Max bytes of preview text for small artifacts (text/csv/json/stdout)."
    )
    enable_cache: bool = Field(True, description="Use disk cache for identical runs.")
    cache_ttl_s: Optional[conint(ge=1, le=7 * 24 * 3600)] = Field(
        None, description="Optional TTL seconds; if None, cache never expires."
    )

    strip_imports: bool = Field(True,
                                description="If True, strip 'import' statements before execution (preloaded modules only).")

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
    callable_fn: Optional[Any]
    files: List[Dict[str, str]]
    preview_bytes: int
    enable_cache: bool
    cache_ttl_s: Optional[int]
    strip_imports: bool = True   # ← no comma, give default



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


def _stage_files(files, scratch_dir):
    staged = []
    for f in files:
        name = f.get("name")
        b64 = f.get("data_b64")
        if not name or not b64:
            continue

        base = os.path.basename(name)

        # Always write the given basename:
        primary_out = os.path.join(scratch_dir, base)
        os.makedirs(os.path.dirname(primary_out), exist_ok=True)
        raw = base64.b64decode(b64)
        with open(primary_out, "wb") as fh:
            fh.write(raw)
        staged.append(primary_out)

        # If the basename looks like "<uuid>_original.ext", also write "original.ext"
        m = _re.match(r"^[0-9a-fA-F-]+_(.+)$", base)
        if m:
            alias = m.group(1)
            alias_out = os.path.join(scratch_dir, alias)
            if alias_out != primary_out:
                with open(alias_out, "wb") as fh:
                    fh.write(raw)
                staged.append(alias_out)

    return staged

def _short_code_excerpt(code: str, lineno: Optional[int], context: int = 2) -> Optional[str]:
    if not code or not lineno or lineno <= 0:
        return None
    lines = code.splitlines()
    i = lineno - 1
    start = max(0, i - context)
    end = min(len(lines), i + context + 1)
    excerpt = []
    for idx in range(start, end):
        prefix = "→ " if idx == i else "  "
        excerpt.append(f"{prefix}{idx+1:4d}: {lines[idx]}")
    return "\n".join(excerpt)

def _friendly_error_payload(err: Exception, code: str, scratch_dir: str, allowed_modules: List[str]) -> Dict[str, Any]:
    etype = type(err).__name__
    msg = str(err)
    hints: List[str] = []
    diagnostics: Dict[str, Any] = {"error_class": etype}
    blame_line = None

    # Try to find the sandbox line number
    if isinstance(err, SyntaxError):
        blame_line = getattr(err, "lineno", None)
        diagnostics["syntax_offset"] = getattr(err, "offset", None)
    else:
        try:
            tb = traceback.TracebackException.from_exception(err)
            for frame in reversed(tb.stack):
                if frame.filename == "<sandbox>":
                    blame_line = frame.lineno
                    break
        except Exception:
            pass

    if blame_line:
        diagnostics["line"] = blame_line
        excerpt = _short_code_excerpt(code, blame_line)
        if excerpt:
            diagnostics["code_excerpt"] = excerpt

    low = msg.lower()

    if isinstance(err, FileNotFoundError):
        hints += [
            "The file isn’t in the sandbox working directory.",
            "If you uploaded via chat, reference it by basename only (e.g., 'invoice_sample.txt').",
            "Or pass it to the tool call via the 'files' array; the sandbox will stage it for you."
        ]
        try:
            diagnostics["workdir_listing"] = sorted(os.listdir(scratch_dir))
        except Exception:
            pass

    if isinstance(err, PermissionError):
        hints += [
            "Only modes {'r','rb','w','wb'} are allowed.",
            "Paths must remain within the sandbox working directory.",
        ]

    if isinstance(err, (ImportError, ModuleNotFoundError)):
        # Extract module name if present
        import re
        m = re.search(r"No module named ['\"]([^'\"]+)['\"]", msg) or re.search(r"cannot import name ['\"]([^'\"]+)['\"]", msg)
        mod = m.group(1) if m else None
        if mod:
            diagnostics["module"] = mod
        hints += [
            "Third-party imports are restricted in the sandbox.",
            "Rewrite using built-in/allowed libraries, or ask the owner to allow-list/install this module.",
        ]
        diagnostics["allowed_modules"] = sorted(allowed_modules)

    if isinstance(err, NameError):
        hints += [
            "A symbol is used but not defined. Did you forget an import or variable assignment?",
        ]

    if isinstance(err, SyntaxError):
        hints += ["Fix the syntax at the highlighted line."]

    if not hints:
        hints.append("Review the error message and traceback; adjust code or inputs accordingly.")

    tb_text = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    return {
        "error_type": etype,
        "error_message": msg,
        "hints": hints,
        "diagnostics": diagnostics,
        "traceback_text": tb_text,
    }



def _mk_artifact(path, mime, preview_bytes=None):
    with open(path, "rb") as fh:
        raw = fh.read()
    art = {
        "kind": "file",
        "name": os.path.basename(path),
        "path": path,
        "mime": mime,
        "size": len(raw),
        "data_b64": base64.b64encode(raw).decode(),
    }
    if preview_bytes and mime in ("text/plain", "text/csv", "application/json"):
        try:
            art["preview_text"] = raw[:preview_bytes].decode(errors="replace")
        except Exception:
            pass
    return art




def _capture_plots(scratch_dir: str) -> List[Dict[str, Any]]:
    if _plt is None:
        return []
    artifacts: List[Dict[str, Any]] = []
    for i, num in enumerate(_plt.get_fignums()):
        fig = _plt.figure(num)
        out = os.path.join(scratch_dir, f"plot_{i + 1}.png")
        fig.savefig(out, bbox_inches="tight")
        with open(out, "rb") as f:
            data = f.read()
        artifacts.append(
            {
                "kind": "plot",
                "name": os.path.basename(out),
                "path": out,
                "mime": "image/png",
                "size": len(data),
                "data_b64": base64.b64encode(data).decode(),
            }
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


def _make_globals(cfg: _JobConfig, scratch_dir: str, upload_roots: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build RestrictedPython globals with hardened importer + helpers."""
    rb = safe_builtins.copy()
    rb.update(_SAFE_EXTRA_BUILTINS)
    rb.update(utility_builtins)

    # make sure RestrictedPython's tuple-unpack helpers are always present
    rb["_iter_unpack_sequence_"] = _iter_unpack_fallback
    rb["_unpack_sequence_"]      = _iter_unpack_fallback


    def list_files():
        return sorted(os.listdir(scratch_dir))

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
        "list_files": list_files,

        # sandbox info & APIs
        "workdir": scratch_dir,
        "open": _safe_open_factory(scratch_dir, upload_roots or []),
        "path_join": os.path.join,

    }

    g["_iter_unpack_sequence_"] = _iter_unpack_fallback
    g["_unpack_sequence_"]      = _iter_unpack_fallback

    # handy short aliases if available
    if _np is not None:
        g["np"] = _np
    if _pd is not None:
        g["pd"] = _pd
    if _plt is not None:
        g["plt"] = _plt

    try:
        if _pd is not None and not hasattr(_pd.DataFrame, "__guarded_setitem__"):
            def _df_guarded_setitem(self, key, value):
                # delegate to the real setter
                self.__setitem__(key, value)

            _pd.DataFrame.__guarded_setitem__ = _df_guarded_setitem

        # (optional) support .loc / .iloc assignment too
        try:
            from pandas.core.indexing import _LocIndexer, _iLocIndexer
            if not hasattr(_LocIndexer, "__guarded_setitem__"):
                def _idx_guarded_setitem(self, key, value):
                    self.__setitem__(key, value)

                _LocIndexer.__guarded_setitem__ = _idx_guarded_setitem
            if not hasattr(_iLocIndexer, "__guarded_setitem__"):
                _iLocIndexer.__guarded_setitem__ = _idx_guarded_setitem
        except Exception:
            # If pandas internals move, just skip indexer support
            pass
    except Exception:
        pass

    g.update(utility_builtins)

    if _guarded_unpack is not None:
        rb["_iter_unpack_sequence_"] = _guarded_unpack
        rb["_unpack_sequence_"] = _guarded_unpack
        g["_iter_unpack_sequence_"] = _guarded_unpack
        g["_unpack_sequence_"] = _guarded_unpack

    g["list_files"] = list_files
    return g

def _discover_upload_roots() -> List[str]:
    """
    Find likely chat-upload roots by walking up from this file and checking
    for '.ragnetic/.ragnetic_temp_clones/chat_uploads'.
    """
    roots: List[str] = []
    here = pathlib.Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        base = p if p.is_dir() else p.parent
        cand = base / ".ragnetic" / ".ragnetic_temp_clones" / "chat_uploads"
        try:
            if cand.exists() and cand.is_dir():
                roots.append(str(cand))
        except Exception:
            pass
    # de-dup while preserving order
    seen = set()
    uniq = []
    for r in roots:
        if r not in seen:
            uniq.append(r); seen.add(r)
    return uniq

def _worker(cfg: _JobConfig, result_q: Queue):
    """Executes RestrictedPython code or registered function in a subprocess."""
    scratch_dir = tempfile.mkdtemp(prefix="lambda_", dir=tempfile.gettempdir())
    t_wall_start = time.time()
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

        # Stage uploads
        logger.info("LambdaTool incoming files count=%d", len(cfg.files))

        _staged = _stage_files(cfg.files, scratch_dir)
        try:
            logger.info("LambdaTool staged %d file(s): %s", len(_staged), [os.path.basename(p) for p in _staged])
        except Exception:
            pass

        try:
            upload_roots = _discover_upload_roots()
        except Exception:
            upload_roots = []

        prev_cwd = os.getcwd()
        try:
            os.chdir(scratch_dir)
        except Exception:
            pass

        # Cache check
        cached = _cache_get(cfg)
        if cached:
            result_q.put(cached)
            return

        safe_globals = _make_globals(cfg, scratch_dir, upload_roots)
        local_ns: Dict[str, Any] = {}

        if cfg.mode == "function":
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
            safe_globals["_print_"] = PrintCollector
            cleaned = _strip_imports(cfg.code or "") if cfg.strip_imports else (cfg.code or "")

            try:
                byte_code = compile_restricted(cleaned, filename="<sandbox>", mode="exec")
            except SyntaxError as e:
                msg_lower = str(e).lower()
                if ("unexpected indent" in msg_lower
                        or "expected an indented block" in msg_lower
                        or "line continuation character" in msg_lower):
                    # Rebuild from the original, normalize and re-wrap
                    original = cfg.code or ""
                    src = original

                    if src.startswith("if True:\n"):
                        lines = src.splitlines()
                        body = lines[1:]
                        body = [ln[4:] if ln.startswith("    ") else ln for ln in body]
                        src = "\n".join(body)

                    if "\n" not in src and "\\n" in src:
                        try:
                            src = src.encode("utf-8").decode("unicode_escape")
                        except Exception:
                            src = src.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ")

                    src = "\n".join(ln.lstrip() for ln in src.splitlines())
                    src = _strip_imports(src) if cfg.strip_imports else src
                    body = "\n".join(("    " + ln) if ln.strip() else ln for ln in src.splitlines())
                    repaired = "if True:\n" + body
                    byte_code = compile_restricted(repaired, filename="<sandbox>", mode="exec")
                else:
                    raise

            logger.info(
                "unpack helpers present? rb:%s g:%s",
                "_iter_unpack_sequence_" in safe_globals["__builtins__"],
                "_iter_unpack_sequence_" in safe_globals,
            )

            exec(byte_code, safe_globals, local_ns)

            # robust print capture across RestrictedPython variants
            printed_text = ""
            obj = local_ns.get("printed", None)
            if obj is not None:
                try:
                    if hasattr(obj, "getvalue"):
                        printed_text = obj.getvalue()
                    else:
                        printed_text = str(obj)
                except Exception:
                    pass
            if not printed_text:
                maybe_fn = local_ns.get("_print", None)
                try:
                    if callable(maybe_fn):
                        printed_text = maybe_fn()
                except Exception:
                    pass
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

            if return_value is None and _pd is not None:
                for _name in ("df", "dataframe", "result", "results"):
                    _obj = local_ns.get(_name)
                    try:
                        if isinstance(_obj, _pd.DataFrame):
                            return_value = _obj
                            break
                    except Exception:
                        pass

        else:
            raise ValueError("Unsupported mode (expected 'code' or 'function')")

        # Artifacts: plots
        if cfg.capture_plots:
            artifacts.extend(_capture_plots(scratch_dir))

        # Artifacts: stdout as text file
        if stdout:
            so_path = os.path.join(scratch_dir, "stdout.txt")
            with open(so_path, "w", encoding="utf-8") as fh:
                fh.write(stdout)
            artifacts.append(_mk_artifact(so_path, "text/plain", cfg.preview_bytes))

        # Artifacts: return value (df/json/text)
        if return_value is not None:
            try:
                if _pd is not None and isinstance(return_value, _pd.DataFrame):
                    df_path = os.path.join(scratch_dir, "dataframe.csv")
                    return_value.to_csv(df_path, index=False)
                    artifacts.append(_mk_artifact(df_path, "text/csv", cfg.preview_bytes))
                elif isinstance(return_value, (list, dict)):
                    j_path = os.path.join(scratch_dir, "return.json")
                    with open(j_path, "w", encoding="utf-8") as fh:
                        _json.dump(return_value, fh, ensure_ascii=False, indent=2)
                    artifacts.append(_mk_artifact(j_path, "application/json", cfg.preview_bytes))
                elif isinstance(return_value, (int, float, bool)):
                    t_path = os.path.join(scratch_dir, "return.txt")
                    with open(t_path, "w", encoding="utf-8") as fh:
                        fh.write(str(return_value))
                    artifacts.append(_mk_artifact(t_path, "text/plain", cfg.preview_bytes))

            except Exception:
                pass

        runtime_ms = int((time.time() - t_wall_start) * 1000)

        # metrics
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF)
            cpu_user = ru.ru_utime
            cpu_sys = ru.ru_stime
            max_rss_kb = getattr(ru, "ru_maxrss", 0)
        except Exception:
            cpu_user = cpu_sys = 0.0
            max_rss_kb = 0

        metrics = {
            "runtime_ms": runtime_ms,
            "cpu_user_s": cpu_user,
            "cpu_sys_s": cpu_sys,
            "peak_rss_kb": max_rss_kb,
            "used_gpu": used_gpu,
        }

        payload = {
            "ok": True,
            "summary": _summarize(return_value, stdout, artifacts),
            "stdout": stdout,
            "return_value": return_value,
            "artifacts": artifacts,
            "runtime_ms": runtime_ms,  # back-compat
            "metrics": metrics,
            "used_gpu": used_gpu,
            "error_type": None,
            "error_message": None,
            "traceback": None,
        }

        # Cache save
        _cache_put(cfg, payload)

        # Return
        result_q.put(payload)


    except Exception:

        err = sys.exc_info()[1]
        tb = traceback.format_exc()
        runtime_ms = int((time.time() - t_wall_start) * 1000)

        try:

            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF)
            cpu_user = ru.ru_utime
            cpu_sys = ru.ru_stime
            max_rss_kb = getattr(ru, "ru_maxrss", 0)

        except Exception:

            cpu_user = cpu_sys = 0.0

            max_rss_kb = 0

        metrics = {

            "runtime_ms": runtime_ms,
            "cpu_user_s": cpu_user,
            "cpu_sys_s": cpu_sys,
            "peak_rss_kb": max_rss_kb,
            "used_gpu": False,

        }

        # Build a friendly, structured error with hints/diagnostics

        fe = _friendly_error_payload(

            err=err,
            code=cfg.code or "",
            scratch_dir=scratch_dir,
            allowed_modules=list(_ALLOWED_MODULES.keys()),

        )

        # Also attach an error.txt artifact for the UI

        try:

            err_path = os.path.join(scratch_dir, "error.txt")
            with open(err_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "ERROR: "
                    + fe["error_type"]
                    + ": "
                    + fe["error_message"]
                    + "\n\nHints:\n- "
                    + "\n- ".join(fe["hints"])
                )

                if "diagnostics" in fe and fe["diagnostics"].get("code_excerpt"):
                    fh.write("\n\nCode excerpt:\n" + fe["diagnostics"]["code_excerpt"] + "\n")

                fh.write("\n\nTraceback:\n" + fe["traceback_text"])

            artifacts.append(_mk_artifact(err_path, "text/plain", preview_bytes=cfg.preview_bytes))

        except Exception:

            pass

        result_q.put(

            {
                "ok": False,
                "summary": f"{fe['error_type']}: {fe['error_message']}",
                "stdout": "",
                "return_value": None,
                "artifacts": artifacts,
                "runtime_ms": runtime_ms,
                "metrics": metrics,
                "used_gpu": False,
                "error_type": fe["error_type"],
                "error_message": fe["error_message"],
                "traceback": fe["traceback_text"],
                # Extra structure your UI/agent can surface:
                "hints": fe["hints"],
                "diagnostics": fe["diagnostics"],

            }

        )
def _extract_fenced_code(s: str) -> str:
    # If the user/LLM included ```python ... ``` fences, grab the inner code
    if "```" not in s:
        return s
    m = _re.search(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", s, flags=_re.DOTALL)
    if m:
        return m.group(1)
    # No closing fence? Just drop the opening/closing markers if present
    return s.replace("```python", "").replace("```", "")


class LambdaTool(_LCBaseTool):
    name: str = "lambda_tool"
    description: str = (
        "Executes Python in a sandbox. Supports file uploads, plots, CSV/JSON/text artifacts with previews, "
        "on-disk caching, and metrics (runtime, CPU, RAM)."
    )
    args_schema: Type[BaseModel] = PythonScriptToolInput

    # --- shared helper to extract configurable dict ---
    @staticmethod
    def _get_configurable(config: Optional[RunnableConfig], kwargs: dict) -> dict:
        cfg = {}
        if isinstance(config, dict):
            cfg = config.get("configurable") or {}
        if not cfg:
            maybe_cfg = kwargs.get("config")
            if isinstance(maybe_cfg, dict):
                cfg = maybe_cfg.get("configurable") or {}
        if not cfg:
            cfg = kwargs.get("configurable") or {}
        return cfg

    # --- sync path (rare, but keep it complete) ---
    def invoke(self, input: dict, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        configurable = self._get_configurable(config, {})
        db = configurable.get("db_session")
        user_id = configurable.get("user_id")
        thread_id = configurable.get("thread_id")

        files_payload = []
        if db and user_id is not None and thread_id is not None:
            import asyncio
            try:
                files_payload = asyncio.get_event_loop().run_until_complete(
                    get_temp_files_as_payload(db, user_id, thread_id)
                ) or []
            except RuntimeError:
                # if we're in a running loop, use a new one
                loop = asyncio.new_event_loop()
                try:
                    files_payload = loop.run_until_complete(get_temp_files_as_payload(db, user_id, thread_id)) or []
                finally:
                    loop.close()
            if files_payload:
                logger.info("LambdaTool attached %d temp file(s) (sync).", len(files_payload))
        else:
            logger.warning(
                "LambdaTool context missing (sync). Cannot attach temp files. (_db=%s, _user_id=%s, _thread_id=%s)",
                bool(db), user_id, thread_id
            )

        if files_payload:
            input = dict(input or {})
            input["files"] = (input.get("files") or []) + files_payload

        # now call BaseTool to do schema validation and route to _run
        return super().invoke(input, config)

    # --- async path (this is what ToolNode uses) ---
    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        configurable = self._get_configurable(config, {})
        db = configurable.get("db_session")
        user_id = configurable.get("user_id")
        thread_id = configurable.get("thread_id")

        files_payload = []
        if db and user_id is not None and thread_id is not None:
            files_payload = await get_temp_files_as_payload(db, user_id, thread_id) or []
            if files_payload:
                logger.info("LambdaTool attached %d temp file(s) (async).", len(files_payload))
        else:
            logger.warning(
                "LambdaTool context missing. Cannot attach temp files. (_db=%s, _user_id=%s, _thread_id=%s)",
                bool(db), user_id, thread_id
            )

        if files_payload:
            input = dict(input or {})
            input["files"] = (input.get("files") or []) + files_payload

        # now call BaseTool to do schema validation and route to _run
        return await super().ainvoke(input, config)


    # keep your existing _run + run methods as-is
    def _run(self, **kwargs) -> Dict[str, Any]:
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
            "files",
            "preview_bytes",
            "enable_cache",
            "cache_ttl_s",
            "strip_imports",
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed_keys}

        # auto-pick function mode if user passed function or callable
        mode = filtered.get("mode")
        if not mode:
            mode = "function" if (filtered.get("function") or filtered.get("callable_fn")) else "code"
        filtered["mode"] = mode

        # If someone set mode='function' but forgot 'function' name, fall back to code
        if filtered["mode"] == "function" and not (filtered.get("function") or filtered.get("callable_fn")):
            logger.info("LambdaTool: falling back to code mode (empty function in function mode).")
            filtered["mode"] = "code"

        # Normalize the script (strip ``` fences, dedent, clean whitespace)
        raw_script = script if script is not None else ""
        normalized_script = _normalize_script(raw_script)

        needs_wrap = any(ln.startswith((" ", "\t")) for ln in normalized_script.splitlines() if ln.strip())
        if needs_wrap:
            body = "\n".join(("    " + ln) if ln.strip() else ln for ln in normalized_script.splitlines())
            normalized_script = "if True:\n" + body

        if "strip_imports" not in filtered:
            filtered["strip_imports"] = True
        if normalized_script and ("import " in normalized_script or "from " in normalized_script) and not filtered.get(
                "strip_imports", True):
            filtered["strip_imports"] = True

        # --- sanitize function/mode before building the input ---
        fn_name = (filtered.get("function") or "").strip() or None
        has_script = bool(normalized_script)

        # If the function name equals the tool name (common LLM slip), treat it as not set.
        if filtered.get("mode") == "function" and fn_name in {self.name, "lambda_tool"}:
            logger.info("LambdaTool: ignoring function=='%s' (tool name); switching to code mode.", fn_name)
            filtered["mode"] = "code"
            filtered["function"] = None

        # If we're in function mode but the function isn't registered and no callable is given,
        # fall back to code mode when we *do* have a script. Otherwise keep the error.
        if filtered.get("mode") == "function" and not filtered.get("callable_fn"):
            if fn_name and fn_name not in _FUNCTIONS:
                if has_script:
                    logger.info("LambdaTool: function '%s' not registered; script provided; switching to code mode.",
                                fn_name)
                    filtered["mode"] = "code"
                    filtered["function"] = None
                else:
                    # optional: downgrade to a friendly error payload instead of raising later
                    logger.warning("LambdaTool: function '%s' not registered and no script provided.", fn_name)

        # Build the input using the normalized script
        inp = PythonScriptToolInput(script=normalized_script, **filtered)

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
            files=inp.files,
            preview_bytes=inp.preview_bytes,
            enable_cache=inp.enable_cache,
            cache_ttl_s=inp.cache_ttl_s,
            strip_imports=inp.strip_imports,

        )

        # Use fork on POSIX to avoid extra pickling pain; spawn on Windows
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
                "metrics": {
                    "runtime_ms": cfg.timeout_s * 1000,
                    "cpu_user_s": 0.0,
                    "cpu_sys_s": 0.0,
                    "peak_rss_kb": 0,
                    "used_gpu": False,
                },
                "used_gpu": False,
                "error_type": "TimeoutError",
                "error_message": f"Exceeded {cfg.timeout_s}s",
                "traceback": None,
            }

        try:
            res = result_q.get_nowait()
            if not res.get("ok"):
                logger.error("LambdaTool error: %s\nTraceback:\n%s", res.get("error_message"), res.get("traceback"))
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
                "metrics": {
                    "runtime_ms": inp.timeout_s * 1000,
                    "cpu_user_s": 0.0,
                    "cpu_sys_s": 0.0,
                    "peak_rss_kb": 0,
                    "used_gpu": False,
                },
                "used_gpu": False,
                "error_type": "Crash",
                "error_message": "No result from worker",
                "traceback": None,
            }
