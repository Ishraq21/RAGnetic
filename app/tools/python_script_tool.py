import logging
import re as _re
import signal
from typing import Any, Dict, Type

from pydantic.v1 import BaseModel, Field
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
from RestrictedPython.PrintCollector import PrintCollector

logger = logging.getLogger(__name__)


_ALLOWED_MODULES = {
    "re": __import__("re"),
    "json": __import__("json"),
    "math": __import__("math"),
}


class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("Script execution timed out")

def with_timeout(seconds: int):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(seconds)
            try:
                return fn(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
        return wrapper
    return decorator

class PythonScriptToolInput(BaseModel):
    script: str = Field(..., description="The Python code to be executed in the sandbox.")


class PythonScriptTool:
    name = "python_script_tool"
    description = (
        "Executes a snippet of Python code in a secure sandbox. "
        "Only a small whitelist of modules is available (re, json, math). "
        "You MUST use `print()` to output results. The return value of the script is not captured."
    )
    args_schema: Type[BaseModel] = PythonScriptToolInput

    def get_input_schema(self) -> Dict[str, Any]:
        return self.args_schema.schema()

    @with_timeout(2)  # fail if user script runs longer than 2s
    def run(self, script: str, **kwargs: Any) -> Dict[str, str]:
        """
        1. Strip out any import statements (we preload only the safe ones).
        2. Compile with RestrictedPython.
        3. Exec in a locked-down globals that only expose safe_builtins, a restricted __import__,
           PrintCollector as _print_, and our whitelisted modules.
        4. Capture and return stdout/stderr.
        """
        logger.info("Original script:\n%s", script)

        # 1) Remove all import lines
        cleaned = _re.sub(r'^\s*import\s+\w+.*$', '# stripped import', script, flags=_re.MULTILINE)

        # 2) Build the sandbox globals
        rb = safe_builtins.copy()
        # re-inject a restricted __import__ that only allows our whitelist
        def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in _ALLOWED_MODULES:
                return _ALLOWED_MODULES[name]
            raise ImportError(f"Module '{name}' is not allowed in sandbox")
        rb["__import__"] = _restricted_import

        safe_globals = {
            "__builtins__": rb,
            "_print_": PrintCollector,
            # preload modules so user code can refer directly to `re`, `json`, etc.
            **_ALLOWED_MODULES,
        }

        try:
            byte_code = compile_restricted(cleaned, filename="<sandbox>", mode="exec")
            local_ns: Dict[str, Any] = {}
            exec(byte_code, safe_globals, local_ns)

            # Pull out whatever was printed
            printed = local_ns.get("_print_", lambda: "")() or ""
            logger.info("Sandbox stdout:\n%s", printed)
            return {"stdout": printed, "stderr": ""}
        except TimeoutException as te:
            logger.warning("Script timed out: %s", te)
            return {"stdout": "", "stderr": f"Timeout: {te}"}
        except Exception as e:
            # Capture full traceback
            import traceback
            tb = traceback.format_exc()
            logger.error("Script error:\n%s", tb)
            return {"stdout": "", "stderr": tb}
