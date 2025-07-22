# app/tools/python_script_tool.py

import sys
from io import StringIO
from typing import Dict, Any
from pydantic.v1 import BaseModel, Field


class ToolInput(BaseModel):
    script: str = Field(...,
                        description="The Python script to execute. The result of the last line will be returned as stdout.")


class PythonScriptTool:
    """A tool for executing sandboxed Python scripts."""

    def run(self, script: str) -> Dict[str, Any]:
        """
        Executes a Python script in a sandboxed environment and captures its output.

        Args:
            script: The Python script to execute.

        Returns:
            A dictionary containing the standard output and standard error of the script.
        """
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured_stdout = StringIO()
        sys.stderr = captured_stderr = StringIO()

        try:
            # WARNING: This uses exec() and is not a secure sandbox.
            # This is a placeholder and should be replaced with a secure execution environment.
            exec(script, {})
        except Exception as e:
            captured_stderr.write(str(e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return {
            "stdout": captured_stdout.getvalue(),
            "stderr": captured_stderr.getvalue()
        }

    # This method allows the engine to discover the tool's input schema
    def get_input_schema(self) -> Dict[str, Any]:
        return ToolInput.schema()