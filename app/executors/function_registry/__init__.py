# app/executors/function_registry/__init__.py
import logging
from typing import Callable, Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class FunctionRegistration:
    """A decorator and registry for managing pre-vetted functions.

    The registry maps function names to a dictionary containing the function
    itself, a description, and its JSON schema for validation and LLM tool-use.
    """

    registry: Dict[str, Dict[str, Any]] = {}

    def __init__(self, name: str, description: str, args_schema: Dict[str, Any]):
        """Initializes the decorator with function metadata."""
        if not name or not description or not args_schema:
            raise ValueError("Function name, description, and args_schema are required.")
        self.name = name
        self.description = description
        self.args_schema = args_schema

    def __call__(self, func: Callable) -> Callable:
        """The decorator itself, which registers the function upon definition."""
        if self.name in self.registry:
            logger.warning(f"Function '{self.name}' is already registered. Overwriting.")
        self.registry[self.name] = {
            "function": func,
            "description": self.description,
            "args_schema": self.args_schema,
        }
        logger.info(f"Function '{self.name}' registered successfully.")
        return func

    @classmethod
    def get_function(cls, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves a registered function and its metadata."""
        return cls.registry.get(name)

    @classmethod
    def list_functions(cls) -> List[Dict[str, Any]]:
        """Lists all registered functions and their metadata."""
        return [
            {
                "name": name,
                "description": data["description"],
                "args_schema": data["args_schema"],
            }
            for name, data in cls.registry.items()
        ]

# Important: To ensure all functions are registered, we must import the modules here.
# This makes the registry self-contained and easy to manage.
from . import basic_utilities, data_analysis, file_utilities