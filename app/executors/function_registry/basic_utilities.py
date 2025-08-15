import logging
import datetime
import json
import uuid
import os
import hashlib
import random
from typing import Any, Dict

from .__init__ import FunctionRegistration

logger = logging.getLogger(__name__)

@FunctionRegistration(
    name="get_datetime",
    description="Returns the current UTC date and time in ISO 8601 format.",
    args_schema={"type": "object", "properties": {}}
)
def get_datetime() -> Dict[str, Any]:
    return {"current_utc_time": datetime.datetime.utcnow().isoformat()}

@FunctionRegistration(
    name="math_calculator",
    description="Safely evaluates a basic mathematical expression (e.g., '2 + 2 * 5').",
    args_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The mathematical expression to evaluate."}
        },
        "required": ["expression"]
    }
)
def math_calculator(expression: str) -> Dict[str, Any]:
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        logger.error(f"Error evaluating math expression: {e}")
        return {"error": str(e), "status": "failed"}

@FunctionRegistration(
    name="string_search",
    description="Searches for a substring in a given text.",
    args_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to search within."},
            "substring": {"type": "string", "description": "The substring to search for."}
        },
        "required": ["text", "substring"]
    }
)
def string_search(text: str, substring: str) -> Dict[str, Any]:
    found = substring in text
    return {"found": found, "count": text.count(substring)}

@FunctionRegistration(
    name="uuid_generator",
    description="Generates a unique UUID4 string.",
    args_schema={"type": "object", "properties": {}}
)
def uuid_generator() -> Dict[str, Any]:
    return {"uuid": str(uuid.uuid4())}

@FunctionRegistration(
    name="json_validator",
    description="Validates a JSON string and returns a pretty-printed version if valid.",
    args_schema={
        "type": "object",
        "properties": {
            "json_string": {"type": "string", "description": "The JSON string to validate."}
        },
        "required": ["json_string"]
    }
)
def json_validator(json_string: str) -> Dict[str, Any]:
    try:
        obj = json.loads(json_string)
        return {"valid": True, "pretty": json.dumps(obj, indent=2)}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e)}

@FunctionRegistration(
    name="random_number",
    description="Generates a random number between min and max.",
    args_schema={
        "type": "object",
        "properties": {
            "min": {"type": "number", "description": "Minimum value."},
            "max": {"type": "number", "description": "Maximum value."}
        },
        "required": ["min", "max"]
    }
)
def random_number(min: float, max: float) -> Dict[str, Any]:
    return {"random_number": random.uniform(min, max)}

@FunctionRegistration(
    name="env_var_reader",
    description="Returns the value of an environment variable.",
    args_schema={
        "type": "object",
        "properties": {
            "var_name": {"type": "string", "description": "The environment variable name."}
        },
        "required": ["var_name"]
    }
)
def env_var_reader(var_name: str) -> Dict[str, Any]:
    return {"value": os.environ.get(var_name)}

@FunctionRegistration(
    name="hash_string",
    description="Returns the SHA256 hash of a given string.",
    args_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to hash."}
        },
        "required": ["text"]
    }
)
def hash_string(text: str) -> Dict[str, Any]:
    return {"sha256": hashlib.sha256(text.encode()).hexdigest()}

