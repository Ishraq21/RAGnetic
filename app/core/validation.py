import re
from fastapi import HTTPException

def sanitize_for_path(filename: str) -> str:
    """
    Sanitizes a string to be safe for use as a file path component.
    - Removes characters that could lead to path traversal (e.g., '/', '..').
    - Keeps only alphanumeric characters, underscores, and hyphens.
    This is a strict allow-list approach.
    """
    if not isinstance(filename, str):
        return ""
    return re.sub(r'[^a-zA-Z0-9_-]', '', filename)

def validate_agent_name(agent_name: str):
    """
    Validates an agent name to ensure it's safe for file paths.
    - Must be between 3 and 50 characters.
    - Can only contain letters, numbers, underscores, and hyphens.

    Raises:
        HTTPException: If the agent name is invalid.
    """
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", agent_name):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid agent_name '{agent_name}'. Name must be 3-50 characters "
                "and can only contain letters, numbers, underscores, and hyphens."
            )
        )

def is_valid_agent_name_cli(agent_name: str) -> bool:
    """
    CLI-safe version for validation. Returns True or False.
    """
    return bool(re.match(r"^[a-zA-Z0-9_-]{3,50}$", agent_name))