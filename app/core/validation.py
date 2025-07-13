import re

def validate_agent_name(agent_name: str):
    """
    Validates an agent name to ensure it's safe for file paths.
    - Must be between 3 and 50 characters.
    - Can only contain letters, numbers, underscores, and hyphens.

    Raises:
        HTTPException: If the agent name is invalid.
    """
    # This regex matches strings that contain only a-z, A-Z, 0-9, _, and -
    # and are between 3 and 50 characters long.
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", agent_name):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid agent_name '{agent_name}'. Name must be 3-50 characters "
                "and can only contain letters, numbers, underscores, and hyphens."
            )
        )

# Create a separate helper for the CLI to avoid the HTTPException dependency
def is_valid_agent_name_cli(agent_name: str) -> bool:
    """
    CLI-safe version for validation. Returns True or False.
    """
    return bool(re.match(r"^[a-zA-Z0-9_-]{3,50}$", agent_name))