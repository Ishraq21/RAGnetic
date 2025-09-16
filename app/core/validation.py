import re
import html
import shlex
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

def sanitize_sql_input(input_str: str) -> str:
    """
    Sanitizes input to prevent SQL injection attacks.
    Escapes dangerous SQL characters and patterns.
    """
    if not isinstance(input_str, str):
        return ""
    
    # Remove or escape dangerous SQL patterns
    dangerous_patterns = [
        r"(\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC|EXECUTE)\b)",
        r"(\bUNION\b)",
        r"(--|#)",
        r"(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+",
        r"(\bOR\b|\bAND\b)\s+['\"]\s*=\s*['\"]",
    ]
    
    sanitized = input_str
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    
    # Escape single quotes
    sanitized = sanitized.replace("'", "''")
    
    return sanitized

def sanitize_html_input(input_str: str) -> str:
    """
    Sanitizes input to prevent XSS attacks.
    Escapes HTML special characters and removes dangerous tags.
    """
    if not isinstance(input_str, str):
        return ""
    
    # Remove dangerous HTML tags and attributes
    dangerous_patterns = [
        r"<script[^>]*>.*?</script>",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
        r"<embed[^>]*>.*?</embed>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
        r"javascript:",
        r"on\w+\s*=",  # Event handlers like onclick, onload, etc.
    ]
    
    sanitized = input_str
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    # Escape HTML special characters
    sanitized = html.escape(sanitized)
    
    return sanitized

def sanitize_path_input(input_str: str) -> str:
    """
    Sanitizes input to prevent path traversal attacks.
    Removes dangerous path sequences.
    """
    if not isinstance(input_str, str):
        return ""
    
    # Remove path traversal sequences
    dangerous_patterns = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",  # URL encoded ../
        r"%2e%2e%5c",  # URL encoded ..\
        r"\.\.%2f",    # Mixed encoding
        r"\.\.%5c",    # Mixed encoding
    ]
    
    sanitized = input_str
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    
    # Remove any remaining path separators
    sanitized = sanitized.replace("/", "").replace("\\", "")
    
    return sanitized

def sanitize_command_input(input_str: str) -> str:
    """
    Sanitizes input to prevent command injection attacks.
    Removes dangerous shell metacharacters.
    """
    if not isinstance(input_str, str):
        return ""
    
    # Remove dangerous shell metacharacters
    dangerous_chars = [";", "|", "&", "`", "$", "(", ")", "<", ">", "\n", "\r"]
    
    sanitized = input_str
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")
    
    # Use shlex.quote for additional safety
    try:
        sanitized = shlex.quote(sanitized)
    except ValueError:
        # If shlex.quote fails, return empty string
        sanitized = ""
    
    return sanitized