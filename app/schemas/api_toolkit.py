# app/schemas/api_toolkit.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

class PaginationConfig(BaseModel):
    """
    Optional pagination settings. Use only what your API supports.
    - type: "page" (page=N), "cursor" (cursor=...), or "link" (RFC5988 Link header).
    """
    type: Literal["page", "cursor", "link"] = Field(..., description="Pagination strategy.")
    # Page-style
    page_param: Optional[str] = Field(None, description="Query param name for page index (e.g., 'page').")
    per_page_param: Optional[str] = Field(None, description="Query param name for items per page.")
    per_page_value: Optional[int] = Field(None, description="Value to set for per_page_param.")
    page_start: int = Field(1, description="Starting page number (default 1).")
    # Cursor-style
    cursor_param: Optional[str] = Field(None, description="Query param name for the cursor.")
    next_cursor_jsonpath: Optional[str] = Field(
        None,
        description="JSONPath to next cursor in the response (e.g., '$.next_cursor')."
    )
    # Link-style
    next_link_jsonpath: Optional[str] = Field(
        None,
        description="JSONPath to the HTTP URL for the next page (from body) if provided."
    )
    # Common limits
    max_pages: int = Field(1, description="Max number of pages to fetch (default 1 = no pagination).")
    item_limit: Optional[int] = Field(
        None,
        description="Optional hard cap on total items to collect across pages (tool won’t slice bodies; the agent should)."
    )

class APIRequestToolInput(BaseModel):
    """
    Input schema for a general-purpose API request tool.
    """
    url: str = Field(..., description="The full URL for the API endpoint.")
    method: Literal['GET', 'POST', 'PUT', 'PATCH', 'DELETE'] = Field('GET', description="HTTP method.")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom headers.")
    params: Optional[Dict[str, str]] = Field(None, description="Query parameters.")
    payload: Optional[Dict[str, Any]] = Field(None, description="JSON payload for POST/PUT/PATCH/DELETE.")

    # New (optional) enhancements — backward compatible
    response_mode: Literal['auto', 'json', 'text', 'bytes'] = Field(
        'auto',
        description="How to parse the response body."
    )
    pagination: Optional[PaginationConfig] = Field(
        None,
        description="Optional pagination settings."
    )
