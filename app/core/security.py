# app/core/security.py
from fastapi import Security, HTTPException, status, WebSocket, WebSocketException
from fastapi.security import APIKeyHeader, APIKeyQuery
from app.core.config import get_server_api_keys

API_KEY_HEADER_NAME = "X-API-Key"
API_KEY_QUERY_NAME = "api_key"

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY_NAME, auto_error=False)


async def get_http_api_key(
        key_from_header: str = Security(api_key_header),
        key_from_query: str = Security(api_key_query),
) -> str:
    """
    A dependency that checks for a valid API key in either the X-API-Key header
    or the 'api_key' query parameter for standard HTTP requests.
    """
    valid_api_keys = get_server_api_keys()
    if not valid_api_keys:
        return "development_mode_unsecured"

    if key_from_header and key_from_header in valid_api_keys:
        return key_from_header

    if key_from_query and key_from_query in valid_api_keys:
        return key_from_query

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )


async def get_websocket_api_key(
        websocket: WebSocket
) -> str:
    """
    A dependency that checks for a valid API key in the 'api_key' query parameter
    for WebSocket connections.
    """
    valid_api_keys = get_server_api_keys()
    if not valid_api_keys:
        return "development_mode_unsecured"

    key_from_query = websocket.query_params.get(API_KEY_QUERY_NAME)

    if key_from_query and key_from_query in valid_api_keys:
        return key_from_query

    # Raising WebSocketException will cause FastAPI to automatically and cleanly
    # close the connection with the specified code and reason.
    raise WebSocketException(
        code=status.WS_1008_POLICY_VIOLATION,
        reason="Invalid or missing API Key"
    )