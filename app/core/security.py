from fastapi import Security, HTTPException, status, Request, WebSocket # Import WebSocket
from fastapi.security import APIKeyHeader, APIKeyQuery
from app.core.config import get_server_api_keys

API_KEY_HEADER_NAME = "X-API-Key"
API_KEY_QUERY_NAME = "api_key"

# Define the two ways a key can be provided for HTTP requests
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY_NAME, auto_error=False)


async def get_http_api_key(
        key_from_header: str = Security(api_key_header),
        key_from_query: str = Security(api_key_query),
) -> str:
    """
    A dependency that checks for a valid API key in either the X-API-Key header
    or the 'api_key' query parameter for standard HTTP requests.

    If no server keys are configured, this dependency allows the request to proceed,
    enabling an insecure "development mode".
    """
    valid_api_keys = get_server_api_keys()
    if not valid_api_keys:
        # No keys configured on the server, so we allow the request.
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
        websocket: WebSocket # Directly inject the WebSocket object
) -> str:
    """
    A dependency that checks for a valid API key in the 'api_key' query parameter
    for WebSocket connections.

    If no server keys are configured, this dependency allows the request to proceed,
    enabling an insecure "development mode".
    """
    valid_api_keys = get_server_api_keys()
    if not valid_api_keys:
        # No keys configured on the server, so we allow the request.
        return "development_mode_unsecured"

    key_from_query = websocket.query_params.get(API_KEY_QUERY_NAME)

    if key_from_query and key_from_query in valid_api_keys:
        return key_from_query

    # For WebSockets, we can't raise HTTPException directly as it expects an HTTP response.
    # Instead, we'll indicate an authentication failure via closing the connection.
    # The calling function in main.py will need to handle this.
    return "" # Return empty string to signify failure, handle closing in main.py