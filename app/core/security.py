# app/core/security.py

import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import Header, HTTPException, status, Depends
from fastapi import WebSocket, WebSocketException, status

from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.core.config import get_server_api_keys
from app.db.models import user_api_keys_table, users_table
from app.db import get_db
from app.db import dao as db_dao
from app.schemas.security import User

logger = logging.getLogger("ragnetic")


# --- API Key Authentication (Global & User-Specific) ---

async def get_current_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Dependency to get the API key from the request header.
    This acts as a basic authentication layer for all API calls.
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing. Please provide X-API-Key header.",
        )
    return x_api_key


async def get_http_api_key(
        api_key: str = Depends(get_current_api_key),
        db: AsyncSession = Depends(get_db)
) -> str:
    """
    Authenticates an HTTP request using the RAGNETIC_API_KEYS (master key)
    or a user-specific API key.
    """
    # Check if it's the master RAGNETIC_API_KEYS
    server_keys = get_server_api_keys()
    if api_key in server_keys:
        return api_key

    # If not a master key, try to authenticate as a user-specific API key
    authenticated_user = await db_dao.get_user_by_api_key(db, api_key)

    if authenticated_user and authenticated_user.get("is_active"):
        return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key provided. Access denied.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_websocket_api_key(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db)
) -> str:
    """
    Authenticates a WebSocket connection using the API key from either:
    - query string (?api_key=...)
    - or header (x-api-key)
    """
    api_key = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key")

    if not api_key:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="WebSocket API Key missing. Use ?api_key=... or x-api-key header.",
        )

    server_keys = get_server_api_keys()
    if api_key in server_keys:
        return api_key

    authenticated_user = await db_dao.get_user_by_api_key(db, api_key)
    if authenticated_user and authenticated_user.get("is_active"):
        return api_key

    raise WebSocketException(
        code=status.WS_1008_POLICY_VIOLATION,
        reason="Invalid WebSocket API Key provided. Connection denied.",
    )


# --- User Authentication and Authorization (RBAC) ---

async def get_current_user_from_api_key(
        api_key: str = Depends(get_http_api_key),
        db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency to retrieve the current authenticated user based on their API key.
    This also handles the 'master' RAGNETIC_API_KEYS giving superuser access.
    """
    # If the master key is used, construct a virtual superuser
    server_keys = get_server_api_keys()
    if api_key in server_keys:
        return User(
            id=0,  # Placeholder ID for master key user
            user_id="master_admin",
            username="master_admin",
            email="admin@ragnetic.ai",
            hashed_password="",  # Not applicable for virtual user
            is_active=True,
            is_superuser=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            roles=[
                {"id": 0, "name": "master_admin_role", "description": "Master administrative role",
                 "permissions": ["*"]}
            ]
        )

    # Otherwise, retrieve the actual user from the database
    user_data = await db_dao.get_user_by_api_key(db, api_key)
    if not user_data or not user_data.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials or user is inactive.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return User(**user_data)


class PermissionChecker:
    """
    FastAPI dependency to check if the current user has the required permissions.
    Usage: Depends(PermissionChecker(["agent:create", "workflow:trigger"]))
    """

    def __init__(self, required_permissions: List[str]):
        self.required_permissions = set(required_permissions)
        logger.debug(f"PermissionChecker initialized for: {required_permissions}")

    async def __call__(self, current_user: User = Depends(
        get_current_user_from_api_key)) -> User:  # Add -> User return type hint
        logger.debug(
            f"PermissionChecker: Checking user '{current_user.username}' (ID: {current_user.id}, Superuser: {current_user.is_superuser}) for required permissions: {self.required_permissions}")
        # Superusers bypass all permission checks
        if current_user.is_superuser:
            logger.debug(f"Superuser '{current_user.username}' bypassing permission check.")
            return current_user  # IMPORTANT: Return the user object here!

        user_permissions = set()
        for role in current_user.roles:
            logger.debug(f"  PermissionChecker: Processing role '{role.name}' with permissions: {role.permissions}")
            if "*" in role.permissions:  # Role has all permissions
                user_permissions.add("*")
                break  # No need to check further roles
            user_permissions.update(role.permissions)

        # If user has a '*' permission, they can do anything
        if "*" in user_permissions:
            logger.debug(
                f"PermissionChecker: User '{current_user.username}' aggregated permissions: {user_permissions}")
            logger.debug(f"User '{current_user.username}' has '*' permission, bypassing specific checks.")
            return current_user  # IMPORTANT: Return the user object here!

        # Check if user has all required permissions
        if not self.required_permissions.issubset(user_permissions):
            missing_permissions = ", ".join(self.required_permissions - user_permissions)
            logger.warning(
                f"User '{current_user.username}' (ID: {current_user.id}) lacks required permissions: {missing_permissions}. "
                f"User's permissions: {', '.join(user_permissions)}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required: {', '.join(self.required_permissions)}. Missing: {missing_permissions}",
            )
        logger.debug(
            f"User '{current_user.username}' has all required permissions: {', '.join(self.required_permissions)}.")
        return current_user  # IMPORTANT: Return the user object here!