# app/core/security.py

import logging
from typing import Optional, List
from fastapi import Header, HTTPException, status, Depends, WebSocket, WebSocketException

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from app.core.config import get_server_api_keys
from app.db.models import user_api_keys_table, users_table
from app.db import get_db
from app.db import dao as db_dao
from app.schemas.security import User

logger = logging.getLogger("ragnetic")


async def get_current_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Dependency to get the API key from the request header.
    This acts as a basic authentication layer for all HTTP calls.
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing. Please provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


async def get_http_api_key(
    api_key: str = Depends(get_current_api_key),
    db: AsyncSession = Depends(get_db),
) -> str:
    server_keys = get_server_api_keys()
    if api_key in server_keys:
        return api_key

    user_api_key_stmt = select(user_api_keys_table).where(
        user_api_keys_table.c.api_key == api_key,
        user_api_keys_table.c.revoked == False,
    )
    user_api_key_record = (await db.execute(user_api_key_stmt)).mappings().first()

    if user_api_key_record:
        # Check if the associated user is active
        user_id = user_api_key_record["user_id"]
        user_stmt = select(users_table.c.is_active).where(users_table.c.id == user_id)
        is_user_active = (await db.execute(user_stmt)).scalar_one_or_none()

        if is_user_active:
            await db_dao.update_api_key_usage(db, api_key)
            return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key provided or user is inactive. Access denied.",
        headers={"WWW-Authenticate": "ApiKey"},
    )



# --- WebSocket helpers (kept for compatibility) ---

async def get_websocket_api_key(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db),
) -> str:
    api_key = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key")
    if not api_key:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="WebSocket API Key missing. Use ?api_key=... or x-api-key header.",
        )

    server_keys = get_server_api_keys()
    if api_key in server_keys:
        return api_key

    user_api_key_stmt = select(user_api_keys_table).where(
        user_api_keys_table.c.api_key == api_key,
        user_api_keys_table.c.revoked == False,
    )
    user_api_key_record = (await db.execute(user_api_key_stmt)).mappings().first()

    if user_api_key_record:
        user_id = user_api_key_record["user_id"]
        user_stmt = select(users_table.c.is_active).where(users_table.c.id == user_id)
        is_user_active = (await db.execute(user_stmt)).scalar_one_or_none()

        if is_user_active:
            return api_key

    raise WebSocketException(
        code=status.WS_1008_POLICY_VIOLATION,
        reason="Invalid WebSocket API Key provided. Connection denied.",
    )


async def get_current_user_from_api_key(
    api_key: str = Depends(get_http_api_key),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Retrieves the current authenticated user for HTTP flows.
    Handles master key by returning a virtual superuser.
    """
    server_keys = get_server_api_keys()
    if api_key in server_keys:
        return User(
            id=0,
            user_id="master_admin",
            username="master_admin",
            email="admin@ragnetic.ai",
            hashed_password="",
            is_active=True,
            is_superuser=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            roles=[
                {
                    "id": 0,
                    "name": "master_admin_role",
                    "description": "Master administrative role",
                    "permissions": ["*"],
                }
            ],
            scope="admin",  # ensure PermissionChecker sees admin scope
        )

    user_data = await db_dao.get_user_by_api_key(db, api_key)
    if not user_data or not user_data.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials or user is inactive.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    user_data.setdefault("scope", "viewer")

    return User(**user_data)


async def get_current_user_from_websocket(
    api_key: str = Depends(get_websocket_api_key),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Retrieves the current authenticated user for WebSocket flows.
    Handles master key by returning a virtual superuser.
    """
    server_keys = get_server_api_keys()
    if api_key in server_keys:
        return User(
            id=0,
            user_id="master_admin",
            username="master_admin",
            email="admin@ragnetic.ai",
            hashed_password="",
            is_active=True,
            is_superuser=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            roles=[
                {
                    "id": 0,
                    "name": "master_admin_role",
                    "description": "Master administrative role",
                    "permissions": ["*"],
                }
            ],
            scope="admin",
        )

    user_data = await db_dao.get_user_by_api_key(db, api_key)
    if not user_data or not user_data.get("is_active"):
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Could not validate credentials or user is inactive.",
        )
    user_data.setdefault("scope", "viewer")

    return User(**user_data)


class PermissionChecker:
    """
    FastAPI dependency to check if the current user has the required permissions.
    Usage: Depends(PermissionChecker(["agent:create", "workflow:trigger"]))
    """

    def __init__(self, required_permissions: List[str]):
        self.required_permissions = set(required_permissions)
        logger.debug(f"PermissionChecker initialized for: {required_permissions}")
        # Minimal expansions so viewer/editor can use the app
        self.scope_permission_map = {
            "admin": [
                "lambda:execute",
                "lambda:read_run_details",
                "analytics:read_lambda_runs",
                # CRUD-style perms
                "read:workflows", "create:workflows", "update:workflows", "delete:workflows",
                "read:agents", "create:agents", "update:agents", "delete:agents",
                "read:users", "create:users", "update:users", "delete:users",
                "read:roles", "create:roles", "update:roles", "delete:roles",
                "read:api_keys", "create:api_keys", "revoke:api_keys",
                "security:create_user", "security:read_users", "security:update_users", "security:delete_users",
                "security:create_role", "security:read_roles", "security:delete_roles",
                "security:manage_api_keys", "security:manage_user_roles", "security:manage_role_permissions",
                "session:create", "document:upload",
                "*",
            ],
            "editor": [
                "lambda:execute",
                "lambda:read_run_details",
                "analytics:read_lambda_runs",
                "read:workflows", "create:workflows", "update:workflows",
                "read:agents", "create:agents", "update:agents",
                "session:create", "document:upload",
                "history:read", "sessions:read", "sessions:update", "sessions:delete",
            ],
            "viewer": [
                "lambda:read_run_details",
                "analytics:read_lambda_runs",
                "read:workflows",
                "read:agents",
                "session:create", "document:upload",
                "history:read", "sessions:read",
            ],
        }

    async def __call__(self, current_user: User = Depends(get_current_user_from_api_key)) -> User:
        logger.debug(
            f"PermissionChecker: Checking user '{current_user.username}' (ID: {current_user.id}, Superuser: {current_user.is_superuser}) for: {self.required_permissions}"
        )

        # Superusers bypass all checks
        if current_user.is_superuser:
            return current_user

        api_key_scope = getattr(current_user, "scope", "viewer")
        scope_permissions = set(self.scope_permission_map.get(api_key_scope, []))

        # If scope grants wildcard or covers required, allow immediately
        if "*" in scope_permissions or self.required_permissions.issubset(scope_permissions):
            return current_user

        # Otherwise fall back to role-derived permissions
        user_permissions = set()
        for role in getattr(current_user, "roles", []):
            r_perms = getattr(role, "permissions", None) or (role.get("permissions") if isinstance(role, dict) else [])
            if isinstance(r_perms, list):
                if "*" in r_perms:
                    return current_user
                user_permissions.update(r_perms)

        if self.required_permissions.issubset(user_permissions):
            return current_user

        # Not enough permissions via scope nor roles
        missing = ", ".join(self.required_permissions - (scope_permissions | user_permissions))
        logger.warning(
            f"User '{current_user.username}' lacks required permissions: {missing}. "
            f"Scope={api_key_scope}, scope_perms={', '.join(scope_permissions)}, "
            f"user_perms={', '.join(user_permissions)}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not enough permissions. Missing: {missing}",
        )
