# app/core/security.py

import logging
import os
import hashlib
from typing import Optional, List, Tuple
from fastapi import Header, HTTPException, status, Depends, WebSocket, WebSocketException
from functools import lru_cache
import time

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from app.core.config import get_server_api_keys
from app.db.models import user_api_keys_table, users_table, api_keys_table
from app.db import get_db
from app.db import dao as db_dao
from app.schemas.security import User

logger = logging.getLogger("ragnetic")


# ----------------------
# Crypto/key helpers
# ----------------------

def hash_password(password: str) -> str:
    import bcrypt
    if password is None:
        password = ""
    # Generate salt and hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: Optional[str], hashed: str) -> bool:
    import bcrypt
    if password is None or not isinstance(password, str):
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_api_key(user_id: int, name: str, scope: str = "viewer") -> dict:
    import secrets
    # Generate a longer, more secure API key
    random_part = secrets.token_urlsafe(32)
    raw = f"sk_{user_id}_{name}_{int(time.time())}_{random_part}"
    return {"key": raw, "key_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(), "scope": scope}


def get_user_permissions(user: User) -> List[str]:
    perms: List[str] = []
    for role in getattr(user, "roles", []) or []:
        r_perms = getattr(role, "permissions", None) or (role.get("permissions") if isinstance(role, dict) else [])
        if isinstance(r_perms, list):
            perms.extend(r_perms)
    return perms


def get_api_key_from_hash(key_hash: str):
    """Placeholder for DB lookup; tests patch this function."""
    return None


def is_session_valid(last_seen_at: datetime, timeout_minutes: int = 30) -> bool:
    try:
        return (datetime.utcnow() - last_seen_at).total_seconds() <= timeout_minutes * 60
    except Exception:
        return False


def generate_session_id() -> str:
    return hashlib.sha256(f"session-{time.time()}".encode("utf-8")).hexdigest()


def generate_secure_token(length: int = 32) -> str:
    return hashlib.sha256(os.urandom(64)).hexdigest()[:max(32, length)]


def constant_time_compare(a: str, b: str) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a.encode("utf-8"), b.encode("utf-8")):
        result |= x ^ y
    return result == 0


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
    Usage: Depends(PermissionChecker(["agent:create"]))
    """

    def __init__(self, required_permissions: List[str]):
        # Ensure we have a list, not a string
        if isinstance(required_permissions, str):
            required_permissions = [required_permissions]
        self.required_permissions = set(required_permissions)
        logger.debug(f"PermissionChecker initialized for: {required_permissions}")
        logger.debug(f"PermissionChecker required_permissions set: {self.required_permissions}")
        # Minimal expansions so viewer/editor can use the app
        self.scope_permission_map = {
            "admin": [
                "lambda:execute",
                "lambda:read_run_details",
                "analytics:read_lambda_runs",
                # CRUD-style perms
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
                "read:agents", "create:agents", "update:agents",
                "session:create", "document:upload",
                "history:read", "sessions:read", "sessions:update", "sessions:delete",
            ],
            "viewer": [
                "lambda:read_run_details",
                "analytics:read_lambda_runs",
                "read:agents",
                "session:create", "document:upload",
                "history:read", "sessions:read",
                "audit:read_agent_runs",
                "fine_tune:read_status", "fine_tune:list_models",
                "agent:query", "agent:read",
                "evaluation:read_benchmarks",
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


# API Key verification with caching
_api_key_cache = {}
_cache_ttl = 60  # 60 seconds cache


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage and verification."""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def verify_api_key(raw_api_key: str, db: AsyncSession) -> Optional[Tuple[int, int, str]]:
    """
    Verify API key and return (api_key_id, user_id, scope).
    Uses 60-second caching for performance.
    
    Args:
        raw_api_key: The raw API key to verify
        db: Database session
        
    Returns:
        Tuple of (api_key_id, user_id, scope) if valid, None otherwise
    """
    if not raw_api_key:
        return None
    
    # Check cache first
    current_time = time.time()
    cache_key = raw_api_key
    
    if cache_key in _api_key_cache:
        cached_data, cache_time = _api_key_cache[cache_key]
        if current_time - cache_time < _cache_ttl:
            logger.debug(f"API key verification cache hit for key ending in ...{raw_api_key[-4:]}")
            return cached_data
    
    # Hash the API key for database lookup
    key_hash = hash_api_key(raw_api_key)
    
    try:
        # Check server API keys first
        server_keys = get_server_api_keys()
        if raw_api_key in server_keys:
            # Server key - return special values
            result = (0, 0, "admin")  # api_key_id=0, user_id=0, scope=admin
            _api_key_cache[cache_key] = (result, current_time)
            return result
        
        # Check deployment API keys
        result = await db.execute(
            select(
                api_keys_table.c.id,
                api_keys_table.c.user_id,
                api_keys_table.c.scope
            ).where(
                api_keys_table.c.key_hash == key_hash,
                api_keys_table.c.is_active == True
            )
        )
        api_key_record = result.fetchone()
        
        if api_key_record:
            # Verify user is still active
            user_result = await db.execute(
                select(users_table.c.is_active).where(
                    users_table.c.id == api_key_record.user_id
                )
            )
            user_active = user_result.scalar_one_or_none()
            
            if user_active:
                result_data = (
                    api_key_record.id,
                    api_key_record.user_id,
                    api_key_record.scope or "viewer"
                )
                # Cache the result
                _api_key_cache[cache_key] = (result_data, current_time)
                logger.debug(f"API key verified for user {api_key_record.user_id}")
                return result_data
            else:
                logger.warning(f"API key valid but user {api_key_record.user_id} is inactive")
                return None
        else:
            logger.debug(f"API key not found in database")
            return None
            
    except Exception as e:
        logger.error(f"Error verifying API key: {e}")
        return None


def clear_api_key_cache():
    """Clear the API key cache. Useful for testing or when keys are revoked."""
    global _api_key_cache
    _api_key_cache.clear()
    logger.info("API key cache cleared")
