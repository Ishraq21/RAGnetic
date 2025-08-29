# app/api/security.py

import logging
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, NoResultFound
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field  # Import Field for LoginRequest
from sqlalchemy import select
from app.core.rate_limit import rate_limiter

from app.db import get_db
from app.db import dao as db_dao  # Alias to avoid name collision with security.py in future
from app.db.models import user_api_keys_table
from app.schemas.security import UserCreate, UserUpdate, User, RoleCreate, Role, Token, TokenData, \
    LoginRequest, UserPublic
from app.core.security import PermissionChecker, get_http_api_key, get_current_user_from_api_key

logger = logging.getLogger("ragnetic")

router = APIRouter(prefix="/api/v1/security", tags=["Security API"])

@router.post("/login", response_model=Token, dependencies=[Depends(rate_limiter("login", 5, 60))])
async def login_for_access_token(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticates a user with username and password and returns a user-specific API key.
    This API key can then be used for subsequent requests.
    """
    user_data = await db_dao.get_user_by_username(db, request.username)
    if not user_data:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password.")

    # Verify password
    if not db_dao.verify_password(request.password, user_data.get("hashed_password", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password.")

    if not user_data.get("is_active"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive.")

    try:
        active_key_stmt = select(user_api_keys_table.c.api_key).where(
            user_api_keys_table.c.user_id == user_data["id"],
            user_api_keys_table.c.revoked == False
        ).order_by(user_api_keys_table.c.created_at.desc()).limit(1)

        existing_key = (await db.execute(active_key_stmt)).scalar_one_or_none()

        if existing_key:
            api_key_str = existing_key
            logger.info(f"User '{request.username}' logged in, reusing existing API key.")
        else:

            # When creating a new key, a default scope must be provided.
            # Here we default to 'viewer' which is a safe, minimal permission scope.
            api_key_str = await db_dao.create_user_api_key(db, user_data["id"], scope="viewer")
            logger.info(f"User '{request.username}' logged in, new API key generated.")

        return Token(access_token=api_key_str, token_type="api_key")
    except Exception as e:
        logger.error(f"Failed to generate/retrieve API key for user {request.username}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to generate access token.")

@router.get("/me", response_model=UserPublic)
async def read_current_user(
        current_user: User = Depends(get_current_user_from_api_key)
):
    return UserPublic(**current_user.model_dump())


# --- User Management Endpoints ---

@router.post("/users", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def create_user(
        user_in: UserCreate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:create_user"]))  # Requires specific permission
):
    """
    Creates a new user account.
    Requires: 'security:create_user' permission.
    """
    try:
        new_user_data = await db_dao.create_user(db, user_in)
        if not new_user_data:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create user.")
        logger.info(
            f"User '{current_user.username}' created new user '{new_user_data['username']}' (ID: {new_user_data['id']}).")
        return UserPublic(**new_user_data)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create user {user_in.username}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while creating the user.")


@router.get("/users", response_model=List[UserPublic])

async def get_all_users(
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=100),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:read_users"]))  # Requires specific permission
):
    """
    Retrieves a paginated list of all users.
    Requires: 'security:read_users' permission.
    """
    try:
        users_data = await db_dao.get_all_users(db, skip=skip, limit=limit)
        logger.info(f"User '{current_user.username}' listed all users.")
        return [UserPublic(**user_data) for user_data in users_data]

    except Exception as e:
        logger.error(f"Failed to retrieve all users: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while retrieving users.")


@router.get("/users/{user_id}", response_model=UserPublic)

async def get_user_by_id(
        user_id: int,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:read_users"]))  # Requires specific permission
):
    """
    Retrieves a single user by their ID.
    Requires: 'security:read_users' permission.
    """
    try:
        user_data = await db_dao.get_user_by_id(db, user_id)
        if not user_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        logger.info(f"User '{current_user.username}' retrieved details for user ID: {user_id}.")
        return UserPublic(**user_data)

    except Exception as e:
        logger.error(f"Failed to retrieve user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while retrieving the user.")


@router.put("/users/{user_id}", response_model=UserPublic)

async def update_user(
        user_id: int,
        user_update: UserUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:update_users"]))  # Requires specific permission
):
    """
    Updates an existing user's details.
    Requires: 'security:update_users' permission.
    """
    try:
        updated_user_data = await db_dao.update_user(db, user_id, user_update)
        if not updated_user_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        logger.info(f"User '{current_user.username}' updated user ID: {user_id}.")
        return UserPublic(**updated_user_data)

    except ValueError as e:  # Catching specific ValueErrors from DAO for duplicates/not found
        if "not found" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        else:  # Duplicate username/email
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while updating the user.")


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
        user_id: int,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:delete_users"]))  # Requires specific permission
):
    """
    Deletes a user account.
    Requires: 'security:delete_users' permission.
    """
    try:
        deleted = await db_dao.delete_user(db, user_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        logger.info(f"User '{current_user.username}' deleted user ID: {user_id}.")
        return  # 204 No Content
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while deleting the user.")


# --- Role Management Endpoints ---

@router.post("/roles", response_model=Role, status_code=status.HTTP_201_CREATED)
async def create_role(
        role_in: RoleCreate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:create_role"]))
):
    """
    Creates a new role.
    Requires: 'security:create_role' permission.
    """
    try:
        new_role_data = await db_dao.create_role(db, role_in)
        logger.info(
            f"User '{current_user.username}' created new role '{new_role_data['name']}' (ID: {new_role_data['id']}).")
        return Role(**new_role_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create role {role_in.name}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while creating the role.")


@router.get("/roles", response_model=List[Role])
async def get_all_roles(
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=100),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:read_roles"]))  # Requires specific permission
):
    """
    Retrieves a paginated list of all roles.
    Requires: 'security:read_roles' permission.
    """
    try:
        roles_data = await db_dao.get_all_roles(db, skip=skip, limit=limit)
        logger.info(f"User '{current_user.username}' listed all roles.")
        return [Role(**role_data) for role_data in roles_data]
    except Exception as e:
        logger.error(f"Failed to retrieve all roles: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while retrieving roles.")


@router.get("/roles/{role_name}", response_model=Role)
async def get_role_by_name(
        role_name: str,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:read_roles"]))  # Requires specific permission
):
    """
    Retrieves a single role by its name.
    Requires: 'security:read_roles' permission.
    """
    try:
        role_data = await db_dao.get_role_by_name(db, role_name)
        if not role_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found.")
        logger.info(f"User '{current_user.username}' retrieved details for role '{role_name}'.")
        return Role(**role_data)
    except Exception as e:
        logger.error(f"Failed to retrieve role {role_name}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while retrieving the role.")


@router.delete("/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(
        role_id: int,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:delete_roles"]))  # Requires specific permission
):
    """
    Deletes a role.
    Requires: 'security:delete_roles' permission.
    """
    try:
        deleted = await db_dao.delete_role(db, role_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found.")
        logger.info(f"User '{current_user.username}' deleted role ID: {role_id}.")
        return  # 204 No Content
    except Exception as e:
        logger.error(f"Failed to delete role {role_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while deleting the role.")

class APIKeyCreate(BaseModel):
    scope: Literal["admin", "editor", "viewer"] = "viewer"

class APIKeyRevoke(BaseModel):
    api_key: str

# --- User API Key Endpoints ---

@router.post(
    "/users/{user_id}/api-keys",
    response_model=Token,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limiter("apikey_create", 10, 60))]
)
async def create_user_api_key(
        user_id: int,
        key_create: APIKeyCreate = Body(...),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:manage_api_keys"]))
):
    """
    Generates a new API key for a specific user.
    Requires: 'security:manage_api_keys' permission.
    """
    try:
        # Verify user exists first
        user_exists = await db_dao.get_user_by_id(db, user_id)
        if not user_exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

        # Pass the scope from the request body to the DAO function
        new_api_key = await db_dao.create_user_api_key(db, user_id, scope=key_create.scope)
        logger.info(f"User '{current_user.username}' generated API key for user ID: {user_id}.")
        return Token(access_token=new_api_key, token_type="api_key")

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create API key for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while creating the API key.")


@router.delete(
    "/api-keys",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(rate_limiter("apikey_revoke", 20, 60))]
)
async def revoke_user_api_key(
        payload: APIKeyRevoke,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:manage_api_keys"]))  # Requires specific permission
):
    """
    Revokes a specific user API key.
    Requires: 'security:manage_api_keys' permission.
    """
    try:
        revoked = await db_dao.revoke_user_api_key(db, payload.api_key)
        if not revoked:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found or already revoked.")
        logger.info(f"User '{current_user.username}' revoked API key: {payload.api_key[:8]}...")
        return  # 204 No Content
    except Exception as e:
        logger.error(f"Failed to revoke API key {payload.api_key[:8]}...: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while revoking the API key.")

# --- User-Role Assignment Endpoints ---

class UserRoleAssignment(BaseModel):
    role_name: str
    organization_name: Optional[str] = "default"  # Default to "default" organization


@router.post("/users/{user_id}/roles", status_code=status.HTTP_200_OK)
async def assign_role_to_user(
        user_id: int,
        assignment: UserRoleAssignment,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:manage_user_roles"]))  # Requires specific permission
):
    """
    Assigns a role to a user.
    Requires: 'security:manage_user_roles' permission.
    """
    try:
        assigned = await db_dao.assign_role_to_user(db, user_id, assignment.role_name, assignment.organization_name)
        if not assigned:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail=f"Role '{assignment.role_name}' is already assigned to user {user_id} in organization '{assignment.organization_name}'.")
        logger.info(f"User '{current_user.username}' assigned role '{assignment.role_name}' to user ID: {user_id}.")
        return {"message": f"Role '{assignment.role_name}' assigned to user {user_id}."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to assign role '{assignment.role_name}' to user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while assigning the role.")


@router.delete("/users/{user_id}/roles", status_code=status.HTTP_200_OK)
async def remove_role_from_user(
        user_id: int,
        assignment: UserRoleAssignment,  # Re-use UserRoleAssignment for consistency
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:manage_user_roles"]))  # Requires specific permission
):
    """
    Removes a role from a user.
    Requires: 'security:manage_user_roles' permission.
    """
    try:
        removed = await db_dao.remove_role_from_user(db, user_id, assignment.role_name, assignment.organization_name)
        if not removed:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Role '{assignment.role_name}' not found for user {user_id} in organization '{assignment.organization_name}'.")
        logger.info(f"User '{current_user.username}' removed role '{assignment.role_name}' from user ID: {user_id}.")
        return {"message": f"Role '{assignment.role_name}' removed from user {user_id}."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to remove role '{assignment.role_name}' from user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while removing the role.")


# --- Role-Permission Assignment Endpoints ---

class RolePermissionAssignment(BaseModel):
    permission: str


@router.post("/roles/{role_id}/permissions", status_code=status.HTTP_200_OK)
async def assign_permission_to_role(
        role_id: int,
        assignment: RolePermissionAssignment,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:manage_role_permissions"]))
        # Requires specific permission
):
    """
    Assigns a permission to a role.
    Requires: 'security:manage_role_permissions' permission.
    """
    try:
        # Verify role exists first
        role_exists = await db_dao.get_role_by_id(db, role_id)  # Need get_role_by_id in DAO
        if not role_exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found.")

        assigned = await db_dao.assign_permission_to_role(db, role_id, assignment.permission)
        if not assigned:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail=f"Permission '{assignment.permission}' is already assigned to role {role_id}.")
        logger.info(
            f"User '{current_user.username}' assigned permission '{assignment.permission}' to role ID: {role_id}.")
        return {"message": f"Permission '{assignment.permission}' assigned to role {role_id}."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to assign permission '{assignment.permission}' to role {role_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while assigning the permission.")


@router.delete("/roles/{role_id}/permissions", status_code=status.HTTP_200_OK)
async def remove_permission_from_role(
        role_id: int,
        assignment: RolePermissionAssignment,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(PermissionChecker(["security:manage_role_permissions"]))
        # Requires specific permission
):
    """
    Removes a permission from a role.
    Requires: 'security:manage_role_permissions' permission.
    """
    try:
        # Verify role exists first
        role_exists = await db_dao.get_role_by_id(db, role_id)  # Need get_role_by_id in DAO
        if not role_exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found.")

        removed = await db_dao.remove_permission_from_role(db, role_id, assignment.permission)
        if not removed:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Permission '{assignment.permission}' not found for role {role_id}.")
        logger.info(
            f"User '{current_user.username}' removed permission '{assignment.permission}' from role ID: {role_id}.")
        return {"message": f"Permission '{assignment.permission}' removed from role {role_id}."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to remove permission '{assignment.permission}' from role {role_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while removing the permission.")
