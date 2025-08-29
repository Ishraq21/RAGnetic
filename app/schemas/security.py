# app/schemas/security.py

from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Literal
from datetime import datetime

# ---------- Roles ----------

class RoleBase(BaseModel):
    """Base schema for a role."""
    name: str = Field(..., min_length=3, max_length=50, description="Unique name of the role (e.g., 'admin', 'developer').")
    description: Optional[str] = Field(None, max_length=255, description="A brief description of the role.")

class RoleCreate(RoleBase):
    """Schema for creating a new role."""
    pass

class Role(RoleBase):
    """Schema for a role as stored in the database, including its ID."""
    id: int = Field(..., description="Unique database ID of the role.")
    permissions: List[str] = Field([], description="List of permissions assigned to this role.")

    class Config:
        from_attributes = True  # Pydantic v2 ORM mode

# Public representation for responses
class RolePublic(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    permissions: List[str] = []

    class Config:
        from_attributes = True

# ---------- Users ----------

class UserBase(BaseModel):
    """Base schema for a user."""
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$", description="Unique username for the user.")
    email: Optional[EmailStr] = Field(None, description="Optional email address for the user.")
    first_name: Optional[str] = Field(None, max_length=100, description="User's first name.")
    last_name: Optional[str] = Field(None, max_length=100, description="User's last name.")
    is_active: bool = Field(True, description="Indicates if the user account is active.")
    is_superuser: bool = Field(False, description="Indicates if the user has superuser privileges.")

class UserCreate(UserBase):
    """Schema for creating a new user, including password."""
    password: str = Field(..., min_length=8, description="User's password.")
    roles: Optional[List[str]] = Field(None, description="List of role names to assign to the user upon creation.")

class UserUpdate(UserBase):
    """Schema for updating an existing user's details."""
    username: Optional[str] = Field(None, min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$", description="New unique username for the user.")
    password: Optional[str] = Field(None, min_length=8, description="New password for the user.")
    first_name: Optional[str] = Field(None, max_length=100, description="User's first name.")
    last_name: Optional[str] = Field(None, max_length=100, description="User's last name.")
    roles: Optional[List[str]] = Field(None, description="List of role names to update for the user.")

class User(UserBase):
    """
    Internal schema used inside the app (includes hashed_password).
    NEVER return this from API endpoints.
    """
    id: int = Field(..., description="Unique database ID of the user.")
    hashed_password: str = Field(..., description="Hashed password of the user.")
    created_at: datetime = Field(..., description="Timestamp when the user was created.")
    updated_at: datetime = Field(..., description="Timestamp when the user was last updated.")
    roles: List[Role] = Field([], description="List of roles assigned to the user.")
    # API key scope propagated to aid PermissionChecker
    scope: Optional[str] = Field(default="viewer", description="API key scope for this request, if applicable.")

    class Config:
        from_attributes = True  # Pydantic v2 ORM mode

# Public-safe version for responses (no hashed_password)
class UserPublic(BaseModel):
    id: int
    username: str
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime
    roles: List[RolePublic] = []

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: Literal["api_key"] = "api_key"


class TokenData(BaseModel):
    """Schema for data contained in a JWT token (e.g., username, user ID, roles)."""
    username: Optional[str] = None
    user_id: Optional[int] = None
    roles: List[str] = []

class LoginRequest(BaseModel):
    """Schema for user login request."""
    username: str = Field(..., description="The username of the user.")
    password: str = Field(..., description="The password of the user.")
