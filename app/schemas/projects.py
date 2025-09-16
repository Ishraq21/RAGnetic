from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    """Schema for creating a new project."""
    name: str = Field(..., description="Project name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Project description")


class ProjectUpdate(BaseModel):
    """Schema for updating an existing project."""
    name: Optional[str] = Field(None, description="Project name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Project description")


class Project(BaseModel):
    """Schema for a complete project."""
    id: int = Field(..., description="Project ID")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    user_id: int = Field(..., description="User ID who owns the project")
    created_at: datetime = Field(..., description="Project creation timestamp")
    updated_at: datetime = Field(..., description="Project last update timestamp")
    billing_session_id: Optional[int] = Field(None, description="Associated billing session ID")

    class Config:
        from_attributes = True


class ProjectListItem(BaseModel):
    """Schema for project list items with optional cost information."""
    id: int = Field(..., description="Project ID")
    name: str = Field(..., description="Project name")
    created_at: datetime = Field(..., description="Project creation timestamp")
    total_cost: Optional[float] = Field(None, description="Total cost for the project")

    class Config:
        from_attributes = True
