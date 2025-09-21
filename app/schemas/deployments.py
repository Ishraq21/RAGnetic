from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class DeploymentCreate(BaseModel):
    """Schema for creating a new deployment."""
    agent_id: int = Field(..., description="Agent ID to deploy")
    deployment_type: Literal["chat", "api"] = Field(..., description="Type of deployment")


class Deployment(BaseModel):
    """Schema for deployment information."""
    id: int = Field(..., description="Deployment ID")
    agent_id: int = Field(..., description="Agent ID")
    deployment_type: Literal["api", "webhook", "streaming"] = Field(..., description="Type of deployment")
    status: Literal["pending", "active", "inactive", "failed"] = Field(..., description="Deployment status")
    endpoint_path: Optional[str] = Field(None, description="API endpoint path", max_length=255)
    created_at: datetime = Field(..., description="Deployment creation timestamp")

    class Config:
        from_attributes = True


class APIKeyCreate(BaseModel):
    """Schema for creating a new API key."""
    name: str = Field(..., description="API key name", min_length=1, max_length=255)
    scope: Literal["admin", "editor", "viewer"] = Field(..., description="API key scope/permissions")


class APIKeyInfo(BaseModel):
    """Schema for API key information."""
    id: int = Field(..., description="API key ID")
    name: str = Field(..., description="API key name")
    scope: Literal["admin", "editor", "viewer"] = Field(..., description="API key scope/permissions")
    is_active: bool = Field(..., description="Whether the API key is active")
    created_at: datetime = Field(..., description="API key creation timestamp")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")

    class Config:
        from_attributes = True


class APIKeyRotateResponse(BaseModel):
    """Schema for API key rotation response."""
    masked_key: str = Field(..., description="Masked API key for display purposes")
