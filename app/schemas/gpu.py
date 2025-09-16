from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class GPUProvider(BaseModel):
    """Schema for GPU provider information."""
    name: str = Field(..., description="Provider name", max_length=255)
    gpu_type: str = Field(..., description="GPU type/model", max_length=255)
    cost_per_hour: float = Field(..., description="Cost per hour in USD", gt=0)
    availability: bool = Field(..., description="Whether the GPU is currently available")

    class Config:
        from_attributes = True


class GPUPricing(BaseModel):
    """Schema for GPU pricing information."""
    gpu_type: str = Field(..., description="GPU type/model", max_length=255)
    provider: str = Field(..., description="Provider name", max_length=255)
    cost_per_hour: float = Field(..., description="Cost per hour in USD", gt=0)


class GPUProvisionRequest(BaseModel):
    """Schema for GPU provisioning requests."""
    project_id: int = Field(..., description="Project ID to provision GPU for")
    gpu_type: str = Field(..., description="GPU type to provision", max_length=255)
    provider: str = Field(..., description="GPU provider", max_length=255)
    max_hours: float = Field(..., description="Maximum hours to run the GPU", gt=0)
    purpose: Literal["training", "inference"] = Field(..., description="Purpose of GPU usage")


class GPUInstance(BaseModel):
    """Schema for GPU instance information."""
    id: int = Field(..., description="GPU instance ID")
    gpu_type: str = Field(..., description="GPU type/model", max_length=255)
    provider: str = Field(..., description="Provider name", max_length=255)
    status: Literal["pending", "running", "stopped", "failed", "terminated"] = Field(..., description="Instance status")
    cost_per_hour: float = Field(..., description="Cost per hour in USD", gt=0)
    total_cost: float = Field(..., description="Total cost incurred", ge=0)
    started_at: Optional[datetime] = Field(None, description="Instance start timestamp")
    stopped_at: Optional[datetime] = Field(None, description="Instance stop timestamp")

    class Config:
        from_attributes = True


class GPUUsageEntry(BaseModel):
    """Schema for GPU usage tracking entries."""
    instance_id: int = Field(..., description="GPU instance ID")
    usage_type: Literal["training", "inference", "deployment"] = Field(..., description="Type of usage")
    duration_minutes: int = Field(..., description="Duration in minutes", gt=0)
    cost: float = Field(..., description="Cost for this usage entry", ge=0)
    created_at: datetime = Field(..., description="Usage entry timestamp")

    class Config:
        from_attributes = True
