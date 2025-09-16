from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class UserCredits(BaseModel):
    """Schema for user credit information."""
    balance: float = Field(..., description="Current credit balance", ge=0)
    daily_limit: float = Field(..., description="Daily spending limit", ge=0)
    total_spent: float = Field(..., description="Total amount spent", ge=0)
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class CreditTopUpRequest(BaseModel):
    """Schema for credit top-up requests."""
    amount: float = Field(..., description="Amount to add to credits", gt=0)


class CreditTransaction(BaseModel):
    """Schema for credit transaction records."""
    id: int = Field(..., description="Transaction ID")
    amount: float = Field(..., description="Transaction amount")
    transaction_type: Literal["credit", "debit", "refund"] = Field(..., description="Type of transaction")
    description: str = Field(..., description="Transaction description")
    created_at: datetime = Field(..., description="Transaction timestamp")

    class Config:
        from_attributes = True


class BillingUsage(BaseModel):
    """Schema for billing usage summary."""
    gpu_hours: float = Field(..., description="Total GPU hours used", ge=0)
    gpu_cost: float = Field(..., description="Total GPU cost", ge=0)
    api_calls: int = Field(..., description="Total API calls made", ge=0)
    api_cost: float = Field(..., description="Total API cost", ge=0)
    total_cost: float = Field(..., description="Total cost across all services", ge=0)


class SpendingLimitUpdate(BaseModel):
    """Schema for updating spending limits."""
    daily_limit: float = Field(..., description="New daily spending limit", ge=0)
