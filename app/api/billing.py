# app/api/billing.py
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, func, and_

from app.core.security import PermissionChecker, get_current_user_from_api_key
from app.core.rate_limit import rate_limiter as rate_limit_dep
from app.schemas.billing import (
    UserCredits, CreditTopUpRequest, CreditTransaction, 
    BillingUsage, SpendingLimitUpdate
)
from app.schemas.security import User
from app.db import get_db
from app.db.models import (
    user_credits_table, credit_transactions_table, 
    gpu_instances_table, gpu_usage_table, api_usage_table
)

router = APIRouter(prefix="/api/v1/billing", tags=["Billing API"])


@router.get("/credits", response_model=UserCredits)
async def get_user_credits(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["billing:read"]))
):
    """Get current user's credit information."""
    try:
        result = await db.execute(
            select(user_credits_table).where(
                user_credits_table.c.user_id == current_user.id
            )
        )
        credits = result.fetchone()
        
        if not credits:
            # Create default credits record if none exists
            await db.execute(
                insert(user_credits_table).values(
                    user_id=current_user.id,
                    balance=0.0,
                    daily_limit=100.0,  # Default daily limit
                    total_spent=0.0,
                    updated_at=datetime.utcnow()
                )
            )
            await db.commit()
            
            # Fetch the newly created record
            result = await db.execute(
                select(user_credits_table).where(
                    user_credits_table.c.user_id == current_user.id
                )
            )
            credits = result.fetchone()
        
        return UserCredits(
            balance=credits.balance,
            daily_limit=credits.daily_limit,
            total_spent=credits.total_spent,
            updated_at=credits.updated_at
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get credits: {str(e)}"
        )


@router.post("/credits/topup", response_model=dict, dependencies=[Depends(rate_limit_dep("billing_topup", 5, 60))])
async def top_up_credits(
    topup_data: CreditTopUpRequest,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["billing:update"]))
):
    """Top up user credits (stub implementation)."""
    try:
        # This is a stub - in a real implementation, you would:
        # 1. Process payment through a payment provider
        # 2. Verify payment success
        # 3. Update credits and create transaction record
        
        # For now, just create a transaction record
        await db.execute(
            insert(credit_transactions_table).values(
                user_id=current_user.id,
                amount=topup_data.amount,
                transaction_type="credit",
                description=f"Credit top-up of ${topup_data.amount}",
                created_at=datetime.utcnow()
            )
        )
        
        # Update user credits
        await db.execute(
            update(user_credits_table).where(
                user_credits_table.c.user_id == current_user.id
            ).values(
                balance=user_credits_table.c.balance + topup_data.amount,
                updated_at=datetime.utcnow()
            )
        )
        
        await db.commit()
        
        return {
            "message": f"Credits topped up successfully. Amount: ${topup_data.amount}",
            "amount": topup_data.amount
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to top up credits: {str(e)}"
        )


@router.get("/transactions", response_model=List[CreditTransaction])
async def get_credit_transactions(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: None = Depends(PermissionChecker(["billing:read"]))
):
    """Get user's credit transaction history."""
    try:
        result = await db.execute(
            select(credit_transactions_table).where(
                credit_transactions_table.c.user_id == current_user.id
            ).order_by(
                credit_transactions_table.c.created_at.desc()
            ).limit(limit).offset(offset)
        )
        transactions = result.fetchall()
        
        return [
            CreditTransaction(
                id=txn.id,
                amount=txn.amount,
                transaction_type=txn.transaction_type,
                description=txn.description,
                created_at=txn.created_at
            )
            for txn in transactions
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transactions: {str(e)}"
        )


@router.post("/limits", response_model=dict, dependencies=[Depends(rate_limit_dep("billing_limits", 3, 60))])
async def update_spending_limits(
    limits_data: SpendingLimitUpdate,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["billing:update"]))
):
    """Update user's spending limits."""
    try:
        # Ensure user has a credits record
        result = await db.execute(
            select(user_credits_table).where(
                user_credits_table.c.user_id == current_user.id
            )
        )
        credits = result.fetchone()
        
        if not credits:
            # Create default credits record
            await db.execute(
                insert(user_credits_table).values(
                    user_id=current_user.id,
                    balance=0.0,
                    daily_limit=limits_data.daily_limit,
                    total_spent=0.0,
                    updated_at=datetime.utcnow()
                )
            )
        else:
            # Update existing record
            await db.execute(
                update(user_credits_table).where(
                    user_credits_table.c.user_id == current_user.id
                ).values(
                    daily_limit=limits_data.daily_limit,
                    updated_at=datetime.utcnow()
                )
            )
        
        await db.commit()
        
        return {
            "message": "Spending limits updated successfully",
            "daily_limit": limits_data.daily_limit
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update limits: {str(e)}"
        )


@router.get("/summary", response_model=UserCredits)
async def get_billing_summary(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["billing:read"]))
):
    """Get comprehensive billing summary for the current user."""
    try:
        result = await db.execute(
            select(user_credits_table).where(
                user_credits_table.c.user_id == current_user.id
            )
        )
        credits = result.fetchone()
        
        if not credits:
            # Create default credits record if none exists
            await db.execute(
                insert(user_credits_table).values(
                    user_id=current_user.id,
                    balance=0.0,
                    daily_limit=100.0,  # Default daily limit
                    total_spent=0.0,
                    updated_at=datetime.utcnow()
                )
            )
            await db.commit()
            
            # Fetch the newly created record
            result = await db.execute(
                select(user_credits_table).where(
                    user_credits_table.c.user_id == current_user.id
                )
            )
            credits = result.fetchone()
        
        return UserCredits(
            balance=credits.balance,
            daily_limit=credits.daily_limit,
            total_spent=credits.total_spent,
            updated_at=credits.updated_at
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch billing summary: {str(e)}"
        )


@router.get("/usage", response_model=BillingUsage)
async def get_billing_usage(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=1, le=365),
    _: None = Depends(PermissionChecker(["billing:read"]))
):
    """Get user's billing usage summary."""
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get GPU usage
        gpu_result = await db.execute(
            select(
                func.sum(gpu_usage_table.c.duration_minutes / 60.0).label('gpu_hours'),
                func.sum(gpu_usage_table.c.cost).label('gpu_cost')
            ).select_from(
                gpu_usage_table.join(
                    gpu_instances_table, 
                    gpu_usage_table.c.instance_id == gpu_instances_table.c.id
                )
            ).where(
                and_(
                    gpu_instances_table.c.user_id == current_user.id,
                    gpu_usage_table.c.created_at >= start_date,
                    gpu_usage_table.c.created_at <= end_date
                )
            )
        )
        gpu_usage = gpu_result.fetchone()
        
        # Get API usage
        api_result = await db.execute(
            select(
                func.sum(api_usage_table.c.request_count).label('api_calls'),
                func.sum(api_usage_table.c.total_cost).label('api_cost')
            ).where(
                and_(
                    api_usage_table.c.created_at >= start_date,
                    api_usage_table.c.created_at <= end_date
                )
            )
        )
        api_usage = api_result.fetchone()
        
        gpu_hours = float(gpu_usage.gpu_hours) if gpu_usage.gpu_hours else 0.0
        gpu_cost = float(gpu_usage.gpu_cost) if gpu_usage.gpu_cost else 0.0
        api_calls = int(api_usage.api_calls) if api_usage.api_calls else 0
        api_cost = float(api_usage.api_cost) if api_usage.api_cost else 0.0
        total_cost = gpu_cost + api_cost
        
        return BillingUsage(
            gpu_hours=gpu_hours,
            gpu_cost=gpu_cost,
            api_calls=api_calls,
            api_cost=api_cost,
            total_cost=total_cost
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage: {str(e)}"
        )
