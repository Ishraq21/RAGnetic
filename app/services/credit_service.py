# app/services/credit_service.py

import logging
from datetime import datetime
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update

from app.db.models import user_credits_table, credit_transactions_table

logger = logging.getLogger(__name__)


class CreditService:
    """Service for managing user credits and transactions."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_user_credits(self, user_id: int) -> Optional[dict]:
        """Get user's current credit information."""
        try:
            result = await self.db.execute(
                select(user_credits_table).where(
                    user_credits_table.c.user_id == user_id
                )
            )
            credits = result.fetchone()
            
            if not credits:
                return None
            
            return {
                "balance": credits.balance,
                "daily_limit": credits.daily_limit,
                "total_spent": credits.total_spent,
                "updated_at": credits.updated_at
            }
        except Exception as e:
            logger.error(f"Failed to get user credits for user {user_id}: {str(e)}")
            raise
    
    async def within_limits(self, user_id: int, amount: float) -> bool:
        """Check if the amount is within user's daily spending limit."""
        try:
            credits = await self.get_user_credits(user_id)
            if not credits:
                return False
            
            # If no daily limit is set, allow any amount
            if credits["daily_limit"] is None:
                return True
            
            return amount <= credits["daily_limit"]
        except Exception as e:
            logger.error(f"Failed to check limits for user {user_id}: {str(e)}")
            return False
    
    async def ensure_balance(self, user_id: int, amount: float) -> None:
        """Ensure user has sufficient balance. Raises exception if insufficient."""
        try:
            credits = await self.get_user_credits(user_id)
            if not credits:
                raise ValueError("No credit account found. Please contact support.")
            
            if credits["balance"] < amount:
                raise ValueError(
                    f"Insufficient credits. Required: ${amount:.2f}, "
                    f"Available: ${credits['balance']:.2f}"
                )
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to check balance for user {user_id}: {str(e)}")
            raise ValueError(f"Failed to verify credits: {str(e)}")
    
    async def deduct(self, user_id: int, amount: float, description: str, gpu_instance_id: Optional[int] = None) -> None:
        """Deduct credits from user's account and create transaction record."""
        try:
            # Start transaction
            await self.db.begin()
            
            # Get current credits
            credits = await self.get_user_credits(user_id)
            if not credits:
                raise ValueError("No credit account found")
            
            if credits["balance"] < amount:
                raise ValueError(
                    f"Insufficient credits. Required: ${amount:.2f}, "
                    f"Available: ${credits['balance']:.2f}"
                )
            
            # Update user credits
            new_balance = credits["balance"] - amount
            new_total_spent = credits["total_spent"] + amount
            
            await self.db.execute(
                update(user_credits_table).where(
                    user_credits_table.c.user_id == user_id
                ).values(
                    balance=new_balance,
                    total_spent=new_total_spent,
                    updated_at=datetime.utcnow()
                )
            )
            
            # Create transaction record
            await self.db.execute(
                insert(credit_transactions_table).values(
                    user_id=user_id,
                    amount=amount,
                    transaction_type="debit",
                    description=description,
                    gpu_instance_id=gpu_instance_id,
                    created_at=datetime.utcnow()
                )
            )
            
            await self.db.commit()
            logger.info(f"Deducted ${amount:.2f} from user {user_id}: {description}")
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to deduct credits for user {user_id}: {str(e)}")
            raise
    
    async def top_up(self, user_id: int, amount: float) -> None:
        """Add credits to user's account and create transaction record."""
        try:
            # Start transaction
            await self.db.begin()
            
            # Get or create user credits
            credits = await self.get_user_credits(user_id)
            if not credits:
                # Create new credit account
                await self.db.execute(
                    insert(user_credits_table).values(
                        user_id=user_id,
                        balance=amount,
                        daily_limit=100.0,  # Default daily limit
                        total_spent=0.0,
                        updated_at=datetime.utcnow()
                    )
                )
                new_balance = amount
                new_total_spent = 0.0
            else:
                # Update existing account
                new_balance = credits["balance"] + amount
                new_total_spent = credits["total_spent"]
                
                await self.db.execute(
                    update(user_credits_table).where(
                        user_credits_table.c.user_id == user_id
                    ).values(
                        balance=new_balance,
                        updated_at=datetime.utcnow()
                    )
                )
            
            # Create transaction record
            await self.db.execute(
                insert(credit_transactions_table).values(
                    user_id=user_id,
                    amount=amount,
                    transaction_type="credit",
                    description=f"Credit top-up of ${amount:.2f}",
                    created_at=datetime.utcnow()
                )
            )
            
            await self.db.commit()
            logger.info(f"Topped up ${amount:.2f} for user {user_id}")
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to top up credits for user {user_id}: {str(e)}")
            raise
    
    async def create_default_credits(self, user_id: int) -> None:
        """Create a default credit account for a new user."""
        try:
            await self.db.execute(
                insert(user_credits_table).values(
                    user_id=user_id,
                    balance=0.0,
                    daily_limit=100.0,  # Default daily limit
                    total_spent=0.0,
                    updated_at=datetime.utcnow()
                )
            )
            await self.db.commit()
            logger.info(f"Created default credit account for user {user_id}")
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to create default credits for user {user_id}: {str(e)}")
            raise


# ----------------------------
# Top-level convenience helpers (DB-agnostic; patch-friendly for tests)
# ----------------------------

class UserCredits:
    def __init__(self, balance: float, daily_limit: float, total_spent: float):
        self.balance = float(balance)
        self.daily_limit = float(daily_limit) if daily_limit is not None else None
        self.total_spent = float(total_spent)


# In-memory store for tests (user_id -> credits)
_CREDITS_STORE: dict = {}
_RESERVED_STORE: dict = {}
_TRANSACTIONS: List[dict] = []


async def get_user_credits(user_id: int) -> UserCredits:
    c = _CREDITS_STORE.get(user_id)
    if not c:
        c = UserCredits(balance=0.0, daily_limit=100.0, total_spent=0.0)
        _CREDITS_STORE[user_id] = c
    return c


async def top_up(user_id: int, amount: float) -> None:
    if amount <= 0:
        raise ValueError("Amount must be positive")
    credits = await get_user_credits(user_id)
    credits.balance += float(amount)
    _TRANSACTIONS.append({"user_id": user_id, "amount": float(amount), "type": "topup", "created_at": datetime.utcnow()})


async def deduct(user_id: int, amount: float, description: str) -> None:
    if amount <= 0:
        raise ValueError("Amount must be positive")
    credits = await get_user_credits(user_id)
    if credits.balance < amount:
        raise ValueError("Insufficient credits")
    credits.balance -= float(amount)
    credits.total_spent += float(amount)
    _TRANSACTIONS.append({"user_id": user_id, "amount": float(amount), "type": "usage", "desc": description, "created_at": datetime.utcnow()})


async def ensure_balance(user_id: int, amount: float) -> None:
    credits = await get_user_credits(user_id)
    if credits.balance < amount:
        raise ValueError("Insufficient credits")


async def within_limits(user_id: int, amount: float) -> bool:
    credits = await get_user_credits(user_id)
    if credits.daily_limit is None:
        return True
    return float(amount) <= float(credits.daily_limit)


async def reserve_credits(user_id: int, amount: float, description: str) -> None:
    if amount <= 0:
        raise ValueError("Amount must be positive")
    credits = await get_user_credits(user_id)
    if credits.balance < amount:
        raise ValueError("Insufficient credits")
    credits.balance -= float(amount)
    _RESERVED_STORE[user_id] = _RESERVED_STORE.get(user_id, 0.0) + float(amount)
    _TRANSACTIONS.append({"user_id": user_id, "amount": float(amount), "type": "reserve", "desc": description, "created_at": datetime.utcnow()})


async def release_credits(user_id: int, amount: float, description: str) -> None:
    reserved = _RESERVED_STORE.get(user_id, 0.0)
    release_amt = min(float(amount), reserved)
    _RESERVED_STORE[user_id] = reserved - release_amt
    credits = await get_user_credits(user_id)
    credits.balance += release_amt
    _TRANSACTIONS.append({"user_id": user_id, "amount": float(amount), "type": "release", "desc": description, "created_at": datetime.utcnow()})


async def charge_reserved(user_id: int, amount: float, description: str) -> None:
    reserved = _RESERVED_STORE.get(user_id, 0.0)
    if reserved < amount:
        raise ValueError("Insufficient reserved credits")
    _RESERVED_STORE[user_id] = reserved - float(amount)
    credits = await get_user_credits(user_id)
    credits.total_spent += float(amount)
    _TRANSACTIONS.append({"user_id": user_id, "amount": float(amount), "type": "usage", "desc": description, "created_at": datetime.utcnow()})


async def update_daily_limit(user_id: int, new_limit: float) -> None:
    credits = await get_user_credits(user_id)
    credits.daily_limit = float(new_limit)


async def get_transaction_history(user_id: int, limit: int = 50) -> List[dict]:
    items = [t for t in _TRANSACTIONS if t["user_id"] == user_id]
    return items[-limit:]


async def get_daily_spending(user_id: int) -> float:
    # Simplified for tests: sum of today's usage
    today = datetime.utcnow().date()
    total = 0.0
    for t in _TRANSACTIONS:
        if t["user_id"] == user_id and t["type"] == "usage" and t["created_at"].date() == today:
            total += float(t["amount"])
    return total
