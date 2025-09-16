# app/services/gpu_rate_limiter.py

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.models import gpu_instances_table, gpu_usage_table, user_credits_table
from app.core.rate_limit import RateLimitConfig, RateLimitResult, get_rate_limiter

logger = logging.getLogger(__name__)


class GPURateLimiter:
    """GPU-specific rate limiting service."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.rate_limiter = get_rate_limiter()
    
    async def check_gpu_instance_limit(
        self,
        user_id: int,
        max_instances: int = 5
    ) -> RateLimitResult:
        """Check if user can create more GPU instances."""
        try:
            # Count current running instances
            result = await self.db.execute(
                select(func.count(gpu_instances_table.c.id)).where(
                    gpu_instances_table.c.user_id == user_id
                ).where(
                    gpu_instances_table.c.status.in_(["running", "pending", "starting"])
                )
            )
            current_instances = result.scalar() or 0
            
            # Check if under limit
            if current_instances >= max_instances:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=int(datetime.utcnow().timestamp() + 3600),  # 1 hour
                    limit=max_instances,
                    retry_after=3600,
                    message=f"Maximum GPU instances limit reached ({max_instances})"
                )
            
            return RateLimitResult(
                allowed=True,
                remaining=max_instances - current_instances,
                reset_time=int(datetime.utcnow().timestamp() + 3600),
                limit=max_instances,
                message=f"GPU instance limit OK ({current_instances}/{max_instances})"
            )
            
        except Exception as e:
            logger.error(f"Error checking GPU instance limit: {e}")
            # Allow on error to avoid blocking legitimate requests
            return RateLimitResult(
                allowed=True,
                remaining=max_instances,
                reset_time=int(datetime.utcnow().timestamp() + 3600),
                limit=max_instances,
                message="GPU instance limit check failed, allowing request"
            )
    
    async def check_gpu_hours_limit(
        self,
        user_id: int,
        max_hours_per_day: float = 100.0
    ) -> RateLimitResult:
        """Check if user has exceeded daily GPU hours limit."""
        try:
            # Get today's date range
            today = datetime.utcnow().date()
            start_of_day = datetime.combine(today, datetime.min.time())
            end_of_day = datetime.combine(today, datetime.max.time())
            
            # Calculate total GPU hours used today
            result = await self.db.execute(
                select(func.sum(gpu_usage_table.c.duration_minutes)).join(
                    gpu_instances_table,
                    gpu_usage_table.c.instance_id == gpu_instances_table.c.id
                ).where(
                    gpu_instances_table.c.user_id == user_id
                ).where(
                    gpu_usage_table.c.created_at >= start_of_day
                ).where(
                    gpu_usage_table.c.created_at <= end_of_day
                )
            )
            total_minutes = result.scalar() or 0
            total_hours = total_minutes / 60.0
            
            # Check if under limit
            if total_hours >= max_hours_per_day:
                # Calculate reset time (next day)
                tomorrow = today + timedelta(days=1)
                reset_time = datetime.combine(tomorrow, datetime.min.time())
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=int(reset_time.timestamp()),
                    limit=max_hours_per_day,
                    retry_after=int((reset_time - datetime.utcnow()).total_seconds()),
                    message=f"Daily GPU hours limit reached ({total_hours:.1f}/{max_hours_per_day})"
                )
            
            return RateLimitResult(
                allowed=True,
                remaining=max_hours_per_day - total_hours,
                reset_time=int((end_of_day + timedelta(seconds=1)).timestamp()),
                limit=max_hours_per_day,
                message=f"Daily GPU hours OK ({total_hours:.1f}/{max_hours_per_day})"
            )
            
        except Exception as e:
            logger.error(f"Error checking GPU hours limit: {e}")
            # Allow on error
            return RateLimitResult(
                allowed=True,
                remaining=max_hours_per_day,
                reset_time=int((datetime.utcnow() + timedelta(days=1)).timestamp()),
                limit=max_hours_per_day,
                message="GPU hours limit check failed, allowing request"
            )
    
    async def check_gpu_cost_limit(
        self,
        user_id: int,
        max_cost_per_day: float = 50.0
    ) -> RateLimitResult:
        """Check if user has exceeded daily GPU cost limit."""
        try:
            # Get today's date range
            today = datetime.utcnow().date()
            start_of_day = datetime.combine(today, datetime.min.time())
            end_of_day = datetime.combine(today, datetime.max.time())
            
            # Calculate total GPU costs today
            result = await self.db.execute(
                select(func.sum(gpu_usage_table.c.cost)).join(
                    gpu_instances_table,
                    gpu_usage_table.c.instance_id == gpu_instances_table.c.id
                ).where(
                    gpu_instances_table.c.user_id == user_id
                ).where(
                    gpu_usage_table.c.created_at >= start_of_day
                ).where(
                    gpu_usage_table.c.created_at <= end_of_day
                )
            )
            total_cost = result.scalar() or 0.0
            
            # Check if under limit
            if total_cost >= max_cost_per_day:
                # Calculate reset time (next day)
                tomorrow = today + timedelta(days=1)
                reset_time = datetime.combine(tomorrow, datetime.min.time())
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=int(reset_time.timestamp()),
                    limit=max_cost_per_day,
                    retry_after=int((reset_time - datetime.utcnow()).total_seconds()),
                    message=f"Daily GPU cost limit reached (${total_cost:.2f}/${max_cost_per_day})"
                )
            
            return RateLimitResult(
                allowed=True,
                remaining=max_cost_per_day - total_cost,
                reset_time=int((end_of_day + timedelta(seconds=1)).timestamp()),
                limit=max_cost_per_day,
                message=f"Daily GPU cost OK (${total_cost:.2f}/${max_cost_per_day})"
            )
            
        except Exception as e:
            logger.error(f"Error checking GPU cost limit: {e}")
            # Allow on error
            return RateLimitResult(
                allowed=True,
                remaining=max_cost_per_day,
                reset_time=int((datetime.utcnow() + timedelta(days=1)).timestamp()),
                limit=max_cost_per_day,
                message="GPU cost limit check failed, allowing request"
            )
    
    async def check_gpu_provision_rate_limit(
        self,
        user_id: int,
        max_provisions_per_hour: int = 10
    ) -> RateLimitResult:
        """Check if user is provisioning GPUs too frequently."""
        try:
            # Use the existing rate limiter with a custom key
            key = f"gpu_provision:user_{user_id}"
            config = RateLimitConfig(
                requests_per_minute=max_provisions_per_hour,
                requests_per_hour=max_provisions_per_hour * 2,
                requests_per_day=max_provisions_per_hour * 24,
                burst_limit=5
            )
            
            result = self.rate_limiter.check_rate_limit(key, config)
            
            if not result.allowed:
                result.message = f"GPU provisioning rate limit exceeded ({max_provisions_per_hour}/hour)"
            else:
                result.message = f"GPU provisioning rate OK"
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking GPU provision rate limit: {e}")
            # Allow on error
            return RateLimitResult(
                allowed=True,
                remaining=max_provisions_per_hour,
                reset_time=int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
                limit=max_provisions_per_hour,
                message="GPU provision rate limit check failed, allowing request"
            )
    
    async def check_comprehensive_gpu_limits(
        self,
        user_id: int,
        max_instances: int = 5,
        max_hours_per_day: float = 100.0,
        max_cost_per_day: float = 50.0,
        max_provisions_per_hour: int = 10
    ) -> Dict[str, Any]:
        """Check all GPU-related rate limits."""
        try:
            # Check all limits
            instance_limit = await self.check_gpu_instance_limit(user_id, max_instances)
            hours_limit = await self.check_gpu_hours_limit(user_id, max_hours_per_day)
            cost_limit = await self.check_gpu_cost_limit(user_id, max_cost_per_day)
            provision_limit = await self.check_gpu_provision_rate_limit(user_id, max_provisions_per_hour)
            
            # Determine overall result
            all_limits = [instance_limit, hours_limit, cost_limit, provision_limit]
            any_exceeded = any(not limit.allowed for limit in all_limits)
            
            return {
                "allowed": not any_exceeded,
                "limits": {
                    "instances": {
                        "allowed": instance_limit.allowed,
                        "remaining": instance_limit.remaining,
                        "limit": instance_limit.limit,
                        "message": instance_limit.message
                    },
                    "hours_per_day": {
                        "allowed": hours_limit.allowed,
                        "remaining": hours_limit.remaining,
                        "limit": hours_limit.limit,
                        "message": hours_limit.message
                    },
                    "cost_per_day": {
                        "allowed": cost_limit.allowed,
                        "remaining": cost_limit.remaining,
                        "limit": cost_limit.limit,
                        "message": cost_limit.message
                    },
                    "provisions_per_hour": {
                        "allowed": provision_limit.allowed,
                        "remaining": provision_limit.remaining,
                        "limit": provision_limit.limit,
                        "message": provision_limit.message
                    }
                },
                "overall_message": "All GPU limits OK" if not any_exceeded else "One or more GPU limits exceeded"
            }
            
        except Exception as e:
            logger.error(f"Error checking comprehensive GPU limits: {e}")
            # Return permissive result on error
            return {
                "allowed": True,
                "limits": {
                    "instances": {"allowed": True, "remaining": max_instances, "limit": max_instances, "message": "Check failed, allowing"},
                    "hours_per_day": {"allowed": True, "remaining": max_hours_per_day, "limit": max_hours_per_day, "message": "Check failed, allowing"},
                    "cost_per_day": {"allowed": True, "remaining": max_cost_per_day, "limit": max_cost_per_day, "message": "Check failed, allowing"},
                    "provisions_per_hour": {"allowed": True, "remaining": max_provisions_per_hour, "limit": max_provisions_per_hour, "message": "Check failed, allowing"}
                },
                "overall_message": "GPU limit checks failed, allowing request"
            }
    
    async def get_user_gpu_usage_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user's current GPU usage statistics."""
        try:
            # Get current running instances
            result = await self.db.execute(
                select(func.count(gpu_instances_table.c.id)).where(
                    gpu_instances_table.c.user_id == user_id
                ).where(
                    gpu_instances_table.c.status.in_(["running", "pending", "starting"])
                )
            )
            current_instances = result.scalar() or 0
            
            # Get today's usage
            today = datetime.utcnow().date()
            start_of_day = datetime.combine(today, datetime.min.time())
            end_of_day = datetime.combine(today, datetime.max.time())
            
            # Hours used today
            hours_result = await self.db.execute(
                select(func.sum(gpu_usage_table.c.duration_minutes)).join(
                    gpu_instances_table,
                    gpu_usage_table.c.instance_id == gpu_instances_table.c.id
                ).where(
                    gpu_instances_table.c.user_id == user_id
                ).where(
                    gpu_usage_table.c.created_at >= start_of_day
                ).where(
                    gpu_usage_table.c.created_at <= end_of_day
                )
            )
            hours_used_today = (hours_result.scalar() or 0) / 60.0
            
            # Cost today
            cost_result = await self.db.execute(
                select(func.sum(gpu_usage_table.c.cost)).join(
                    gpu_instances_table,
                    gpu_usage_table.c.instance_id == gpu_instances_table.c.id
                ).where(
                    gpu_instances_table.c.user_id == user_id
                ).where(
                    gpu_usage_table.c.created_at >= start_of_day
                ).where(
                    gpu_usage_table.c.created_at <= end_of_day
                )
            )
            cost_today = cost_result.scalar() or 0.0
            
            return {
                "user_id": user_id,
                "current_instances": current_instances,
                "hours_used_today": hours_used_today,
                "cost_today": cost_today,
                "date": today.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user GPU usage stats: {e}")
            return {
                "user_id": user_id,
                "current_instances": 0,
                "hours_used_today": 0.0,
                "cost_today": 0.0,
                "date": datetime.utcnow().date().isoformat(),
                "error": str(e)
            }
