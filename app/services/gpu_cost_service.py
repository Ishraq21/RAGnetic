# app/services/gpu_cost_service.py

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, func

from app.db.models import gpu_instances_table, gpu_usage_table, user_credits_table, credit_transactions_table
from app.services.credit_service import CreditService

logger = logging.getLogger(__name__)


class GPUCostService:
    """Service for GPU cost calculation and tracking."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.credit_service = CreditService(db)
    
    async def calculate_gpu_cost(
        self,
        gpu_type: str,
        provider: str,
        start_time: datetime,
        end_time: datetime
    ) -> float:
        """Calculate GPU usage cost based on time and pricing."""
        try:
            # Get cost per hour from GPU service
            from app.services.gpu_service_factory import get_gpu_service_instance
            gpu_service = get_gpu_service_instance()
            providers = await gpu_service.get_gpu_providers()
            
            cost_per_hour = 1.89  # Default fallback
            for p in providers:
                if p["gpu_type"] == gpu_type and p["name"] == provider:
                    cost_per_hour = p["cost_per_hour"]
                    break
            
            # Calculate hours used
            duration = end_time - start_time
            hours_used = duration.total_seconds() / 3600
            
            # Calculate total cost
            total_cost = hours_used * cost_per_hour
            
            logger.info(f"GPU cost calculation: {gpu_type} via {provider} for {hours_used:.2f} hours = ${total_cost:.2f}")
            return total_cost
            
        except Exception as e:
            logger.error(f"Error calculating GPU cost: {e}")
            return 0.0
    
    async def track_gpu_usage(
        self,
        instance_id: int,
        usage_type: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Track GPU usage and create usage entry."""
        try:
            if end_time is None:
                end_time = datetime.utcnow()
            
            # Calculate duration in minutes
            duration = end_time - start_time
            duration_minutes = int(duration.total_seconds() / 60)
            
            # Get instance details
            result = await self.db.execute(
                select(gpu_instances_table).where(
                    gpu_instances_table.c.id == instance_id
                )
            )
            instance = result.fetchone()
            
            if not instance:
                raise ValueError(f"GPU instance {instance_id} not found")
            
            # Calculate cost
            cost = await self.calculate_gpu_cost(
                instance.gpu_type,
                instance.provider,
                start_time,
                end_time
            )
            
            # Create usage entry
            usage_entry = {
                "instance_id": instance_id,
                "usage_type": usage_type,
                "duration_minutes": duration_minutes,
                "cost": cost,
                "created_at": datetime.utcnow()
            }
            
            await self.db.execute(
                insert(gpu_usage_table).values(**usage_entry)
            )
            
            # Update instance total cost
            new_total_cost = instance.total_cost + cost
            await self.db.execute(
                update(gpu_instances_table).where(
                    gpu_instances_table.c.id == instance_id
                ).values(
                    total_cost=new_total_cost,
                    stopped_at=end_time if usage_type == "stopped" else None
                )
            )
            
            await self.db.commit()
            
            logger.info(f"Tracked GPU usage: {duration_minutes} minutes, ${cost:.2f}")
            return usage_entry
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error tracking GPU usage: {e}")
            raise
    
    async def start_gpu_session(
        self,
        instance_id: int,
        user_id: int,
        usage_type: str = "training"
    ) -> Dict[str, Any]:
        """Start a GPU usage session."""
        try:
            # Check if user has sufficient credits
            instance_result = await self.db.execute(
                select(gpu_instances_table).where(
                    gpu_instances_table.c.id == instance_id
                )
            )
            instance = instance_result.fetchone()
            
            if not instance:
                raise ValueError(f"GPU instance {instance_id} not found")
            
            # Estimate cost for 1 hour (minimum billing)
            estimated_cost = instance.cost_per_hour
            
            # Check user credits
            await self.credit_service.ensure_balance(user_id, estimated_cost)
            
            # Update instance status
            await self.db.execute(
                update(gpu_instances_table).where(
                    gpu_instances_table.c.id == instance_id
                ).values(
                    status="running",
                    started_at=datetime.utcnow()
                )
            )
            
            await self.db.commit()
            
            logger.info(f"Started GPU session for instance {instance_id}")
            return {
                "instance_id": instance_id,
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "estimated_cost_per_hour": estimated_cost
            }
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error starting GPU session: {e}")
            raise
    
    async def stop_gpu_session(
        self,
        instance_id: int,
        user_id: int,
        usage_type: str = "training"
    ) -> Dict[str, Any]:
        """Stop a GPU usage session and calculate final cost."""
        try:
            # Get instance details
            result = await self.db.execute(
                select(gpu_instances_table).where(
                    gpu_instances_table.c.id == instance_id
                )
            )
            instance = result.fetchone()
            
            if not instance:
                raise ValueError(f"GPU instance {instance_id} not found")
            
            if not instance.started_at:
                raise ValueError("GPU instance was not started")
            
            # Calculate final cost
            end_time = datetime.utcnow()
            final_cost = await self.calculate_gpu_cost(
                instance.gpu_type,
                instance.provider,
                instance.started_at,
                end_time
            )
            
            # Track usage
            usage_entry = await self.track_gpu_usage(
                instance_id,
                usage_type,
                instance.started_at,
                end_time
            )
            
            # Deduct credits from user
            await self.credit_service.deduct(
                user_id=user_id,
                amount=final_cost,
                description=f"GPU usage: {instance.gpu_type} via {instance.provider}",
                gpu_instance_id=instance_id
            )
            
            # Update instance status
            await self.db.execute(
                update(gpu_instances_table).where(
                    gpu_instances_table.c.id == instance_id
                ).values(
                    status="stopped",
                    stopped_at=end_time
                )
            )
            
            await self.db.commit()
            
            logger.info(f"Stopped GPU session for instance {instance_id}, cost: ${final_cost:.2f}")
            return {
                "instance_id": instance_id,
                "status": "stopped",
                "stopped_at": end_time.isoformat(),
                "final_cost": final_cost,
                "usage_entry": usage_entry
            }
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error stopping GPU session: {e}")
            raise
    
    async def get_user_gpu_costs(
        self,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get user's GPU costs for a date range."""
        try:
            if start_date is None:
                start_date = datetime.utcnow() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.utcnow()
            
            # Get GPU instances for user
            result = await self.db.execute(
                select(gpu_instances_table).where(
                    gpu_instances_table.c.user_id == user_id
                ).where(
                    gpu_instances_table.c.created_at >= start_date
                ).where(
                    gpu_instances_table.c.created_at <= end_date
                )
            )
            instances = result.fetchall()
            
            # Get usage entries
            usage_result = await self.db.execute(
                select(gpu_usage_table).join(
                    gpu_instances_table,
                    gpu_usage_table.c.instance_id == gpu_instances_table.c.id
                ).where(
                    gpu_instances_table.c.user_id == user_id
                ).where(
                    gpu_usage_table.c.created_at >= start_date
                ).where(
                    gpu_usage_table.c.created_at <= end_date
                )
            )
            usage_entries = usage_result.fetchall()
            
            # Calculate totals
            total_cost = sum(entry.cost for entry in usage_entries)
            total_hours = sum(entry.duration_minutes for entry in usage_entries) / 60
            
            # Group by GPU type
            gpu_costs = {}
            for entry in usage_entries:
                gpu_type = entry.gpu_type if hasattr(entry, 'gpu_type') else 'Unknown'
                if gpu_type not in gpu_costs:
                    gpu_costs[gpu_type] = {"cost": 0.0, "hours": 0.0}
                gpu_costs[gpu_type]["cost"] += entry.cost
                gpu_costs[gpu_type]["hours"] += entry.duration_minutes / 60
            
            return {
                "user_id": user_id,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_cost": total_cost,
                    "total_hours": total_hours,
                    "instance_count": len(instances)
                },
                "gpu_breakdown": gpu_costs,
                "instances": [
                    {
                        "id": inst.id,
                        "gpu_type": inst.gpu_type,
                        "provider": inst.provider,
                        "status": inst.status,
                        "total_cost": inst.total_cost,
                        "started_at": inst.started_at.isoformat() if inst.started_at else None,
                        "stopped_at": inst.stopped_at.isoformat() if inst.stopped_at else None
                    }
                    for inst in instances
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting user GPU costs: {e}")
            raise
    
    async def estimate_gpu_cost(
        self,
        gpu_type: str,
        provider: str,
        estimated_hours: float
    ) -> Dict[str, Any]:
        """Estimate GPU cost for a given duration."""
        try:
            from app.services.gpu_service_factory import get_gpu_service_instance
            gpu_service = get_gpu_service_instance()
            providers = await gpu_service.get_gpu_providers()
            
            cost_per_hour = 1.89  # Default fallback
            for p in providers:
                if p["gpu_type"] == gpu_type and p["name"] == provider:
                    cost_per_hour = p["cost_per_hour"]
                    break
            
            estimated_cost = estimated_hours * cost_per_hour
            
            return {
                "gpu_type": gpu_type,
                "provider": provider,
                "estimated_hours": estimated_hours,
                "cost_per_hour": cost_per_hour,
                "estimated_total_cost": estimated_cost,
                "estimated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error estimating GPU cost: {e}")
            raise
