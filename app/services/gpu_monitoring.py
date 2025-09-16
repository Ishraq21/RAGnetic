# app/services/gpu_monitoring.py

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, func, and_, update

from app.db.models import gpu_instances_table, gpu_usage_table

logger = logging.getLogger(__name__)


class GPUEventType(Enum):
    """Types of GPU events to monitor."""
    INSTANCE_PROVISIONED = "instance_provisioned"
    INSTANCE_STARTED = "instance_started"
    INSTANCE_STOPPED = "instance_stopped"
    INSTANCE_FAILED = "instance_failed"
    COST_CALCULATED = "cost_calculated"
    RATE_LIMIT_HIT = "rate_limit_hit"
    API_ERROR = "api_error"
    PROVIDER_SWITCH = "provider_switch"


@dataclass
class GPUMetrics:
    """GPU metrics data structure."""
    timestamp: datetime
    event_type: GPUEventType
    user_id: int
    instance_id: Optional[str] = None
    gpu_type: Optional[str] = None
    provider: Optional[str] = None
    cost: Optional[float] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GPUMonitoringService:
    """Service for monitoring GPU operations and generating metrics."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.logger = logging.getLogger(f"{__name__}.GPUMonitoringService")
    
    async def log_gpu_event(self, metrics: GPUMetrics) -> None:
        """Log a GPU event for monitoring and analytics."""
        try:
            # Log to structured logging
            self.logger.info(
                f"GPU Event: {metrics.event_type.value}",
                extra={
                    "event_type": metrics.event_type.value,
                    "user_id": metrics.user_id,
                    "instance_id": metrics.instance_id,
                    "gpu_type": metrics.gpu_type,
                    "provider": metrics.provider,
                    "cost": metrics.cost,
                    "duration_seconds": metrics.duration_seconds,
                    "error_message": metrics.error_message,
                    "metadata": metrics.metadata,
                    "timestamp": metrics.timestamp.isoformat()
                }
            )
            
            # Store in database for analytics
            await self._store_gpu_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Failed to log GPU event: {e}")
    
    async def _store_gpu_metrics(self, metrics: GPUMetrics) -> None:
        """Store GPU metrics in database."""
        try:
            # Store in proper GPU tables for analytics
            if metrics.event_type == GPUEventType.INSTANCE_CREATED:
                # Create GPU instance record
                instance_data = {
                    "user_id": metrics.user_id,
                    "gpu_type": metrics.gpu_type,
                    "provider": metrics.provider,
                    "status": "running",
                    "instance_id": metrics.instance_id,
                    "cost_per_hour": metrics.cost or 0.0,
                    "total_cost": 0.0,
                    "started_at": metrics.timestamp,
                    "created_at": metrics.timestamp
                }
                await self.db.execute(
                    insert(gpu_instances_table).values(**instance_data)
                )
                
            elif metrics.event_type == GPUEventType.INSTANCE_STOPPED:
                # Update instance status and create usage record
                if metrics.instance_id:
                    # Update instance
                    await self.db.execute(
                        update(gpu_instances_table)
                        .where(gpu_instances_table.c.instance_id == metrics.instance_id)
                        .values(
                            status="stopped",
                            stopped_at=metrics.timestamp,
                            total_cost=metrics.cost or 0.0
                        )
                    )
                    
                    # Create usage record
                    if metrics.duration_seconds:
                        usage_data = {
                            "instance_id": metrics.instance_id,
                            "usage_type": "stopped",
                            "duration_minutes": int(metrics.duration_seconds / 60),
                            "cost": metrics.cost or 0.0,
                            "created_at": metrics.timestamp
                        }
                        await self.db.execute(
                            insert(gpu_usage_table).values(**usage_data)
                        )
                        
            elif metrics.event_type == GPUEventType.INSTANCE_ERROR:
                # Update instance status to error
                if metrics.instance_id:
                    await self.db.execute(
                        update(gpu_instances_table)
                        .where(gpu_instances_table.c.instance_id == metrics.instance_id)
                        .values(
                            status="error",
                            stopped_at=metrics.timestamp
                        )
                    )
            
            await self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store GPU metrics: {e}")
            await self.db.rollback()
    
    async def get_gpu_usage_metrics(
        self, 
        user_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get GPU usage metrics for monitoring dashboard."""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=7)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Get instance statistics
            query = select(
                func.count(gpu_instances_table.c.id).label("total_instances"),
                func.count().filter(gpu_instances_table.c.status == "running").label("active_instances"),
                func.count().filter(gpu_instances_table.c.status == "stopped").label("stopped_instances"),
                func.count().filter(gpu_instances_table.c.status == "error").label("failed_instances"),
                func.sum(gpu_instances_table.c.total_cost).label("total_cost"),
                func.avg(gpu_instances_table.c.total_cost).label("avg_cost_per_instance")
            ).where(
                and_(
                    gpu_instances_table.c.created_at >= start_date,
                    gpu_instances_table.c.created_at <= end_date
                )
            )
            
            if user_id:
                query = query.where(gpu_instances_table.c.user_id == user_id)
            
            result = await self.db.execute(query)
            stats = result.fetchone()
            
            # Get usage statistics
            usage_query = select(
                func.count(gpu_usage_table.c.id).label("total_usage_entries"),
                func.sum(gpu_usage_table.c.duration_minutes).label("total_minutes"),
                func.sum(gpu_usage_table.c.cost).label("total_usage_cost"),
                func.avg(gpu_usage_table.c.cost).label("avg_cost_per_session")
            ).where(
                and_(
                    gpu_usage_table.c.created_at >= start_date,
                    gpu_usage_table.c.created_at <= end_date
                )
            )
            
            if user_id:
                usage_query = usage_query.where(
                    gpu_instances_table.c.user_id == user_id
                ).join(
                    gpu_instances_table, 
                    gpu_usage_table.c.instance_id == gpu_instances_table.c.id
                )
            
            usage_result = await self.db.execute(usage_query)
            usage_stats = usage_result.fetchone()
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "instances": {
                    "total": stats.total_instances or 0,
                    "active": stats.active_instances or 0,
                    "stopped": stats.stopped_instances or 0,
                    "failed": stats.failed_instances or 0,
                    "success_rate": (
                        (stats.stopped_instances or 0) / max(stats.total_instances or 1, 1) * 100
                    )
                },
                "costs": {
                    "total_cost": float(stats.total_cost or 0),
                    "avg_cost_per_instance": float(stats.avg_cost_per_instance or 0),
                    "total_usage_cost": float(usage_stats.total_usage_cost or 0),
                    "avg_cost_per_session": float(usage_stats.avg_cost_per_session or 0)
                },
                "usage": {
                    "total_sessions": usage_stats.total_usage_entries or 0,
                    "total_minutes": usage_stats.total_minutes or 0,
                    "avg_session_duration": (
                        (usage_stats.total_minutes or 0) / max(usage_stats.total_usage_entries or 1, 1)
                    )
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get GPU usage metrics: {e}")
            return {}
    
    async def get_gpu_provider_metrics(self) -> Dict[str, Any]:
        """Get metrics comparing different GPU providers."""
        try:
            # Get provider statistics
            query = select(
                gpu_instances_table.c.provider,
                gpu_instances_table.c.gpu_type,
                func.count().label("instance_count"),
                func.count().filter(gpu_instances_table.c.status == "running").label("active_count"),
                func.sum(gpu_instances_table.c.total_cost).label("total_cost"),
                func.avg(gpu_instances_table.c.total_cost).label("avg_cost")
            ).group_by(
                gpu_instances_table.c.provider,
                gpu_instances_table.c.gpu_type
            )
            
            result = await self.db.execute(query)
            provider_stats = result.fetchall()
            
            providers = {}
            for stat in provider_stats:
                provider = stat.provider
                if provider not in providers:
                    providers[provider] = {
                        "total_instances": 0,
                        "active_instances": 0,
                        "total_cost": 0.0,
                        "gpu_types": {}
                    }
                
                providers[provider]["total_instances"] += stat.instance_count
                providers[provider]["active_instances"] += stat.active_count
                providers[provider]["total_cost"] += float(stat.total_cost or 0)
                providers[provider]["gpu_types"][stat.gpu_type] = {
                    "instances": stat.instance_count,
                    "active": stat.active_count,
                    "total_cost": float(stat.total_cost or 0),
                    "avg_cost": float(stat.avg_cost or 0)
                }
            
            return providers
            
        except Exception as e:
            self.logger.error(f"Failed to get GPU provider metrics: {e}")
            return {}
    
    async def get_gpu_cost_trends(
        self, 
        days: int = 30,
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get daily GPU cost trends for the last N days."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get daily cost trends
            query = select(
                func.date(gpu_usage_table.c.created_at).label("date"),
                func.sum(gpu_usage_table.c.cost).label("daily_cost"),
                func.count(gpu_usage_table.c.id).label("daily_sessions"),
                func.sum(gpu_usage_table.c.duration_minutes).label("daily_minutes")
            ).where(
                and_(
                    gpu_usage_table.c.created_at >= start_date,
                    gpu_usage_table.c.created_at <= end_date
                )
            ).group_by(
                func.date(gpu_usage_table.c.created_at)
            ).order_by(
                func.date(gpu_usage_table.c.created_at)
            )
            
            if user_id:
                query = query.join(
                    gpu_instances_table,
                    gpu_usage_table.c.instance_id == gpu_instances_table.c.id
                ).where(gpu_instances_table.c.user_id == user_id)
            
            result = await self.db.execute(query)
            trends = result.fetchall()
            
            return [
                {
                    "date": trend.date.isoformat(),
                    "cost": float(trend.daily_cost or 0),
                    "sessions": trend.daily_sessions,
                    "minutes": trend.daily_minutes or 0
                }
                for trend in trends
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get GPU cost trends: {e}")
            return []
    
    async def get_gpu_alerts(self) -> List[Dict[str, Any]]:
        """Get GPU-related alerts and warnings."""
        alerts = []
        
        try:
            # Check for high cost instances
            high_cost_query = select(gpu_instances_table).where(
                gpu_instances_table.c.total_cost > 100.0  # Alert if cost > $100
            )
            result = await self.db.execute(high_cost_query)
            high_cost_instances = result.fetchall()
            
            for instance in high_cost_instances:
                alerts.append({
                    "type": "high_cost",
                    "severity": "warning",
                    "message": f"Instance {instance.id} has high cost: ${instance.total_cost:.2f}",
                    "instance_id": instance.id,
                    "cost": instance.total_cost,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Check for failed instances
            failed_query = select(gpu_instances_table).where(
                gpu_instances_table.c.status == "error"
            )
            result = await self.db.execute(failed_query)
            failed_instances = result.fetchall()
            
            for instance in failed_instances:
                alerts.append({
                    "type": "instance_failed",
                    "severity": "error",
                    "message": f"Instance {instance.id} failed to provision or run",
                    "instance_id": instance.id,
                    "gpu_type": instance.gpu_type,
                    "provider": instance.provider,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Check for long-running instances
            long_running_query = select(gpu_instances_table).where(
                and_(
                    gpu_instances_table.c.status == "running",
                    gpu_instances_table.c.started_at < datetime.utcnow() - timedelta(hours=24)
                )
            )
            result = await self.db.execute(long_running_query)
            long_running_instances = result.fetchall()
            
            for instance in long_running_instances:
                runtime_hours = (datetime.utcnow() - instance.started_at).total_seconds() / 3600
                alerts.append({
                    "type": "long_running",
                    "severity": "info",
                    "message": f"Instance {instance.id} has been running for {runtime_hours:.1f} hours",
                    "instance_id": instance.id,
                    "runtime_hours": runtime_hours,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
        except Exception as e:
            self.logger.error(f"Failed to get GPU alerts: {e}")
        
        return alerts


# Context manager for timing GPU operations
class GPUTimingContext:
    """Context manager for timing GPU operations."""
    
    def __init__(self, monitoring_service: GPUMonitoringService, event_type: GPUEventType, user_id: int, **metadata):
        self.monitoring_service = monitoring_service
        self.event_type = event_type
        self.user_id = user_id
        self.metadata = metadata
        self.start_time = None
        self.error = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0
        
        if exc_type:
            self.error = str(exc_val)
            event_type = GPUEventType.API_ERROR
        else:
            event_type = self.event_type
        
        metrics = GPUMetrics(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            user_id=self.user_id,
            duration_seconds=duration,
            error_message=self.error,
            metadata=self.metadata
        )
        
        await self.monitoring_service.log_gpu_event(metrics)
