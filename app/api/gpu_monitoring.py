# app/api/gpu_monitoring.py

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.core.security import PermissionChecker, get_current_user_from_api_key
from app.schemas.security import User
from app.services.gpu_monitoring import GPUMonitoringService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/gpu/monitoring", tags=["GPU Monitoring API"])


@router.get("/metrics")
async def get_gpu_metrics(
    user_id: Optional[int] = Query(None, description="Filter by specific user ID (admin only)"),
    start_date: Optional[datetime] = Query(None, description="Start date for metrics (default: 7 days ago)"),
    end_date: Optional[datetime] = Query(None, description="End date for metrics (default: now)"),
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get comprehensive GPU usage metrics."""
    try:
        monitoring_service = GPUMonitoringService(db)
        
        # Non-admin users can only see their own metrics
        if not current_user.is_admin and user_id and user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only view your own GPU metrics"
            )
        
        # Use current user's ID if no user_id specified
        if not user_id:
            user_id = current_user.id
        
        metrics = await monitoring_service.get_gpu_usage_metrics(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "success": True,
            "data": metrics,
            "user_id": user_id,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get GPU metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU metrics: {str(e)}"
        )


@router.get("/provider-metrics")
async def get_gpu_provider_metrics(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get metrics comparing different GPU providers."""
    try:
        monitoring_service = GPUMonitoringService(db)
        provider_metrics = await monitoring_service.get_gpu_provider_metrics()
        
        return {
            "success": True,
            "data": provider_metrics,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get GPU provider metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU provider metrics: {str(e)}"
        )


@router.get("/cost-trends")
async def get_gpu_cost_trends(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze (1-365)"),
    user_id: Optional[int] = Query(None, description="Filter by specific user ID (admin only)"),
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get daily GPU cost trends for the last N days."""
    try:
        monitoring_service = GPUMonitoringService(db)
        
        # Non-admin users can only see their own trends
        if not current_user.is_admin and user_id and user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only view your own cost trends"
            )
        
        # Use current user's ID if no user_id specified
        if not user_id:
            user_id = current_user.id
        
        trends = await monitoring_service.get_gpu_cost_trends(
            days=days,
            user_id=user_id
        )
        
        return {
            "success": True,
            "data": trends,
            "user_id": user_id,
            "days_analyzed": days,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get GPU cost trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU cost trends: {str(e)}"
        )


@router.get("/alerts")
async def get_gpu_alerts(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get GPU-related alerts and warnings."""
    try:
        monitoring_service = GPUMonitoringService(db)
        alerts = await monitoring_service.get_gpu_alerts()
        
        # Filter alerts by user if not admin
        if not current_user.is_admin:
            # This would require joining with instance data to filter by user
            # For now, return all alerts but mark them appropriately
            pass
        
        return {
            "success": True,
            "data": alerts,
            "count": len(alerts),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get GPU alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU alerts: {str(e)}"
        )


@router.get("/health")
async def get_gpu_health_status(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get overall GPU system health status."""
    try:
        monitoring_service = GPUMonitoringService(db)
        
        # Get recent metrics for health assessment
        recent_metrics = await monitoring_service.get_gpu_usage_metrics(
            start_date=datetime.utcnow() - timedelta(hours=24)
        )
        
        # Get alerts
        alerts = await monitoring_service.get_gpu_alerts()
        
        # Calculate health score
        health_score = 100
        health_issues = []
        
        # Check success rate
        success_rate = recent_metrics.get("instances", {}).get("success_rate", 100)
        if success_rate < 90:
            health_score -= 20
            health_issues.append(f"Low success rate: {success_rate:.1f}%")
        
        # Check for failed instances
        failed_instances = recent_metrics.get("instances", {}).get("failed", 0)
        if failed_instances > 0:
            health_score -= 10 * min(failed_instances, 5)  # Max 50 points deduction
            health_issues.append(f"{failed_instances} failed instances")
        
        # Check for high-cost alerts
        high_cost_alerts = [a for a in alerts if a.get("type") == "high_cost"]
        if high_cost_alerts:
            health_score -= 5 * len(high_cost_alerts)
            health_issues.append(f"{len(high_cost_alerts)} high-cost instances")
        
        # Check for error alerts
        error_alerts = [a for a in alerts if a.get("severity") == "error"]
        if error_alerts:
            health_score -= 15 * len(error_alerts)
            health_issues.append(f"{len(error_alerts)} error alerts")
        
        # Determine health status
        if health_score >= 90:
            status_text = "healthy"
            status_color = "green"
        elif health_score >= 70:
            status_text = "warning"
            status_color = "yellow"
        else:
            status_text = "critical"
            status_color = "red"
        
        return {
            "success": True,
            "data": {
                "health_score": max(0, health_score),
                "status": status_text,
                "status_color": status_color,
                "issues": health_issues,
                "recent_metrics": recent_metrics,
                "alert_count": len(alerts),
                "last_updated": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get GPU health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU health status: {str(e)}"
        )


@router.get("/dashboard")
async def get_gpu_monitoring_dashboard(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get comprehensive GPU monitoring dashboard data."""
    try:
        monitoring_service = GPUMonitoringService(db)
        
        # Get all monitoring data
        metrics = await monitoring_service.get_gpu_usage_metrics(
            user_id=current_user.id,
            start_date=datetime.utcnow() - timedelta(days=7)
        )
        
        provider_metrics = await monitoring_service.get_gpu_provider_metrics()
        cost_trends = await monitoring_service.get_gpu_cost_trends(days=7, user_id=current_user.id)
        alerts = await monitoring_service.get_gpu_alerts()
        
        # Calculate summary statistics
        total_cost = metrics.get("costs", {}).get("total_cost", 0)
        total_instances = metrics.get("instances", {}).get("total_instances", 0)
        active_instances = metrics.get("instances", {}).get("active_instances", 0)
        success_rate = metrics.get("instances", {}).get("success_rate", 100)
        
        return {
            "success": True,
            "data": {
                "summary": {
                    "total_cost": total_cost,
                    "total_instances": total_instances,
                    "active_instances": active_instances,
                    "success_rate": success_rate,
                    "alert_count": len(alerts)
                },
                "metrics": metrics,
                "provider_metrics": provider_metrics,
                "cost_trends": cost_trends,
                "alerts": alerts[:10],  # Limit to 10 most recent alerts
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get GPU monitoring dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU monitoring dashboard: {str(e)}"
        )
