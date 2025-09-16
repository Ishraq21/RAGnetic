# app/api/gpu.py
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, and_

from app.core.security import PermissionChecker, get_current_user_from_api_key
from app.core.rate_limit import rate_limiter as rate_limit_dep
from app.schemas.gpu import (
    GPUProvider, GPUPricing, GPUProvisionRequest, 
    GPUInstance, GPUUsageEntry
)
from app.schemas.security import User
from app.db import get_db
from app.db.models import (
    gpu_providers_table, gpu_instances_table, gpu_usage_table,
    user_credits_table, projects_table
)
from app.services.gpu_service_factory import get_gpu_service_instance
from app.services.gpu_cost_service import GPUCostService
from app.services.gpu_rate_limiter import GPURateLimiter

router = APIRouter(prefix="/api/v1/gpu", tags=["GPU API"])


@router.get("/providers", response_model=List[GPUProvider])
async def get_gpu_providers(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get available GPU providers."""
    try:
        # Try to get from database first
        result = await db.execute(
            select(gpu_providers_table).where(
                gpu_providers_table.c.availability == True
            ).order_by(gpu_providers_table.c.name, gpu_providers_table.c.gpu_type)
        )
        providers = result.fetchall()
        
        if providers:
            return [
                GPUProvider(
                    name=provider.name,
                    gpu_type=provider.gpu_type,
                    cost_per_hour=provider.cost_per_hour,
                    availability=provider.availability
                )
                for provider in providers
            ]
        else:
            # Fall back to service data (mock or real based on environment)
            gpu_service = get_gpu_service_instance()
            service_providers = await gpu_service.get_gpu_providers()
            return [
                GPUProvider(
                    name=provider["name"],
                    gpu_type=provider["gpu_type"],
                    cost_per_hour=provider["cost_per_hour"],
                    availability=provider["availability"]
                )
                for provider in service_providers
            ]
    except Exception as e:
        # Fall back to service data on error
        try:
            gpu_service = get_gpu_service_instance()
            service_providers = await gpu_service.get_gpu_providers()
            return [
                GPUProvider(
                    name=provider["name"],
                    gpu_type=provider["gpu_type"],
                    cost_per_hour=provider["cost_per_hour"],
                    availability=provider["availability"]
                )
                for provider in service_providers
            ]
        except Exception as service_error:
            logger.error(f"Failed to get GPU providers from service: {service_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve GPU providers"
            )


@router.get("/pricing", response_model=List[GPUPricing])
async def get_gpu_pricing(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get GPU pricing information."""
    try:
        # Try to get from database first
        result = await db.execute(
            select(gpu_providers_table).where(
                gpu_providers_table.c.availability == True
            ).order_by(gpu_providers_table.c.cost_per_hour)
        )
        providers = result.fetchall()
        
        if providers:
            return [
                GPUPricing(
                    gpu_type=provider.gpu_type,
                    provider=provider.name,
                    cost_per_hour=provider.cost_per_hour
                )
                for provider in providers
            ]
        else:
            # Fall back to service data (mock or real based on environment)
            gpu_service = get_gpu_service_instance()
            service_pricing = await gpu_service.get_gpu_pricing()
            return [
                GPUPricing(
                    gpu_type=pricing["gpu_type"],
                    provider=pricing["provider"],
                    cost_per_hour=pricing["cost_per_hour"]
                )
                for pricing in service_pricing
            ]
    except Exception as e:
        # Fall back to service data on error
        try:
            gpu_service = get_gpu_service_instance()
            service_pricing = await gpu_service.get_gpu_pricing()
            return [
                GPUPricing(
                    gpu_type=pricing["gpu_type"],
                    provider=pricing["provider"],
                    cost_per_hour=pricing["cost_per_hour"]
                )
                for pricing in service_pricing
            ]
        except Exception as mock_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get GPU pricing: {str(e)}. Mock fallback also failed: {str(mock_error)}"
            )


@router.post("/provision", response_model=GPUInstance, status_code=status.HTTP_201_CREATED, dependencies=[Depends(rate_limit_dep("gpu_provision", 3, 60))])
async def provision_gpu(
    provision_data: GPUProvisionRequest,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:create"]))
):
    """Provision a new GPU instance with credit and limit checks."""
    try:
        # Verify project exists and belongs to user
        project_result = await db.execute(
            select(projects_table).where(
                and_(
                    projects_table.c.id == provision_data.project_id,
                    projects_table.c.user_id == current_user.id
                )
            )
        )
        project = project_result.fetchone()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Get provider pricing
        provider_result = await db.execute(
            select(gpu_providers_table).where(
                and_(
                    gpu_providers_table.c.name == provision_data.provider,
                    gpu_providers_table.c.gpu_type == provision_data.gpu_type,
                    gpu_providers_table.c.availability == True
                )
            )
        )
        provider = provider_result.fetchone()
        
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GPU provider/type not available"
            )
        
        # Calculate estimated cost
        estimated_cost = provider.cost_per_hour * provision_data.max_hours
        
        # Check user credits and daily limit
        credits_result = await db.execute(
            select(user_credits_table).where(
                user_credits_table.c.user_id == current_user.id
            )
        )
        credits = credits_result.fetchone()
        
        if not credits:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No credit account found. Please set up billing first."
            )
        
        if credits.balance < estimated_cost:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient credits. Required: ${estimated_cost:.2f}, Available: ${credits.balance:.2f}"
            )
        
        if credits.daily_limit and estimated_cost > credits.daily_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Request exceeds daily limit. Required: ${estimated_cost:.2f}, Daily limit: ${credits.daily_limit:.2f}"
            )
        
        # Create GPU instance
        instance_result = await db.execute(
            insert(gpu_instances_table).values(
                project_id=provision_data.project_id,
                user_id=current_user.id,
                gpu_type=provision_data.gpu_type,
                provider=provision_data.provider,
                status="pending",
                cost_per_hour=provider.cost_per_hour,
                total_cost=0.0,
                created_at=datetime.utcnow()
            ).returning(gpu_instances_table)
        )
        instance = instance_result.fetchone()
        
        await db.commit()
        
        return GPUInstance(
            id=instance.id,
            gpu_type=instance.gpu_type,
            provider=instance.provider,
            status=instance.status,
            cost_per_hour=instance.cost_per_hour,
            total_cost=instance.total_cost,
            started_at=instance.started_at,
            stopped_at=instance.stopped_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to provision GPU: {str(e)}"
        )


@router.get("/instances", response_model=List[GPUInstance])
async def get_gpu_instances(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get user's GPU instances."""
    try:
        query = select(gpu_instances_table).where(
            gpu_instances_table.c.user_id == current_user.id
        )
        
        if project_id:
            query = query.where(gpu_instances_table.c.project_id == project_id)
        
        if status:
            query = query.where(gpu_instances_table.c.status == status)
        
        query = query.order_by(gpu_instances_table.c.created_at.desc())
        
        result = await db.execute(query)
        instances = result.fetchall()
        
        return [
            GPUInstance(
                id=instance.id,
                gpu_type=instance.gpu_type,
                provider=instance.provider,
                status=instance.status,
                cost_per_hour=instance.cost_per_hour,
                total_cost=instance.total_cost,
                started_at=instance.started_at,
                stopped_at=instance.stopped_at
            )
            for instance in instances
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU instances: {str(e)}"
        )


@router.post("/instances/{instance_id}/stop", response_model=dict, dependencies=[Depends(rate_limit_dep("gpu_stop", 10, 60))])
async def stop_gpu_instance(
    instance_id: int,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:update"]))
):
    """Stop a GPU instance."""
    try:
        # Verify instance exists and belongs to user
        result = await db.execute(
            select(gpu_instances_table).where(
                and_(
                    gpu_instances_table.c.id == instance_id,
                    gpu_instances_table.c.user_id == current_user.id
                )
            )
        )
        instance = result.fetchone()
        
        if not instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="GPU instance not found"
            )
        
        if instance.status not in ["running", "pending"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot stop instance with status: {instance.status}"
            )
        
        # Update instance status
        await db.execute(
            update(gpu_instances_table).where(
                gpu_instances_table.c.id == instance_id
            ).values(
                status="stopped",
                stopped_at=datetime.utcnow()
            )
        )
        
        await db.commit()
        
        return {"message": "GPU instance stopped successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop GPU instance: {str(e)}"
        )


@router.get("/usage", response_model=List[GPUUsageEntry])
async def get_gpu_usage(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    instance_id: Optional[int] = Query(None, description="Filter by instance ID"),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get GPU usage entries."""
    try:
        query = select(gpu_usage_table).join(
            gpu_instances_table, 
            gpu_usage_table.c.instance_id == gpu_instances_table.c.id
        ).where(
            gpu_instances_table.c.user_id == current_user.id
        )
        
        if instance_id:
            query = query.where(gpu_usage_table.c.instance_id == instance_id)
        
        query = query.order_by(gpu_usage_table.c.created_at.desc())
        
        result = await db.execute(query)
        usage_entries = result.fetchall()
        
        return [
            GPUUsageEntry(
                instance_id=entry.instance_id,
                usage_type=entry.usage_type,
                duration_minutes=entry.duration_minutes,
                cost=entry.cost,
                created_at=entry.created_at
            )
            for entry in usage_entries
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU usage: {str(e)}"
        )


# Mock-specific endpoints for development and testing
@router.get("/mock/gpus", response_model=List[dict])
async def get_mock_gpus(
    current_user: User = Depends(get_current_user_from_api_key),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get all available mock GPU types in RunPod format."""
    try:
        gpu_service = get_gpu_service_instance()
        return await gpu_service.get_available_gpus()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get mock GPUs: {str(e)}"
        )


@router.get("/mock/gpus/{gpu_id}", response_model=dict)
async def get_mock_gpu_by_id(
    gpu_id: str,
    current_user: User = Depends(get_current_user_from_api_key),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get a specific mock GPU by its ID."""
    try:
        gpu_service = get_gpu_service_instance()
        gpu = await gpu_service.get_gpu_by_id(gpu_id)
        if not gpu:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"GPU {gpu_id} not found"
            )
        return gpu
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get mock GPU: {str(e)}"
        )


@router.post("/mock/provision", response_model=GPUInstance, status_code=status.HTTP_201_CREATED)
async def provision_mock_gpu(
    provision_data: GPUProvisionRequest,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:create"]))
):
    """Provision a GPU instance (mock or real based on environment)."""
    try:
        # Check GPU rate limits
        rate_limiter = GPURateLimiter(db)
        limits_check = await rate_limiter.check_comprehensive_gpu_limits(
            user_id=current_user.id
        )
        
        if not limits_check["allowed"]:
            # Find the first exceeded limit for error message
            exceeded_limits = []
            for limit_name, limit_info in limits_check["limits"].items():
                if not limit_info["allowed"]:
                    exceeded_limits.append(f"{limit_name}: {limit_info['message']}")
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"GPU rate limits exceeded: {'; '.join(exceeded_limits)}"
            )
        
        # Create instance (mock or real based on environment)
        gpu_service = get_gpu_service_instance()
        instance = await gpu_service.create_instance(
            gpu_type=provision_data.gpu_type,
            provider=provision_data.provider,
            user_id=current_user.id,
            project_id=provision_data.project_id,
            container_disk_gb=getattr(provision_data, 'container_disk_gb', 50),
            volume_gb=getattr(provision_data, 'volume_gb', 0),
            ports=getattr(provision_data, 'ports', '8000/http'),
            environment_vars=getattr(provision_data, 'environment_vars', {}),
            docker_args=getattr(provision_data, 'docker_args', ''),
            start_jupyter=getattr(provision_data, 'start_jupyter', False),
            start_ssh=getattr(provision_data, 'start_ssh', True)
        )
        
        return GPUInstance(
            id=instance["id"],
            gpu_type=instance["gpu_type"],
            provider=instance["provider"],
            status=instance["status"],
            cost_per_hour=instance["cost_per_hour"],
            total_cost=0.0,  # Will be calculated based on usage
            started_at=instance.get("started_at"),
            stopped_at=instance.get("stopped_at")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to provision mock GPU: {str(e)}"
        )


@router.get("/mock/instances", response_model=List[GPUInstance])
async def get_mock_gpu_instances(
    current_user: User = Depends(get_current_user_from_api_key),
    project_id: Optional[int] = Query(None, description="Filter by project ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get user's mock GPU instances."""
    try:
        gpu_service = get_gpu_service_instance()
        mock_instances = await gpu_service.get_user_instances(
            user_id=current_user.id,
            project_id=project_id,
            status=status
        )
        
        return [
            GPUInstance(
                id=instance["id"],
                gpu_type=instance["gpu_type"],
                provider=instance["provider"],
                status=instance["status"],
                cost_per_hour=instance["cost_per_hour"],
                total_cost=instance["total_cost"],
                started_at=instance["started_at"],
                stopped_at=instance["stopped_at"]
            )
            for instance in mock_instances
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get mock GPU instances: {str(e)}"
        )


@router.post("/mock/instances/{instance_id}/stop", response_model=dict)
async def stop_mock_gpu_instance(
    instance_id: int,
    current_user: User = Depends(get_current_user_from_api_key),
    _: None = Depends(PermissionChecker(["gpu:update"]))
):
    """Stop a mock GPU instance."""
    try:
        gpu_service = get_gpu_service_instance()
        success = await gpu_service.stop_instance(instance_id, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Mock GPU instance not found or cannot be stopped"
            )
        
        return {"message": "Mock GPU instance stopped successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop mock GPU instance: {str(e)}"
        )


@router.get("/mock/instances/{instance_id}/status", response_model=dict)
async def get_mock_instance_status(
    instance_id: str,
    current_user: User = Depends(get_current_user_from_api_key),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get mock instance status (RunPod compatible)."""
    try:
        gpu_service = get_gpu_service_instance()
        status_info = await gpu_service.get_instance_status(instance_id)
        return status_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get mock instance status: {str(e)}"
        )


@router.get("/mock/usage", response_model=List[GPUUsageEntry])
async def get_mock_gpu_usage(
    current_user: User = Depends(get_current_user_from_api_key),
    instance_id: Optional[int] = Query(None, description="Filter by instance ID"),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get mock GPU usage entries."""
    try:
        # Get user's instances
        gpu_service = get_gpu_service_instance()
        user_instances = await gpu_service.get_user_instances(current_user.id)
        
        if instance_id:
            user_instances = [inst for inst in user_instances if inst["id"] == instance_id]
        
        # Generate usage entries for each instance
        all_usage_entries = []
        for instance in user_instances:
            usage_entries = await gpu_service.create_usage_entries(
                instance["id"], 
                "training"
            )
            all_usage_entries.extend(usage_entries)
        
        return [
            GPUUsageEntry(
                instance_id=entry["instance_id"],
                usage_type=entry["usage_type"],
                duration_minutes=entry["duration_minutes"],
                cost=entry["cost"],
                created_at=entry["created_at"]
            )
            for entry in all_usage_entries
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get mock GPU usage: {str(e)}"
        )


@router.post("/instances/{instance_id}/start")
async def start_gpu_session(
    instance_id: int,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:update"]))
):
    """Start a GPU usage session."""
    try:
        cost_service = GPUCostService(db)
        result = await cost_service.start_gpu_session(
            instance_id=instance_id,
            user_id=current_user.id,
            usage_type="training"
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start GPU session: {str(e)}"
        )


@router.post("/instances/{instance_id}/stop")
async def stop_gpu_session(
    instance_id: int,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:update"]))
):
    """Stop a GPU usage session and calculate final cost."""
    try:
        cost_service = GPUCostService(db)
        result = await cost_service.stop_gpu_session(
            instance_id=instance_id,
            user_id=current_user.id,
            usage_type="training"
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop GPU session: {str(e)}"
        )


@router.get("/costs")
async def get_user_gpu_costs(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get user's GPU costs for a date range."""
    try:
        from datetime import datetime
        
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        cost_service = GPUCostService(db)
        result = await cost_service.get_user_gpu_costs(
            user_id=current_user.id,
            start_date=start_dt,
            end_date=end_dt
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU costs: {str(e)}"
        )


@router.post("/estimate-cost")
async def estimate_gpu_cost(
    gpu_type: str = Body(..., description="GPU type"),
    provider: str = Body(..., description="GPU provider"),
    estimated_hours: float = Body(..., description="Estimated hours of usage"),
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Estimate GPU cost for a given duration."""
    try:
        cost_service = GPUCostService(db)
        result = await cost_service.estimate_gpu_cost(
            gpu_type=gpu_type,
            provider=provider,
            estimated_hours=estimated_hours
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to estimate GPU cost: {str(e)}"
        )


@router.get("/rate-limits")
async def get_gpu_rate_limits(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get user's current GPU rate limit status."""
    try:
        rate_limiter = GPURateLimiter(db)
        limits_check = await rate_limiter.check_comprehensive_gpu_limits(
            user_id=current_user.id
        )
        return limits_check
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU rate limits: {str(e)}"
        )


@router.get("/usage-stats")
async def get_gpu_usage_stats(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get user's current GPU usage statistics."""
    try:
        rate_limiter = GPURateLimiter(db)
        stats = await rate_limiter.get_user_gpu_usage_stats(
            user_id=current_user.id
        )
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU usage stats: {str(e)}"
        )


@router.get("/service-status")
async def get_gpu_service_status(
    current_user: User = Depends(get_current_user_from_api_key),
    _: None = Depends(PermissionChecker(["gpu:read"]))
):
    """Get current GPU service status (mock vs real)."""
    try:
        import os
        from app.services.gpu_service_factory import get_gpu_service_instance
        
        # Get environment variables
        environment = os.getenv("ENVIRONMENT", "development")
        runpod_api_key = os.getenv("RUNPOD_API_KEY")
        force_mock = os.getenv("FORCE_MOCK_GPU", "false").lower() == "true"
        
        # Determine service type
        if force_mock:
            service_type = "mock"
            service_name = "Mock GPU Service"
            status_color = "orange"
            status_message = "Using mock GPU service (forced)"
        elif runpod_api_key and environment == "production":
            service_type = "real"
            service_name = "RunPod GPU Service"
            status_color = "green"
            status_message = "Using real RunPod GPU service"
        elif runpod_api_key and os.getenv("ENABLE_REAL_GPU", "false").lower() == "true":
            service_type = "real"
            service_name = "RunPod GPU Service"
            status_color = "green"
            status_message = "Using real RunPod GPU service (explicitly enabled)"
        else:
            service_type = "mock"
            service_name = "Mock GPU Service"
            status_color = "blue"
            status_message = "Using mock GPU service (development mode)"
        
        # Get service instance to verify it's working
        gpu_service = get_gpu_service_instance()
        service_class = gpu_service.__class__.__name__
        
        return {
            "service_type": service_type,
            "service_name": service_name,
            "service_class": service_class,
            "status_color": status_color,
            "status_message": status_message,
            "environment": environment,
            "has_api_key": bool(runpod_api_key),
            "force_mock": force_mock,
            "api_key_masked": f"{'*' * (len(runpod_api_key) - 4) + runpod_api_key[-4:]}" if runpod_api_key else None
        }
    except Exception as e:
        return {
            "service_type": "unknown",
            "service_name": "Unknown Service",
            "service_class": "Unknown",
            "status_color": "red",
            "status_message": f"Error determining service status: {str(e)}",
            "environment": "unknown",
            "has_api_key": False,
            "force_mock": False,
            "api_key_masked": None
        }
