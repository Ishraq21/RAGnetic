# app/api/deployments.py
import secrets
import hashlib
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, and_

from app.core.security import PermissionChecker, get_current_user_from_api_key
from app.core.rate_limit import rate_limiter as rate_limit_dep
from app.schemas.deployments import (
    DeploymentCreate, Deployment, APIKeyCreate, 
    APIKeyInfo, APIKeyRotateResponse
)
from app.schemas.security import User
from app.db import get_db
from app.db.models import (
    deployments_table, api_keys_table, projects_table, agents_table
)

router = APIRouter(prefix="/api/v1/deployments", tags=["Deployments API"])


def generate_api_key() -> str:
    """Generate a secure API key."""
    return f"ragnetic_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def mask_api_key(api_key: str) -> str:
    """Mask an API key for display."""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]


@router.post("/api", response_model=dict, status_code=status.HTTP_201_CREATED, dependencies=[Depends(rate_limit_dep("deployment_create", 3, 60))])
async def create_api_deployment(
    deployment_data: DeploymentCreate,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["deployment:create"]))
):
    """Create an API deployment with an API key."""
    try:
        # Verify project exists and belongs to user
        project_result = await db.execute(
            select(projects_table).where(
                and_(
                    projects_table.c.id == deployment_data.project_id,
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
        
        # Verify agent exists
        agent_result = await db.execute(
            select(agents_table).where(agents_table.c.id == deployment_data.agent_id)
        )
        agent = agent_result.fetchone()
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        # Generate API key
        api_key = generate_api_key()
        key_hash = hash_api_key(api_key)
        
        # Create API key
        key_result = await db.execute(
            insert(api_keys_table).values(
                user_id=current_user.id,
                name=f"Deployment Key for {agent.name}",
                key_hash=key_hash,
                scope="viewer",  # Default scope for API deployments
                is_active=True,
                created_at=datetime.utcnow()
            ).returning(api_keys_table)
        )
        api_key_record = key_result.fetchone()
        
        # Create deployment
        deployment_result = await db.execute(
            insert(deployments_table).values(
                project_id=deployment_data.project_id,
                user_id=current_user.id,
                agent_id=deployment_data.agent_id,
                deployment_type="api",
                status="active",
                api_key_id=api_key_record.id,
                endpoint_path=f"/api/v1/invoke/{deployment_data.agent_id}",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ).returning(deployments_table)
        )
        deployment = deployment_result.fetchone()
        
        await db.commit()
        
        return {
            "deployment_id": deployment.id,
            "api_key": api_key,  # Only returned on creation
            "endpoint_path": deployment.endpoint_path,
            "status": deployment.status,
            "message": "API deployment created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create deployment: {str(e)}"
        )


@router.get("/", response_model=List[Deployment])
async def list_deployments(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["deployment:read"]))
):
    """List user's deployments."""
    try:
        result = await db.execute(
            select(deployments_table).where(
                deployments_table.c.user_id == current_user.id
            ).order_by(deployments_table.c.created_at.desc())
        )
        deployments = result.fetchall()
        
        return [
            Deployment(
                id=deployment.id,
                project_id=deployment.project_id,
                agent_id=deployment.agent_id,
                deployment_type=deployment.deployment_type,
                status=deployment.status,
                endpoint_path=deployment.endpoint_path,
                created_at=deployment.created_at
            )
            for deployment in deployments
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list deployments: {str(e)}"
        )


@router.delete("/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(rate_limit_dep("deployment_delete", 5, 60))])
async def delete_deployment(
    deployment_id: int,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["deployment:delete"]))
):
    """Delete a deployment."""
    try:
        # Verify deployment exists and belongs to user
        result = await db.execute(
            select(deployments_table).where(
                and_(
                    deployments_table.c.id == deployment_id,
                    deployments_table.c.user_id == current_user.id
                )
            )
        )
        deployment = result.fetchone()
        
        if not deployment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found"
            )
        
        # Delete deployment (CASCADE will handle API key if needed)
        await db.execute(
            delete(deployments_table).where(
                deployments_table.c.id == deployment_id
            )
        )
        
        # Also deactivate the associated API key
        if deployment.api_key_id:
            await db.execute(
                update(api_keys_table).where(
                    api_keys_table.c.id == deployment.api_key_id
                ).values(is_active=False)
            )
        
        await db.commit()
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete deployment: {str(e)}"
        )


@router.post("/{deployment_id}/regenerate-key", response_model=APIKeyRotateResponse, dependencies=[Depends(rate_limit_dep("deployment_regenerate", 3, 60))])
async def regenerate_api_key(
    deployment_id: int,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["deployment:update"]))
):
    """Regenerate API key for a deployment."""
    try:
        # Verify deployment exists and belongs to user
        result = await db.execute(
            select(deployments_table).where(
                and_(
                    deployments_table.c.id == deployment_id,
                    deployments_table.c.user_id == current_user.id
                )
            )
        )
        deployment = result.fetchone()
        
        if not deployment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found"
            )
        
        if not deployment.api_key_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Deployment has no associated API key"
            )
        
        # Generate new API key
        new_api_key = generate_api_key()
        new_key_hash = hash_api_key(new_api_key)
        
        # Update API key
        await db.execute(
            update(api_keys_table).where(
                api_keys_table.c.id == deployment.api_key_id
            ).values(
                key_hash=new_key_hash,
                last_used_at=None  # Reset last used
            )
        )
        
        await db.commit()
        
        return APIKeyRotateResponse(masked_key=mask_api_key(new_api_key))
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to regenerate API key: {str(e)}"
        )


@router.post("/{deployment_id}/toggle", response_model=dict, dependencies=[Depends(rate_limit_dep("deployment_toggle", 10, 60))])
async def toggle_deployment(
    deployment_id: int,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["deployment:update"]))
):
    """Toggle deployment status (active/inactive)."""
    try:
        # Verify deployment exists and belongs to user
        result = await db.execute(
            select(deployments_table).where(
                and_(
                    deployments_table.c.id == deployment_id,
                    deployments_table.c.user_id == current_user.id
                )
            )
        )
        deployment = result.fetchone()
        
        if not deployment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Deployment not found"
            )
        
        # Toggle status
        new_status = "inactive" if deployment.status == "active" else "active"
        
        await db.execute(
            update(deployments_table).where(
                deployments_table.c.id == deployment_id
            ).values(
                status=new_status,
                updated_at=datetime.utcnow()
            )
        )
        
        # Also toggle API key status if it exists
        if deployment.api_key_id:
            await db.execute(
                update(api_keys_table).where(
                    api_keys_table.c.id == deployment.api_key_id
                ).values(is_active=(new_status == "active"))
            )
        
        await db.commit()
        
        return {
            "deployment_id": deployment_id,
            "status": new_status,
            "message": f"Deployment {new_status}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to toggle deployment: {str(e)}"
        )
