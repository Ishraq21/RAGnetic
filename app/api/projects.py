# app/api/projects.py
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, func

from app.core.security import PermissionChecker, get_current_user_from_api_key
from app.core.rate_limit import rate_limiter as rate_limit_dep
from app.schemas.projects import ProjectCreate, ProjectUpdate, Project, ProjectListItem
from app.schemas.security import User
from app.db import get_db
from app.db.models import projects_table, project_agents_table, gpu_instances_table

router = APIRouter(prefix="/api/v1/projects", tags=["Projects API"])


@router.post("/", response_model=Project, status_code=status.HTTP_201_CREATED, dependencies=[Depends(rate_limit_dep("project_create", 5, 60))])
async def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["project:create"]))
):
    """Create a new project."""
    try:
        # Insert new project
        result = await db.execute(
            insert(projects_table).values(
                name=project_data.name,
                description=project_data.description,
                user_id=current_user.id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ).returning(projects_table)
        )
        project = result.fetchone()
        await db.commit()
        
        return Project(
            id=project.id,
            name=project.name,
            description=project.description,
            user_id=project.user_id,
            created_at=project.created_at,
            updated_at=project.updated_at,
            billing_session_id=project.billing_session_id
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create project: {str(e)}"
        )


@router.get("/", response_model=List[ProjectListItem])
async def list_projects(
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["project:read"]))
):
    """List all projects for the current user."""
    try:
        # Get projects with optional total cost calculation
        query = select(
            projects_table.c.id,
            projects_table.c.name,
            projects_table.c.created_at,
            func.coalesce(func.sum(gpu_instances_table.c.total_cost), 0).label('total_cost')
        ).select_from(
            projects_table.outerjoin(gpu_instances_table, projects_table.c.id == gpu_instances_table.c.project_id)
        ).where(
            projects_table.c.user_id == current_user.id
        ).group_by(
            projects_table.c.id,
            projects_table.c.name,
            projects_table.c.created_at
        ).order_by(projects_table.c.created_at.desc())
        
        result = await db.execute(query)
        projects = result.fetchall()
        
        return [
            ProjectListItem(
                id=project.id,
                name=project.name,
                created_at=project.created_at,
                total_cost=float(project.total_cost) if project.total_cost else None
            )
            for project in projects
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list projects: {str(e)}"
        )


@router.get("/{project_id}", response_model=Project)
async def get_project(
    project_id: int,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["project:read"]))
):
    """Get a specific project by ID."""
    try:
        result = await db.execute(
            select(projects_table).where(
                projects_table.c.id == project_id,
                projects_table.c.user_id == current_user.id
            )
        )
        project = result.fetchone()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        return Project(
            id=project.id,
            name=project.name,
            description=project.description,
            user_id=project.user_id,
            created_at=project.created_at,
            updated_at=project.updated_at,
            billing_session_id=project.billing_session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get project: {str(e)}"
        )


@router.put("/{project_id}", response_model=Project, dependencies=[Depends(rate_limit_dep("project_update", 10, 60))])
async def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["project:update"]))
):
    """Update a project."""
    try:
        # Check if project exists and belongs to user
        result = await db.execute(
            select(projects_table).where(
                projects_table.c.id == project_id,
                projects_table.c.user_id == current_user.id
            )
        )
        project = result.fetchone()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Update project
        update_data = {"updated_at": datetime.utcnow()}
        if project_data.name is not None:
            update_data["name"] = project_data.name
        if project_data.description is not None:
            update_data["description"] = project_data.description
        
        await db.execute(
            update(projects_table).where(
                projects_table.c.id == project_id
            ).values(**update_data)
        )
        await db.commit()
        
        # Return updated project
        result = await db.execute(
            select(projects_table).where(projects_table.c.id == project_id)
        )
        updated_project = result.fetchone()
        
        return Project(
            id=updated_project.id,
            name=updated_project.name,
            description=updated_project.description,
            user_id=updated_project.user_id,
            created_at=updated_project.created_at,
            updated_at=updated_project.updated_at,
            billing_session_id=updated_project.billing_session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update project: {str(e)}"
        )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(rate_limit_dep("project_delete", 3, 60))])
async def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["project:delete"]))
):
    """Delete a project."""
    try:
        # Check if project exists and belongs to user
        result = await db.execute(
            select(projects_table).where(
                projects_table.c.id == project_id,
                projects_table.c.user_id == current_user.id
            )
        )
        project = result.fetchone()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Delete project (CASCADE will handle related records)
        await db.execute(
            delete(projects_table).where(projects_table.c.id == project_id)
        )
        await db.commit()
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete project: {str(e)}"
        )


@router.post("/{project_id}/agents", status_code=status.HTTP_201_CREATED)
async def attach_agent_to_project(
    project_id: int,
    agent_name: str,
    current_user: User = Depends(get_current_user_from_api_key),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(PermissionChecker(["project:update"]))
):
    """Attach an agent to a project by name."""
    try:
        # Check if project exists and belongs to user
        result = await db.execute(
            select(projects_table).where(
                projects_table.c.id == project_id,
                projects_table.c.user_id == current_user.id
            )
        )
        project = result.fetchone()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Insert agent attachment
        await db.execute(
            insert(project_agents_table).values(
                project_id=project_id,
                agent_name=agent_name,
                created_at=datetime.utcnow()
            )
        )
        await db.commit()
        
        return {"message": f"Agent '{agent_name}' attached to project successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent '{agent_name}' is already attached to this project"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to attach agent: {str(e)}"
        )
