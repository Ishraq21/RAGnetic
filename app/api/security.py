import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy import create_engine, select, insert
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel

# Import new tables
from app.db.models import (
    users_table, organizations_table, roles_table, user_organizations_table
)
from app.core.config import get_db_connection, get_memory_storage_config, get_log_storage_config
from app.core.security import get_http_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


# Helper function to get a synchronous database session
def get_sync_db_session():
    """Helper to get a synchronous SQLAlchemy session for the security APIs."""
    mem_cfg = get_memory_storage_config()
    log_cfg = get_log_storage_config()
    conn_name = mem_cfg.get("connection_name") if mem_cfg.get("type") in ["db", "sqlite"] else log_cfg.get(
        "connection_name")

    if not conn_name:
        raise RuntimeError("No database connection is configured.")

    conn_str = get_db_connection(conn_name)
    sync_conn_str = conn_str.replace('+aiosqlite', '').replace('+asyncpg', '')
    engine = create_engine(sync_conn_str)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


# Pydantic models for request body validation
class OrganizationCreate(BaseModel):
    name: str


class UserCreate(BaseModel):
    user_id: str
    organization_name: str
    role_name: str
    email: Optional[str] = None


@router.post("/organizations", status_code=status.HTTP_201_CREATED)
async def create_organization(
        org: OrganizationCreate,
        api_key: str = Depends(get_http_api_key),
        db: Any = Depends(get_sync_db_session)
):
    """Creates a new organization."""
    try:
        stmt = insert(organizations_table).values(name=org.name)
        result = db.execute(stmt)
        db.commit()
        return {"message": f"Organization '{org.name}' created successfully."}
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating organization: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error occurred.")


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(
        user_data: UserCreate,
        api_key: str = Depends(get_http_api_key),
        db: Any = Depends(get_sync_db_session)
):
    """
    Creates a new user and links them to an organization with a specific role.
    Assumes organizations and roles exist.
    """
    try:
        # Find organization and role IDs
        org_id = db.execute(select(organizations_table.c.id).where(
            organizations_table.c.name == user_data.organization_name)).scalar_one_or_none()
        role_id = db.execute(
            select(roles_table.c.id).where(roles_table.c.name == user_data.role_name)).scalar_one_or_none()

        if not org_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
        if not role_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found.")

        # Create user record
        user_stmt = insert(users_table).values(user_id=user_data.user_id, email=user_data.email).returning(
            users_table.c.id)
        new_user_id = db.execute(user_stmt).scalar_one()

        # Link user to organization and role
        link_stmt = insert(user_organizations_table).values(
            user_id=new_user_id,
            organization_id=org_id,
            role_id=role_id
        )
        db.execute(link_stmt)
        db.commit()

        return {
            "message": f"User '{user_data.user_id}' created and linked to organization '{user_data.organization_name}'."}

    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error occurred.")


@router.get("/organizations/{organization_name}/users", status_code=status.HTTP_200_OK)
async def list_organization_users(
        organization_name: str,
        api_key: str = Depends(get_http_api_key),
        db: Any = Depends(get_sync_db_session)
):
    """Lists all users belonging to a specific organization."""
    try:
        org_id = db.execute(select(organizations_table.c.id).where(
            organizations_table.c.name == organization_name)).scalar_one_or_none()
        if not org_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")

        stmt = select(
            users_table.c.user_id, users_table.c.email, roles_table.c.name.label("role")
        ).join(
            user_organizations_table, users_table.c.id == user_organizations_table.c.user_id
        ).join(
            roles_table, user_organizations_table.c.role_id == roles_table.c.id
        ).where(
            user_organizations_table.c.organization_id == org_id
        )

        users = db.execute(stmt).fetchall()

        user_list = [
            {"user_id": user.user_id, "email": user.email, "role": user.role}
            for user in users
        ]
        return {"organization": organization_name, "users": user_list}

    except SQLAlchemyError as e:
        logger.error(f"Error listing users for organization '{organization_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error occurred.")