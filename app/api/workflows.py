import logging
import yaml
import json
from fastapi import APIRouter, HTTPException, Request, Depends, status
from typing import Dict, Any, Optional, List

from sqlalchemy import create_engine, select, insert, update, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError

from app.db import get_db
from app.db.models import workflows_table, workflow_runs_table, human_tasks_table
from app.core.config import get_db_connection, get_memory_storage_config, get_log_storage_config
from app.workflows.engine import WorkflowEngine
from app.schemas.workflow import WorkflowCreate, WorkflowUpdate, Workflow
from app.workflows.tasks import run_workflow_task
from app.workflows.sync import sync_workflows_from_files
from app.core.security import get_http_api_key, PermissionChecker  # Import PermissionChecker
from app.schemas.security import User  # Import User schema

logger = logging.getLogger("app.api.workflows")
router = APIRouter()


def get_sync_db_engine():
    """Helper to get a synchronous SQLAlchemy engine."""
    mem_cfg = get_memory_storage_config()
    log_cfg = get_log_storage_config()
    conn_name = (
        mem_cfg.get("connection_name")
        if mem_cfg.get("type") in ["db", "sqlite"]
        else log_cfg.get("connection_name")
    )
    if not conn_name:
        raise RuntimeError("No database connection is configured for workflows.")
    conn_str = get_db_connection(conn_name)
    return create_engine(conn_str.replace("+aiosqlite", "").replace("+asyncpg", ""))


class WorkflowResumeRequest(WorkflowUpdate):
    user_input: Optional[Dict[str, Any]] = None


@router.get("/workflows", response_model=List[Workflow])
def list_workflows(
        current_user: User = Depends(PermissionChecker(["workflow:read"]))
):
    """Return all workflow definitions."""
    engine = get_sync_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(select(workflows_table)).mappings().all()
    workflows: List[Workflow] = []
    for row in rows:
        data = dict(row["definition"])
        data["id"] = row["id"]
        workflows.append(Workflow.model_validate(data))
    return workflows


@router.post("/workflows/{workflow_name}/trigger", status_code=status.HTTP_202_ACCEPTED)
async def trigger_workflow(
        workflow_name: str,
        initial_input: Optional[Dict[str, Any]] = None,
        # Users need 'workflow:trigger' permission to trigger workflows
        current_user: User = Depends(PermissionChecker(["workflow:trigger"])),
        db: AsyncSession = Depends(get_db)
):
    """
    Triggers execution of a workflow by dispatching it to a background worker.
    This endpoint is called by webhooks, manual triggers, and the Celery Beat scheduler.
    Requires: 'workflow:trigger' permission.
    """
    # First, validate that the workflow exists in the DB.
    workflow_row = await db.execute(
        select(workflows_table).where(workflows_table.c.name == workflow_name)
    )
    if not workflow_row.first():
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found.")

    # Dispatch the task to the Celery worker.
    logger.info(f"User '{current_user.username}' dispatching workflow '{workflow_name}' to background worker.")
    run_workflow_task.delay(workflow_name=workflow_name, initial_input=initial_input,
                            user_id=current_user.id)  # Pass user_id

    return {"message": f"Workflow '{workflow_name}' has been successfully dispatched for execution."}


@router.post("/workflows", status_code=status.HTTP_201_CREATED, response_model=Workflow)
async def create_workflow(
        request: Request,
        # Users need 'workflow:create' permission to create workflows
        current_user: User = Depends(PermissionChecker(["workflow:create"]))
):
    """
    Creates a new workflow definition (JSON or YAML) in the database.
    Requires: 'workflow:create' permission.
    """
    content_type = request.headers.get("content-type", "").lower()
    try:
        if "yaml" in content_type or "yml" in content_type:
            raw = await request.body()
            payload = yaml.safe_load(raw)
        else:
            payload = await request.json()
        wf_in = WorkflowCreate(**payload)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid format: {e}")

    engine = get_sync_db_engine()
    with engine.connect() as conn:
        stmt = insert(workflows_table).values(
            name=wf_in.name,
            agent_name=wf_in.agent_name,
            description=wf_in.description,
            definition=wf_in.model_dump(),
        )
        try:
            res = conn.execute(stmt)
            conn.commit()
        except IntegrityError:
            raise HTTPException(status_code=409, detail=f"Workflow '{wf_in.name}' already exists.")
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

        row = conn.execute(
            select(workflows_table).where(workflows_table.c.id == res.inserted_primary_key[0])
        ).mappings().first()

    logger.info(f"User '{current_user.username}' created workflow '{wf_in.name}'.")
    data = dict(row["definition"])
    data["id"] = row["id"]
    return Workflow.model_validate(data)


@router.get("/workflows/{workflow_name}", response_model=Workflow)
def get_workflow(
        workflow_name: str,
        # Users need 'workflow:read' permission to read workflow definitions
        current_user: User = Depends(PermissionChecker(["workflow:read"]))
):
    """
    Retrieves a workflow definition by name.
    Requires: 'workflow:read' permission.
    """
    engine = get_sync_db_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(workflows_table).where(workflows_table.c.name == workflow_name)
        ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found.")

    data = dict(row["definition"])
    data["id"] = row["id"]
    return Workflow.model_validate(data)


@router.put("/workflows/{workflow_name}", response_model=Workflow)
async def update_workflow(
        workflow_name: str,
        request: Request,
        # Users need 'workflow:update' permission to update workflows
        current_user: User = Depends(PermissionChecker(["workflow:update"]))
):
    """
    Updates an existing workflow definition (JSON or YAML).
    Requires: 'workflow:update' permission.
    """
    content_type = request.headers.get("content-type", "").lower()
    try:
        if "yaml" in content_type or "yml" in content_type:
            raw = await request.body()
            payload = yaml.safe_load(raw)
        else:
            payload = await request.json()
        upd = WorkflowUpdate(**payload).model_dump(exclude_unset=True)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid format: {e}")

    engine = get_sync_db_engine()
    with engine.connect() as conn:
        existing = conn.execute(
            select(workflows_table).where(workflows_table.c.name == workflow_name)
        ).mappings().first()
        if not existing:
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found.")

        if "steps" in upd:
            existing_definition = json.loads(existing["definition"]) if isinstance(existing["definition"], str) else \
            existing["definition"]
            full = {**existing_definition, **upd}
            upd["definition"] = full

        conn.execute(
            update(workflows_table)
            .where(workflows_table.c.name == workflow_name)
            .values(**upd)
        )
        conn.commit()

    logger.info(f"User '{current_user.username}' updated workflow '{workflow_name}'.")
    return get_workflow(upd.get("name", workflow_name))


@router.delete("/workflows/{workflow_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_workflow(
        workflow_name: str,
        # Users need 'workflow:delete' permission to delete workflows
        current_user: User = Depends(PermissionChecker(["workflow:delete"]))
):
    """
    Deletes a workflow definition.
    Requires: 'workflow:delete' permission.
    """
    engine = get_sync_db_engine()
    with engine.connect() as conn:
        res = conn.execute(
            delete(workflows_table).where(workflows_table.c.name == workflow_name)
        )
        conn.commit()
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found.")

    logger.info(f"User '{current_user.username}' deleted workflow '{workflow_name}'.")


@router.post("/workflows/{run_id}/resume", status_code=status.HTTP_202_ACCEPTED)
async def resume_workflow(
        run_id: str,
        request: WorkflowResumeRequest,
        # Users need 'workflow:resume' permission to resume workflows
        current_user: User = Depends(PermissionChecker(["workflow:resume"])),
):
    """
    Resumes a paused workflow run, supplying any human input.
    Requires: 'workflow:resume' permission.
    """
    engine = get_sync_db_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(
                workflow_runs_table,
                workflows_table.c.name.label("workflow_name")
            )
            .join(workflows_table, workflow_runs_table.c.workflow_id == workflows_table.c.id)
            .where(workflow_runs_table.c.run_id == run_id)
        ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        if row["status"] != "paused":
            raise HTTPException(status_code=409, detail=f"Run not paused (status={row['status']}).")

        conn.execute(
            update(human_tasks_table)
            .where(
                human_tasks_table.c.run_id == row["id"],
                human_tasks_table.c.status == "pending"
            )
            .values(status="completed", resolution_data=request.user_input)
        )
        conn.commit()

    try:
        logger.info(f"User '{current_user.username}' resuming workflow '{row['workflow_name']}' (Run ID: {run_id}).")
        engine = WorkflowEngine(get_sync_db_engine())
        engine.run_workflow(workflow_name=row["workflow_name"], resume_run_id=run_id,
                            user_id=current_user.id)  # Pass user_id
        return {"message": f"Workflow run '{run_id}' resumed successfully."}
    except Exception:
        logger.exception("Error resuming workflow.")
        raise HTTPException(status_code=500, detail="Error resuming workflow.")


@router.post("/workflows/sync", status_code=status.HTTP_200_OK)
def sync_all_workflows(
        # Only users with 'workflow:sync' permission can trigger workflow sync
        current_user: User = Depends(PermissionChecker(["workflow:sync"]))
):
    """
    Triggers an immediate synchronization of all workflow definitions from YAML files
    to the database. Useful for applying changes without a server restart.
    NOTE: Changes to webhook paths may still require a full server restart to apply.
    Requires: 'workflow:sync' permission.
    """
    try:
        sync_workflows_from_files()
        logger.info(f"User '{current_user.username}' triggered workflow synchronization.")
        return {"message": "Workflow definitions synced successfully from files to database."}
    except Exception as e:
        logger.error(f"Failed to sync workflows: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to sync workflows: {e}")

