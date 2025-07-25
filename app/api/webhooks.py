import logging
from fastapi import APIRouter, HTTPException, Request, FastAPI, Depends, status
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Row
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import workflows_table
from app.core.config import get_db_connection, get_memory_storage_config, get_log_storage_config
from app.workflows.tasks import run_workflow_task
import json
from app.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

def setup_dynamic_webhooks(app_instance: FastAPI): # Renamed 'app' to 'app_instance'
    """
    Dynamically creates API webhook routes for workflows with 'trigger.type' = 'api_webhook'.
    This should be called during FastAPI startup after DB sync.
    """
    logger.info("Setting up dynamic webhook routes...")

    # --- Resolve DB connection string ---
    mem_cfg = get_memory_storage_config()
    log_cfg = get_log_storage_config()
    conn_name = (
        mem_cfg.get("connection_name")
        if mem_cfg.get("type") in ["db", "sqlite"]
        else log_cfg.get("connection_name")
    )
    if not conn_name:
        logger.warning("No valid database connection found for webhook setup.")
        return

    conn_str = get_db_connection(conn_name)
    sync_conn_str = conn_str.replace('+aiosqlite', '').replace('+asyncpg', '')
    engine = create_engine(sync_conn_str)

    with engine.connect() as conn:
        workflows = conn.execute(select(workflows_table)).fetchall()

        for row in workflows:
            wf: Dict[str, Any] = row._mapping
            workflow_name = wf.get("name")
            logger.debug(f"Found workflow: {workflow_name}")

            definition = wf.get("definition")
            if not isinstance(definition, dict):
                logger.error(f"Workflow '{workflow_name}' has an invalid definition type: {type(definition)}")
                continue

            trigger = definition.get("trigger", {})
            # CRITICAL CHANGE: Only register routes for the OLD 'api_webhook' type.
            # The new 'api_dispatch_webhook' type will be handled by the static '/dispatch' endpoint.
            if trigger.get("type") != "api_webhook" or not trigger.get("path"):
                continue

            path = trigger["path"]

            def create_webhook_endpoint(wf_name: str, p: str):
                async def webhook_endpoint(request: Request):
                    try:
                        payload = await request.json()
                    except json.JSONDecodeError:
                        payload = None # Changed from `pass` to `None` if JSON decoding fails.

                    logger.info(f"[Webhook Triggered] {wf_name} via {p} with payload: {payload}")
                    run_workflow_task.delay(wf_name, payload)
                    return {
                        "status": "success",
                        "message": f"Workflow '{wf_name}' triggered.",
                    }
                # Assign a unique __name__ to the endpoint function to avoid FastAPI conflicts
                webhook_endpoint.__name__ = f"webhook_dynamic_route_{workflow_name.replace('-', '_').replace(' ', '_')}"
                return webhook_endpoint

            try:
                app_instance.add_api_route( # Use app_instance here
                    path,
                    create_webhook_endpoint(workflow_name, path),
                    methods=["POST"],
                    tags=["Webhooks"],
                    summary=f"Trigger for '{workflow_name}'"
                )
                logger.info(f"[Webhook Ready] {workflow_name} → POST {path}")
            except Exception as e:
                logger.exception(f"Failed to add webhook route for '{workflow_name}' at '{path}': {e}")

    engine.dispose()



@router.post("/dispatch", status_code=status.HTTP_202_ACCEPTED) # This endpoint is part of the 'router'
async def generic_workflow_dispatch(
    request: Request,
    db: AsyncSession = Depends(get_db) # Need db session to check workflow existence
):
    """
    A single, generic webhook endpoint to trigger workflows by name from the payload.
    Payload MUST contain 'workflow_name' and optionally other initial input parameters.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    workflow_name = payload.get("workflow_name")
    if not workflow_name:
        raise HTTPException(status_code=400, detail="'workflow_name' is required in the payload.")

    # The rest of the payload becomes the initial_input for the workflow
    initial_input = {k: v for k, v in payload.items() if k != "workflow_name"}

    # First, validate that the workflow exists in the DB.
    workflow_row = await db.execute(
        select(workflows_table).where(workflows_table.c.name == workflow_name)
    )
    if not workflow_row.first():
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found.")

    # Dispatch the task to the Celery worker.
    logger.info(f"Dispatching workflow '{workflow_name}' via generic dispatcher to background worker.")
    run_workflow_task.delay(workflow_name=workflow_name, initial_input=initial_input)

    return {"message": f"Workflow '{workflow_name}' has been successfully dispatched for execution."}