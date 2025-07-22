import logging
from fastapi import APIRouter, Request, FastAPI
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Row
from typing import Dict, Any

from app.db.models import workflows_table
from app.core.config import get_db_connection, get_memory_storage_config, get_log_storage_config
from app.workflows.tasks import run_workflow_task
import json

logger = logging.getLogger(__name__)


def setup_dynamic_webhooks(app: FastAPI):
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
            logger.info(f"Found workflow: {workflow_name}")

            # The 'definition' column is already a dict because of the JSON type.
            definition = wf.get("definition")
            logger.info(f"Found definition: {definition}")
            if not isinstance(definition, dict):
                logger.error(f"Workflow '{workflow_name}' has an invalid definition type: {type(definition)}")
                continue

            trigger = definition.get("trigger", {})
            if trigger.get("type") != "api_webhook" or not trigger.get("path"):
                continue

            path = trigger["path"]

            def create_webhook_endpoint(wf_name: str, p: str):
                async def webhook_endpoint(request: Request):
                    try:
                        payload = await request.json()
                    except Exception:
                        payload = None

                    logger.info(f"[Webhook Triggered] {wf_name} via {p} with payload: {payload}")
                    run_workflow_task.delay(wf_name, payload)
                    return {
                        "status": "success",
                        "message": f"Workflow '{wf_name}' triggered.",
                    }

                return webhook_endpoint

            # Use unique function reference to avoid closure binding issue
            endpoint_func = create_webhook_endpoint(workflow_name, path)

            try:
                app.add_api_route(
                    path,
                    endpoint_func,
                    methods=["POST"],
                    tags=["Webhooks"],
                    summary=f"Trigger for '{workflow_name}'"
                )
                logger.info(f"[Webhook Ready] {workflow_name} → POST {path}")
            except Exception as e:
                logger.exception(f"Failed to add webhook route for '{workflow_name}' at '{path}': {e}")

    engine.dispose()
