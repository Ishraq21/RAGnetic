import logging
import yaml
import json
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import create_engine, select, insert, update, func

from app.db.models import workflows_table, crontab_schedule_table, periodic_task_table, periodic_task_changed_table
from app.core.config import get_db_connection, get_memory_storage_config, get_log_storage_config
from app.schemas.workflow import WorkflowCreate #
from app.core.config import _get_config_parser, get_path_settings #

logger = logging.getLogger("ragnetic")
_APP_PATHS = get_path_settings()

def is_db_configured_sync() -> bool:
    mem_config = get_memory_storage_config()
    log_config = get_log_storage_config()
    return (mem_config.get("type") in ["db", "sqlite"]) or (log_config.get("type") == "db")

def sync_workflows_from_files():
    """
    Scans the workflows directory and syncs all definitions with the database.
    - Updates the main `workflows_table` for all workflows (enabling webhooks).
    - Updates the Celery Beat tables for scheduled workflows.
    """
    if not is_db_configured_sync():
        logger.warning("Skipping workflow sync: No database is configured.")
        return

    workflows_dir = _APP_PATHS.get("WORKFLOWS_DIR")
    if not workflows_dir or not workflows_dir.is_dir():
        logger.info(f"Workflows directory not found or is not a directory: {workflows_dir}. Skipping sync.")
        return

    logger.info("Starting on-demand sync of workflow definitions and schedules with the database...")

    conn_name = (get_memory_storage_config().get("connection_name") or
                 get_log_storage_config().get("connection_name"))
    sync_conn_str = get_db_connection(conn_name).replace('+aiosqlite', '').replace('+asyncpg', '')
    engine = create_engine(sync_conn_str)

    with engine.connect() as conn:
        made_schedule_changes = False
        all_workflow_names_in_yaml = set()

        # Get all existing periodic tasks to compare against
        existing_periodic_tasks = {row.name: row for row in conn.execute(select(periodic_task_table))}

        for filepath in workflows_dir.glob("*.yaml"):
            try:
                with open(filepath, 'r') as f:
                    payload = yaml.safe_load(f)
                    wf_in = WorkflowCreate(**payload)
                    all_workflow_names_in_yaml.add(wf_in.name)

                # --- PART 1: Sync Core Workflow Definition (for Webhooks, etc.) ---
                existing_workflow = conn.execute(
                    select(workflows_table).where(workflows_table.c.name == wf_in.name)).first()

                wf_values = {
                    "agent_name": wf_in.agent_name,
                    "description": wf_in.description,
                    "definition": payload,  # Store the raw YAML/JSON payload
                    "updated_at": datetime.utcnow()
                }

                if existing_workflow:
                    stmt = update(workflows_table).where(workflows_table.c.id == existing_workflow.id).values(
                        **wf_values)
                else:
                    stmt = insert(workflows_table).values(name=wf_in.name, **wf_values)
                conn.execute(stmt)
                # --- End Part 1 ---

                # --- PART 2: Sync Schedule to Celery Beat DB ---
                trigger = payload.get("trigger", {})
                schedule_info = trigger.get("schedule")
                task_name = f"workflow:{wf_in.name}"

                if trigger.get("type") == "schedule" and isinstance(schedule_info, dict):
                    # Upsert the crontab definition
                    cron_filters = [getattr(crontab_schedule_table.c, k) == v for k, v in schedule_info.items()]
                    existing_cron = conn.execute(select(crontab_schedule_table).where(*cron_filters)).first()
                    crontab_id = existing_cron.id if existing_cron else \
                    conn.execute(insert(crontab_schedule_table).values(**schedule_info)).inserted_primary_key[0]

                    # Prepare the periodic task definition
                    task_args = json.dumps([wf_in.name, payload.get("initial_input") or {}])
                    task_values = {
                        "task": "app.workflows.tasks.run_workflow_task",
                        "crontab_id": crontab_id,
                        "args": task_args,
                        "enabled": True,
                        "date_changed": func.now(),
                    }

                    if task_name in existing_periodic_tasks:
                        stmt = update(periodic_task_table).where(periodic_task_table.c.name == task_name).values(
                            **task_values)
                    else:
                        stmt = insert(periodic_task_table).values(name=task_name, **task_values)

                    if conn.execute(stmt).rowcount > 0:
                        made_schedule_changes = True
                else:
                    # If a schedule is NOT defined in YAML, ensure it's disabled in the DB
                    if task_name in existing_periodic_tasks and existing_periodic_tasks[task_name].enabled:
                        conn.execute(update(periodic_task_table).where(periodic_task_table.c.name == task_name).values(
                            enabled=False, date_changed=func.now()))
                        made_schedule_changes = True
                        logger.info(f"Disabled schedule for '{wf_in.name}' (trigger removed from YAML).")
                # --- End Part 2 ---

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to sync workflow file '{filepath.name}': {e}", exc_info=True)
                # We need to restart the transaction after a rollback
                conn.begin() # Re-establish transaction after rollback

        # Final cleanup: disable any tasks for workflows that no longer have a YAML file
        existing_task_names = set(existing_periodic_tasks.keys())
        all_yaml_task_names = {f"workflow:{name}" for name in all_workflow_names_in_yaml}
        tasks_to_disable = existing_task_names - all_yaml_task_names

        if tasks_to_disable:
            conn.execute(
                update(periodic_task_table)
                .where(periodic_task_table.c.name.in_(tasks_to_disable), periodic_task_table.c.enabled == True)
                .values(enabled=False, date_changed=func.now())
            )
            made_schedule_changes = True
            logger.info(f"Disabled schedules for deleted workflow YAMLs: {', '.join(tasks_to_disable)}")

        if made_schedule_changes:
            # Touch the `last_update` field to force Celery Beat to reload schedules
            if conn.execute(select(periodic_task_changed_table).where(periodic_task_changed_table.c.id == 1)).first():
                conn.execute(update(periodic_task_changed_table).where(periodic_task_changed_table.c.id == 1).values(
                    last_update=func.now()))
            else:
                conn.execute(insert(periodic_task_changed_table).values(id=1, last_update=func.now()))
            logger.info("Notified Celery Beat of schedule changes.")

        conn.commit()

    engine.dispose()
    logger.info("Workflow and schedule sync complete.")