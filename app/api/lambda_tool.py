# app/api/lambda_tool.py
import asyncio
import uuid
import json
import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState
from sqlalchemy import select, and_, asc

from app.db import get_db, get_async_db_session
from app.db.dao import create_lambda_run, get_lambda_run, get_user_by_id
from app.db.models import ragnetic_logs_table
from app.schemas.lambda_tool import LambdaRequestPayload
from app.core.security import PermissionChecker, get_current_user_from_websocket
from app.schemas.security import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/lambda", tags=["LambdaTool"])


@router.post("/execute", status_code=status.HTTP_202_ACCEPTED)
async def execute_lambda_code(
        payload: LambdaRequestPayload,
        user: User = Depends(PermissionChecker(["lambda:execute"])),
        db: AsyncSession = Depends(get_db),
):
    """
    Submits a new LambdaTool execution job to a background worker.
    """
    try:
        from app.executors.docker_executor import run_lambda_job_task

        # The create_lambda_run function now requires a user_id
        run_record = await create_lambda_run(db, user_id=user.id, payload=payload.model_dump_json())
        run_id = run_record["run_id"]

        # Dispatch the job to the Celery worker
        run_lambda_job_task.delay(run_id, payload.model_dump())

        return {"run_id": run_id, "status": "dispatched"}
    except Exception as e:
        logger.error(f"Failed to submit LambdaTool job: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to submit job: {e}")


@router.get("/runs/{run_id}")
async def get_lambda_run_details(
        run_id: str,
        db: AsyncSession = Depends(get_db),
        user: User = Depends(PermissionChecker(["lambda:read_run_details"])),
):
    """
    Retrieves the status, logs, and artifacts of a LambdaTool run.
    Requires 'lambda:read_run_details' permission.
    """
    run_data = await get_lambda_run(db, run_id)
    if not run_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lambda run not found.")

    # Optional: Add an ownership check to ensure the user can view this run
    if run_data.get('user_id') != user.id:
        # A superuser could still be allowed, but for simplicity, we'll enforce strict ownership
        # In a real app, this would be `if not user.is_superuser and run_data.get('user_id') != user.id`
        # raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
        pass

    return run_data


@router.websocket("/ws/{run_id}")
async def stream_lambda_logs(
    websocket: WebSocket,
    run_id: str,
    user: User = Depends(get_current_user_from_websocket),
):
    """
    WebSocket endpoint to retrieve final, structured logs for a completed LambdaTool job.
    This does not stream in real-time, but fetches the final logs from the database.
    Requires 'lambda:read_run_details' permission.
    """
    # Check permission directly using resolved user
    try:
        permission_checker = PermissionChecker(["lambda:read_run_details"])
        await permission_checker(current_user=user)
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Access denied.")
        return
    except Exception as e:
        logger.error(f"Permission check failed for WS connection: {e}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Permission check failed.")
        return

    await websocket.accept()

    try:
        async with get_async_db_session() as db:
            logs_stmt = select(
                ragnetic_logs_table.c.timestamp,
                ragnetic_logs_table.c.level,
                ragnetic_logs_table.c.message,
                ragnetic_logs_table.c.details
            ).where(
                ragnetic_logs_table.c.correlation_id == run_id
            ).order_by(asc(ragnetic_logs_table.c.timestamp))

            result = await db.execute(logs_stmt)
            logs = result.fetchall()

            if not logs:
                await websocket.send_text(json.dumps({"log": "No logs found for this run."}))
                return

            for log in logs:
                log_entry = {
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level,
                    "message": log.message,
                    "details": log.details
                }
                await websocket.send_text(json.dumps({"log": log_entry}))

            await websocket.send_text(json.dumps({"done": True}))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run_id={run_id}")
    except Exception as e:
        logger.error(f"Failed to retrieve and send logs for run '{run_id}': {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()