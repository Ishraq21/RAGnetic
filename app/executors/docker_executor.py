# app/executors/docker_executor.py
import asyncio
import os
import shutil
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import docker
import redis
from docker.errors import ImageNotFound, ContainerError
from sqlalchemy import insert

from app.db import get_async_db_session, initialize_db_connections
from app.db.dao import update_lambda_run_status, get_lambda_run
from app.db.models import ragnetic_logs_table
from app.schemas.lambda_tool import LambdaRequestPayload
from app.core.config import get_path_settings, get_memory_storage_config
from app.services.file_service import FileService

# This is a temporary path modification to ensure the sandbox/runner module is found
_APP_PATHS = get_path_settings()
sys.path.append(str(_APP_PATHS["PROJECT_ROOT"]))

from celery import Celery

logger = logging.getLogger(__name__)

# Define sandbox-related paths from the main project settings
_SANDBOX_DIR = _APP_PATHS["PROJECT_ROOT"] / "sandbox"
_LAMBDA_RUNS_DIR = _APP_PATHS["PROJECT_ROOT"] / ".ragnetic" / "lambda_runs"
_LAMBDA_RUNS_DIR.mkdir(parents=True, exist_ok=True)


celery_app = Celery("lambda_executor", broker="redis://localhost:6379/0")


@celery_app.task(name="lambda_executor.run_job")
def run_lambda_job_task(job_payload: Dict[str, Any]):
    conn_name = get_memory_storage_config().get("connection_name") or "ragnetic_db"
    initialize_db_connections(conn_name)

    executor = LocalDockerExecutor()
    run_id = job_payload["run_id"]
    asyncio.run(executor.execute(run_id, job_payload))


class LocalDockerExecutor:
    def __init__(self):
        try:
            self.client = docker.from_env()
            self.file_service = FileService()
        except Exception as e:
            logger.critical(f"Could not connect to Docker daemon: {e}")
            raise RuntimeError("Docker daemon not running or not accessible.") from e

    async def _ensure_image(self, image_name: str):
        try:
            await asyncio.to_thread(self.client.images.get, image_name)
        except ImageNotFound:
            logger.warning(f"Image '{image_name}' not found. Building...")
            await asyncio.to_thread(self._build_image, image_name)

    async def execute(self, run_id: str, payload: Dict[str, Any]):
        """
        Main execution method. It sets up a temporary workspace, runs the container,
        and handles output collection and cleanup.
        """
        run_payload = LambdaRequestPayload(**payload)

        workspace_dir = _LAMBDA_RUNS_DIR / run_id
        workspace_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._prepare_workspace(workspace_dir, run_id, run_payload)
            await self._run_container(run_id, workspace_dir, run_payload)
        except Exception as e:
            logger.error(f"Execution failed for run {run_id}: {e}", exc_info=True)
            async with get_async_db_session() as db:
                await update_lambda_run_status(db, run_id, "failed", error_message=str(e))
        finally:
            shutil.rmtree(workspace_dir, ignore_errors=True)
            logger.info(f"Cleaned up workspace for run {run_id}.")

    def _prepare_workspace(self, workspace_dir: Path, run_id: str, payload: LambdaRequestPayload):

        request_file_path = workspace_dir / "request.json"
        with open(request_file_path, "w") as f:
            f.write(payload.model_dump_json(indent=2))
        logger.info(f"Prepared request.json for run {run_id} (inputs already pre-staged).")

    async def _run_container(self, run_id: str, workspace_dir: Path, payload: LambdaRequestPayload):
        image_name = "ragnetic-lambda:py310-cpu"
        await self._ensure_image(image_name)

        logger.info(f"Starting Docker container for run {run_id} using image '{image_name}'.")

        # Get resource limits, ensuring they are set before use.
        cpu_limit = payload.resource_spec.cpu if payload.resource_spec.cpu is not None else "1"
        mem_limit = f"{payload.resource_spec.memory_gb}g" if payload.resource_spec.memory_gb is not None else "1g"

        env_vars_for_container = {}
        network_mode = 'none'

        try:
            container = await asyncio.to_thread(
                self.client.containers.run,
                image_name,
                detach=True,
                auto_remove=False,
                volumes={str(workspace_dir): {'bind': '/work', 'mode': 'rw'}},
                network_mode=network_mode,
                environment=env_vars_for_container,
                cpu_count=int(cpu_limit),
                mem_limit=mem_limit,
            )

            await asyncio.to_thread(container.wait)

            try:
                log_output = await asyncio.to_thread(
                    container.logs, stdout=True, stderr=True, stream=False, timestamps=False, tail="all"
                )
                log_output = log_output.decode('utf-8', errors='ignore')
                await self._save_logs(run_id, log_output)
            except docker.errors.APIError as e:
                logger.warning(f"Could not fetch container logs for run {run_id}: {e}")

            await self._process_output(run_id, workspace_dir)

            try:
                await asyncio.to_thread(container.remove, force=True)
            except Exception as e:
                logger.debug(f"Container remove failed for run {run_id}: {e}")

        except ContainerError as e:
            logger.error(f"Container failed for run {run_id}: {e}", exc_info=True)
            async with get_async_db_session() as db:
                await update_lambda_run_status(db, run_id, "failed", error_message=str(e))
        except Exception as e:
            logger.error(f"An unexpected error occurred during container execution for run {run_id}: {e}",
                         exc_info=True)
            raise

    def _build_image(self, image_name: str):
        """
        Builds the Docker image from the Dockerfile.
        """
        try:
            logger.info(f"Building Docker image '{image_name}'. This may take a few minutes...")
            _, logs = self.client.images.build(
                path=str(_SANDBOX_DIR),
                tag=image_name,
                rm=True
            )
            for log in logs:
                if 'stream' in log:
                    logger.info(log['stream'].strip())
            logger.info(f"Successfully built Docker image '{image_name}'.")
        except Exception as e:
            logger.critical(f"Failed to build Docker image '{image_name}': {e}")
            raise

    async def _save_logs(self, run_id: str, log_output: str):
        """Parse JSON log lines; insert only columns that exist in ragnetic_logs_table."""
        async with get_async_db_session() as db:
            log_lines = (log_output or "").strip().split("\n")
            for line in log_lines:
                if not line.strip():
                    continue
                try:
                    log_data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse log line for run {run_id}: {line}")
                    continue

                # Build details and include run_id there (since the table has no correlation_id column)
                base_details = log_data.get("details")
                if isinstance(base_details, dict):
                    details = {**base_details, "run_id": run_id}
                else:
                    details = {"run_id": run_id, "raw_details": base_details}

                # Safe timestamp
                ts_raw = log_data.get("timestamp")
                try:
                    ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.utcnow()
                except Exception:
                    ts = datetime.utcnow()

                # Only the columns that exist in ragnetic_logs_table
                log_entry = {
                    "timestamp": ts,
                    "level": log_data.get("level"),
                    "message": log_data.get("message"),
                    "module": log_data.get("module"),
                    "function": log_data.get("function"),
                    "line": log_data.get("line"),
                    "exc_info": log_data.get("exc_info"),
                    "details": details,
                }

                await db.execute(insert(ragnetic_logs_table).values(**log_entry))

            await db.commit()

    async def _save_final_state(self, run_id: str, workspace_dir: Path):
        output_dir = workspace_dir / "outputs"

        async with get_async_db_session() as db:
            error_file = output_dir / "error.json"
            if error_file.exists():
                with open(error_file, 'r') as f:
                    error_data = json.load(f)
                await update_lambda_run_status(
                    db,
                    run_id,
                    "failed",
                    final_state=error_data,
                    error_message=error_data.get("message")
                )
                return

            final_state = {}
            result_file = output_dir / "result.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    final_state = json.load(f)

            await update_lambda_run_status(db, run_id, "completed", final_state=final_state)

    def _safe_copy(self, src: Path, dst: Path, run_id: str):
        """Copy a file unless src and dst are the same file."""
        try:
            if src.resolve() == dst.resolve():
                logger.debug(f"Skipping redundant copy for run {run_id}: {src}")
                return
            shutil.copy2(src, dst)
        except Exception as e:
            logger.warning(f"Failed to copy output {src} for run {run_id}: {e}")

    async def _process_output(self, run_id: str, workspace_dir: Path):
        """
        Collects all outputs from the sandbox run:
        - Copies everything in /work/outputs back into host outputs dir.
        - Updates DB with result.json or error.json if present.
        """
        try:
            output_dir = workspace_dir / "outputs"
            host_output_dir = _LAMBDA_RUNS_DIR / run_id / "outputs"
            host_output_dir.mkdir(parents=True, exist_ok=True)

            # 1. Copy all files from container outputs → host outputs
            if output_dir.exists():
                for item in output_dir.iterdir():
                    dest = host_output_dir / item.name
                    if item.is_file():
                        self._safe_copy(item, dest, run_id)
                    elif item.is_dir():
                        if dest.resolve() == item.resolve():
                            logger.debug(f"Skipping redundant copy of directory for run {run_id}: {item}")
                            continue
                        shutil.copytree(item, dest, dirs_exist_ok=True)

            # 2. Process result/error JSON to update DB state
            await self._save_final_state(run_id, workspace_dir)

            logger.info(f"Lambda run '{run_id}' completed successfully. Outputs staged to {host_output_dir}.")
        except Exception as e:
            logger.error(f"Error processing output for run {run_id}: {e}", exc_info=True)
            async with get_async_db_session() as db:
                await update_lambda_run_status(
                    db,
                    run_id,
                    "failed",
                    error_message=f"Output processing failed: {e}"
                )