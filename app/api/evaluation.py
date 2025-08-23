# app/api/evaluation.py
import asyncio
import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks

from app.core.security import get_http_api_key, PermissionChecker # Import PermissionChecker
from app.agents.config_manager import load_agent_config
from app.evaluation.dataset_generator import generate_test_set
from app.evaluation.benchmark import run_benchmark
from app.schemas.agent import AgentConfig
from app.core.config import get_path_settings
from app.schemas.security import User # Import User schema
from app.db import get_sync_db_engine
import secrets

logger = logging.getLogger("ragnetic")

router = APIRouter(prefix="/api/v1/evaluate", tags=["Evaluation API"])

_APP_PATHS = get_path_settings()


@router.post("/test-set", status_code=status.HTTP_202_ACCEPTED)
async def generate_test_set_api(
    agent_name: str = Body(..., embed=True, description="The name of the agent."),
    num_questions: int = Body(50, embed=True, description="Number of questions to generate."),
    output_file: str = Body("test_set.json", embed=True, description="Output file name."),
    bg_tasks: BackgroundTasks = None,
    current_user: User = Depends(PermissionChecker(["evaluation:generate_test_set"])),
):
    if bg_tasks is None:
        from fastapi import BackgroundTasks as _BT
        bg_tasks = _BT()

    try:
        agent_config = load_agent_config(agent_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

    logger.info(f"API: User '{current_user.username}' is generating {num_questions} questions for '{agent_name}' in the background...")

    def _generate_and_save_sync():
        try:
            qa_pairs = asyncio.run(generate_test_set(agent_config, num_questions))
            safe_output_file = os.path.join(_APP_PATHS["BENCHMARK_DIR"], os.path.basename(output_file))
            os.makedirs(os.path.dirname(safe_output_file), exist_ok=True)
            with open(safe_output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2)
            logger.info(f"Saved {len(qa_pairs)} questions to '{safe_output_file}'.")
        except Exception as e:
            logger.error(f"Background task for test set generation failed for '{agent_name}': {e}", exc_info=True)

    bg_tasks.add_task(_generate_and_save_sync)
    return {"status": "Test set generation started.", "agent": agent_name, "output_file": output_file}



@router.post("/benchmark", status_code=status.HTTP_202_ACCEPTED)
async def run_benchmark_api(
    agent_name: str = Body(..., embed=True, description="The name of the agent."),
    test_set: List[Dict[str, Any]] = Body(..., description="The test set as a JSON array of objects."),
    bg_tasks: BackgroundTasks = None,
    current_user: User = Depends(PermissionChecker(["evaluation:run_benchmark"])),
):
    if bg_tasks is None:
        from fastapi import BackgroundTasks as _BT
        bg_tasks = _BT()

    try:
        agent_config = load_agent_config(agent_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

    logger.info(f"API: User '{current_user.username}' is running benchmark for '{agent_name}' in the background...")

    run_id = f"bench_{secrets.token_hex(6)}"

    def _run_and_save_benchmark_sync():
        try:
            engine = get_sync_db_engine()
            # run_benchmark is sync; it may use asyncio.run() internally — safe here in a background thread
            run_benchmark(
                agent_config,
                test_set,
                run_id=run_id,
                dataset_id=None,
                sync_engine=engine,
                export_csv_path=None
            )
            logger.info(f"Benchmark for '{agent_name}' completed. run_id={run_id}")
        except Exception as e:
            logger.error(f"Background task for benchmark run failed for '{agent_name}': {e}", exc_info=True)

    bg_tasks.add_task(_run_and_save_benchmark_sync)
    return {"status": "Benchmark started.", "agent": agent_name, "run_id": run_id}

