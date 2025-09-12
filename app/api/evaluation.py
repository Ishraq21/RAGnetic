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
import re

logger = logging.getLogger("ragnetic")

router = APIRouter(prefix="/api/v1/evaluate", tags=["Evaluation API"])

_APP_PATHS = get_path_settings()
_slug = lambda s: re.sub(r'[^A-Za-z0-9_.-]+', '_', s).strip('._')



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

    # NEW: clamp num_questions to a sane range to avoid accidental huge jobs
    num_questions = max(1, min(int(num_questions), 500))

    # NEW: generate a job id and write to a unique file name to avoid collisions
    job_id = f"testset_{secrets.token_hex(6)}"
    base = os.path.splitext(os.path.basename(output_file))[0]
    safe_output_file = os.path.join(_APP_PATHS["BENCHMARK_DIR"], f"{_slug(base)}_{job_id}.json")

    logger.info(
        f"API: User '{current_user.username}' is generating {num_questions} questions for '{agent_name}' "
        f"in the background... job_id={job_id}"
    )

    def _generate_and_save_sync():
        try:
            qa_pairs = asyncio.run(generate_test_set(agent_config, num_questions))
            os.makedirs(os.path.dirname(safe_output_file), exist_ok=True)
            with open(safe_output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2)
            logger.info(f"Saved {len(qa_pairs)} questions to '{safe_output_file}'. job_id={job_id}")
        except Exception as e:
            logger.error(
                f"Background task for test set generation failed for '{agent_name}' (job_id={job_id}): {e}",
                exc_info=True
            )

    bg_tasks.add_task(_generate_and_save_sync)
    # NEW: return the resolved absolute path + job id
    return {"status": "Test set generation started.", "agent": agent_name, "job_id": job_id, "output_file": safe_output_file}



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

    # NEW: quick shape validation to fail fast on bad payloads
    if not isinstance(test_set, list) or any(not isinstance(x, dict) or "question" not in x for x in test_set):
        raise HTTPException(status_code=422, detail="`test_set` must be a list of objects each containing a `question` field.")

    run_id = f"bench_{secrets.token_hex(6)}"
    # NEW: pre-compute expected CSV path (matches benchmark’s default)
    csv_path = os.path.join(_APP_PATHS["BENCHMARK_DIR"], f"benchmark_{_slug(agent_config.name)}_{run_id}.csv")

    logger.info(
        f"API: User '{current_user.username}' is running benchmark for '{agent_name}' "
        f"in the background... run_id={run_id}"
    )

    def _run_and_save_benchmark_sync():
        try:
            engine = get_sync_db_engine()
            run_benchmark(
                agent_config,
                test_set,
                run_id=run_id,
                dataset_id=None,
                sync_engine=engine,
                export_csv_path=None  # benchmark will use the same csv_path we computed above
            )
            logger.info(f"Benchmark for '{agent_name}' completed. run_id={run_id}")
        except Exception as e:
            logger.error(f"Background task for benchmark run failed for '{agent_name}' (run_id={run_id}): {e}", exc_info=True)

    bg_tasks.add_task(_run_and_save_benchmark_sync)
    # NEW: include csv_path so clients can poll for it
    return {"status": "Benchmark started.", "agent": agent_name, "run_id": run_id, "csv_path": csv_path}


@router.get("/benchmarks")
async def list_benchmarks(
    current_user: User = Depends(PermissionChecker(["evaluation:read_benchmarks"])),
):
    """
    List available benchmark results.
    """
    try:
        benchmark_dir = _APP_PATHS["BENCHMARK_DIR"]
        if not os.path.exists(benchmark_dir):
            return {"benchmarks": []}
        
        benchmark_files = []
        for filename in os.listdir(benchmark_dir):
            if filename.endswith('.csv') and filename.startswith('benchmark_'):
                # Parse filename: benchmark_{agent_name}_{run_id}.csv
                parts = filename.replace('.csv', '').split('_')
                if len(parts) >= 3:
                    agent_name = parts[1]
                    run_id = parts[2]
                    file_path = os.path.join(benchmark_dir, filename)
                    file_stat = os.stat(file_path)
                    
                    benchmark_files.append({
                        "filename": filename,
                        "agent_name": agent_name,
                        "run_id": run_id,
                        "file_path": file_path,
                        "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        "size_bytes": file_stat.st_size
                    })
        
        # Sort by creation time, newest first
        benchmark_files.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {"benchmarks": benchmark_files}
        
    except Exception as e:
        logger.error(f"Error listing benchmarks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list benchmarks")

