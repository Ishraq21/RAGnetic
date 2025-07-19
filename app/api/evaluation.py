# app/api/evaluation.py
import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks

from app.core.security import get_http_api_key
from app.agents.config_manager import load_agent_config
from app.evaluation.dataset_generator import generate_test_set
from app.evaluation.benchmark import run_benchmark
from app.schemas.agent import AgentConfig
from app.core.config import get_path_settings

logger = logging.getLogger("ragnetic")

router = APIRouter(prefix="/api/v1/evaluate", tags=["Evaluation API"])

_APP_PATHS = get_path_settings()


@router.post("/test-set", status_code=status.HTTP_202_ACCEPTED)
async def generate_test_set_api(
    agent_name: str = Body(..., embed=True, description="The name of the agent."),
    num_questions: int = Body(50, embed=True, description="Number of questions to generate."),
    output_file: str = Body("test_set.json", embed=True, description="Output file name."),
    bg_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(get_http_api_key),
):
    """
    Generates a Q&A test set for a given agent and saves it to a file.
    Runs in the background to prevent a timeout on large datasets.
    """
    try:
        agent_config = load_agent_config(agent_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

    logger.info(f"API: Generating {num_questions} questions for agent '{agent_name}' in the background...")

    async def _generate_and_save():
        try:
            qa_pairs = await generate_test_set(agent_config, num_questions)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, indent=2)
            logger.info(f"Successfully generated and saved {len(qa_pairs)} questions to '{output_file}'.")
        except Exception as e:
            logger.error(f"Background task for test set generation failed for '{agent_name}': {e}", exc_info=True)

    bg_tasks.add_task(_generate_and_save)

    return {"status": "Test set generation started in the background.", "agent": agent_name, "output_file": output_file}


@router.post("/benchmark", status_code=status.HTTP_202_ACCEPTED)
async def run_benchmark_api(
    agent_name: str = Body(..., embed=True, description="The name of the agent."),
    test_set: List[Dict[str, Any]] = Body(..., description="The test set as a JSON array of objects."),
    bg_tasks: BackgroundTasks = BackgroundTasks(),
    api_key: str = Depends(get_http_api_key),
):
    """
    Runs a benchmark on an agent using the provided test set.
    The task runs in the background to prevent a timeout.
    """
    try:
        agent_config = load_agent_config(agent_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

    logger.info(f"API: Running benchmark for agent '{agent_name}' in the background...")

    async def _run_and_save_benchmark():
        try:
            results_df = run_benchmark(agent_config, test_set)
            results_filename = f"benchmark_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            # Use the correct path from centralized settings
            output_path = os.path.join(_APP_PATHS["BENCHMARK_DIR"], results_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Benchmark for '{agent_name}' completed. Results saved to '{output_path}'.")
        except Exception as e:
            logger.error(f"Background task for benchmark run failed for '{agent_name}': {e}", exc_info=True)

    bg_tasks.add_task(_run_and_save_benchmark)

    return {"status": "Benchmark run started in the background.", "agent": agent_name}