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


@router.get("/test-sets")
async def list_test_sets(
    current_user: User = Depends(PermissionChecker(["evaluation:read_benchmarks"])),
):
    """
    List available test sets from benchmark directory.
    """
    try:
        benchmark_dir = _APP_PATHS["BENCHMARK_DIR"]
        if not os.path.exists(benchmark_dir):
            return {"test_sets": []}
        
        test_set_files = []
        for filename in os.listdir(benchmark_dir):
            if filename.endswith('.json') and 'testset_' in filename:
                file_path = os.path.join(benchmark_dir, filename)
                file_stat = os.stat(file_path)
                
                # Parse filename: {base_name}_testset_{job_id}.json
                parts = filename.replace('.json', '').split('_testset_')
                base_name = parts[0] if len(parts) > 0 else "test_set"
                job_id = parts[1] if len(parts) > 1 else "unknown"
                
                # Try to get basic info from the file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        num_questions = len(data) if isinstance(data, list) else 0
                except Exception:
                    num_questions = 0
                
                test_set_files.append({
                    "filename": filename,
                    "display_name": base_name.replace('_', ' ').title(),
                    "base_name": base_name,
                    "job_id": job_id,
                    "file_path": file_path,
                    "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    "size_bytes": file_stat.st_size,
                    "num_questions": num_questions
                })
        
        # Sort by creation time, newest first
        test_set_files.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {"test_sets": test_set_files}
        
    except Exception as e:
        logger.error(f"Error listing test sets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list test sets")


@router.get("/test-sets/{filename}")
async def get_test_set(
    filename: str,
    current_user: User = Depends(PermissionChecker(["evaluation:read_benchmarks"])),
):
    """
    Get a specific test set by filename.
    """
    try:
        benchmark_dir = _APP_PATHS["BENCHMARK_DIR"]
        file_path = os.path.join(benchmark_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Test set not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            test_set_data = json.load(f)
        
        return {
            "filename": filename,
            "data": test_set_data,
            "num_questions": len(test_set_data) if isinstance(test_set_data, list) else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading test set {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read test set")


@router.delete("/test-sets/{filename}")
async def delete_test_set(
    filename: str,
    current_user: User = Depends(PermissionChecker(["evaluation:generate_test_set"])),
):
    """
    Delete a test set by filename.
    """
    try:
        benchmark_dir = _APP_PATHS["BENCHMARK_DIR"]
        file_path = os.path.join(benchmark_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Test set not found")
        
        # Validate filename to prevent directory traversal
        if not filename.endswith('.json') or 'testset_' not in filename:
            raise HTTPException(status_code=400, detail="Invalid test set filename")
        
        os.remove(file_path)
        
        return {"message": f"Test set {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting test set {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete test set")


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
                # Remove .csv extension and split by underscore
                name_without_ext = filename.replace('.csv', '')
                parts = name_without_ext.split('_')
                if len(parts) >= 3:
                    agent_name = parts[1]
                    # run_id is everything after the agent name (parts[2:])
                    run_id = '_'.join(parts[2:])
                    # normalize legacy filenames that include a leading 'agent_' token before the real run_id
                    if run_id.startswith('agent_'):
                        run_id = run_id[len('agent_'):]
                    file_path = os.path.join(benchmark_dir, filename)
                    file_stat = os.stat(file_path)
                    
                    benchmark_files.append({
                        "filename": filename,
                        "agent_name": agent_name,
                        "run_id": run_id,
                        "file_path": file_path,
                        "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        "size_bytes": file_stat.st_size,
                        "status": "completed"  # CSV files exist only when benchmark is completed
                    })
        
        # Sort by creation time, newest first
        benchmark_files.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {"benchmarks": benchmark_files}
        
    except Exception as e:
        logger.error(f"Error listing benchmarks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list benchmarks")


@router.get("/benchmarks/{run_id}/results")
async def get_benchmark_results(
    run_id: str,
    current_user: User = Depends(PermissionChecker(["evaluation:read_benchmarks"])),
):
    """
    Get detailed results for a specific benchmark run.
    """
    try:
        benchmark_dir = _APP_PATHS["BENCHMARK_DIR"]
        if not os.path.exists(benchmark_dir):
            raise HTTPException(status_code=404, detail="Benchmark directory not found")
        
        # Find the benchmark file
        benchmark_file = None
        agent_name = None
        
        for filename in os.listdir(benchmark_dir):
            if filename.endswith('.csv') and filename.startswith('benchmark_'):
                name_without_ext = filename.replace('.csv', '')
                parts = name_without_ext.split('_')
                if len(parts) >= 3:
                    file_agent_name = parts[1]
                    file_run_id = '_'.join(parts[2:])
                    # Strip 'agent_' prefix if present
                    if file_run_id.startswith('agent_'):
                        file_run_id = file_run_id[len('agent_'):]
                    
                    if file_run_id == run_id:
                        benchmark_file = os.path.join(benchmark_dir, filename)
                        agent_name = file_agent_name
                        break
        
        if not benchmark_file:
            raise HTTPException(status_code=404, detail=f"Benchmark with run_id '{run_id}' not found")
        
        # Read the CSV file and extract metrics
        import pandas as pd
        
        try:
            df = pd.read_csv(benchmark_file)
            
            # Calculate summary metrics from the CSV data
            summary_metrics = {}
            
            # Retrieval metrics
            if 'retrieval' in df.columns:
                # Parse JSON strings in retrieval column
                retrieval_data = []
                for idx, row in df.iterrows():
                    if pd.notna(row['retrieval']) and row['retrieval']:
                        try:
                            import ast
                            retrieval_metrics = ast.literal_eval(row['retrieval']) if isinstance(row['retrieval'], str) else row['retrieval']
                            retrieval_data.append(retrieval_metrics)
                        except:
                            continue
                
                if retrieval_data:
                    # Calculate average MRR and nDCG@10
                    mrr_values = [r.get('mrr', 0) for r in retrieval_data]
                    ndcg_values = [r.get('ndcg@10', 0) for r in retrieval_data]
                    summary_metrics['avg_mrr'] = sum(mrr_values) / len(mrr_values) if mrr_values else 0
                    summary_metrics['avg_ndcg_at_10'] = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0
            
            # Judge metrics
            if 'judge' in df.columns:
                judge_data = []
                for idx, row in df.iterrows():
                    if pd.notna(row['judge']) and row['judge']:
                        try:
                            import ast
                            judge_metrics = ast.literal_eval(row['judge']) if isinstance(row['judge'], str) else row['judge']
                            judge_data.append(judge_metrics)
                        except:
                            continue
                
                if judge_data:
                    # Calculate average judge scores
                    faithfulness_values = [j.get('faithfulness', 0) for j in judge_data]
                    relevance_values = [j.get('relevance', 0) for j in judge_data]
                    conciseness_values = [j.get('conciseness', 0) for j in judge_data]
                    coherence_values = [j.get('coherence', 0) for j in judge_data]
                    
                    summary_metrics['avg_faithfulness'] = sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0
                    summary_metrics['avg_relevance'] = sum(relevance_values) / len(relevance_values) if relevance_values else 0
                    summary_metrics['avg_conciseness'] = sum(conciseness_values) / len(conciseness_values) if conciseness_values else 0
                    summary_metrics['avg_coherence'] = sum(coherence_values) / len(coherence_values) if coherence_values else 0
            
            # Performance metrics
            if 'durations' in df.columns:
                duration_data = []
                for idx, row in df.iterrows():
                    if pd.notna(row['durations']) and row['durations']:
                        try:
                            import ast
                            durations = ast.literal_eval(row['durations']) if isinstance(row['durations'], str) else row['durations']
                            duration_data.append(durations)
                        except:
                            continue
                
                if duration_data:
                    retrieval_times = [d.get('retrieval_s', 0) for d in duration_data]
                    generation_times = [d.get('generation_s', 0) for d in duration_data]
                    
                    summary_metrics['avg_retrieval_s'] = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
                    summary_metrics['avg_generation_s'] = sum(generation_times) / len(generation_times) if generation_times else 0
            
            # Get file stats
            file_stat = os.stat(benchmark_file)
            
            return {
                "run_id": run_id,
                "agent_name": agent_name,
                "filename": os.path.basename(benchmark_file),
                "status": "completed",
                "total_items": len(df),
                "completed_items": len(df),
                "started_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "ended_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "summary_metrics": summary_metrics,
                "size_bytes": file_stat.st_size
            }
            
        except Exception as e:
            logger.error(f"Error reading benchmark CSV file '{benchmark_file}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to read benchmark results")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting benchmark results for run_id '{run_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get benchmark results")


@router.delete("/benchmarks/{run_id}")
async def delete_benchmark(
    run_id: str,
    current_user: User = Depends(PermissionChecker(["evaluation:delete_benchmarks"])),
):
    """
    Delete a benchmark result file and all associated database records.
    """
    try:
        from sqlalchemy import delete
        from app.db.models import benchmark_runs_table, benchmark_items_table
        
        benchmark_dir = _APP_PATHS["BENCHMARK_DIR"]
        if not os.path.exists(benchmark_dir):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Benchmark directory not found"
            )
        
        # Find the benchmark file with the given run_id
        benchmark_file = None
        for filename in os.listdir(benchmark_dir):
            if filename.endswith('.csv') and filename.startswith('benchmark_'):
                # Parse filename to extract run_id
                name_without_ext = filename.replace('.csv', '')
                parts = name_without_ext.split('_')
                if len(parts) >= 3:
                    file_run_id = '_'.join(parts[2:])
                    if file_run_id.startswith('agent_'):
                        file_run_id = file_run_id[len('agent_'):]
                    if file_run_id == run_id:
                        benchmark_file = os.path.join(benchmark_dir, filename)
                        break
        
        if not benchmark_file or not os.path.exists(benchmark_file):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Benchmark with run_id '{run_id}' not found"
            )
        
        # Delete from database first (CASCADE will handle benchmark_items)
        engine = get_sync_db_engine()
        with engine.connect() as connection:
            # Delete benchmark run (CASCADE will automatically delete benchmark_items)
            result = connection.execute(
                delete(benchmark_runs_table).where(benchmark_runs_table.c.run_id == run_id)
            )
            connection.commit()
            
            if result.rowcount == 0:
                logger.info(f"No database record found for benchmark run_id: {run_id}; removed CSV if present")
            else:
                logger.info(f"Deleted database records for benchmark run_id: {run_id}")
        
        # Delete the CSV file
        os.remove(benchmark_file)
        logger.info(f"Deleted benchmark file: {benchmark_file}")
        
        return {"message": f"Benchmark '{run_id}' deleted successfully from both database and filesystem"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting benchmark {run_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete benchmark: {str(e)}"
        )

