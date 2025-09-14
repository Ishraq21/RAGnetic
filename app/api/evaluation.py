# app/api/evaluation.py
import asyncio
import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks
from fastapi import Path as _Path

from app.core.security import get_http_api_key, PermissionChecker # Import PermissionChecker
from app.agents.config_manager import load_agent_config
from app.evaluation.dataset_generator import generate_test_set
from app.evaluation.benchmark import run_benchmark
from app.schemas.agent import AgentConfig
from app.core.config import get_path_settings
from app.schemas.security import User # Import User schema
from app.db import get_sync_db_engine
from sqlalchemy import update as _update
from app.db.models import benchmark_runs_table
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
    purpose: str = Body("", embed=True, description="Purpose or change note for audit."),
    tags: List[str] = Body([], embed=True, description="Tags for audit/governance."),
    dataset_id: Optional[str] = Body(None, embed=True, description="Dataset identifier for provenance and audit."),
    test_set_file: Optional[str] = Body(None, embed=True, description="Original test set filename for metadata."),
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
            
            # Improve dataset_id if not provided
            effective_dataset_id = dataset_id
            if not effective_dataset_id and test_set_file:
                # Extract dataset name from filename (remove extension)
                import os
                effective_dataset_id = os.path.splitext(test_set_file)[0]
            
            run_benchmark(
                agent_config,
                test_set,
                run_id=run_id,
                dataset_id=effective_dataset_id,
                sync_engine=engine,
                export_csv_path=None,  # benchmark will use the same csv_path we computed above
                audit_info={
                    "requested_by": current_user.username,
                    "purpose": purpose or None,  # Convert empty string to None
                    "tags": tags or [],
                    "ip": getattr(current_user, "last_ip", None),
                    "test_set_file": test_set_file  # Add original filename to audit info
                }
            )
            logger.info(f"Benchmark for '{agent_name}' completed. run_id={run_id}")
        except Exception as e:
            logger.error(f"Background task for benchmark run failed for '{agent_name}' (run_id={run_id}): {e}", exc_info=True)

    bg_tasks.add_task(_run_and_save_benchmark_sync)
    # NEW: include csv_path so clients can poll for it
    return {"status": "Benchmark started.", "agent": agent_name, "run_id": run_id, "csv_path": csv_path}
@router.post("/benchmark/{run_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_benchmark(
    run_id: str = _Path(..., description="Benchmark run ID to cancel."),
    current_user: User = Depends(PermissionChecker(["evaluation:run_benchmark"]))
):
    try:
        engine = get_sync_db_engine()
        with engine.begin() as conn:
            res = conn.execute(
                _update(benchmark_runs_table)
                .where(benchmark_runs_table.c.run_id == run_id)
                .where(benchmark_runs_table.c.status == "running")
                .values(status="aborted")
            )
            if res.rowcount == 0:
                # Either not found or not running
                return {"status": "noop", "message": "Run not found or not in running state."}
        logger.info(f"User '{current_user.username}' requested cancel for run_id={run_id}")
        return {"status": "cancelled", "run_id": run_id}
    except Exception as e:
        logger.error(f"Failed to cancel benchmark {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to cancel benchmark")


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
            
            # Enterprise metrics from database
            try:
                from sqlalchemy import select
                from app.db.models import benchmark_runs_table
                
                engine = get_sync_db_engine()
                with engine.connect() as conn:
                    stmt = select(benchmark_runs_table).where(benchmark_runs_table.c.run_id == run_id)
                    result = conn.execute(stmt).fetchone()
                    
                    if result and result.summary_metrics:
                        import json
                        try:
                            db_summary = json.loads(result.summary_metrics) if isinstance(result.summary_metrics, str) else result.summary_metrics
                            # Add enterprise metrics from database
                            enterprise_metrics = ['safety_pass_rate', 'pii_leak_rate', 'toxicity_rate', 'multi_turn_items', 'bias_delta']
                            for metric in enterprise_metrics:
                                if metric in db_summary:
                                    summary_metrics[metric] = db_summary[metric]
                            
                            # Add cost metrics
                            if 'avg_total_cost_usd' in db_summary:
                                summary_metrics['total_cost_usd'] = db_summary['avg_total_cost_usd']
                            
                            # Add hit@k metrics
                            if 'avg_retrieval_hit@5' in db_summary:
                                summary_metrics['avg_hit_at_k'] = db_summary['avg_retrieval_hit@5']
                            
                        except Exception as e:
                            logger.warning(f"Could not parse database summary metrics: {e}")
                
            except Exception as e:
                logger.warning(f"Could not fetch enterprise metrics from database: {e}")
            
            # Get file stats
            file_stat = os.stat(benchmark_file)
            
            # Try to get additional data from database
            config_snapshot = {}
            audit_info = {}
            dataset_meta = {}
            
            try:
                from sqlalchemy import select
                from app.db.models import benchmark_runs_table
                
                # Get database connection
                engine = get_sync_db_engine()
                with engine.connect() as conn:
                    # Query database for additional metadata
                    stmt = select(benchmark_runs_table).where(benchmark_runs_table.c.run_id == run_id)
                    result = conn.execute(stmt).fetchone()
                    
                    if result:
                        # Parse config_snapshot JSON
                        if result.config_snapshot:
                            import json
                            try:
                                config_snapshot = json.loads(result.config_snapshot) if isinstance(result.config_snapshot, str) else result.config_snapshot
                                audit_info = config_snapshot.get('audit', {})
                                dataset_meta = config_snapshot.get('dataset_meta', {})
                            except:
                                pass
                        
                        # Use database timestamps if available
                        started_at = result.started_at.isoformat() if result.started_at else datetime.fromtimestamp(file_stat.st_ctime).isoformat()
                        ended_at = result.ended_at.isoformat() if result.ended_at else datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    else:
                        started_at = datetime.fromtimestamp(file_stat.st_ctime).isoformat()
                        ended_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                
            except Exception as e:
                logger.warning(f"Could not fetch database metadata for run_id '{run_id}': {e}")
                started_at = datetime.fromtimestamp(file_stat.st_ctime).isoformat()
                ended_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            
            # Process config_snapshot to fix frontend compatibility issues
            processed_config = config_snapshot.copy() if config_snapshot else {}
            
            # Fix dataset metadata field names for frontend compatibility
            if 'dataset_meta' in processed_config:
                dataset_meta = processed_config['dataset_meta']
                # Map backend field names to frontend expected names
                dataset_meta['dataset_size'] = dataset_meta.get('size')
                dataset_meta['dataset_checksum'] = dataset_meta.get('checksum')
                
                # Improve dataset_id with fallback
                dataset_id = dataset_meta.get('dataset_id')
                if not dataset_id:
                    # Try to derive from checksum or provide meaningful fallback
                    checksum = dataset_meta.get('checksum')
                    if checksum:
                        dataset_id = f"dataset_{checksum[:8]}"
                    else:
                        dataset_id = f"benchmark_{run_id}"
                dataset_meta['dataset_id'] = dataset_id
            
            # Fix vector store display - convert object to readable string
            if 'vector_store' in processed_config and isinstance(processed_config['vector_store'], dict):
                vs = processed_config['vector_store']
                vs_type = vs.get('type', 'Unknown')
                vs_strategy = vs.get('retrieval_strategy', '')
                if vs_strategy:
                    processed_config['vector_store'] = f"{vs_type.upper()} ({vs_strategy})"
                else:
                    processed_config['vector_store'] = vs_type.upper()
            
            # Add temperature from config snapshot if available
            temperature = processed_config.get('temperature')
            if temperature is None:
                # Try other possible sources
                if 'benchmark' in processed_config:
                    benchmark_config = processed_config['benchmark']
                    temperature = benchmark_config.get('temperature')
                
                if temperature is None:
                    temperature = processed_config.get('llm_temperature')
            
            # Set temperature with fallback for deterministic evaluation
            if temperature is not None:
                processed_config['temperature'] = temperature
            else:
                # For benchmarks, default is usually 0.0 (deterministic)
                processed_config['temperature'] = 0.0
            
            # Fix audit info - convert empty strings to null for better frontend display
            if 'audit' in processed_config:
                audit = processed_config['audit']
                if audit.get('purpose') == '':
                    audit['purpose'] = None
                if audit.get('tags') == []:
                    audit['tags'] = None
            
            # Add test set file name from audit info or derive it
            test_set_file = None
            if 'audit' in processed_config and processed_config['audit'].get('test_set_file'):
                original_filename = processed_config['audit']['test_set_file']
                
                # If filename has the pattern {name}_testset_{id}.json, extract the original name
                if '_testset_' in original_filename and original_filename.endswith('.json'):
                    # Extract the base name before '_testset_'
                    base_name = original_filename.split('_testset_')[0]
                    test_set_file = f"{base_name}.json"
                else:
                    # Use the filename as-is
                    test_set_file = original_filename
                    
            elif 'dataset_meta' in processed_config and processed_config['dataset_meta'].get('dataset_id'):
                dataset_id_val = processed_config['dataset_meta']['dataset_id']
                if dataset_id_val and not dataset_id_val.startswith('dataset_') and not dataset_id_val.startswith('benchmark_'):
                    # If it's a real dataset name, add .json extension
                    test_set_file = f"{dataset_id_val}.json"
                else:
                    # For generated dataset IDs, provide a more generic name
                    test_set_file = "test_set.json"
            
            if test_set_file:
                processed_config['test_set_file'] = test_set_file

            return {
                "run_id": run_id,
                "agent_name": agent_name,
                "filename": os.path.basename(benchmark_file),
                "status": "completed",
                "total_items": len(df),
                "completed_items": len(df),
                "started_at": started_at,
                "ended_at": ended_at,
                "summary_metrics": summary_metrics,
                "config_snapshot": processed_config,
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


@router.post("/benchmarks/{run_id}/insights")
async def generate_benchmark_insights(
    run_id: str,
    current_user: User = Depends(PermissionChecker(["evaluation:read_benchmarks"])),
):
    """
    Generate AI-powered insights for a benchmark run.
    """
    try:
        # Get benchmark results first
        benchmark_dir = _APP_PATHS["BENCHMARK_DIR"]
        if not os.path.exists(benchmark_dir):
            raise HTTPException(status_code=404, detail="Benchmark directory not found")
        
        # Find the benchmark file
        benchmark_file = None
        for filename in os.listdir(benchmark_dir):
            if filename.endswith('.csv') and filename.startswith('benchmark_'):
                name_without_ext = filename.replace('.csv', '')
                parts = name_without_ext.split('_')
                if len(parts) >= 3:
                    file_run_id = '_'.join(parts[2:])
                    if file_run_id.startswith('agent_'):
                        file_run_id = file_run_id[len('agent_'):]
                    
                    if file_run_id == run_id:
                        benchmark_file = os.path.join(benchmark_dir, filename)
                        break
        
        if not benchmark_file:
            raise HTTPException(status_code=404, detail=f"Benchmark with run_id '{run_id}' not found")
        
        # Get summary metrics from database or calculate from CSV
        summary_metrics = {}
        
        try:
            from sqlalchemy import select
            
            engine = get_sync_db_engine()
            with engine.connect() as conn:
                stmt = select(benchmark_runs_table).where(benchmark_runs_table.c.run_id == run_id)
                result = conn.execute(stmt).fetchone()
            
            if result and result.summary_metrics:
                import json
                summary_metrics = json.loads(result.summary_metrics) if isinstance(result.summary_metrics, str) else result.summary_metrics
            else:
                # Legacy benchmark - calculate metrics from CSV
                logger.info(f"No database record found for {run_id}, calculating metrics from CSV")
                try:
                    import pandas as pd
                    logger.info(f"Successfully imported pandas, reading CSV: {benchmark_file}")
                except ImportError as e:
                    logger.error(f"Failed to import pandas: {e}")
                    raise HTTPException(status_code=500, detail="Failed to import pandas for CSV processing")
                
                try:
                    df = pd.read_csv(benchmark_file)
                    logger.info(f"Successfully read CSV with {len(df)} rows and columns: {list(df.columns)}")
                except Exception as e:
                    logger.error(f"Failed to read CSV file {benchmark_file}: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to read CSV file: {e}")
                
                # Calculate basic metrics from CSV
                if 'retrieval' in df.columns:
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
                        mrr_values = [r.get('mrr', 0) for r in retrieval_data]
                        ndcg_values = [r.get('ndcg@10', 0) for r in retrieval_data]
                        summary_metrics['avg_mrr'] = sum(mrr_values) / len(mrr_values) if mrr_values else 0
                        summary_metrics['avg_ndcg_at_10'] = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0
                
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
                        faithfulness_values = [j.get('faithfulness', 0) for j in judge_data]
                        relevance_values = [j.get('relevance', 0) for j in judge_data]
                        conciseness_values = [j.get('conciseness', 0) for j in judge_data]
                        coherence_values = [j.get('coherence', 0) for j in judge_data]
                        
                        summary_metrics['avg_faithfulness'] = sum(faithfulness_values) / len(faithfulness_values) if faithfulness_values else 0
                        summary_metrics['avg_relevance'] = sum(relevance_values) / len(relevance_values) if relevance_values else 0
                        summary_metrics['avg_conciseness'] = sum(conciseness_values) / len(conciseness_values) if conciseness_values else 0
                        summary_metrics['avg_coherence'] = sum(coherence_values) / len(coherence_values) if coherence_values else 0
                
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
            
            # Connection closed automatically with context manager
            
        except Exception as e:
            logger.error(f"Error fetching summary metrics: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch summary metrics")
        
        # Generate fallback insights (LLM integration can be added later)
        fallback_insights = []
        
        # Performance insights
        mrr = summary_metrics.get('avg_mrr', 0)
        if mrr > 0.7:
            fallback_insights.append({
                "title": "Excellent Retrieval Performance",
                "description": f"MRR of {mrr:.3f} indicates strong retrieval accuracy. The system is finding relevant documents effectively.",
                "type": "positive",
                "priority": "medium"
            })
        elif mrr < 0.4:
            fallback_insights.append({
                "title": "Retrieval Performance Needs Improvement",
                "description": f"MRR of {mrr:.3f} suggests the retrieval system may need tuning or better document indexing.",
                "type": "warning",
                "priority": "high"
            })
        else:
            fallback_insights.append({
                "title": "Moderate Retrieval Performance",
                "description": f"MRR of {mrr:.3f} indicates moderate retrieval performance. Consider optimizing document indexing or retrieval parameters.",
                "type": "info",
                "priority": "medium"
            })
        
        # Response quality insights
        faithfulness = summary_metrics.get('avg_faithfulness', 0)
        if faithfulness > 0.8:
            fallback_insights.append({
                "title": "High Response Faithfulness",
                "description": f"Faithfulness score of {faithfulness:.3f} shows responses are well-grounded in retrieved content.",
                "type": "positive",
                "priority": "medium"
            })
        
        # Performance insights
        retrieval_time = summary_metrics.get('avg_retrieval_s', 0)
        generation_time = summary_metrics.get('avg_generation_s', 0)
        
        if retrieval_time > 2.0:
            fallback_insights.append({
                "title": "Slow Retrieval Performance",
                "description": f"Average retrieval time of {retrieval_time:.1f}s is high. Consider optimizing vector search or reducing document count.",
                "type": "warning",
                "priority": "medium"
            })
        
        if generation_time > 5.0:
            fallback_insights.append({
                "title": "Slow Generation Performance",
                "description": f"Average generation time of {generation_time:.1f}s is high. Consider using a faster model or optimizing prompts.",
                "type": "warning",
                "priority": "medium"
            })
        
        # Add a general summary insight
        fallback_insights.append({
            "title": "Benchmark Summary",
            "description": f"Completed {summary_metrics.get('total_items', 0)} items with {len(fallback_insights)} key areas identified for optimization.",
            "type": "info",
            "priority": "low"
        })
        
        return {"insights": fallback_insights}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating insights for run_id '{run_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate insights")


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

