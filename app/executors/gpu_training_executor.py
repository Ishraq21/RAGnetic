# app/executors/gpu_training_executor.py

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.db.models import fine_tuned_models_table, gpu_instances_table
from app.services.credit_service import CreditService
from app.services.gpu_orchestrator import GPUOrchestrator
from app.schemas.fine_tuning import FineTuningStatus

logger = logging.getLogger(__name__)


class GPUTrainingExecutor:
    """Executes fine-tuning jobs on remote GPU instances."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.credit_service = CreditService(db)
        self.gpu_orchestrator = GPUOrchestrator(db)
    
    async def execute_training_job(
        self, 
        job_config: Dict[str, Any], 
        user_id: int
    ) -> bool:
        """Execute a training job on a GPU instance."""
        try:
            adapter_id = job_config.get("adapter_id")
            gpu_instance_id = job_config.get("gpu_instance_id")
            
            if not adapter_id or not gpu_instance_id:
                raise ValueError("Missing adapter_id or gpu_instance_id in job config")
            
            logger.info(f"Starting GPU training execution for job {adapter_id}")
            
            # Update job status to running
            await self._update_job_status(adapter_id, FineTuningStatus.RUNNING, "Training started on GPU")
            
            # Get GPU instance details
            gpu_instance = await self._get_gpu_instance(int(gpu_instance_id), user_id)
            if not gpu_instance:
                raise ValueError(f"GPU instance {gpu_instance_id} not found")
            
            # Prepare training request
            training_request = self._prepare_training_request(job_config)
            
            # Push request to GPU instance
            success = await self._push_training_request(gpu_instance, training_request)
            if not success:
                raise Exception("Failed to push training request to GPU instance")
            
            # Monitor training progress
            training_success = await self._monitor_training_progress(
                adapter_id, 
                int(gpu_instance_id), 
                user_id
            )
            
            if training_success:
                # Update job status to completed
                await self._update_job_status(
                    adapter_id, 
                    FineTuningStatus.COMPLETED, 
                    "Training completed successfully"
                )
                
                # Calculate final cost and deduct credits
                await self._finalize_billing(int(gpu_instance_id), user_id)
                
                logger.info(f"GPU training job {adapter_id} completed successfully")
                return True
            else:
                # Update job status to failed
                await self._update_job_status(
                    adapter_id, 
                    FineTuningStatus.FAILED, 
                    "Training failed on GPU instance"
                )
                
                logger.error(f"GPU training job {adapter_id} failed")
                return False
                
        except Exception as e:
            logger.error(f"GPU training execution failed for job {adapter_id}: {e}", exc_info=True)
            
            # Update job status to failed
            if adapter_id:
                await self._update_job_status(
                    adapter_id, 
                    FineTuningStatus.FAILED, 
                    f"Training execution error: {str(e)}"
                )
            
            return False
    
    async def _update_job_status(
        self, 
        adapter_id: str, 
        status: FineTuningStatus, 
        message: str = ""
    ) -> None:
        """Update the fine-tuning job status in the database."""
        try:
            update_data = {
                "training_status": status.value,
                "updated_at": datetime.utcnow()
            }
            
            if message:
                # Store message in hyperparameters or create a separate field
                # For now, we'll log it
                logger.info(f"Job {adapter_id} status: {status.value} - {message}")
            
            await self.db.execute(
                update(fine_tuned_models_table).where(
                    fine_tuned_models_table.c.adapter_id == adapter_id
                ).values(**update_data)
            )
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
            await self.db.rollback()
    
    async def _get_gpu_instance(self, instance_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        """Get GPU instance details."""
        try:
            result = await self.db.execute(
                select(gpu_instances_table).where(
                    gpu_instances_table.c.id == instance_id,
                    gpu_instances_table.c.user_id == user_id
                )
            )
            instance = result.fetchone()
            
            if instance:
                return {
                    "id": instance.id,
                    "instance_id": instance.instance_id,
                    "provider": instance.provider,
                    "gpu_type": instance.gpu_type,
                    "status": instance.status,
                    "cost_per_hour": instance.cost_per_hour
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get GPU instance: {e}")
            return None
    
    def _prepare_training_request(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the training request for the GPU instance."""
        return {
            "adapter_id": job_config.get("adapter_id"),
            "job_name": job_config.get("job_name"),
            "base_model_name": job_config.get("base_model_name"),
            "dataset_path": job_config.get("dataset_path"),
            "eval_dataset_path": job_config.get("eval_dataset_path"),
            "output_base_dir": job_config.get("output_base_dir", "/work/output"),
            "hyperparameters": job_config.get("hyperparameters", {}),
            "use_gpu": job_config.get("use_gpu", True),
            "gpu_type": job_config.get("gpu_type"),
            "max_hours": job_config.get("max_hours")
        }
    
    async def _push_training_request(
        self, 
        gpu_instance: Dict[str, Any], 
        training_request: Dict[str, Any]
    ) -> bool:
        """Push the training request to the GPU instance."""
        try:
            # This would involve copying the request.json file to the GPU instance
            # For now, we'll simulate this process
            
            logger.info(f"Pushing training request to GPU instance {gpu_instance['instance_id']}")
            
            # In a real implementation, this would:
            # 1. Copy request.json to the GPU instance
            # 2. Ensure the training dataset is accessible
            # 3. Start the training container
            
            # Simulate the push operation
            time.sleep(1)  # Simulate network delay
            
            logger.info("Training request pushed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push training request: {e}")
            return False
    
    async def _monitor_training_progress(
        self, 
        adapter_id: str, 
        gpu_instance_id: int, 
        user_id: int
    ) -> bool:
        """Monitor the training progress on the GPU instance."""
        try:
            logger.info(f"Starting training progress monitoring for job {adapter_id}")
            
            max_wait_time = 3600  # 1 hour max wait
            check_interval = 30   # Check every 30 seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                # Get GPU instance status
                instance_status = await self.gpu_orchestrator.get_instance_status(
                    gpu_instance_id, 
                    user_id
                )
                
                if not instance_status:
                    logger.error(f"Could not get status for GPU instance {gpu_instance_id}")
                    return False
                
                # Check if instance is still running
                if instance_status["status"] not in ["running", "provisioning"]:
                    logger.error(f"GPU instance {gpu_instance_id} is not running: {instance_status['status']}")
                    return False
                
                # Get training logs to check progress
                logs = await self.gpu_orchestrator.get_instance_logs(
                    gpu_instance_id, 
                    user_id, 
                    tail_kb=10
                )
                
                if logs and "Training completed successfully" in logs:
                    logger.info(f"Training completed for job {adapter_id}")
                    return True
                elif logs and "Training failed" in logs:
                    logger.error(f"Training failed for job {adapter_id}")
                    return False
                
                # Update progress based on logs (simplified)
                progress = self._extract_progress_from_logs(logs)
                if progress > 0:
                    await self._update_job_status(
                        adapter_id, 
                        FineTuningStatus.RUNNING, 
                        f"Training progress: {progress}%"
                    )
                
                # Wait before next check
                time.sleep(check_interval)
            
            logger.error(f"Training monitoring timeout for job {adapter_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to monitor training progress: {e}")
            return False
    
    def _extract_progress_from_logs(self, logs: str) -> float:
        """Extract training progress from logs."""
        try:
            if not logs:
                return 0.0
            
            # Look for progress indicators in logs
            # This is a simplified implementation
            if "Epoch 1/" in logs:
                return 25.0
            elif "Epoch 2/" in logs:
                return 50.0
            elif "Epoch 3/" in logs:
                return 75.0
            elif "Training completed" in logs:
                return 100.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to extract progress from logs: {e}")
            return 0.0
    
    async def _finalize_billing(self, gpu_instance_id: int, user_id: int) -> None:
        """Finalize billing and deduct final costs."""
        try:
            # Get final GPU instance details
            result = await self.db.execute(
                select(gpu_instances_table).where(
                    gpu_instances_table.c.id == gpu_instance_id,
                    gpu_instances_table.c.user_id == user_id
                )
            )
            instance = result.fetchone()
            
            if not instance:
                logger.error(f"GPU instance {gpu_instance_id} not found for billing")
                return
            
            # Calculate final cost
            if instance.started_at:
                uptime = datetime.utcnow() - instance.started_at
                uptime_hours = uptime.total_seconds() / 3600
                final_cost = instance.cost_per_hour * uptime_hours
                
                # Deduct final cost
                await self.credit_service.deduct(
                    user_id,
                    final_cost,
                    f"Final GPU training cost for instance {gpu_instance_id}",
                    gpu_instance_id
                )
                
                # Update instance with final cost
                await self.db.execute(
                    update(gpu_instances_table).where(
                        gpu_instances_table.c.id == gpu_instance_id
                    ).values(
                        total_cost=final_cost,
                        stopped_at=datetime.utcnow()
                    )
                )
                
                await self.db.commit()
                
                logger.info(f"Finalized billing for GPU instance {gpu_instance_id}: ${final_cost:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to finalize billing: {e}")
            await self.db.rollback()
    
    async def stop_training_job(self, adapter_id: str, user_id: int) -> bool:
        """Stop a running training job."""
        try:
            # Get job details
            result = await self.db.execute(
                select(fine_tuned_models_table).where(
                    fine_tuned_models_table.c.adapter_id == adapter_id,
                    fine_tuned_models_table.c.created_by_user_id == user_id
                )
            )
            job = result.fetchone()
            
            if not job:
                logger.error(f"Training job {adapter_id} not found")
                return False
            
            # Get GPU instance ID from job config
            gpu_instance_id = job.get("gpu_instance_id")
            if not gpu_instance_id:
                logger.error(f"No GPU instance ID found for job {adapter_id}")
                return False
            
            # Stop the GPU instance
            success = await self.gpu_orchestrator.stop_instance(int(gpu_instance_id), user_id)
            
            if success:
                # Update job status
                await self._update_job_status(
                    adapter_id, 
                    FineTuningStatus.FAILED, 
                    "Training job stopped by user"
                )
                
                logger.info(f"Training job {adapter_id} stopped successfully")
                return True
            else:
                logger.error(f"Failed to stop training job {adapter_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to stop training job {adapter_id}: {e}")
            return False
