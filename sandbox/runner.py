#!/usr/bin/env python3
"""
GPU Training Runner for RAGnetic
Executes fine-tuning jobs in GPU containers with proper logging and status reporting.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# Add the app directory to Python path for imports
sys.path.insert(0, '/work')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/work/logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrainingRunner:
    """Handles the execution of fine-tuning jobs in GPU containers."""
    
    def __init__(self, work_dir: str = "/work"):
        self.work_dir = Path(work_dir)
        self.request_file = self.work_dir / "request.json"
        self.status_file = self.work_dir / "status.json"
        self.logs_file = self.work_dir / "logs.txt"
        
        # Ensure directories exist
        self.work_dir.mkdir(exist_ok=True)
        (self.work_dir / "logs").mkdir(exist_ok=True)
        (self.work_dir / "data").mkdir(exist_ok=True)
        (self.work_dir / "output").mkdir(exist_ok=True)
    
    def load_request(self) -> Dict[str, Any]:
        """Load the training request configuration."""
        try:
            if not self.request_file.exists():
                raise FileNotFoundError(f"Request file not found: {self.request_file}")
            
            with open(self.request_file, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded training request: {config.get('job_name', 'unknown')}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load request: {e}")
            raise
    
    def update_status(self, status: str, message: str = "", progress: float = 0.0, **kwargs):
        """Update the status file with current job state."""
        try:
            status_data = {
                "status": status,
                "message": message,
                "progress": progress,
                "timestamp": time.time(),
                "gpu_info": self._get_gpu_info(),
                **kwargs
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
            
            logger.info(f"Status updated: {status} - {message}")
            
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information for monitoring."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                    "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                    "gpu_memory_cached": torch.cuda.memory_reserved(0)
                }
            else:
                return {"gpu_available": False}
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
            return {"error": str(e)}
    
    def log_to_file(self, message: str):
        """Log message to the logs file."""
        try:
            with open(self.logs_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        except Exception as e:
            logger.error(f"Failed to write to logs file: {e}")
    
    def run_training(self, config: Dict[str, Any]) -> bool:
        """Execute the fine-tuning job."""
        try:
            self.update_status("running", "Starting fine-tuning job", 0.0)
            self.log_to_file("Starting fine-tuning job")
            
            # Import the fine-tuner (this would be from the actual RAGnetic codebase)
            # For now, we'll simulate the training process
            from app.training.llm_fine_tuner import LLMFineTuner
            
            # Initialize the fine-tuner
            fine_tuner = LLMFineTuner()
            
            # Update status
            self.update_status("running", "Initializing training environment", 10.0)
            
            # Prepare training configuration
            training_config = self._prepare_training_config(config)
            
            # Start training
            self.update_status("running", "Training in progress", 20.0)
            self.log_to_file("Training started")
            
            # Run the actual fine-tuning
            result = fine_tuner.fine_tune_llm(training_config)
            
            if result.get("success", False):
                self.update_status(
                    "completed", 
                    "Training completed successfully", 
                    100.0,
                    final_loss=result.get("final_loss"),
                    validation_loss=result.get("validation_loss"),
                    gpu_hours_consumed=result.get("gpu_hours_consumed", 0.0)
                )
                self.log_to_file("Training completed successfully")
                return True
            else:
                error_msg = result.get("error", "Unknown training error")
                self.update_status("failed", f"Training failed: {error_msg}", 0.0)
                self.log_to_file(f"Training failed: {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Training execution error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.update_status("failed", error_msg, 0.0)
            self.log_to_file(error_msg)
            return False
    
    def _prepare_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the training configuration for the fine-tuner."""
        # Map the request config to the format expected by LLMFineTuner
        training_config = {
            "adapter_id": config.get("adapter_id"),
            "job_name": config.get("job_name"),
            "base_model_name": config.get("base_model_name"),
            "dataset_path": config.get("dataset_path"),
            "eval_dataset_path": config.get("eval_dataset_path"),
            "output_base_dir": config.get("output_base_dir", "/work/output"),
            "hyperparameters": config.get("hyperparameters", {}),
            "device": "cuda" if config.get("use_gpu", False) else "cpu",
            "gpu_type": config.get("gpu_type"),
            "max_hours": config.get("max_hours")
        }
        
        # Ensure output directory exists
        output_dir = Path(training_config["output_base_dir"]) / training_config["job_name"] / training_config["adapter_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        training_config["adapter_path"] = str(output_dir)
        
        return training_config
    
    def run(self) -> int:
        """Main execution method."""
        try:
            logger.info("Starting GPU training runner")
            
            # Load the training request
            config = self.load_request()
            
            # Update initial status
            self.update_status("initializing", "Loading training configuration", 0.0)
            
            # Run the training
            success = self.run_training(config)
            
            if success:
                logger.info("Training completed successfully")
                return 0
            else:
                logger.error("Training failed")
                return 1
                
        except Exception as e:
            logger.error(f"Runner failed: {e}", exc_info=True)
            self.update_status("failed", f"Runner error: {str(e)}", 0.0)
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAGnetic GPU Training Runner")
    parser.add_argument("--work-dir", default="/work", help="Working directory path")
    parser.add_argument("--config", help="Path to training configuration file")
    
    args = parser.parse_args()
    
    # If config file is provided, copy it to request.json
    if args.config:
        import shutil
        shutil.copy2(args.config, "/work/request.json")
    
    # Create and run the training runner
    runner = TrainingRunner(args.work_dir)
    exit_code = runner.run()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()