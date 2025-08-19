# app/services/file_service.py
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from app.core.config import get_path_settings

logger = logging.getLogger(__name__)

_APP_PATHS = get_path_settings()
_LAMBDA_RUNS_DIR = _APP_PATHS["PROJECT_ROOT"] / ".ragnetic" / "lambda_runs"
_TEMP_CHAT_UPLOADS_DIR = _APP_PATHS["TEMP_CLONES_DIR"] / "chat_uploads"

# Ensure necessary directories exist
_LAMBDA_RUNS_DIR.mkdir(parents=True, exist_ok=True)
_TEMP_CHAT_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

class FileService:
    """
    Manages the lifecycle of files for the LambdaTool.
    This includes staging input files for the sandbox.
    """
    def __init__(self):
        self.base_dir = _LAMBDA_RUNS_DIR
        self.upload_dir = _TEMP_CHAT_UPLOADS_DIR

    def stage_input_file(
            self,
            temp_doc_id: str,
            user_id: int,
            thread_id: str,
            run_id: str,
            file_name: str
    ) -> Dict[str, str]:
        """
        Copies a source file (from a user upload) to the run's input dir,
        and returns both host + sandbox paths.
        """
        # 1. Build the path to the original uploaded file
        original_file_path = self.upload_dir / str(user_id) / thread_id / f"{temp_doc_id}_{file_name}"
        if not original_file_path.exists():
            raise FileNotFoundError(f"Original temporary file not found for ID: {temp_doc_id}")

        # 2. Build the path for the new staged file in the run-specific directory
        run_input_dir = self.base_dir / run_id / "inputs"
        run_input_dir.mkdir(parents=True, exist_ok=True)

        staged_name = f"{temp_doc_id}_{file_name}"
        destination_path = run_input_dir / staged_name

        # 3. Perform the copy operation
        shutil.copy(original_file_path, destination_path)
        logger.info(f"Staged input file from {original_file_path} to {destination_path}")

        # 4. Return the metadata including the sandbox path
        return {
            "host_path": str(destination_path),
            "sandbox_path": f"/work/inputs/{staged_name}",
        }
#22
    def cleanup_run_data(self, run_id: str):
        """
        Removes the entire directory for a finished or failed run.
        This is a hard deletion and should be used with caution.
        """
        run_dir = self.base_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
            logger.info(f"Cleaned up directory for run: {run_id}")

    def get_result_file(self, run_id: str, file_name: str) -> Path:
        """
        Retrieve a file from the run's outputs directory.
        Ensures it exists and prevents directory traversal.
        """
        outputs_dir = self.base_dir / run_id / "outputs"
        file_path = outputs_dir / file_name

        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found in outputs for run {run_id}: {file_name}")

        if not file_path.resolve().parent == outputs_dir.resolve():
            raise PermissionError("Invalid file path (possible traversal).")

        return file_path
