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
    This includes staging input files for the sandbox and storing output artifacts.
    """
    def __init__(self):
        self.base_dir = _LAMBDA_RUNS_DIR
        self.upload_dir = _TEMP_CHAT_UPLOADS_DIR

    def stage_input_file(self, temp_doc_id: str, user_id: int, thread_id: str, run_id: str, file_name: str) -> Path:
        """
        Copies a source file (from a user upload) to the specific run's input directory.
        Returns the path to the staged file.
        """
        # Resolve the original temp document file path from the upload directory
        original_file_path = self.upload_dir / str(user_id) / thread_id / f"{temp_doc_id}_{file_name}"
        if not original_file_path.exists():
            raise FileNotFoundError(f"Original temporary file not found for ID: {temp_doc_id}")

        run_input_dir = self.base_dir / run_id / "inputs"
        run_input_dir.mkdir(parents=True, exist_ok=True)

        destination_path = run_input_dir / file_name
        shutil.copy(original_file_path, destination_path)
        logger.info(f"Staged input file from {original_file_path} to {destination_path}")
        return destination_path

    def collect_artifact(self, run_id: str, relative_path: str) -> Dict[str, Any]:
        """
        Collects metadata and returns a file path for a generated artifact.
        """
        artifact_path = self.base_dir / run_id / relative_path
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found at expected path: {artifact_path}")

        # In a real implementation, this would involve uploading to a central store (S3, etc.)
        return {
            "file_name": artifact_path.name,
            "size_bytes": artifact_path.stat().st_size,
            "mime_type": "application/octet-stream",
            "signed_url": f"/api/v1/lambda/artifacts/{run_id}/{artifact_path.name}"
        }

    def cleanup_run_data(self, run_id: str):
        """
        Removes the entire directory for a finished or failed run.
        This is a hard deletion and should be used with caution.
        """
        run_dir = self.base_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
            logger.info(f"Cleaned up directory for run: {run_id}")

    def get_file_for_download(self, run_id: str, file_name: str) -> Path:
        """
        Retrieves a file path for a completed run's artifact.
        Performs a security check to ensure path traversal is not possible.
        """
        run_dir = self.base_dir / run_id
        file_path = run_dir / file_name

        if not file_path.exists() or not file_path.is_file() or file_path.parent != run_dir:
            raise FileNotFoundError("File not found or access denied.")

        return file_path