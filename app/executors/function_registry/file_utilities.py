import logging
import os
import shutil
import pandas as pd
from typing import Dict, Any
from pathlib import Path  # CHANGE: Added for sandbox checks
from .__init__ import FunctionRegistration

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(handler)

SANDBOX_ROOT = Path("/work")

def _is_within_sandbox(path: str) -> bool:
    try:
        return Path(path).resolve().is_relative_to(SANDBOX_ROOT)
    except AttributeError:
        # Python < 3.9 fallback
        return str(SANDBOX_ROOT) in str(Path(path).resolve())

@FunctionRegistration(
    name="read_text_file",
    description="Reads a text file from the sandbox and returns its content.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "The path to the text file in the sandbox."}
        },
        "required": ["file_path"]
    }
)
def read_text_file(file_path: str) -> Dict[str, Any]:
    if not _is_within_sandbox(file_path):  # CHANGE: Security check
        return {"error": "Access denied: outside sandbox", "status": "failed"}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:  # CHANGE: Explicit encoding
            content = f.read()
        logger.info(f"Read file: {file_path}")  # CHANGE: Log success
        return {"file_path": file_path, "content": content}
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")  # CHANGE: Log error
        return {"error": f"File not found at '{file_path}'.", "status": "failed"}
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return {"error": str(e), "status": "failed"}

@FunctionRegistration(
    name="list_files_in_dir",
    description="Lists all files in a given directory within the sandbox.",
    args_schema={
        "type": "object",
        "properties": {
            "dir_path": {"type": "string", "description": "The directory path."}
        },
        "required": ["dir_path"]
    }
)
def list_files_in_dir(dir_path: str) -> Dict[str, Any]:
    if not _is_within_sandbox(dir_path):
        return {"error": "Access denied: outside sandbox", "status": "failed"}
    try:
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]  # CHANGE: Only files
        logger.info(f"Listed files in: {dir_path}")
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files in {dir_path}: {e}")
        return {"error": str(e), "status": "failed"}

@FunctionRegistration(
    name="delete_file",
    description="Deletes a file from the sandbox.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to delete."}
        },
        "required": ["file_path"]
    }
)
def delete_file(file_path: str) -> Dict[str, Any]:
    if not _is_within_sandbox(file_path):
        return {"error": "Access denied: outside sandbox", "status": "failed"}
    if not os.path.isfile(file_path):  # CHANGE: Ensure file exists
        return {"error": f"Not a valid file: {file_path}", "status": "failed"}
    try:
        os.remove(file_path)
        logger.info(f"Deleted file: {file_path}")
        return {"status": "success", "deleted": file_path}
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return {"error": str(e), "status": "failed"}

@FunctionRegistration(
    name="append_text_file",
    description="Appends text to an existing file.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "text": {"type": "string"}
        },
        "required": ["file_path", "text"]
    }
)
def append_text_file(file_path: str, text: str) -> Dict[str, Any]:
    if not _is_within_sandbox(file_path):
        return {"error": "Access denied: outside sandbox", "status": "failed"}
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}", "status": "failed"}
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Appended to file: {file_path}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error appending to file {file_path}: {e}")
        return {"error": str(e), "status": "failed"}

@FunctionRegistration(
    name="copy_file",
    description="Copies a file within the sandbox.",
    args_schema={
        "type": "object",
        "properties": {
            "src_path": {"type": "string"},
            "dest_path": {"type": "string"}
        },
        "required": ["src_path", "dest_path"]
    }
)
def copy_file(src_path: str, dest_path: str) -> Dict[str, Any]:
    if not _is_within_sandbox(src_path) or not _is_within_sandbox(dest_path):
        return {"error": "Access denied: outside sandbox", "status": "failed"}
    if not os.path.isfile(src_path):
        return {"error": f"Source file not found: {src_path}", "status": "failed"}
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # CHANGE: Ensure dest dir exists
        shutil.copy(src_path, dest_path)
        logger.info(f"Copied file from {src_path} to {dest_path}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error copying file {src_path} -> {dest_path}: {e}")
        return {"error": str(e), "status": "failed"}

@FunctionRegistration(
    name="convert_csv_to_json",
    description="Converts a CSV file to JSON format.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"}
        },
        "required": ["file_path"]
    }
)
def convert_csv_to_json(file_path: str) -> Dict[str, Any]:
    if not _is_within_sandbox(file_path):
        return {"error": "Access denied: outside sandbox", "status": "failed"}
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}", "status": "failed"}
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Converted CSV to JSON: {file_path}")
        return {"json_data": df.to_json(orient='records')}
    except Exception as e:
        logger.error(f"Error converting CSV to JSON {file_path}: {e}")
        return {"error": str(e), "status": "failed"}

@FunctionRegistration(
    name="get_file_metadata",
    description="Returns metadata about a file.",
    args_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"}
        },
        "required": ["file_path"]
    }
)
def get_file_metadata(file_path: str) -> Dict[str, Any]:
    if not _is_within_sandbox(file_path):
        return {"error": "Access denied: outside sandbox", "status": "failed"}
    if not os.path.exists(file_path):
        return {"error": f"Path not found: {file_path}", "status": "failed"}
    try:
        stats = os.stat(file_path)
        logger.info(f"Got metadata for: {file_path}")
        return {
            "size_bytes": stats.st_size,
            "modified_time": stats.st_mtime,
            "is_directory": os.path.isdir(file_path)
        }
    except Exception as e:
        logger.error(f"Error getting metadata for {file_path}: {e}")
        return {"error": str(e), "status": "failed"}
