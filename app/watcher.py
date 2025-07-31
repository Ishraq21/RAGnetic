import time
import os
import logging
import requests
import json
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from multiprocessing import Process
from pathlib import Path
import configparser

# Import core components
from app.agents.config_manager import get_agent_configs
from app.pipelines.embed import embed_agent_data
from app.core.config import get_path_settings, get_memory_storage_config, get_log_storage_config
from app.db import initialize_db_connections, \
    AsyncSessionLocal  # Import AsyncSessionLocal and initialize_db_connections

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_APP_PATHS = get_path_settings()
_PROJECT_ROOT = _APP_PATHS["PROJECT_ROOT"]
_CONFIG_FILE = _APP_PATHS["CONFIG_FILE_PATH"]
_WORKFLOWS_DIR = _APP_PATHS["WORKFLOWS_DIR"]


class AgentDataEventHandler(FileSystemEventHandler):
    """Handles file system events and triggers re-embedding for affected agents,
       and now also triggers workflow sync for workflow definition changes."""

    def __init__(self):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(_CONFIG_FILE)
        self.server_url = self._get_server_url()
        self.last_event_time = {}  # To debounce events
        self.workflows_dir = _WORKFLOWS_DIR  # Ensure _WORKFLOWS_DIR is accessible as an attribute

        self._initialize_db_for_watcher()

    def _initialize_db_for_watcher(self):
        """Initializes database connections for the watcher process."""
        try:
            conn_name = get_memory_storage_config().get("connection_name") or get_log_storage_config().get(
                "connection_name")
            if not conn_name:
                logger.error(
                    "Database connection name not found for watcher DB initialization. DB-dependent features might fail.")
                return
            initialize_db_connections(conn_name)
            logger.info("Watcher: Database connections initialized.")
        except Exception as e:
            logger.error(f"Watcher: Failed to initialize database connections: {e}", exc_info=True)

    def _get_server_url(self) -> str:
        """Constructs the server URL from the config file."""
        host = self.config.get('SERVER', 'host', fallback='127.0.0.1')
        port = self.config.get('SERVER', 'port', fallback='8000')
        return f"http://{host}:{port}/api/v1"

    # NEW: Helper to run an async function with DB session from a sync context
    async def _run_async_with_db_session(self, coro_func, *args, **kwargs):
        """Runs an async coroutine, providing it with a DB session."""
        if AsyncSessionLocal is None:
            logger.error("AsyncSessionLocal is not initialized in watcher process. Cannot run async with DB session.")
            return

        async with AsyncSessionLocal() as session:
            try:
                # Pass the session explicitly to the coroutine function
                return await coro_func(*args, db=session, **kwargs)
            except Exception as e:
                logger.error(f"Error during async DB operation in watcher: {e}", exc_info=True)
                raise  # Re-raise to be caught by the calling context

    def on_any_event(self, event):
        if event.is_directory or event.event_type not in ['created', 'deleted', 'modified']:
            return

        src_path = Path(event.src_path)
        logger.info(f"File change detected: {src_path} ({event.event_type})")

        # Debounce events to prevent multiple triggers for a single save operation
        current_time = time.time()
        # Increased debounce time for better reliability with file systems
        if str(src_path) in self.last_event_time and (current_time - self.last_event_time[str(src_path)]) < 2:
            logger.debug(f"Debouncing event for {src_path}")
            return
        self.last_event_time[str(src_path)] = current_time

        # --- Check for workflow YAML file changes and trigger sync ---
        if src_path.is_file() and src_path.suffix in ['.yaml', '.yml']:
            try:
                # Check if the file is within the workflows directory
                src_path.relative_to(self.workflows_dir)  # Use self.workflows_dir here
                logger.info(f"Workflow definition file changed: {src_path}. Triggering workflow sync API.")
                self._trigger_workflow_sync_api()
                return  # Workflow change is handled, no need to proceed to agent/file_ingestion logic for this file
            except ValueError:
                # Not a workflow file in the workflows directory
                pass

        # --- Agent re-deployment logic (only if not a workflow file) ---
        affected_agents = self._find_affected_agents(str(src_path))
        if not affected_agents:
            return

        for agent_config in affected_agents:
            logger.info(f"Agent '{agent_config.name}' is affected. Triggering re-deployment...")
            try:
                # FIX: Pass the agent_config and acquire a DB session
                asyncio.run(self._run_async_with_db_session(embed_agent_data, config=agent_config))
                logger.info(f"Successfully re-deployed agent '{agent_config.name}'.")
            except Exception as e:
                logger.error(f"Failed to re-deploy agent '{agent_config.name}': {e}", exc_info=True)

    def _find_affected_agents(self, changed_path: str):
        """Scans all agent configs to see if their data sources include the changed path."""
        affected = []
        all_agent_configs = get_agent_configs()
        normalized_changed_path = os.path.normpath(changed_path)

        for config in all_agent_configs:
            for source in config.sources:
                if source.type == 'local' and source.path:
                    normalized_source_path = os.path.normpath(source.path)
                    # Check if the changed path is within the source path
                    # or if the source path is the changed path itself (for single file sources)
                    if normalized_changed_path.startswith(normalized_source_path) or \
                            normalized_source_path == normalized_changed_path:
                        affected.append(config)
                        break  # Only need to add an agent once if multiple sources affected
        return affected

    def _trigger_workflow_on_new_file(self, file_path: Path):
        """
        Triggers a configured workflow for a new file.
        This is a placeholder for a more robust configuration system.
        """
        workflow_name = "file_ingestion"

        try:
            url = f"{self.server_url}/workflows/{workflow_name}/trigger"
            payload = {
                "initial_input": {
                    "file_path": str(file_path.relative_to(_PROJECT_ROOT))
                }
            }
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Successfully triggered workflow '{workflow_name}' for new file: {file_path}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to trigger workflow '{workflow_name}' for file {file_path}: {e}")

    def _trigger_workflow_sync_api(self):
        """Calls the API endpoint to sync workflow definitions from files to database."""
        try:
            url = f"{self.server_url}/workflows/sync"
            # In a real app, this API call would need authentication (e.g., API key)
            # For simplicity, assuming local access without auth for watcher or webhook sync.
            response = requests.post(url, timeout=5)
            response.raise_for_status()
            logger.info(f"Successfully triggered workflow sync API. Response: {response.json()}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to trigger workflow sync API: {e}", exc_info=True)


def start_watcher(directories: list[str]):
    """Starts the file system watcher on the specified directories."""
    event_handler = AgentDataEventHandler()  # DB initialization happens here now
    observer = Observer()

    for directory in directories:
        path_to_monitor = Path(directory)
        if path_to_monitor.is_dir():
            observer.schedule(event_handler, str(path_to_monitor), recursive=True)
            logger.info(f"RAGnetic watcher is now monitoring the '{path_to_monitor}' directory...")
        else:
            logger.warning(f"Watcher: Directory not found, skipping monitoring: {path_to_monitor}")

    observer.start()
    try:
        while True:
            time.sleep(1)  # Keep the main thread alive for the observer
    except KeyboardInterrupt:
        observer.stop()
    observer.join()