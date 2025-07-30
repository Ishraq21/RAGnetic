import time
import os
import logging
import requests
import json
import asyncio # NEW: Import asyncio for running async functions
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from multiprocessing import Process
from pathlib import Path
import configparser

# Import core components
from app.agents.config_manager import get_agent_configs
from app.pipelines.embed import embed_agent_data # This is an async function
from app.core.config import get_path_settings

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
        self.last_event_time = {} # To debounce events
        self.workflows_dir = _WORKFLOWS_DIR # Ensure _WORKFLOWS_DIR is accessible as an attribute

    def _get_server_url(self) -> str:
        """Constructs the server URL from the config file."""
        host = self.config.get('SERVER', 'host', fallback='127.0.0.1')
        port = self.config.get('SERVER', 'port', fallback='8000')
        return f"http://{host}:{port}/api/v1"

    # NEW: Helper to run an async function from a sync context
    def _run_async_in_sync(self, coro):
        """Runs a coroutine in a new or existing event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError: # No running loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        else: # Already a running loop, schedule the task
            # For a synchronous watchdog handler, we typically don't want to block
            # the loop indefinitely. So, we'll schedule it and let it run.
            # However, for `embed_agent_data` we want the watcher to effectively "wait"
            # for it to finish for proper sequencing from the watcher's perspective.
            # `asyncio.run` is the simplest way to do that for a single call in a sync context.
            # But watchdog's observer.start() is sync and blocks. So, a new loop each time is necessary.
            return loop.run_until_complete(coro) # Use run_until_complete if within main thread

    def on_any_event(self, event):
        if event.is_directory or event.event_type not in ['created', 'deleted', 'modified']:
            return

        src_path = Path(event.src_path)
        logger.info(f"File change detected: {src_path} ({event.event_type})")

        # Debounce events to prevent multiple triggers for a single save operation
        current_time = time.time()
        if str(src_path) in self.last_event_time and (current_time - self.last_event_time[str(src_path)]) < 1: # 1 second debounce
            logger.debug(f"Debouncing event for {src_path}")
            return
        self.last_event_time[str(src_path)] = current_time

        # --- Check for workflow YAML file changes and trigger sync ---
        if src_path.is_file() and src_path.suffix in ['.yaml', '.yml']:
            try:
                # Check if the file is within the workflows directory
                src_path.relative_to(self.workflows_dir) # Use self.workflows_dir here
                logger.info(f"Workflow definition file changed: {src_path}. Triggering workflow sync API.")
                self._trigger_workflow_sync_api()
                return # Workflow change is handled, no need to proceed to agent/file_ingestion logic for this file
            except ValueError:
                # Not a workflow file in the workflows directory
                pass

        # --- Agent re-deployment logic (only if not a workflow file) ---
        # Note: The API-based local file upload already triggers embed_agent_data
        # directly via the PUT /agents/{name} endpoint. This watcher logic is
        # primarily for direct file system changes outside of the API (e.g., manual copy).
        affected_agents = self._find_affected_agents(str(src_path))
        if not affected_agents:
            return

        for agent_config in affected_agents:
            logger.info(f"Agent '{agent_config.name}' is affected. Triggering re-deployment...")
            try:
                # --- FIX: Running async embed_agent_data in a sync context ---
                self._run_async_in_sync(embed_agent_data(agent_config))
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
                        break # Only need to add an agent once if multiple sources affected
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
    event_handler = AgentDataEventHandler()
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
            time.sleep(1) # Keep the main thread alive for the observer
    except KeyboardInterrupt:
        observer.stop()
    observer.join()