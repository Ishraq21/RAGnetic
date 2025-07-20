import time
import os
import logging
import requests
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from multiprocessing import Process
from pathlib import Path
import configparser

# Import core components
from app.agents.config_manager import get_agent_configs
from app.pipelines.embed import embed_agent_data
from app.core.config import get_path_settings

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_APP_PATHS = get_path_settings()
_PROJECT_ROOT = _APP_PATHS["PROJECT_ROOT"]
_CONFIG_FILE = _APP_PATHS["CONFIG_FILE_PATH"]


class AgentDataEventHandler(FileSystemEventHandler):
    """Handles file system events and triggers re-embedding for affected agents."""

    def __init__(self):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(_CONFIG_FILE)
        self.server_url = self._get_server_url()

    def _get_server_url(self) -> str:
        """Constructs the server URL from the config file."""
        host = self.config.get('SERVER', 'host', fallback='127.0.0.1')
        port = self.config.get('SERVER', 'port', fallback='8000')
        return f"http://{host}:{port}/api/v1"

    def on_any_event(self, event):
        if event.is_directory or event.event_type not in ['created', 'deleted', 'modified']:
            return

        src_path = event.src_path
        logger.info(f"File change detected: {src_path} ({event.event_type})")

        # Check if the event should trigger a workflow
        if event.event_type == 'created':
            self._trigger_workflow_on_new_file(src_path)

        # Trigger the existing agent re-deployment logic
        affected_agents = self._find_affected_agents(src_path)
        if not affected_agents:
            return

        for agent_config in affected_agents:
            logger.info(f"Agent '{agent_config.name}' is affected. Triggering re-deployment...")
            try:
                # This should probably be run in a separate process/thread to not block the watcher
                embed_agent_data(agent_config)
                logger.info(f"Successfully re-deployed agent '{agent_config.name}'.")
            except Exception as e:
                logger.error(f"Failed to re-deploy agent '{agent_config.name}': {e}")

    def _find_affected_agents(self, changed_path: str):
        """Scans all agent configs to see if their data sources include the changed path."""
        affected = []
        all_agent_configs = get_agent_configs()
        normalized_changed_path = os.path.normpath(changed_path)

        for config in all_agent_configs:
            for source in config.sources:
                if source.type == 'local' and source.path:
                    normalized_source_path = os.path.normpath(source.path)
                    if normalized_changed_path.startswith(normalized_source_path):
                        affected.append(config)
                        break
        return affected

    def _trigger_workflow_on_new_file(self, file_path: str):
        """
        Triggers a configured workflow for a new file.
        This is a placeholder for a more robust configuration system.
        """
        # Placeholder logic: Assumes a workflow named 'file_ingestion' exists
        # and should be triggered for any new file.
        workflow_name = "file_ingestion"

        try:
            url = f"{self.server_url}/workflows/{workflow_name}/trigger"
            payload = {
                "initial_input": {
                    "file_path": str(Path(file_path).relative_to(_PROJECT_ROOT))
                }
            }
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Successfully triggered workflow '{workflow_name}' for new file: {file_path}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to trigger workflow '{workflow_name}' for file {file_path}: {e}")


def start_watcher(directory: str):
    """Starts the file system watcher on the specified directory."""
    event_handler = AgentDataEventHandler()
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)

    logger.info(f"RAGnetic watcher is now monitoring the '{directory}' directory...")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()