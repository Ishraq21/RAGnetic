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
import logging.config

# Import core components
from app.agents.config_manager import get_agent_configs
from app.pipelines.embed import embed_agent_data
from app.core.config import get_path_settings, get_memory_storage_config, get_log_storage_config, get_server_api_keys
from app.db import initialize_db_connections, AsyncSessionLocal

logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
_APP_PATHS = get_path_settings()
_PROJECT_ROOT = _APP_PATHS["PROJECT_ROOT"]
_CONFIG_FILE = _APP_PATHS["CONFIG_FILE_PATH"]
_DATA_DIR = _APP_PATHS["DATA_DIR"]

# Helper function to get file extension safely
def _get_file_extension(filepath: str) -> str:
    return Path(filepath).suffix.lower().lstrip('.')


class AgentDataEventHandler(FileSystemEventHandler):
    """Handles file system events and triggers re-embedding for affected agents."""

    def __init__(self):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(_CONFIG_FILE)
        self.server_url = self._get_server_url()
        self.api_key = self._get_api_key()
        self.last_event_time = {} # To debounce events
        self.uploaded_temp_dir = _DATA_DIR / "uploaded_temp"

        self._initialize_db_for_watcher()

    def _initialize_db_for_watcher(self):
        """Initializes database connections for the watcher process."""
        try:
            conn_name = get_memory_storage_config().get("connection_name") or get_log_storage_config().get("connection_name")
            if not conn_name:
                logger.error("Database connection name not found for watcher DB initialization. DB-dependent features might fail.")
                return
            initialize_db_connections(conn_name)
            logger.info("Watcher: Database connections initialized.")
        except Exception as e:
            logger.error(f"Watcher: Failed to initialize database connections: {e}", exc_info=True)

    def _get_api_key(self) -> str:
        """Retrieves the first server API key for authentication."""
        keys = get_server_api_keys()
        if not keys:
            logger.warning("No RAGNETIC_API_KEYS found. API calls will likely fail with 401 Unauthorized errors.")
            return ""
        return keys[0]

    def _get_server_url(self) -> str:
        """Constructs the server URL from the config file."""
        host = self.config.get('SERVER', 'host', fallback='127.0.0.1')
        port = self.config.get('SERVER', 'port', fallback='8000')
        return f"http://{host}:{port}/api/v1"

    async def _run_async_with_db_session(self, coro_func, *args, **kwargs):
        """Runs an async coroutine, providing it with a DB session."""
        if AsyncSessionLocal is None:
            logger.error("AsyncSessionLocal is not initialized in watcher process. Cannot run async with DB session.")
            return

        async with AsyncSessionLocal() as session:
            try:
                return await coro_func(*args, db=session, **kwargs)
            except Exception as e:
                logger.error(f"Error during async DB operation in watcher: {e}", exc_info=True)
                raise

    def on_any_event(self, event):
        if event.is_directory or event.event_type not in ['created', 'deleted', 'modified']:
            return

        src_path = Path(event.src_path)
        logger.info(f"File change detected: {src_path} ({event.event_type})")

        if src_path.is_relative_to(self.uploaded_temp_dir):
            logger.info(f"Ignoring file change in temporary upload directory: {src_path}. API handles this.")
            return

        # Debounce events to prevent multiple triggers for a single save operation
        current_time = time.time()
        if str(src_path) in self.last_event_time and (current_time - self.last_event_time[str(src_path)]) < 2:
            logger.debug(f"Debouncing event for {src_path}")
            return
        self.last_event_time[str(src_path)] = current_time

        # --- Agent re-deployment logic ---
        affected_agents = self._find_affected_agents(str(src_path))
        if not affected_agents:
            return

        for agent_config in affected_agents:
            logger.info(f"Agent '{agent_config.name}' is affected. Triggering re-deployment...")
            try:
                asyncio.run(self._run_async_with_db_session(embed_agent_data, config=agent_config))
                logger.info(f"Successfully re-deployed agent '{agent_config.name}'.")
            except Exception as e:
                logger.error(f"Failed to re-deploy agent '{agent_config.name}': {e}", exc_info=True)


    def _find_affected_agents(self, changed_path: str):
        """
        Scans all agent configs to see if their local data sources include the changed path,
        and if the file type matches (if specified).
        """
        affected = []
        all_agent_configs = get_agent_configs()
        normalized_changed_path = os.path.normpath(changed_path)
        changed_file_extension = _get_file_extension(changed_path)

        for config in all_agent_configs:
            for source in config.sources:
                if source.type == 'local' and source.path:
                    normalized_source_path = os.path.normpath(source.path)

                    is_exact_match = (normalized_changed_path == normalized_source_path)

                    is_within_directory = False
                    if Path(normalized_source_path).is_dir() and normalized_changed_path.startswith(normalized_source_path + os.sep):
                        is_within_directory = True

                    path_matches = is_exact_match or is_within_directory

                    file_type_matches = True
                    if source.file_types:
                        file_type_matches = changed_file_extension in [ext.lower().lstrip('.') for ext in source.file_types]

                    if path_matches and file_type_matches:
                        affected.append(config)
                        break
        return affected


def start_watcher(directories: list[str]):
    """Starts the file system watcher on the specified directories."""
    from app.core.structured_logging import get_logging_config
    logging.config.dictConfig(get_logging_config())

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
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()