import time
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from app.agents.config_manager import get_agent_configs
from app.pipelines.embed import embed_agent_data

logger = logging.getLogger(__name__)


class AgentDataEventHandler(FileSystemEventHandler):
    """Handles file system events and triggers re-embedding for affected agents."""

    def on_any_event(self, event):
        if event.is_directory or event.event_type not in ['created', 'deleted', 'modified']:
            return

        src_path = event.src_path
        logger.info(f"File change detected: {src_path} ({event.event_type})")

        affected_agents = self._find_affected_agents(src_path)
        if not affected_agents:
            return

        for agent_config in affected_agents:
            logger.info(f"Agent '{agent_config.name}' is affected. Triggering re-deployment...")
            try:
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