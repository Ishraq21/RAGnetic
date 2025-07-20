import logging
import requests
import json
import time  # <-- Add this import
from typing import Dict, Any, Optional, List
from apscheduler.schedulers.background import BackgroundScheduler
from app.core.config import get_path_settings
import configparser

logger = logging.getLogger(__name__)

_APP_PATHS = get_path_settings()
_CONFIG_FILE = _APP_PATHS["CONFIG_FILE_PATH"]


class WorkflowScheduler:
    """
    Schedules and runs workflows using APScheduler for cron-like scheduling.
    This version is resilient and fetches jobs from the database via the API.
    """

    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone="UTC")
        self.config = configparser.ConfigParser()
        self.config.read(_CONFIG_FILE)
        self.server_url = self._get_server_url()
        self.schedule_config = self._load_schedules_from_db()

    def _get_server_url(self) -> str:
        host = self.config.get('SERVER', 'host', fallback='127.0.0.1')
        port = self.config.get('SERVER', 'port', fallback='8000')
        return f"http://{self.config.get('SERVER', 'host', fallback='127.0.0.1')}:{self.config.get('SERVER', 'port', fallback='8000')}/api/v1"

    def _load_schedules_from_db(self) -> List[Dict[str, Any]]:
        """Fetches the active schedules from the database via API, with retries."""
        # --- FIX 1: Add a delay and retry logic ---
        schedules_url = f"{self.server_url}/schedules"
        for attempt in range(5):  # Try 5 times
            try:
                time.sleep(2)  # Wait 2 seconds for the main server to start
                response = requests.get(schedules_url, timeout=10)
                response.raise_for_status()
                logger.info("Successfully loaded schedules from the database.")
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Scheduler could not connect to API (attempt {attempt + 1}/5): {e}. Retrying...")

        logger.error("Failed to load schedules from database after multiple retries. Scheduler will have no jobs.")
        return []

    def _run_workflow(self, workflow_name: str, initial_input: Optional[Dict[str, Any]] = None):
        """Triggers a workflow via the API."""
        try:
            url = f"{self.server_url}/workflows/{workflow_name}/trigger"
            response = requests.post(url, json=initial_input, timeout=10)
            response.raise_for_status()
            logger.info(f"Successfully triggered scheduled workflow '{workflow_name}'.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to trigger scheduled workflow '{workflow_name}': {e}")

    def setup_jobs(self):
        """Adds jobs to the scheduler from the loaded database schedule."""
        for job_info in self.schedule_config:
            # --- FIX 2: Add defensive checks ---
            if not isinstance(job_info, dict):
                logger.warning(f"Skipping invalid schedule item (not a dict): {job_info}")
                continue

            name = job_info.get("workflow_name")
            cron = job_info.get("cron_schedule")

            if not name or not cron:
                logger.warning(f"Skipping invalid schedule item (missing name or cron_schedule): {job_info}")
                continue
            # --- End of defensive checks ---

            self.scheduler.add_job(
                self._run_workflow,
                trigger='cron',
                args=[name],
                kwargs={"initial_input": job_info.get("initial_input")},
                id=name,  # Use the workflow name as the job ID
                replace_existing=True,
                **cron
            )
            logger.info(f"Scheduled job '{name}' with schedule: {cron}")

    def start(self):
        """Starts the scheduler."""
        self.setup_jobs()
        self.scheduler.start()
        logger.info("RAGnetic scheduler process started.")

    def stop(self):
        """Stops the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("RAGnetic scheduler process stopped.")


_scheduler_instance: Optional[WorkflowScheduler] = None


def start_scheduler_process():
    """Starts the scheduler loop directly."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = WorkflowScheduler()
        _scheduler_instance.start()


def stop_scheduler_process():
    """Stops the scheduler."""
    global _scheduler_instance
    if _scheduler_instance:
        _scheduler_instance.stop()