import logging
import requests
import json
import time
from typing import Dict, Any, Optional, List
from apscheduler.schedulers.background import BackgroundScheduler
from app.core.config import get_path_settings
import configparser

logger = logging.getLogger(__name__)

_APP_PATHS = get_path_settings()
_CONFIG_FILE = _APP_PATHS["CONFIG_FILE_PATH"]


class WorkflowScheduler:
    """
    Schedules and runs workflows by periodically fetching schedule
    definitions from the database via the main application's API.
    """

    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone="UTC")
        self.config = configparser.ConfigParser()
        self.config.read(_CONFIG_FILE)
        self.server_url = self._get_server_url()
        # Initial load is no longer needed here, as the sync job will handle it.

    def _get_server_url(self) -> str:
        """Constructs the base URL for the RAGnetic API."""
        host = self.config.get('SERVER', 'host', fallback='127.0.0.1')
        port = self.config.get('SERVER', 'port', fallback='8000')
        return f"http://{host}:{port}/api/v1"

    def _run_workflow(self, workflow_name: str, initial_input: Optional[Dict[str, Any]] = None):
        """Triggers a workflow's execution via an API call to the main server."""
        trigger_url = f"{self.server_url}/workflows/{workflow_name}/trigger"
        try:
            response = requests.post(trigger_url, json=initial_input, timeout=15)
            response.raise_for_status()
            logger.info(f"Successfully triggered scheduled workflow '{workflow_name}'.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Scheduler failed to trigger workflow '{workflow_name}': {e}")

    def _sync_jobs_from_db(self):
        """
        Fetches all active schedules from the DB and updates the running scheduler.
        This is the core of the dynamic updates.
        """
        logger.info("Syncing schedules from database...")
        schedules_url = f"{self.server_url}/schedules"
        try:
            response = requests.get(schedules_url, timeout=10)
            response.raise_for_status()
            db_schedules = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Scheduler could not sync schedules from API: {e}. Will retry on next interval.")
            return

        # Create sets of job IDs for efficient comparison
        scheduled_job_ids = {job.id for job in self.scheduler.get_jobs() if job.id != 'sync_schedules_job'}
        db_job_ids = {f"wf_schedule_{s.get('id')}" for s in db_schedules}

        # Remove jobs from the scheduler that are no longer in the database
        for job_id in scheduled_job_ids - db_job_ids:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed stale schedule: '{job_id}'")

        # Add or update jobs from the database definitions
        for schedule_info in db_schedules:
            if not isinstance(schedule_info, dict):
                logger.warning(f"Skipping invalid schedule item (not a dict): {schedule_info}")
                continue

            name = schedule_info.get("workflow_name")
            cron_config = schedule_info.get("cron_schedule")
            schedule_id = schedule_info.get("id")

            if not all([name, cron_config, schedule_id]):
                logger.warning(f"Skipping invalid schedule item (missing name, cron_schedule, or id): {schedule_info}")
                continue

            job_id = f"wf_schedule_{schedule_id}"
            self.scheduler.add_job(
                self._run_workflow,
                trigger='cron',
                id=job_id,
                name=name,
                args=[name],
                kwargs={"initial_input": schedule_info.get("initial_input")},
                replace_existing=True,
                **cron_config
            )
        logger.info(f"Schedule sync complete. Current jobs: {len(self.scheduler.get_jobs()) - 1}")

    def start(self):
        """Starts the scheduler and the periodic sync job."""
        # Add the recurring job that keeps the scheduler in sync with the DB
        self.scheduler.add_job(
            self._sync_jobs_from_db,
            trigger='interval',
            minutes=5,
            id='sync_schedules_job'
        )
        # Run an initial sync immediately on startup
        self.scheduler.add_job(self._sync_jobs_from_db)

        self.scheduler.start()
        logger.info("RAGnetic scheduler process started and is now monitoring schedules.")

    def stop(self):
        """Stops the scheduler gracefully."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("RAGnetic scheduler process stopped.")


_scheduler_instance: Optional[WorkflowScheduler] = None


def start_scheduler_process():
    """Entry point for starting the scheduler process."""
    global _scheduler_instance
    # Add a delay to ensure the main FastAPI server is up before the first sync attempt.
    logger.info("Scheduler process starting in 5 seconds...")
    time.sleep(5)
    if _scheduler_instance is None:
        _scheduler_instance = WorkflowScheduler()
        _scheduler_instance.start()


def stop_scheduler_process():
    """Entry point for stopping the scheduler process."""
    global _scheduler_instance
    if _scheduler_instance:
        _scheduler_instance.stop()