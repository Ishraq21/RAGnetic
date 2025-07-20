import logging
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
from sqlalchemy import create_engine, select, insert, update, delete

from app.db.models import schedules_table
from app.core.config import get_db_connection, get_memory_storage_config, get_log_storage_config
from app.schemas.schedule import ScheduleCreate, ScheduleUpdate, Schedule

logger = logging.getLogger(__name__)
router = APIRouter()


def get_sync_db_engine():
    """Helper to get a synchronous SQLAlchemy engine."""
    mem_cfg = get_memory_storage_config()
    log_cfg = get_log_storage_config()
    conn_name = mem_cfg.get("connection_name") if mem_cfg.get("type") in ["db", "sqlite"] else log_cfg.get(
        "connection_name")
    if not conn_name:
        raise RuntimeError("No database connection is configured.")
    conn_str = get_db_connection(conn_name)
    sync_conn_str = conn_str.replace('+aiosqlite', '').replace('+asyncpg', '')
    return create_engine(sync_conn_str)


@router.post("/schedules", status_code=status.HTTP_201_CREATED, response_model=Schedule)
def create_schedule(schedule: ScheduleCreate):
    """Creates a new workflow schedule in the database."""
    db_engine = get_sync_db_engine()
    with db_engine.connect() as connection:
        stmt = insert(schedules_table).values(**schedule.model_dump())
        try:
            result = connection.execute(stmt)
            connection.commit()
            return {**schedule.model_dump(), "id": result.inserted_primary_key[0]}
        except Exception as e:
            connection.rollback()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to create schedule: {e}")


@router.get("/schedules", response_model=List[Schedule])
def list_schedules():
    """Lists all configured schedules."""
    db_engine = get_sync_db_engine()
    with db_engine.connect() as connection:
        stmt = select(schedules_table)
        results = connection.execute(stmt).mappings().fetchall()
        return [Schedule(**row) for row in results]


@router.get("/schedules/{schedule_id}", response_model=Schedule)
def get_schedule(schedule_id: int):
    """Retrieves a specific schedule by its ID."""
    db_engine = get_sync_db_engine()
    with db_engine.connect() as connection:
        stmt = select(schedules_table).where(schedules_table.c.id == schedule_id)
        result = connection.execute(stmt).mappings().first()
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found.")
        return Schedule(**result)


@router.put("/schedules/{schedule_id}", response_model=Schedule)
def update_schedule(schedule_id: int, schedule: ScheduleUpdate):
    """Updates an existing schedule."""
    db_engine = get_sync_db_engine()
    with db_engine.connect() as connection:
        update_data = schedule.model_dump(exclude_unset=True)
        if not update_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided.")

        stmt = update(schedules_table).where(schedules_table.c.id == schedule_id).values(**update_data)
        result = connection.execute(stmt)
        connection.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found.")

        return get_schedule(schedule_id)  # Return the updated object


@router.delete("/schedules/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_schedule(schedule_id: int):
    """Deletes a schedule."""
    db_engine = get_sync_db_engine()
    with db_engine.connect() as connection:
        stmt = delete(schedules_table).where(schedules_table.c.id == schedule_id)
        result = connection.execute(stmt)
        connection.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found.")