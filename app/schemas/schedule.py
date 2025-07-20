from pydantic import BaseModel
from typing import Dict, Any, Optional

class ScheduleBase(BaseModel):
    name: str
    workflow_name: str
    cron_schedule: Dict[str, Any]
    initial_input: Optional[Dict[str, Any]] = None
    is_enabled: bool = True

class ScheduleCreate(ScheduleBase):
    pass

class ScheduleUpdate(BaseModel):
    name: Optional[str] = None
    workflow_name: Optional[str] = None
    cron_schedule: Optional[Dict[str, Any]] = None
    initial_input: Optional[Dict[str, Any]] = None
    is_enabled: Optional[bool] = None

class Schedule(ScheduleBase):
    id: int

    class Config:
        from_attributes = True