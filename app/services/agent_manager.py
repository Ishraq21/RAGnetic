"""
Agent State Machine Manager

Handles agent state transitions following AWS-like patterns:
- Created → Configured → Deploying → Deployed
- Deployed → Idle (after inactivity)
- Deployed/Idle → Stopped (manual)
- Deploying → Error (if startup fails)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from app.db.models import agents_table, gpu_instances_table
from app.schemas.agent import AgentDeploymentStatus

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent deployment states following AWS-like patterns."""
    CREATED = "created"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    IDLE = "idle"
    ERROR = "error"
    STOPPED = "stopped"

class AgentManager:
    """Manages agent state transitions and lifecycle."""
    
    def __init__(self):
        self.idle_timeout_minutes = 30  # Auto-idle after 30 minutes of inactivity
        self._state_transitions = {
            AgentState.CREATED: [AgentState.DEPLOYING, AgentState.STOPPED],
            AgentState.DEPLOYING: [AgentState.DEPLOYED, AgentState.ERROR, AgentState.STOPPED],
            AgentState.DEPLOYED: [AgentState.IDLE, AgentState.STOPPED, AgentState.ERROR, AgentState.DEPLOYING],
            AgentState.IDLE: [AgentState.DEPLOYED, AgentState.STOPPED, AgentState.ERROR],
            AgentState.ERROR: [AgentState.DEPLOYING, AgentState.STOPPED],
            AgentState.STOPPED: [AgentState.DEPLOYING]
        }
    
    def can_transition(self, from_state: AgentState, to_state: AgentState) -> bool:
        """Check if a state transition is valid."""
        return to_state in self._state_transitions.get(from_state, [])
    
    async def transition_agent_status(
        self, 
        db: AsyncSession, 
        agent_name: str, 
        new_status: AgentState,
        gpu_instance_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Transition an agent to a new status with validation."""
        try:
            # Get current agent status
            result = await db.execute(
                select(agents_table).where(agents_table.c.name == agent_name)
            )
            agent = result.fetchone()
            
            if not agent:
                logger.error(f"Agent {agent_name} not found in database")
                return False
            
            current_status = AgentState(agent.status)
            
            # Validate transition
            if not self.can_transition(current_status, new_status):
                logger.warning(
                    f"Invalid transition from {current_status.value} to {new_status.value} for agent {agent_name}"
                )
                return False
            
            # Update agent status
            update_data = {
                "status": new_status.value,
                "last_updated": datetime.utcnow()
            }
            
            if gpu_instance_id:
                update_data["gpu_instance_id"] = gpu_instance_id
            
            await db.execute(
                update(agents_table)
                .where(agents_table.c.name == agent_name)
                .values(**update_data)
            )
            
            await db.commit()
            
            logger.info(f"Agent {agent_name} transitioned from {current_status.value} to {new_status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to transition agent {agent_name}: {e}")
            await db.rollback()
            return False
    
    async def deploy_agent(
        self, 
        db: AsyncSession, 
        agent_name: str,
        gpu_instance_id: Optional[str] = None
    ) -> bool:
        """Deploy an agent (transition to deploying then deployed)."""
        try:
            # Start deployment
            success = await self.transition_agent_status(
                db, agent_name, AgentState.DEPLOYING, gpu_instance_id
            )
            
            if not success:
                return False
            
            # Create vectorstore directory for the agent
            from app.core.config import get_path_settings
            _APP_PATHS = get_path_settings()
            _VECTORSTORE_DIR = _APP_PATHS["VECTORSTORE_DIR"]
            
            vs_dir = _VECTORSTORE_DIR / agent_name
            vs_dir.mkdir(parents=True, exist_ok=True)
            
            # Remove offline marker if it exists (to bring agent back online)
            offline_marker = vs_dir / ".offline"
            if offline_marker.exists():
                offline_marker.unlink()
                logger.info(f"Removed offline marker for agent {agent_name}")
            
            logger.info(f"Created vectorstore directory for agent {agent_name}: {vs_dir}")
            
            # Simulate deployment process (in real implementation, this would be async)
            # For now, we'll immediately transition to deployed
            await asyncio.sleep(1)  # Simulate deployment time
            
            success = await self.transition_agent_status(
                db, agent_name, AgentState.DEPLOYED, gpu_instance_id
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to deploy agent {agent_name}: {e}")
            # Transition to error state
            await self.transition_agent_status(
                db, agent_name, AgentState.ERROR, error_message=str(e)
            )
            return False
    
    async def stop_agent(
        self, 
        db: AsyncSession, 
        agent_name: str
    ) -> bool:
        """Stop an agent (transition to stopped)."""
        return await self.transition_agent_status(
            db, agent_name, AgentState.STOPPED
        )
    
    async def resume_agent(
        self, 
        db: AsyncSession, 
        agent_name: str
    ) -> bool:
        """Resume an idle agent (transition to deployed)."""
        return await self.transition_agent_status(
            db, agent_name, AgentState.DEPLOYED
        )
    
    async def retry_deployment(
        self, 
        db: AsyncSession, 
        agent_name: str
    ) -> bool:
        """Retry deployment for an agent in error state."""
        return await self.deploy_agent(db, agent_name)
    
    async def check_idle_agents(self, db: AsyncSession) -> List[str]:
        """Check for agents that should be transitioned to idle state."""
        try:
            # Get agents that have been deployed for more than idle_timeout_minutes
            cutoff_time = datetime.utcnow() - timedelta(minutes=self.idle_timeout_minutes)
            
            result = await db.execute(
                select(agents_table.c.name)
                .where(
                    agents_table.c.status == AgentState.DEPLOYED.value,
                    agents_table.c.last_updated < cutoff_time
                )
            )
            
            idle_agents = [row[0] for row in result.fetchall()]
            
            # Transition agents to idle
            for agent_name in idle_agents:
                await self.transition_agent_status(
                    db, agent_name, AgentState.IDLE
                )
                logger.info(f"Agent {agent_name} transitioned to idle due to inactivity")
            
            return idle_agents
            
        except Exception as e:
            logger.error(f"Failed to check idle agents: {e}")
            return []
    
    async def get_agent_status(self, db: AsyncSession, agent_name: str) -> Optional[Dict]:
        """Get current agent status and metadata."""
        try:
            result = await db.execute(
                select(agents_table)
                .where(agents_table.c.name == agent_name)
            )
            agent = result.fetchone()
            
            if not agent:
                return None
            
            return {
                "name": agent.name,
                "status": agent.status,
                "created_at": agent.created_at,
                "total_cost": agent.total_cost,
                "gpu_instance_id": agent.gpu_instance_id,
                "display_name": agent.display_name,
                "description": agent.description,
                "model_name": agent.model_name,
                "embedding_model": agent.embedding_model,
                "last_run": agent.last_run,
                "deployment_type": agent.deployment_type,
                "project_id": agent.project_id,
                "tags": agent.tags
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent status for {agent_name}: {e}")
            return None
    
    async def get_all_agents_status(self, db: AsyncSession) -> List[Dict]:
        """Get status for all agents with real analytics data."""
        try:
            result = await db.execute(
                select(agents_table)
                .order_by(agents_table.c.created_at.desc())
            )
            agents = result.fetchall()
            
            # Get analytics data for all agents
            from app.api.analytics import get_usage_summary_for_agents
            agent_names = [agent.name for agent in agents]
            analytics_data = await get_usage_summary_for_agents(db, agent_names)
            
            # Create lookup by agent name - sum costs for agents with multiple entries
            analytics_lookup = {}
            for data in analytics_data:
                agent_name = data['agent_name']
                if agent_name not in analytics_lookup:
                    analytics_lookup[agent_name] = 0.0
                analytics_lookup[agent_name] += data['total_estimated_cost_usd']
            
            return [
                {
                    "name": agent.name,
                    "status": agent.status,
                    "created_at": agent.created_at,
                    "total_cost": analytics_lookup.get(agent.name, 0.0),
                    "gpu_instance_id": agent.gpu_instance_id,
                    "display_name": agent.display_name,
                    "description": agent.description,
                    "model_name": agent.model_name,
                    "embedding_model": agent.embedding_model,
                    "last_run": agent.last_run,
                    "deployment_type": agent.deployment_type,
                    "project_id": agent.project_id,
                    "tags": agent.tags
                }
                for agent in agents
            ]
            
        except Exception as e:
            logger.error(f"Failed to get all agents status: {e}")
            return []
    
    async def update_agent_cost(
        self, 
        db: AsyncSession, 
        agent_name: str, 
        additional_cost: float
    ) -> bool:
        """Update agent total cost."""
        try:
            await db.execute(
                update(agents_table)
                .where(agents_table.c.name == agent_name)
                .values(
                    total_cost=agents_table.c.total_cost + additional_cost,
                    last_updated=datetime.utcnow()
                )
            )
            await db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent cost for {agent_name}: {e}")
            await db.rollback()
            return False
    
    def get_status_badge_info(self, status: str) -> Dict[str, str]:
        """Get badge styling information for agent status."""
        badge_info = {
            AgentState.CREATED.value: {
                "class": "badge badge-secondary",
                "text": "Created",
                "icon": ""
            },
            AgentState.CONFIGURED.value: {
                "class": "badge badge-info", 
                "text": "Configured",
                "icon": ""
            },
            AgentState.DEPLOYING.value: {
                "class": "badge badge-warning",
                "text": "Deploying",
                "icon": ""
            },
            AgentState.DEPLOYED.value: {
                "class": "badge badge-success",
                "text": "Deployed", 
                "icon": ""
            },
            AgentState.IDLE.value: {
                "class": "badge badge-primary",
                "text": "Idle",
                "icon": ""
            },
            AgentState.ERROR.value: {
                "class": "badge badge-danger",
                "text": "Error",
                "icon": ""
            },
            AgentState.STOPPED.value: {
                "class": "badge badge-secondary",
                "text": "Stopped",
                "icon": ""
            }
        }
        
        return badge_info.get(status, {
            "class": "badge badge-secondary",
            "text": status.title(),
            "icon": "❓"
        })
    
    def get_available_actions(self, status: str) -> List[Dict[str, str]]:
        """Get available actions for an agent based on its status."""
        actions = {
            AgentState.CREATED.value: [
                {"action": "deploy", "text": "Deploy", "class": "btn btn-primary btn-sm"}
            ],
            AgentState.CONFIGURED.value: [
                {"action": "deploy", "text": "Deploy", "class": "btn btn-primary btn-sm"},
                {"action": "stop", "text": "Stop", "class": "btn btn-secondary btn-sm"}
            ],
            AgentState.DEPLOYING.value: [
                {"action": "stop", "text": "Stop", "class": "btn btn-secondary btn-sm"}
            ],
            AgentState.DEPLOYED.value: [
                {"action": "stop", "text": "Shutdown", "class": "btn btn-secondary btn-sm"},
                {"action": "logs", "text": "View Logs", "class": "btn btn-info btn-sm"}
            ],
            AgentState.IDLE.value: [
                {"action": "resume", "text": "Resume", "class": "btn btn-primary btn-sm"},
                {"action": "stop", "text": "Shutdown", "class": "btn btn-secondary btn-sm"},
                {"action": "logs", "text": "View Logs", "class": "btn btn-info btn-sm"}
            ],
            AgentState.ERROR.value: [
                {"action": "retry", "text": "Retry", "class": "btn btn-warning btn-sm"},
                {"action": "stop", "text": "Stop", "class": "btn btn-secondary btn-sm"},
                {"action": "logs", "text": "View Logs", "class": "btn btn-info btn-sm"}
            ],
            AgentState.STOPPED.value: [
                {"action": "deploy", "text": "Deploy", "class": "btn btn-primary btn-sm"},
                {"action": "delete", "text": "Delete", "class": "btn btn-danger btn-sm"}
            ]
        }
        
        return actions.get(status, [])

# Global agent manager instance
agent_manager = AgentManager()
