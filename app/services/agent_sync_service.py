"""
Agent Synchronization Service

Handles synchronization between agent YAML files and database records.
Ensures consistency between file system and database state.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, and_
from sqlalchemy.exc import IntegrityError

from app.db.models import agents_table, agent_tools_table
from app.schemas.agent import AgentConfig
from app.agents.config_manager import load_agent_config, get_agent_configs

logger = logging.getLogger(__name__)

class AgentSyncService:
    """Handles synchronization between agent YAML files and database records."""
    
    def __init__(self):
        self.agents_dir = "agents"
    
    async def sync_agent_to_db(
        self, 
        db: AsyncSession, 
        agent_config: AgentConfig,
        user_id: int = 1  # Default user ID, should be passed from request
    ) -> Dict[str, Any]:
        """Sync an agent configuration to the database."""
        try:
            # Check if agent exists in database
            result = await db.execute(
                select(agents_table).where(agents_table.c.name == agent_config.name)
            )
            existing_agent = result.fetchone()
            
            if existing_agent:
                # Update existing agent
                await self._update_agent_in_db(db, agent_config, existing_agent.id)
                return {"action": "updated", "agent_id": existing_agent.id}
            else:
                # Create new agent
                agent_id = await self._create_agent_in_db(db, agent_config, user_id)
                return {"action": "created", "agent_id": agent_id}
                
        except Exception as e:
            logger.error(f"Failed to sync agent {agent_config.name} to database: {e}")
            raise
    
    async def _create_agent_in_db(
        self, 
        db: AsyncSession, 
        agent_config: AgentConfig, 
        user_id: int
    ) -> int:
        """Create a new agent record in the database."""
        try:
            # Insert agent record
            agent_data = {
                "name": agent_config.name,
                "display_name": agent_config.display_name,
                "description": agent_config.description,
                "model_name": agent_config.llm_model,
                "status": "created",
                "last_updated": datetime.utcnow(),
                "total_cost": 0.0,
                "gpu_instance_id": None,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await db.execute(
                insert(agents_table).values(**agent_data)
            )
            agent_id = result.inserted_primary_key[0]
            
            # Insert agent tools
            if agent_config.tools:
                await self._sync_agent_tools(db, agent_id, agent_config.tools)
            
            await db.commit()
            logger.info(f"Created agent {agent_config.name} in database with ID {agent_id}")
            return agent_id
            
        except IntegrityError as e:
            await db.rollback()
            logger.error(f"Integrity error creating agent {agent_config.name}: {e}")
            raise
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create agent {agent_config.name} in database: {e}")
            raise
    
    async def _update_agent_in_db(
        self, 
        db: AsyncSession, 
        agent_config: AgentConfig, 
        agent_id: int
    ) -> None:
        """Update an existing agent record in the database."""
        try:
            # Update agent record
            update_data = {
                "display_name": agent_config.display_name,
                "description": agent_config.description,
                "model_name": agent_config.llm_model,
                "updated_at": datetime.utcnow()
            }
            
            await db.execute(
                update(agents_table)
                .where(agents_table.c.id == agent_id)
                .values(**update_data)
            )
            
            # Update agent tools
            if agent_config.tools:
                await self._sync_agent_tools(db, agent_id, agent_config.tools)
            
            await db.commit()
            logger.info(f"Updated agent {agent_config.name} in database")
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to update agent {agent_config.name} in database: {e}")
            raise
    
    async def _sync_agent_tools(
        self, 
        db: AsyncSession, 
        agent_id: int, 
        tools: List[str]
    ) -> None:
        """Sync agent tools to the database."""
        try:
            # Delete existing tools
            await db.execute(
                delete(agent_tools_table).where(agent_tools_table.c.agent_id == agent_id)
            )
            
            # Insert new tools
            if tools:
                tool_records = [
                    {"agent_id": agent_id, "tool_name": tool, "tool_config": None}
                    for tool in tools
                ]
                await db.execute(insert(agent_tools_table), tool_records)
            
        except Exception as e:
            logger.error(f"Failed to sync tools for agent {agent_id}: {e}")
            raise
    
    async def sync_all_agents_to_db(
        self, 
        db: AsyncSession, 
        user_id: int = 1
    ) -> Dict[str, Any]:
        """Sync all YAML agents to database."""
        try:
            # Get all agent configs from YAML files
            agent_configs = get_agent_configs()
            
            results = {
                "synced": [],
                "errors": [],
                "total": len(agent_configs)
            }
            
            for agent_config in agent_configs:
                try:
                    result = await self.sync_agent_to_db(db, agent_config, user_id)
                    results["synced"].append({
                        "name": agent_config.name,
                        "action": result["action"],
                        "agent_id": result["agent_id"]
                    })
                except Exception as e:
                    results["errors"].append({
                        "name": agent_config.name,
                        "error": str(e)
                    })
                    logger.error(f"Failed to sync agent {agent_config.name}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to sync all agents: {e}")
            raise
    
    async def get_agent_from_db(
        self, 
        db: AsyncSession, 
        agent_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get agent from database."""
        try:
            result = await db.execute(
                select(agents_table).where(agents_table.c.name == agent_name)
            )
            agent = result.fetchone()
            
            if not agent:
                return None
            
            return {
                "id": agent.id,
                "name": agent.name,
                "display_name": agent.display_name,
                "description": agent.description,
                "model_name": agent.model_name,
                "status": agent.status,
                "last_updated": agent.last_updated,
                "total_cost": agent.total_cost,
                "gpu_instance_id": agent.gpu_instance_id,
                "created_at": agent.created_at,
                "updated_at": agent.updated_at,
                "user_id": agent.user_id
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent {agent_name} from database: {e}")
            return None
    
    async def delete_agent_from_db(
        self, 
        db: AsyncSession, 
        agent_name: str,
        user_id: int = None
    ) -> bool:
        """Delete agent from database."""
        try:
            if user_id is not None:
                # Delete only the specific user's agent
                result = await db.execute(
                    delete(agents_table).where(
                        and_(agents_table.c.name == agent_name, agents_table.c.user_id == user_id)
                    )
                )
            else:
                # Delete all agents with this name (backward compatibility)
                result = await db.execute(
                    delete(agents_table).where(agents_table.c.name == agent_name)
                )
            
            if result.rowcount == 0:
                return False
            
            await db.commit()
            logger.info(f"Deleted agent {agent_name} from database")
            return True
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to delete agent {agent_name} from database: {e}")
            return False
    
    async def ensure_agent_exists_in_db(
        self, 
        db: AsyncSession, 
        agent_name: str,
        user_id: int = 1
    ) -> Optional[int]:
        """Ensure agent exists in database, create if missing."""
        try:
            # Check if agent exists
            agent_data = await self.get_agent_from_db(db, agent_name)
            if agent_data:
                return agent_data["id"]
            
            # Try to load from YAML and sync
            try:
                agent_config = load_agent_config(agent_name)
                result = await self.sync_agent_to_db(db, agent_config, user_id)
                return result["agent_id"]
            except FileNotFoundError:
                logger.warning(f"Agent {agent_name} not found in YAML files")
                return None
                
        except Exception as e:
            logger.error(f"Failed to ensure agent {agent_name} exists in database: {e}")
            return None

# Global sync service instance
agent_sync_service = AgentSyncService()
