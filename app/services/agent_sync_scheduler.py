"""
Agent Synchronization Scheduler

Automatically syncs agents from the file system to the database.
Runs as a background task to keep the database in sync with agent files.
"""

import os
import sqlite3
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AgentSyncScheduler:
    """Handles automatic synchronization of agents from file system to database."""
    
    def __init__(self, db_path: str = "memory/ragnetic.db", agents_dir: str = "agents"):
        self.db_path = db_path
        self.agents_dir = Path(agents_dir)
    
    def sync_agents_from_filesystem(self) -> Dict[str, Any]:
        """Sync agents from file system to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get existing agents from database
            cursor.execute('SELECT name FROM agents')
            existing_agents = {row[0] for row in cursor.fetchall()}
            
            synced_count = 0
            errors = []
            
            # Scan agents directory for YAML files
            for agent_file in self.agents_dir.glob('*.yaml'):
                agent_name = agent_file.stem
                
                if agent_name in existing_agents:
                    continue
                    
                try:
                    # Load agent config
                    with open(agent_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Extract agent info
                    display_name = config.get('display_name', agent_name.replace('-', ' ').replace('_', ' ').title())
                    description = config.get('description', f'Agent {display_name}')
                    model_name = config.get('model_name', 'gpt-4o-mini')
                    
                    # Check if agent is deployed (has vector store)
                    from app.core.user_paths import get_agent_vectorstore_path
                    vectorstore_path = get_agent_vectorstore_path(agent_name, 1)  # Default to admin user for legacy sync
                    status = 'deployed' if vectorstore_path.exists() else 'created'
                    
                    # Insert into database
                    now = datetime.now().isoformat()
                    cursor.execute('''
                        INSERT INTO agents (name, display_name, description, model_name, status, created_at, updated_at, last_updated, total_cost, user_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (agent_name, display_name, description, model_name, status, now, now, now, 0.0, 1))  # Default to admin user
                    
                    synced_count += 1
                    logger.info(f'Synced agent: {agent_name}')
                    
                except Exception as e:
                    error_msg = f'Error syncing {agent_name}: {e}'
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            conn.commit()
            
            # Check final count
            cursor.execute('SELECT COUNT(*) FROM agents')
            total_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'success': True,
                'synced_count': synced_count,
                'total_count': total_count,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f'Failed to sync agents: {e}')
            return {
                'success': False,
                'error': str(e),
                'synced_count': 0,
                'total_count': 0
            }
    
    def sync_user_agents(self, user_id: int, user_agents_dir: str = None) -> Dict[str, Any]:
        """Sync agents for a specific user from their agents directory."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Use user-specific agents directory if provided
            if user_agents_dir:
                agents_dir = Path(user_agents_dir)
            else:
                agents_dir = Path(f"agents/users/{user_id}")
            
            if not agents_dir.exists():
                return {
                    'success': True,
                    'synced_count': 0,
                    'total_count': 0,
                    'message': f'No agents directory found for user {user_id}'
                }
            
            # Get existing agents for this user
            cursor.execute('SELECT name FROM agents WHERE user_id = ?', (user_id,))
            existing_agents = {row[0] for row in cursor.fetchall()}
            
            synced_count = 0
            errors = []
            
            # Scan user's agents directory
            for agent_file in agents_dir.glob('*.yaml'):
                agent_name = agent_file.stem
                
                if agent_name in existing_agents:
                    continue
                    
                try:
                    # Load agent config
                    with open(agent_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Extract agent info
                    display_name = config.get('display_name', agent_name.replace('-', ' ').replace('_', ' ').title())
                    description = config.get('description', f'Agent {display_name}')
                    model_name = config.get('model_name', 'gpt-4o-mini')
                    
                    # Check if agent is deployed (has vector store)
                    from app.core.user_paths import get_user_vectorstore_path
                    vectorstore_path = get_user_vectorstore_path(user_id, agent_name)
                    status = 'deployed' if vectorstore_path.exists() else 'created'
                    
                    # Insert into database with user_id
                    now = datetime.now().isoformat()
                    cursor.execute('''
                        INSERT INTO agents (name, display_name, description, model_name, status, created_at, updated_at, last_updated, total_cost, user_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (agent_name, display_name, description, model_name, status, now, now, now, 0.0, user_id))
                    
                    synced_count += 1
                    logger.info(f'Synced agent {agent_name} for user {user_id}')
                    
                except Exception as e:
                    error_msg = f'Error syncing {agent_name} for user {user_id}: {e}'
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            conn.commit()
            
            # Check final count for this user
            cursor.execute('SELECT COUNT(*) FROM agents WHERE user_id = ?', (user_id,))
            user_agent_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'success': True,
                'synced_count': synced_count,
                'user_agent_count': user_agent_count,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f'Failed to sync agents for user {user_id}: {e}')
            return {
                'success': False,
                'error': str(e),
                'synced_count': 0,
                'user_agent_count': 0
            }

    def update_agent_status(self, agent_name: str, status: str) -> bool:
        """Update agent status in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            cursor.execute('''
                UPDATE agents 
                SET status = ?, updated_at = ?, last_updated = ?
                WHERE name = ?
            ''', (status, now, now, agent_name))
            
            conn.commit()
            conn.close()
            
            logger.info(f'Updated agent {agent_name} status to {status}')
            return True
            
        except Exception as e:
            logger.error(f'Failed to update agent {agent_name} status: {e}')
            return False

# Global instance
agent_sync_scheduler = AgentSyncScheduler()
