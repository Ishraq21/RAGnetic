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
                    vectorstore_path = Path(f"vectorstore/{agent_name}")
                    status = 'deployed' if vectorstore_path.exists() else 'created'
                    
                    # Insert into database
                    now = datetime.now().isoformat()
                    cursor.execute('''
                        INSERT INTO agents (name, display_name, description, model_name, status, created_at, updated_at, last_updated, total_cost)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (agent_name, display_name, description, model_name, status, now, now, now, 0.0))
                    
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
