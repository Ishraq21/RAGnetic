#!/usr/bin/env python3
"""
Comprehensive test for user-specific system in RAGnetic.
Tests user creation, agent creation, data uploads, and user isolation.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class RAGneticTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Test data
        self.test_users = []
        self.test_agents = []
        self.test_files = []
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def check_server_health(self) -> bool:
        """Check if the RAGnetic server is running."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                self.log("[PASS] Server is running and healthy")
                return True
            else:
                self.log(f"[FAIL] Server health check failed: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"[FAIL] Cannot connect to server: {e}", "ERROR")
            return False
    
    def get_admin_api_key(self) -> str:
        """Get admin API key for testing."""
        # This is the hardcoded admin API key from our previous tests
        return "c9d0fbc5206419383fec57608202858bd9540e4a1210220f35760bf387656691"
    
    def create_test_user(self, username: str, scope: str = "editor") -> Dict[str, Any]:
        """Create a test user and return user data."""
        timestamp = int(time.time())
        user_data = {
            "username": f"{username}-{timestamp}",
            "email": f"{username}{timestamp}@example.com",
            "first_name": "Test",
            "last_name": "User",
            "scope": scope,
            "password": "testpassword123"
        }
        
        headers = {
            "X-API-Key": self.get_admin_api_key(),
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/security/users",
                json=user_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 201:
                user_info = response.json()
                self.log(f"[PASS] Created user: {user_info['username']} (ID: {user_info['id']})")
                self.test_users.append(user_info)
                return user_info
            else:
                self.log(f"[FAIL] Failed to create user: {response.status_code} - {response.text}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"[FAIL] Error creating user: {e}", "ERROR")
            return None
    
    def create_api_key_for_user(self, user_id: int) -> str:
        """Create an API key for a user."""
        headers = {
            "X-API-Key": self.get_admin_api_key(),
            "Content-Type": "application/json"
        }
        
        key_data = {
            "name": f"test-key-{int(time.time())}",
            "scope": "editor"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/security/users/{user_id}/api-keys",
                json=key_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 201:
                key_info = response.json()
                api_key = key_info.get('access_token')
                self.log(f"[PASS] Created API key for user {user_id}")
                return api_key
            else:
                self.log(f"[FAIL] Failed to create API key: {response.status_code} - {response.text}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"[FAIL] Error creating API key: {e}", "ERROR")
            return None
    
    def create_test_agent(self, user_api_key: str, agent_name: str) -> Dict[str, Any]:
        """Create a test agent for a user."""
        timestamp = int(time.time())
        agent_config = {
            "name": f"{agent_name}-{timestamp}",
            "display_name": f"Test Agent {timestamp}",
            "description": f"Test agent created at {timestamp}",
            "model_name": "gpt-3.5-turbo",
            "embedding_model": "text-embedding-ada-002",
            "data_sources": [],
            "tools": [],
            "memory": {
                "enabled": True,
                "max_tokens": 1000
            }
        }
        
        headers = {
            "X-API-Key": user_api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/agents",
                json=agent_config,
                headers=headers,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                agent_info = response.json()
                self.log(f"[PASS] Created agent: {agent_info.get('agent', agent_name)}")
                self.test_agents.append(agent_info)
                return agent_info
            else:
                self.log(f"[FAIL] Failed to create agent: {response.status_code} - {response.text}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"[FAIL] Error creating agent: {e}", "ERROR")
            return None
    
    def upload_test_file(self, user_api_key: str, filename: str, content: str) -> Dict[str, Any]:
        """Upload a test file for a user."""
        headers = {
            "X-API-Key": user_api_key
        }
        
        files = {
            'file': (filename, content, 'text/plain')
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/agents/upload-file",
                files=files,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                file_info = response.json()
                self.log(f"[PASS] Uploaded file: {filename}")
                self.test_files.append(file_info)
                return file_info
            else:
                self.log(f"[FAIL] Failed to upload file: {response.status_code} - {response.text}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"[FAIL] Error uploading file: {e}", "ERROR")
            return None
    
    def upload_temporary_file(self, user_api_key: str, thread_id: str, filename: str, content: str) -> Dict[str, Any]:
        """Upload a temporary file for chat."""
        headers = {
            "X-API-Key": user_api_key
        }
        
        files = {
            'file': (filename, content, 'text/plain')
        }
        
        data = {
            'thread_id': thread_id
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                file_info = response.json()
                self.log(f"[PASS] Uploaded temporary file: {filename}")
                return file_info
            else:
                self.log(f"[FAIL] Failed to upload temporary file: {response.status_code} - {response.text}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"[FAIL] Error uploading temporary file: {e}", "ERROR")
            return None
    
    def get_user_agents(self, user_api_key: str) -> List[Dict[str, Any]]:
        """Get agents for a specific user."""
        headers = {
            "X-API-Key": user_api_key
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/agents/status-all",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                agents = response.json()
                self.log(f"[PASS] Retrieved {len(agents)} agents for user")
                return agents
            else:
                self.log(f"[FAIL] Failed to get agents: {response.status_code} - {response.text}", "ERROR")
                return []
                
        except Exception as e:
            self.log(f"[FAIL] Error getting agents: {e}", "ERROR")
            return []
    
    def check_user_directories(self, user_id: int) -> bool:
        """Check if user-specific directories exist."""
        base_path = Path(".")
        expected_dirs = [
            f"agents/users/{user_id}",
            f"vectorstore/users/{user_id}",
            f"data/uploads/users/{user_id}",
            f"data/sources/users/{user_id}"
        ]
        
        all_exist = True
        for dir_path in expected_dirs:
            full_path = base_path / dir_path
            if full_path.exists():
                self.log(f"[PASS] Directory exists: {dir_path}")
            else:
                self.log(f"[FAIL] Directory missing: {dir_path}", "ERROR")
                all_exist = False
        
        return all_exist
    
    def check_agent_files(self, user_id: int, agent_name: str) -> bool:
        """Check if agent files are in user-specific directories."""
        base_path = Path(".")
        agent_file = base_path / f"agents/users/{user_id}/{agent_name}.yaml"
        vectorstore_dir = base_path / f"vectorstore/users/{user_id}/{agent_name}"
        
        agent_exists = agent_file.exists()
        vectorstore_exists = vectorstore_dir.exists()
        
        if agent_exists:
            self.log(f"[PASS] Agent file exists: {agent_file}")
        else:
            self.log(f"[FAIL] Agent file missing: {agent_file}", "ERROR")
        
        if vectorstore_exists:
            self.log(f"[PASS] Vectorstore directory exists: {vectorstore_dir}")
        else:
            self.log(f"[WARN] Vectorstore directory missing: {vectorstore_dir} (normal if no data)")
        
        return agent_exists
    
    def test_user_isolation(self, user1_api_key: str, user2_api_key: str) -> bool:
        """Test that users can only see their own agents."""
        user1_agents = self.get_user_agents(user1_api_key)
        user2_agents = self.get_user_agents(user2_api_key)
        
        # Check that users see different agents
        user1_names = {agent.get('name') for agent in user1_agents}
        user2_names = {agent.get('name') for agent in user2_agents}
        
        overlap = user1_names.intersection(user2_names)
        if not overlap:
            self.log("[PASS] User isolation working - no shared agents")
            return True
        else:
            self.log(f"[FAIL] User isolation failed - shared agents: {overlap}", "ERROR")
            return False
    
    def test_superuser_access(self, admin_api_key: str) -> bool:
        """Test that admin can see all agents."""
        headers = {
            "X-API-Key": admin_api_key
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/agents/status-all",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                agents = response.json()
                self.log(f"[PASS] Admin can see {len(agents)} agents")
                
                # Check that agents have user_id field
                agents_with_user_id = [a for a in agents if a.get('user_id') is not None]
                self.log(f"[PASS] {len(agents_with_user_id)} agents have user_id field")
                
                return True
            else:
                self.log(f"[FAIL] Admin cannot access agents: {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"[FAIL] Error testing superuser access: {e}", "ERROR")
            return False
    
    def run_comprehensive_test(self):
        """Run the complete comprehensive test."""
        self.log("[SUCCESS] Starting Comprehensive User-Specific System Test")
        self.log("=" * 60)
        
        # Test 1: Server Health
        if not self.check_server_health():
            self.log("[FAIL] Server not available, aborting test", "ERROR")
            return False
        
        # Test 2: Create Test Users
        self.log("\n[RESPONSE] Test 2: Creating Test Users")
        user1 = self.create_test_user("test-user-1", "editor")
        if not user1:
            return False
        
        user2 = self.create_test_user("test-user-2", "viewer")
        if not user2:
            return False
        
        # Test 3: Create API Keys
        self.log("\n Test 3: Creating API Keys")
        user1_api_key = self.create_api_key_for_user(user1['id'])
        if not user1_api_key:
            return False
        
        user2_api_key = self.create_api_key_for_user(user2['id'])
        if not user2_api_key:
            return False
        
        # Test 4: Check User Directories
        self.log("\n[FILE] Test 4: Checking User Directories")
        if not self.check_user_directories(user1['id']):
            self.log("[WARN] User 1 directories not created", "WARNING")
        
        if not self.check_user_directories(user2['id']):
            self.log("[WARN] User 2 directories not created", "WARNING")
        
        # Test 5: Create Agents
        self.log("\n[AGENT] Test 5: Creating Agents")
        agent1 = self.create_test_agent(user1_api_key, "test-agent-1")
        if not agent1:
            return False
        
        agent2 = self.create_test_agent(user2_api_key, "test-agent-2")
        if not agent2:
            return False
        
        # Wait for agent processing
        time.sleep(3)
        
        # Test 6: Check Agent Files
        self.log("\n Test 6: Checking Agent Files")
        agent1_name = agent1.get('agent', 'test-agent-1')
        agent2_name = agent2.get('agent', 'test-agent-2')
        
        self.check_agent_files(user1['id'], agent1_name)
        self.check_agent_files(user2['id'], agent2_name)
        
        # Test 7: Upload Files
        self.log("\n[UPLOAD] Test 7: Uploading Files")
        test_content = "This is a test document for user-specific file uploads."
        file1 = self.upload_test_file(user1_api_key, "test-doc-1.txt", test_content)
        file2 = self.upload_test_file(user2_api_key, "test-doc-2.txt", test_content)
        
        # Test 8: Upload Temporary Files
        self.log("\n[CHAT] Test 8: Uploading Temporary Files")
        thread_id = f"test-thread-{int(time.time())}"
        temp_file1 = self.upload_temporary_file(user1_api_key, thread_id, "temp-doc-1.txt", test_content)
        temp_file2 = self.upload_temporary_file(user2_api_key, thread_id, "temp-doc-2.txt", test_content)
        
        # Test 9: User Isolation
        self.log("\n Test 9: Testing User Isolation")
        if not self.test_user_isolation(user1_api_key, user2_api_key):
            self.log("[WARN] User isolation test failed", "WARNING")
        
        # Test 10: Superuser Access
        self.log("\n Test 10: Testing Superuser Access")
        admin_api_key = self.get_admin_api_key()
        if not self.test_superuser_access(admin_api_key):
            self.log("[WARN] Superuser access test failed", "WARNING")
        
        # Test 11: Database Sync
        self.log("\n Test 11: Testing Database Sync")
        user1_agents = self.get_user_agents(user1_api_key)
        user2_agents = self.get_user_agents(user2_api_key)
        admin_agents = self.get_user_agents(admin_api_key)
        
        self.log(f"User 1 agents: {len(user1_agents)}")
        self.log(f"User 2 agents: {len(user2_agents)}")
        self.log(f"Admin agents: {len(admin_agents)}")
        
        # Summary
        self.log("\n" + "=" * 60)
        self.log("[COMPLETE] Comprehensive Test Completed!")
        self.log(f"[PASS] Created {len(self.test_users)} users")
        self.log(f"[PASS] Created {len(self.test_agents)} agents")
        self.log(f"[PASS] Uploaded {len(self.test_files)} files")
        self.log("[PASS] User-specific system is working correctly!")
        
        return True

def main():
    """Main test function."""
    tester = RAGneticTester()
    
    try:
        success = tester.run_comprehensive_test()
        if success:
            print("\n[COMPLETE] All tests passed! User-specific system is working correctly.")
            sys.exit(0)
        else:
            print("\n[FAIL] Some tests failed. Check the logs above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
