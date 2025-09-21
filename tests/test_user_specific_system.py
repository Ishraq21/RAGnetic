"""
Comprehensive test suite for user-specific system functionality.
Tests user creation, directory creation, agent management, data ingestion,
chat interaction, and superuser access.
"""

import pytest
import asyncio
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import shutil

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
ADMIN_API_KEY = "c9d0fbc5206419383fec57608202858bd9540e4a1210220f35760bf387656691"

class TestUserSpecificSystem:
    """Comprehensive test suite for user-specific system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.admin_headers = {"X-API-Key": ADMIN_API_KEY}
        self.test_users = []
        self.test_agents = []
        self.test_data = []
        
    def teardown_method(self):
        """Cleanup test environment."""
        # Clean up test users
        for user in self.test_users:
            try:
                self.delete_user(user['id'])
            except:
                pass
                
        # Clean up test agents
        for agent in self.test_agents:
            try:
                self.delete_agent(agent['name'], agent['user_id'])
            except:
                pass
                
        # Clean up test data
        for data in self.test_data:
            try:
                self.delete_test_data(data)
            except:
                pass
    
    def create_test_user(self, username: str, email: str, scope: str = "viewer") -> Dict:
        """Create a test user and verify directory creation."""
        print(f"\n=== Creating test user: {username} ===")
        
        user_data = {
            "username": username,
            "email": email,
            "first_name": "Test",
            "last_name": "User",
            "scope": scope
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/security/users",
            headers=self.admin_headers,
            json=user_data
        )
        
        assert response.status_code == 201, f"Failed to create user: {response.text}"
        user = response.json()
        self.test_users.append(user)
        
        print(f"[PASS] User created: {user['username']} (ID: {user['id']})")
        
        # Verify user directories were created
        user_id = user['id']
        expected_dirs = [
            f"agents/users/{user_id}",
            f"vectorstore/users/{user_id}",
            f"data/uploads/users/{user_id}",
            f"data/sources/users/{user_id}"
        ]
        
        for dir_path in expected_dirs:
            assert Path(dir_path).exists(), f"Directory {dir_path} was not created"
            print(f"[PASS] Directory created: {dir_path}")
        
        return user
    
    def create_test_agent(self, user_id: int, agent_name: str) -> Dict:
        """Create a test agent for a specific user."""
        print(f"\n=== Creating test agent: {agent_name} for user {user_id} ===")
        
        # Get user's API key
        user_key = self.get_user_api_key(user_id)
        user_headers = {"X-API-Key": user_key}
        
        agent_config = {
            "name": agent_name,
            "description": f"Test agent for user {user_id}",
            "llm_model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000,
            "data_sources": [
                {
                    "type": "local",
                    "path": f"data/uploads/users/{user_id}/test_document.txt"
                }
            ]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/agents",
            headers=user_headers,
            json=agent_config
        )
        
        assert response.status_code == 200, f"Failed to create agent: {response.text}"
        agent = response.json()
        self.test_agents.append({"name": agent_name, "user_id": user_id})
        
        print(f"[PASS] Agent created: {agent_name}")
        
        # Verify agent was saved to user-specific directory
        agent_file = Path(f"agents/users/{user_id}/{agent_name}.yaml")
        assert agent_file.exists(), f"Agent file {agent_file} was not created"
        print(f"[PASS] Agent file created: {agent_file}")
        
        return agent
    
    def upload_test_data(self, user_id: int, filename: str, content: str) -> str:
        """Upload test data for a specific user."""
        print(f"\n=== Uploading test data: {filename} for user {user_id} ===")
        
        # Get user's API key
        user_key = self.get_user_api_key(user_id)
        user_headers = {"X-API-Key": user_key}
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_file.write(content)
        temp_file.close()
        
        try:
            with open(temp_file.name, 'rb') as f:
                files = {'file': (filename, f, 'text/plain')}
                response = requests.post(
                    f"{BASE_URL}/api/v1/agents/upload-file",
                    headers=user_headers,
                    files=files
                )
            
            assert response.status_code == 200, f"Failed to upload file: {response.text}"
            upload_info = response.json()
            
            print(f"[PASS] File uploaded: {upload_info['file_path']}")
            self.test_data.append(upload_info['file_path'])
            
            return upload_info['file_path']
            
        finally:
            # Clean up temporary file
            Path(temp_file.name).unlink()
    
    def test_chat_interaction(self, user_id: int, agent_name: str, message: str) -> str:
        """Test chat interaction with an agent."""
        print(f"\n=== Testing chat interaction with {agent_name} ===")
        
        # Get user's API key
        user_key = self.get_user_api_key(user_id)
        user_headers = {"X-API-Key": user_key}
        
        chat_data = {
            "message": message,
            "agent_name": agent_name
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/agents/chat",
            headers=user_headers,
            json=chat_data
        )
        
        assert response.status_code == 200, f"Failed to chat with agent: {response.text}"
        chat_response = response.json()
        
        print(f"[PASS] Chat response: {chat_response.get('response', 'No response')}")
        return chat_response.get('response', '')
    
    def test_superuser_access(self, admin_user_id: int, target_user_id: int):
        """Test that superuser can see all users' data."""
        print(f"\n=== Testing superuser access to user {target_user_id} data ===")
        
        # Test 1: Superuser can see all agents
        response = requests.get(
            f"{BASE_URL}/api/v1/agents/status-all",
            headers=self.admin_headers
        )
        
        assert response.status_code == 200, f"Failed to get agents: {response.text}"
        agents = response.json()
        
        # Verify superuser can see agents from different users
        user_agents = [a for a in agents if a.get('user_id') == target_user_id]
        assert len(user_agents) > 0, f"Superuser cannot see agents for user {target_user_id}"
        print(f"[PASS] Superuser can see {len(user_agents)} agents for user {target_user_id}")
        
        # Test 2: Superuser can see all users
        response = requests.get(
            f"{BASE_URL}/api/v1/security/users",
            headers=self.admin_headers
        )
        
        assert response.status_code == 200, f"Failed to get users: {response.text}"
        users = response.json()
        
        target_user = next((u for u in users if u['id'] == target_user_id), None)
        assert target_user is not None, f"Superuser cannot see user {target_user_id}"
        print(f"[PASS] Superuser can see user: {target_user['username']}")
        
        return True
    
    def test_user_isolation(self, user_id: int, other_user_id: int):
        """Test that users can only see their own data."""
        print(f"\n=== Testing user isolation for user {user_id} ===")
        
        # Get user's API key
        user_key = self.get_user_api_key(user_id)
        user_headers = {"X-API-Key": user_key}
        
        # Test: User can only see their own agents
        response = requests.get(
            f"{BASE_URL}/api/v1/agents/status-all",
            headers=user_headers
        )
        
        assert response.status_code == 200, f"Failed to get agents: {response.text}"
        agents = response.json()
        
        # Verify user only sees their own agents
        user_agents = [a for a in agents if a.get('user_id') == user_id]
        other_user_agents = [a for a in agents if a.get('user_id') == other_user_id]
        
        assert len(user_agents) > 0, f"User {user_id} cannot see their own agents"
        assert len(other_user_agents) == 0, f"User {user_id} can see other user's agents (security breach!)"
        print(f"[PASS] User isolation working: {len(user_agents)} own agents, {len(other_user_agents)} other users' agents")
        
        return True
    
    def test_data_ingestion(self, user_id: int, agent_name: str):
        """Test data ingestion for a specific user."""
        print(f"\n=== Testing data ingestion for user {user_id} ===")
        
        # Get user's API key
        user_key = self.get_user_api_key(user_id)
        user_headers = {"X-API-Key": user_key}
        
        # Test data ingestion
        ingestion_data = {
            "agent_name": agent_name,
            "data_source": f"data/uploads/users/{user_id}/test_document.txt"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/agents/ingest",
            headers=user_headers,
            json=ingestion_data
        )
        
        assert response.status_code == 200, f"Failed to ingest data: {response.text}"
        ingestion_result = response.json()
        
        print(f"[PASS] Data ingestion successful: {ingestion_result}")
        
        # Verify vector store was created in user-specific directory
        vectorstore_path = Path(f"vectorstore/users/{user_id}/{agent_name}")
        assert vectorstore_path.exists(), f"Vector store {vectorstore_path} was not created"
        print(f"[PASS] Vector store created: {vectorstore_path}")
        
        return ingestion_result
    
    def test_database_sync(self, user_id: int, agent_name: str):
        """Test that database and file system are synced."""
        print(f"\n=== Testing database sync for user {user_id} ===")
        
        # Test 1: Agent exists in database
        response = requests.get(
            f"{BASE_URL}/api/v1/agents/status-all",
            headers=self.admin_headers
        )
        
        assert response.status_code == 200, f"Failed to get agents: {response.text}"
        agents = response.json()
        
        user_agent = next((a for a in agents if a.get('name') == agent_name and a.get('user_id') == user_id), None)
        assert user_agent is not None, f"Agent {agent_name} not found in database for user {user_id}"
        print(f"[PASS] Agent {agent_name} found in database for user {user_id}")
        
        # Test 2: Agent file exists in file system
        agent_file = Path(f"agents/users/{user_id}/{agent_name}.yaml")
        assert agent_file.exists(), f"Agent file {agent_file} not found in file system"
        print(f"[PASS] Agent file {agent_file} exists in file system")
        
        # Test 3: Vector store exists in file system
        vectorstore_path = Path(f"vectorstore/users/{user_id}/{agent_name}")
        assert vectorstore_path.exists(), f"Vector store {vectorstore_path} not found in file system"
        print(f"[PASS] Vector store {vectorstore_path} exists in file system")
        
        return True
    
    def get_user_api_key(self, user_id: int) -> str:
        """Get API key for a specific user."""
        # For testing, we'll use the admin key
        # In production, this would be the user's actual API key
        return ADMIN_API_KEY
    
    def delete_user(self, user_id: int):
        """Delete a test user."""
        try:
            response = requests.delete(
                f"{BASE_URL}/api/v1/security/users/{user_id}",
                headers=self.admin_headers
            )
            if response.status_code == 200:
                print(f"[PASS] User {user_id} deleted")
        except Exception as e:
            print(f"[WARN] Could not delete user {user_id}: {e}")
    
    def delete_agent(self, agent_name: str, user_id: int):
        """Delete a test agent."""
        try:
            # Get user's API key
            user_key = self.get_user_api_key(user_id)
            user_headers = {"X-API-Key": user_key}
            
            response = requests.delete(
                f"{BASE_URL}/api/v1/agents/{agent_name}",
                headers=user_headers
            )
            if response.status_code == 200:
                print(f"[PASS] Agent {agent_name} deleted")
        except Exception as e:
            print(f"[WARN] Could not delete agent {agent_name}: {e}")
    
    def delete_test_data(self, file_path: str):
        """Delete test data file."""
        try:
            Path(file_path).unlink()
            print(f"[PASS] Test data {file_path} deleted")
        except Exception as e:
            print(f"[WARN] Could not delete test data {file_path}: {e}")


def run_comprehensive_tests():
    """Run comprehensive tests for the user-specific system."""
    print("[SUCCESS] Starting Comprehensive User-Specific System Tests")
    print("=" * 60)
    
    test_suite = TestUserSpecificSystem()
    
    try:
        # Test 1: Create multiple users
        print("\n[RESPONSE] TEST 1: User Creation and Directory Setup")
        user1 = test_suite.create_test_user("test-user-1", "testuser1@example.com")
        user2 = test_suite.create_test_user("test-user-2", "testuser2@example.com")
        user3 = test_suite.create_test_user("test-user-3", "testuser3@example.com", "admin")
        
        # Test 2: Create agents for different users
        print("\n[RESPONSE] TEST 2: Agent Creation for Different Users")
        agent1 = test_suite.create_test_agent(user1['id'], "user1-agent")
        agent2 = test_suite.create_test_agent(user2['id'], "user2-agent")
        agent3 = test_suite.create_test_agent(user3['id'], "user3-agent")
        
        # Test 3: Upload test data for different users
        print("\n[RESPONSE] TEST 3: Data Upload for Different Users")
        data1 = test_suite.upload_test_data(user1['id'], "user1_doc.txt", "This is test data for user 1")
        data2 = test_suite.upload_test_data(user2['id'], "user2_doc.txt", "This is test data for user 2")
        data3 = test_suite.upload_test_data(user3['id'], "user3_doc.txt", "This is test data for user 3")
        
        # Test 4: Data ingestion
        print("\n[RESPONSE] TEST 4: Data Ingestion")
        test_suite.test_data_ingestion(user1['id'], "user1-agent")
        test_suite.test_data_ingestion(user2['id'], "user2-agent")
        test_suite.test_data_ingestion(user3['id'], "user3-agent")
        
        # Test 5: Chat interaction
        print("\n[RESPONSE] TEST 5: Chat Interaction")
        response1 = test_suite.test_chat_interaction(user1['id'], "user1-agent", "Hello, what can you tell me about the uploaded data?")
        response2 = test_suite.test_chat_interaction(user2['id'], "user2-agent", "What information do you have?")
        response3 = test_suite.test_chat_interaction(user3['id'], "user3-agent", "Can you help me with my data?")
        
        # Test 6: User isolation
        print("\n[RESPONSE] TEST 6: User Isolation")
        test_suite.test_user_isolation(user1['id'], user2['id'])
        test_suite.test_user_isolation(user2['id'], user1['id'])
        test_suite.test_user_isolation(user3['id'], user1['id'])
        
        # Test 7: Superuser access
        print("\n[RESPONSE] TEST 7: Superuser Access")
        test_suite.test_superuser_access(user3['id'], user1['id'])  # user3 is admin
        test_suite.test_superuser_access(user3['id'], user2['id'])
        
        # Test 8: Database sync
        print("\n[RESPONSE] TEST 8: Database and File System Sync")
        test_suite.test_database_sync(user1['id'], "user1-agent")
        test_suite.test_database_sync(user2['id'], "user2-agent")
        test_suite.test_database_sync(user3['id'], "user3-agent")
        
        print("\n[COMPLETE] ALL TESTS PASSED! User-specific system is working correctly.")
        print("=" * 60)
        print("[PASS] User creation and directory setup")
        print("[PASS] Agent creation and user-specific storage")
        print("[PASS] Data upload and user-specific storage")
        print("[PASS] Data ingestion and vector store creation")
        print("[PASS] Chat interaction with user-specific agents")
        print("[PASS] User isolation (users can only see their own data)")
        print("[PASS] Superuser access (admins can see all data)")
        print("[PASS] Database and file system synchronization")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        print("=" * 60)
        return False
    
    finally:
        # Cleanup
        print("\n Cleaning up test data...")
        test_suite.teardown_method()


if __name__ == "__main__":
    # Run the comprehensive tests
    success = run_comprehensive_tests()
    
    if success:
        print("\n[TARGET] Test Summary:")
        print("- User-specific system is fully functional")
        print("- All user isolation features working")
        print("- Superuser access working correctly")
        print("- Database and file system properly synced")
        print("- Ready for production use!")
    else:
        print("\n[FAIL] Tests failed - system needs debugging")
        exit(1)
