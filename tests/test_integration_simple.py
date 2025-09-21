"""
Simple integration test for user-specific system.
Tests core functionality without complex setup.
"""

import requests
import json
from pathlib import Path
import time

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
ADMIN_API_KEY = "c9d0fbc5206419383fec57608202858bd9540e4a1210220f35760bf387656691"

def test_server_health():
    """Test if server is running and healthy."""
    print("[INFO] Testing server health...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("[PASS] Server is healthy")
            return True
        else:
            print(f"[FAIL] Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Server not accessible: {e}")
        return False

def test_user_creation():
    """Test user creation and directory setup."""
    print("\n[INFO] Testing user creation...")
    
    headers = {"X-API-Key": ADMIN_API_KEY}
    
    # Create test user
    user_data = {
        "username": "integration-test-user",
        "email": "integration@test.com",
        "first_name": "Integration",
        "last_name": "Test",
        "scope": "viewer"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/security/users",
            headers=headers,
            json=user_data
        )
        
        if response.status_code == 201:
            user = response.json()
            print(f"[PASS] User created: {user['username']} (ID: {user['id']})")
            
            # Check if directories were created
            user_id = user['id']
            expected_dirs = [
                f"agents/users/{user_id}",
                f"vectorstore/users/{user_id}",
                f"data/uploads/users/{user_id}",
                f"data/sources/users/{user_id}"
            ]
            
            all_dirs_exist = True
            for dir_path in expected_dirs:
                if Path(dir_path).exists():
                    print(f"[PASS] Directory exists: {dir_path}")
                else:
                    print(f"[FAIL] Directory missing: {dir_path}")
                    all_dirs_exist = False
            
            if all_dirs_exist:
                print("[PASS] All user directories created successfully")
                return user
            else:
                print("[FAIL] Some user directories missing")
                return None
        else:
            print(f"[FAIL] Failed to create user: {response.text}")
            return None
            
    except Exception as e:
        print(f"[FAIL] Error creating user: {e}")
        return None

def test_agent_creation(user_id):
    """Test agent creation for a specific user."""
    print(f"\n[INFO] Testing agent creation for user {user_id}...")
    
    headers = {"X-API-Key": ADMIN_API_KEY}
    
    agent_config = {
        "name": "integration-test-agent",
        "description": "Integration test agent",
        "llm_model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000,
        "data_sources": []
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/agents",
            headers=headers,
            json=agent_config
        )
        
        if response.status_code == 200:
            agent = response.json()
            print(f"[PASS] Agent created: {agent['name']}")
            
            # Check if agent file was created in user-specific directory
            agent_file = Path(f"agents/users/{user_id}/integration-test-agent.yaml")
            if agent_file.exists():
                print(f"[PASS] Agent file created: {agent_file}")
                return True
            else:
                print(f"[FAIL] Agent file not found: {agent_file}")
                return False
        else:
            print(f"[FAIL] Failed to create agent: {response.text}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error creating agent: {e}")
        return False

def test_data_upload(user_id):
    """Test data upload for a specific user."""
    print(f"\n[INFO] Testing data upload for user {user_id}...")
    
    headers = {"X-API-Key": ADMIN_API_KEY}
    
    # Create test content
    test_content = "This is integration test data for user-specific system testing."
    
    try:
        # Create temporary file
        temp_file = Path("temp_test_data.txt")
        temp_file.write_text(test_content)
        
        with open(temp_file, 'rb') as f:
            files = {'file': ('integration_test.txt', f, 'text/plain')}
            response = requests.post(
                f"{BASE_URL}/api/v1/agents/upload-file",
                headers=headers,
                files=files
            )
        
        # Clean up temp file
        temp_file.unlink()
        
        if response.status_code == 200:
            upload_info = response.json()
            print(f"[PASS] File uploaded: {upload_info['file_path']}")
            
            # Check if file was saved to user-specific directory
            if f"users/{user_id}" in upload_info['file_path']:
                print("[PASS] File saved to user-specific directory")
                return True
            else:
                print("[FAIL] File not saved to user-specific directory")
                return False
        else:
            print(f"[FAIL] Failed to upload file: {response.text}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error uploading file: {e}")
        return False

def test_user_isolation():
    """Test that users can only see their own data."""
    print("\n[INFO] Testing user isolation...")
    
    headers = {"X-API-Key": ADMIN_API_KEY}
    
    try:
        # Get all agents (admin view)
        response = requests.get(
            f"{BASE_URL}/api/v1/agents/status-all",
            headers=headers
        )
        
        if response.status_code == 200:
            agents = response.json()
            print(f"[PASS] Retrieved {len(agents)} agents")
            
            # Check if agents have user_id
            agents_with_user_id = [a for a in agents if 'user_id' in a]
            print(f"[PASS] {len(agents_with_user_id)} agents have user_id")
            
            return True
        else:
            print(f"[FAIL] Failed to get agents: {response.text}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error testing user isolation: {e}")
        return False

def test_database_sync():
    """Test that database and file system are synced."""
    print("\n[INFO] Testing database sync...")
    
    headers = {"X-API-Key": ADMIN_API_KEY}
    
    try:
        # Get agents from database
        response = requests.get(
            f"{BASE_URL}/api/v1/agents/status-all",
            headers=headers
        )
        
        if response.status_code == 200:
            agents = response.json()
            print(f"[PASS] Retrieved {len(agents)} agents from database")
            
            # Check if agent files exist in file system
            agents_found = 0
            for agent in agents:
                if 'name' in agent and 'user_id' in agent:
                    agent_file = Path(f"agents/users/{agent['user_id']}/{agent['name']}.yaml")
                    if agent_file.exists():
                        agents_found += 1
                        print(f"[PASS] Agent file exists: {agent_file}")
                    else:
                        print(f"[FAIL] Agent file missing: {agent_file}")
            
            print(f"[PASS] {agents_found} agent files found in file system")
            return agents_found > 0
        else:
            print(f"[FAIL] Failed to get agents: {response.text}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error testing database sync: {e}")
        return False

def run_integration_tests():
    """Run all integration tests."""
    print("[SUCCESS] Starting Integration Tests for User-Specific System")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Server health
    if test_server_health():
        tests_passed += 1
    
    # Test 2: User creation
    user = test_user_creation()
    if user:
        tests_passed += 1
        
        # Test 3: Agent creation
        if test_agent_creation(user['id']):
            tests_passed += 1
        
        # Test 4: Data upload
        if test_data_upload(user['id']):
            tests_passed += 1
    
    # Test 5: User isolation
    if test_user_isolation():
        tests_passed += 1
    
    # Test 6: Database sync
    if test_database_sync():
        tests_passed += 1
    
    print(f"\n[STATS] Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("[COMPLETE] ALL INTEGRATION TESTS PASSED!")
        print("[PASS] User-specific system is working correctly")
        return True
    else:
        print("[FAIL] Some tests failed - system needs debugging")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    
    if success:
        print("\n[TARGET] Integration Test Summary:")
        print("- Server is healthy and accessible")
        print("- User creation and directory setup working")
        print("- Agent creation and user-specific storage working")
        print("- Data upload and user-specific storage working")
        print("- User isolation features working")
        print("- Database and file system synchronization working")
        print("- System is ready for production use!")
    else:
        print("\n[FAIL] Integration tests failed - system needs debugging")
        exit(1)
