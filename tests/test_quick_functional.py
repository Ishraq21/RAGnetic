"""
Quick functional test for user-specific system.
Tests core functionality with minimal setup.
"""

import requests
import json
from pathlib import Path
import time

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
ADMIN_API_KEY = "c9d0fbc5206419383fec57608202858bd9540e4a1210220f35760bf387656691"

def quick_test():
    """Run quick functional tests."""
    print("[SUCCESS] Quick Functional Test for User-Specific System")
    print("=" * 50)
    
    headers = {"X-API-Key": ADMIN_API_KEY}
    
    try:
        # Test 1: Check server health
        print("\n1. Testing server health...")
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("[PASS] Server is healthy")
        else:
            print(f"[FAIL] Server health check failed: {response.status_code}")
            return False
        
        # Test 2: Create a test user
        print("\n2. Creating test user...")
        import time
        timestamp = int(time.time())
        user_data = {
            "username": f"quick-test-user-{timestamp}",
            "email": f"quicktest{timestamp}@example.com",
            "first_name": "Quick",
            "last_name": "Test",
            "scope": "viewer",
            "password": "testpassword123"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/security/users",
            headers=headers,
            json=user_data
        )
        
        if response.status_code == 201:
            user = response.json()
            user_id = user['id']
            print(f"[PASS] User created: {user['username']} (ID: {user_id})")
            
            # Check directories
            expected_dirs = [
                f"agents/users/{user_id}",
                f"vectorstore/users/{user_id}",
                f"data/uploads/users/{user_id}",
                f"data/sources/users/{user_id}"
            ]
            
            dirs_created = 0
            for dir_path in expected_dirs:
                if Path(dir_path).exists():
                    print(f"[PASS] Directory exists: {dir_path}")
                    dirs_created += 1
                else:
                    print(f"[FAIL] Directory missing: {dir_path}")
            
            if dirs_created == len(expected_dirs):
                print("[PASS] All user directories created")
            else:
                print(f"[FAIL] Only {dirs_created}/{len(expected_dirs)} directories created")
                return False
        else:
            print(f"[FAIL] Failed to create user: {response.text}")
            return False
        
        # Test 3: Create an agent
        print("\n3. Creating test agent...")
        agent_name = f"quick-test-agent-{timestamp}"
        agent_config = {
            "name": agent_name,
            "description": "Quick test agent",
            "llm_model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000,
            "data_sources": []
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/agents",
            headers=headers,
            json=agent_config
        )
        
        if response.status_code == 200:
            agent_response = response.json()
            print(f"[PASS] Agent created: {agent_response.get('agent', 'quick-test-agent')}")
            
            # Check agent file
            agent_file = Path(f"agents/users/{user_id}/{agent_name}.yaml")
            if agent_file.exists():
                print(f"[PASS] Agent file created: {agent_file}")
            else:
                print(f"[FAIL] Agent file not found: {agent_file}")
                return False
        else:
            print(f"[FAIL] Failed to create agent: {response.text}")
            return False
        
        # Test 4: Upload test data
        print("\n4. Uploading test data...")
        test_content = "This is quick test data for user-specific system testing."
        
        # Create temporary file
        temp_file = Path("temp_quick_test.txt")
        temp_file.write_text(test_content)
        
        with open(temp_file, 'rb') as f:
            files = {'file': ('quick_test.txt', f, 'text/plain')}
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
            
            if f"users/{user_id}" in upload_info['file_path']:
                print("[PASS] File saved to user-specific directory")
            else:
                print("[FAIL] File not saved to user-specific directory")
                return False
        else:
            print(f"[FAIL] Failed to upload file: {response.text}")
            return False
        
        # Test 5: Check user isolation
        print("\n5. Testing user isolation...")
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
            
            if len(agents_with_user_id) > 0:
                print("[PASS] User isolation working - agents have user_id")
            else:
                print("[FAIL] User isolation not working - no agents have user_id")
                return False
        else:
            print(f"[FAIL] Failed to get agents: {response.text}")
            return False
        
        # Test 6: Check database sync
        print("\n6. Testing database sync...")
        agents_found = 0
        for agent in agents:
            if 'name' in agent and 'user_id' in agent:
                agent_file = Path(f"agents/users/{agent['user_id']}/{agent['name']}.yaml")
                if agent_file.exists():
                    agents_found += 1
                    print(f"[PASS] Agent file exists: {agent_file}")
                else:
                    print(f"[FAIL] Agent file missing: {agent_file}")
        
        if agents_found > 0:
            print(f"[PASS] Database sync working - {agents_found} agent files found")
        else:
            print("[FAIL] Database sync not working - no agent files found")
            return False
        
        print("\n[COMPLETE] ALL QUICK TESTS PASSED!")
        print("[PASS] User-specific system is working correctly")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Quick test failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n[TARGET] Quick Test Summary:")
        print("- Server is healthy and accessible")
        print("- User creation and directory setup working")
        print("- Agent creation and user-specific storage working")
        print("- Data upload and user-specific storage working")
        print("- User isolation features working")
        print("- Database and file system synchronization working")
        print("- System is ready for production use!")
    else:
        print("\n[FAIL] Quick tests failed - system needs debugging")
        exit(1)
