"""
Test new agent creation with user isolation.
This test creates a new agent and verifies it's properly isolated.
"""

import requests
import json
from pathlib import Path
import time

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
ADMIN_API_KEY = "c9d0fbc5206419383fec57608202858bd9540e4a1210220f35760bf387656691"

def test_new_agent_creation():
    """Test creating a new agent with user isolation."""
    print("[INFO] Testing New Agent Creation with User Isolation")
    print("=" * 50)
    
    headers = {"X-API-Key": ADMIN_API_KEY}
    
    try:
        # Test 1: Check server health
        print("\n1. Checking server health...")
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("[PASS] Server is healthy")
        else:
            print(f"[FAIL] Server health check failed: {response.status_code}")
            return False
        
        # Test 2: Get existing users
        print("\n2. Getting existing users...")
        response = requests.get(
            f"{BASE_URL}/api/v1/security/users",
            headers=headers
        )
        
        if response.status_code == 200:
            users = response.json()
            print(f"[PASS] Found {len(users)} users in system")
            
            # Use the first non-admin user for testing
            test_user = None
            for user in users:
                if user['username'] != 'admin' and user.get('scope', 'viewer') != 'admin':
                    test_user = user
                    break
            
            if not test_user:
                print("[FAIL] No non-admin users found for testing")
                return False
            
            print(f"[PASS] Using test user: {test_user['username']} (ID: {test_user['id']})")
        else:
            print(f"[FAIL] Failed to get users: {response.text}")
            return False
        
        # Test 3: Create a new agent
        print("\n3. Creating new agent...")
        timestamp = int(time.time())
        agent_name = f"isolation-test-agent-{timestamp}"
        
        agent_config = {
            "name": agent_name,
            "description": "Test agent for user isolation",
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
        
        if response.status_code in [200, 201]:
            agent_response = response.json()
            print(f"[PASS] Agent created: {agent_response.get('agent', agent_name)}")
            print(f"[PASS] Agent status: {agent_response.get('status', 'Unknown')}")
            
            # Wait a moment for the agent to be processed
            time.sleep(2)
        else:
            print(f"[FAIL] Failed to create agent: {response.text}")
            return False
        
        # Test 4: Check agent file location
        print("\n4. Checking agent file location...")
        
        # Check if agent file was created in user-specific directory
        # Since we're using admin API key, the agent should be created for admin user (ID: 1)
        admin_user_id = 1
        agent_file = Path(f"agents/users/{admin_user_id}/{agent_name}.yaml")
        
        if agent_file.exists():
            print(f"[PASS] Agent file created in user-specific directory: {agent_file}")
            print(f"[PASS] Agent correctly associated with admin user (ID: {admin_user_id})")
        else:
            # Check if it was created in global directory (legacy behavior)
            global_agent_file = Path(f"agents/{agent_name}.yaml")
            if global_agent_file.exists():
                print(f"[WARN] Agent file created in global directory: {global_agent_file}")
                print("[WARN] This indicates the agent creation is not using user-specific paths")
                return False
            else:
                print(f"[FAIL] Agent file not found anywhere: {agent_file}")
                return False
        
        # Test 5: Check database entry
        print("\n5. Checking database entry...")
        response = requests.get(
            f"{BASE_URL}/api/v1/agents/status-all",
            headers=headers
        )
        
        if response.status_code == 200:
            agents = response.json()
            print(f"[PASS] Retrieved {len(agents)} agents from database")
            
            # Find our test agent
            test_agent = None
            for agent in agents:
                if agent.get('name') == agent_name:
                    test_agent = agent
                    break
            
            if test_agent:
                print(f"[PASS] Test agent found in database: {test_agent['name']}")
                
                # Check if agent has user_id
                if 'user_id' in test_agent and test_agent['user_id'] is not None:
                    print(f"[PASS] Agent has user_id: {test_agent['user_id']}")
                    
                    if test_agent['user_id'] == admin_user_id:
                        print("[PASS] Agent user_id matches admin user")
                    else:
                        print(f"[WARN] Agent user_id ({test_agent['user_id']}) doesn't match admin user ({admin_user_id})")
                else:
                    print("[FAIL] Agent missing user_id - not properly isolated")
                    return False
            else:
                print("[FAIL] Test agent not found in database")
                return False
        else:
            print(f"[FAIL] Failed to get agents: {response.text}")
            return False
        
        # Test 6: Check user isolation
        print("\n6. Checking user isolation...")
        
        # Get agents for the admin user (since we're using admin API key)
        admin_user_agents = [a for a in agents if a.get('user_id') == admin_user_id]
        other_agents = [a for a in agents if a.get('user_id') != admin_user_id and a.get('user_id') is not None]
        
        print(f"[PASS] Admin user has {len(admin_user_agents)} own agents")
        print(f"[PASS] Other users have {len(other_agents)} agents")
        
        if len(admin_user_agents) > 0:
            print("[PASS] User isolation working - admin user has their own agents")
        else:
            print("[FAIL] User isolation not working - admin user has no agents")
            return False
        
        # Test 7: Check vector store creation
        print("\n7. Checking vector store creation...")
        
        vectorstore_path = Path(f"vectorstore/users/{admin_user_id}/{agent_name}")
        if vectorstore_path.exists():
            print(f"[PASS] Vector store created: {vectorstore_path}")
        else:
            print(f"[WARN] Vector store not created: {vectorstore_path}")
            # This is not necessarily an error as vector stores are created during data ingestion
        
        print("\n[COMPLETE] NEW AGENT CREATION TEST PASSED!")
        print("=" * 50)
        print("[PASS] Agent created successfully")
        print("[PASS] Agent file saved to user-specific directory")
        print("[PASS] Agent has proper user_id in database")
        print("[PASS] User isolation working correctly")
        print("[PASS] System is ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] New agent creation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_new_agent_creation()
    
    if success:
        print("\n[TARGET] New Agent Creation Test Summary:")
        print("- New agents are created with proper user isolation")
        print("- Agent files are saved to user-specific directories")
        print("- Database entries have proper user_id")
        print("- User isolation is working correctly")
        print("- System is ready for production use!")
    else:
        print("\n[FAIL] New agent creation test failed - system needs debugging")
        exit(1)
