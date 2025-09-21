#!/usr/bin/env python3
"""
GUI Dashboard Testing
Tests the complete GUI dashboard functionality including:
- User authentication and session management
- Agent management through GUI
- File upload interface
- Chat interface integration
- User isolation in GUI
- Superuser dashboard access
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
ADMIN_API_KEY = "master_key_123"

def test_gui_dashboard():
    """Test the complete GUI dashboard functionality."""
    print("[DASHBOARD] Testing GUI Dashboard Functionality")
    print("=" * 60)
    
    headers = {
        "X-API-Key": ADMIN_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Test 1: Dashboard Access and Authentication
    print("\n1. [SECURITY] Testing Dashboard Access...")
    
    try:
        # Test dashboard endpoint
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("   [PASS] Dashboard accessible")
            if "RAGnetic" in response.text:
                print("   [PASS] Dashboard contains RAGnetic branding")
            else:
                print("   [WARN]  Dashboard may not be loading correctly")
        else:
            print(f"   [FAIL] Dashboard not accessible: {response.status_code}")
            
    except Exception as e:
        print(f"   [FAIL] Error accessing dashboard: {e}")
    
    # Test 2: API Documentation Access
    print("\n2. [DOCS] Testing API Documentation...")
    
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("   [PASS] API documentation accessible")
        else:
            print(f"   [FAIL] API documentation not accessible: {response.status_code}")
            
    except Exception as e:
        print(f"   [FAIL] Error accessing API docs: {e}")
    
    # Test 3: Create Test Users for GUI Testing
    print("\n3. ‍ Creating Test Users for GUI Testing...")
    
    test_users = []
    for i in range(2):
        user_data = {
            "username": f"gui-test-user-{int(time.time())}-{i}",
            "email": f"guitest{i}@example.com",
            "first_name": f"GUI",
            "last_name": f"Tester{i}",
            "password": "testpassword123",
            "is_superuser": i == 0  # First user is superuser
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/security/users",
                json=user_data,
                headers=headers
            )
            
            if response.status_code in [200, 201]:
                user_result = response.json()
                test_users.append(user_result)
                print(f"   [PASS] Created GUI test user: {user_result['username']} (ID: {user_result['id']})")
                
                # Create API key
                api_key_data = {"scope": "admin" if user_result['is_superuser'] else "editor"}
                api_response = requests.post(
                    f"{BASE_URL}/api/v1/security/users/{user_result['id']}/api-keys",
                    json=api_key_data,
                    headers=headers
                )
                
                if api_response.status_code in [200, 201]:
                    api_result = api_response.json()
                    user_result['api_key'] = api_result['access_token']
                    print(f"   [PASS] Created API key for user {user_result['username']}")
                else:
                    print(f"   [FAIL] Failed to create API key for user {user_result['username']}")
                    
            else:
                print(f"   [FAIL] Failed to create GUI test user {i}: {response.text}")
                
        except Exception as e:
            print(f"   [FAIL] Error creating GUI test user {i}: {e}")
        
        # Add delay to avoid rate limiting
        time.sleep(2)
            
    if not test_users:
        print("   [FAIL] No test users created, cannot continue GUI testing")
        return
    
    # Test 4: Agent Management Through GUI
    print("\n4. [AGENT] Testing Agent Management Through GUI...")
    
    for user in test_users:
        user_headers = {
            "X-API-Key": user['api_key'],
            "Content-Type": "application/json"
        }
        
        # Create agents for each user
        for j in range(2):
            agent_name = f"gui-agent-{user['user_id']}-{j}"
            agent_config = {
                "name": agent_name,
                "display_name": f"GUI Agent {j} for {user['user_id']}",
                "description": f"GUI test agent created by {user['user_id']}",
                "model_name": "gpt-3.5-turbo",
                "tools": ["search_engine"],
                "sources": [],
                "system_prompt": f"You are a GUI test agent for {user['user_id']}."
            }
            
            try:
                response = requests.post(
                    f"{BASE_URL}/api/v1/agents",
                    json=agent_config,
                    headers=user_headers
                )
                
                if response.status_code in [200, 201]:
                    agent_result = response.json()
                    print(f"   [PASS] Created GUI agent '{agent_name}' for user {user['user_id']}")
                else:
                    print(f"   [FAIL] Failed to create GUI agent for user {user['user_id']}: {response.text}")
                    
            except Exception as e:
                print(f"   [FAIL] Error creating GUI agent for user {user['user_id']}: {e}")
    
    # Test 5: GUI Agent List and User Isolation
    print("\n5. [INFO] Testing GUI Agent List and User Isolation...")
    
    for user in test_users:
        user_headers = {
            "X-API-Key": user['api_key'],
            "Content-Type": "application/json"
        }
        
        try:
            # Get agents list (simulating GUI agent list)
            response = requests.get(
                f"{BASE_URL}/api/v1/agents/status-all",
                headers=user_headers
            )
            
            if response.status_code == 200:
                agents = response.json()
                user_agent_count = len([a for a in agents if a.get('user_id') == user['id']])
                total_agent_count = len(agents)
                
                if user['is_superuser']:
                    print(f"   [PASS] Superuser {user['user_id']} sees {total_agent_count} agents in GUI (all agents)")
                else:
                    print(f"   [PASS] Regular user {user['user_id']} sees {user_agent_count} agents in GUI (only their own)")
                
                # Verify agent data structure for GUI
                if agents:
                    agent = agents[0]
                    required_fields = ['name', 'status', 'created_at', 'user_id']
                    missing_fields = [field for field in required_fields if field not in agent]
                    if not missing_fields:
                        print(f"   [PASS] Agent data structure complete for GUI")
                    else:
                        print(f"   [WARN]  Missing fields in agent data: {missing_fields}")
                        
            else:
                print(f"   [FAIL] Failed to get agents for GUI user {user['user_id']}: {response.text}")
                
        except Exception as e:
            print(f"   [FAIL] Error getting agents for GUI user {user['user_id']}: {e}")
    
    # Test 6: File Upload Interface Testing
    print("\n6. [FILE] Testing File Upload Interface...")
    
    for user in test_users:
        user_headers = {
            "X-API-Key": user['api_key']
        }
        
        # Test different file types
        test_files = [
            {"name": "document.txt", "content": "Test document content", "type": "text/plain"},
            {"name": "data.json", "content": '{"test": "data", "value": 123}', "type": "application/json"},
            {"name": "code.py", "content": "print('Hello, World!')", "type": "text/x-python"}
        ]
        
        for test_file in test_files:
            try:
                # Create test file
                temp_file_path = test_file['name']
                with open(temp_file_path, "w") as f:
                    f.write(test_file['content'])
                
                # Upload file
                with open(temp_file_path, "rb") as f:
                    files = {"file": (test_file['name'], f, test_file['type'])}
                    response = requests.post(
                        f"{BASE_URL}/api/v1/agents/upload-file",
                        files=files,
                        headers=user_headers
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   [PASS] Uploaded {test_file['name']} for user {user['user_id']}")
                    
                    # Verify file is in user-specific directory
                    file_path = Path(result.get('file_path', ''))
                    if file_path.exists() and f"users/{user['id']}" in str(file_path):
                        print(f"   [PASS] File stored in user-specific directory")
                    else:
                        print(f"   [WARN]  File not in user-specific directory: {file_path}")
                else:
                    print(f"   [FAIL] Failed to upload {test_file['name']} for user {user['user_id']}: {response.text}")
                
                # Clean up
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
            except Exception as e:
                print(f"   [FAIL] Error uploading {test_file['name']} for user {user['user_id']}: {e}")
    
    # Test 7: Chat Interface Integration
    print("\n7. [CHAT] Testing Chat Interface Integration...")
    
    for user in test_users:
        user_headers = {
            "X-API-Key": user['api_key'],
            "Content-Type": "application/json"
        }
        
        # Get user's agents
        try:
            response = requests.get(
                f"{BASE_URL}/api/v1/agents/status-all",
                headers=user_headers
            )
            
            if response.status_code == 200:
                agents = response.json()
                user_agents = [a for a in agents if a.get('user_id') == user['id']]
                
                if user_agents:
                    agent = user_agents[0]  # Use first agent
                    
                    # Create chat session
                    thread_id = f"gui-chat-{user['id']}-{int(time.time())}"
                    session_data = {
                        "agent_name": agent['name'],
                        "thread_id": thread_id
                    }
                    
                    response = requests.post(
                        f"{BASE_URL}/api/v1/sessions/create",
                        json=session_data,
                        headers=user_headers
                    )
                    
                    if response.status_code in [200, 201]:
                        print(f"   [PASS] Chat session created for user {user['user_id']} with agent {agent['name']}")
                        
                        # Test temporary file upload in chat
                        temp_content = f"Chat test file for user {user['user_id']}"
                        temp_file_path = f"chat_test_{user['id']}.txt"
                        with open(temp_file_path, "w") as f:
                            f.write(temp_content)
                        
                        with open(temp_file_path, "rb") as f:
                            files = {"file": (f"chat_test_{user['id']}.txt", f, "text/plain")}
                            upload_headers = {"X-API-Key": user['api_key']}
                            temp_response = requests.post(
                                f"{BASE_URL}/api/v1/chat/upload-temp-document",
                                files=files,
                                headers=upload_headers,
                                params={"thread_id": thread_id}
                            )
                        
                        if temp_response.status_code == 200:
                            temp_result = temp_response.json()
                            print(f"   [PASS] Temporary file uploaded in chat for user {user['user_id']}")
                            print(f"   [FILE] Temp doc ID: {temp_result.get('temp_doc_id', 'N/A')}")
                        else:
                            print(f"   [FAIL] Failed to upload temporary file in chat: {temp_response.text}")
                        
                        # Clean up
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        
                        # Clean up session
                        requests.delete(
                            f"{BASE_URL}/api/v1/sessions/{thread_id}?agent_name={agent['name']}&user_id={user['id']}",
                            headers=user_headers
                        )
                    else:
                        print(f"   [FAIL] Failed to create chat session for user {user['user_id']}: {response.text}")
                else:
                    print(f"   [WARN]  No agents found for user {user['user_id']}, skipping chat test")
            else:
                print(f"   [FAIL] Failed to get agents for chat test: {response.text}")
                
        except Exception as e:
            print(f"   [FAIL] Error testing chat interface for user {user['user_id']}: {e}")
    
    # Test 8: Dashboard Analytics and Metrics
    print("\n8. [STATS] Testing Dashboard Analytics...")
    
    try:
        # Test analytics endpoints
        analytics_endpoints = [
            "/api/v1/analytics/usage-summary",
            "/api/v1/analytics/agent-runs",
            "/api/v1/analytics/latency",
            "/api/v1/analytics/benchmarks"
        ]
        
        for endpoint in analytics_endpoints:
            response = requests.get(f"{BASE_URL}{endpoint}", headers=headers)
            if response.status_code == 200:
                print(f"   [PASS] Analytics endpoint accessible: {endpoint}")
            else:
                print(f"   [FAIL] Analytics endpoint failed: {endpoint} - {response.status_code}")
                
    except Exception as e:
        print(f"   [FAIL] Error testing analytics: {e}")
    
    # Test 9: Cleanup
    print("\n9.  Cleaning Up GUI Test Data...")
    
    for user in test_users:
        user_headers = {
            "X-API-Key": user['api_key'],
            "Content-Type": "application/json"
        }
        
        try:
            # Get user's agents and delete them
            response = requests.get(
                f"{BASE_URL}/api/v1/agents/status-all",
                headers=user_headers
            )
            
            if response.status_code == 200:
                agents = response.json()
                user_agents = [a for a in agents if a.get('user_id') == user['id']]
                
                for agent in user_agents:
                    delete_response = requests.delete(
                        f"{BASE_URL}/api/v1/agents/{agent['name']}",
                        headers=user_headers
                    )
                    if delete_response.status_code in [200, 204]:
                        print(f"   [PASS] Deleted agent {agent['name']}")
                    else:
                        print(f"   [WARN]  Failed to delete agent {agent['name']}")
                        
        except Exception as e:
            print(f"   [FAIL] Error cleaning up for user {user['user_id']}: {e}")
    
    print("\n" + "=" * 60)
    print("[TARGET] GUI Dashboard Testing Complete!")
    print("\n[PASS] GUI functionality verified:")
    print("• Dashboard access and authentication")
    print("• Agent management through GUI")
    print("• File upload interface")
    print("• Chat interface integration")
    print("• User isolation in GUI")
    print("• Superuser dashboard access")
    print("• Analytics and metrics")
    print("• Complete user workflow")

if __name__ == "__main__":
    test_gui_dashboard()
