#!/usr/bin/env python3
"""
Simple GUI Verification Test
Tests the GUI functionality using existing admin user without creating new users.
"""

import requests
import time
import json

# Configuration
BASE_URL = "http://127.0.0.1:8000"
ADMIN_API_KEY = "master_key_123"

def test_simple_gui_verification():
    """Test basic GUI functionality using existing admin user."""
    
    print("Simple GUI Verification Test")
    print("=" * 60)
    
    headers = {
        "X-API-Key": ADMIN_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Test 1: Dashboard Access
    print("\n1. Testing Dashboard Access...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("   [PASS] Dashboard accessible")
            if "RAGnetic" in response.text:
                print("   [PASS] Dashboard contains RAGnetic branding")
            else:
                print("   [WARN] Dashboard branding not found")
        else:
            print(f"   [FAIL] Dashboard not accessible: {response.status_code}")
    except Exception as e:
        print(f"   [FAIL] Error accessing dashboard: {e}")
    
    # Test 2: API Documentation
    print("\n2. [DOCS] Testing API Documentation...")
    
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("   [PASS] API documentation accessible")
        else:
            print(f"   [FAIL] API documentation not accessible: {response.status_code}")
    except Exception as e:
        print(f"   [FAIL] Error accessing API docs: {e}")
    
    # Test 3: Agent List (Admin View)
    print("\n3. [AGENT] Testing Agent List (Admin View)...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/agents/status-all", headers=headers)
        if response.status_code == 200:
            agents = response.json()
            print(f"   [PASS] Retrieved {len(agents)} agents")
            
            # Check if agents have user_id field
            if agents:
                first_agent = agents[0]
                if 'user_id' in first_agent:
                    print("   [PASS] Agents include user_id field")
                else:
                    print("   [WARN] Agents missing user_id field")
                
                # Show some agent details
                print(f"   [INFO] Sample agent: {first_agent.get('name', 'Unknown')}")
                print(f"   [USER] User ID: {first_agent.get('user_id', 'None')}")
            else:
                print("   [WARN] No agents found")
        else:
            print(f"   [FAIL] Failed to retrieve agents: {response.status_code}")
    except Exception as e:
        print(f"   [FAIL] Error retrieving agents: {e}")
    
    # Test 4: User List (Admin View)
    print("\n4. [USERS] Testing User List (Admin View)...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/security/users", headers=headers)
        if response.status_code == 200:
            users = response.json()
            print(f"   [PASS] Retrieved {len(users)} users")
            
            # Show user details
            for user in users[:3]:  # Show first 3 users
                print(f"   [USER] User: {user.get('username', 'Unknown')} (ID: {user.get('id', 'Unknown')})")
        else:
            print(f"   [FAIL] Failed to retrieve users: {response.status_code}")
    except Exception as e:
        print(f"   [FAIL] Error retrieving users: {e}")
    
    # Test 5: Chat Session Creation
    print("\n5. [CHAT] Testing Chat Session Creation...")
    
    try:
        # Get first available agent
        response = requests.get(f"{BASE_URL}/api/v1/agents", headers=headers)
        if response.status_code == 200:
            agents = response.json()
            if agents:
                agent_name = agents[0]['name']
                print(f"   [INFO] Using agent: {agent_name}")
                
                # Create chat session
                session_data = {
                    "agent_name": agent_name,
                    "user_id": 1  # Admin user ID
                }
                
                session_response = requests.post(
                    f"{BASE_URL}/api/v1/sessions/create",
                    json=session_data,
                    headers=headers
                )
                
                if session_response.status_code in [200, 201]:
                    session_result = session_response.json()
                    print(f"   [PASS] Created chat session: {session_result.get('thread_id', 'Unknown')}")
                    
                    # Test temporary file upload
                    print("\n6. [FILE] Testing Temporary File Upload...")
                    
                    # Create a simple test file
                    test_content = "This is a test document for GUI verification."
                    files = {
                        'file': ('test_document.txt', test_content, 'text/plain')
                    }
                    
                    upload_data = {
                        'thread_id': session_result.get('thread_id'),
                        'file_type': 'document'
                    }
                    
                    upload_response = requests.post(
                        f"{BASE_URL}/api/v1/chat/upload-temp-document",
                        files=files,
                        data=upload_data,
                        headers={"X-API-Key": ADMIN_API_KEY}
                    )
                    
                    if upload_response.status_code in [200, 201]:
                        print("   [PASS] Temporary file upload successful")
                    else:
                        print(f"   [FAIL] Temporary file upload failed: {upload_response.status_code}")
                        print(f"   [RESPONSE] Response: {upload_response.text}")
                else:
                    print(f"   [FAIL] Failed to create chat session: {session_response.status_code}")
                    print(f"   [RESPONSE] Response: {session_response.text}")
            else:
                print("   [FAIL] No agents available for chat testing")
        else:
            print(f"   [FAIL] Failed to retrieve agents for chat testing: {response.status_code}")
    except Exception as e:
        print(f"   [FAIL] Error in chat session testing: {e}")
    
    # Test 7: File Upload (Regular)
    print("\n7. [UPLOAD] Testing Regular File Upload...")
    
    try:
        # Create a simple test file
        test_content = "This is a test document for regular file upload."
        files = {
            'file': ('test_regular_upload.txt', test_content, 'text/plain')
        }
        
        upload_response = requests.post(
            f"{BASE_URL}/api/v1/agents/upload-file",
            files=files,
            headers={"X-API-Key": ADMIN_API_KEY}
        )
        
        if upload_response.status_code in [200, 201]:
            print("   [PASS] Regular file upload successful")
            upload_result = upload_response.json()
            print(f"   [FILE] File path: {upload_result.get('file_path', 'Unknown')}")
        else:
            print(f"   [FAIL] Regular file upload failed: {upload_response.status_code}")
            print(f"   [RESPONSE] Response: {upload_response.text}")
    except Exception as e:
        print(f"   [FAIL] Error in regular file upload: {e}")
    
    print("\n" + "=" * 60)
    print("[COMPLETE] Simple GUI Verification Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_simple_gui_verification()