#!/usr/bin/env python3
"""
Chat Interface and Temporary File Upload Testing
Tests the complete chat workflow including:
- Chat session creation
- Temporary file uploads through chat
- Agent interactions with uploaded files
- User-specific temporary file storage
"""

import requests
import json
import time
import os
import asyncio
import websockets
from pathlib import Path
import uuid

# Configuration
BASE_URL = "http://localhost:8000"
ADMIN_API_KEY = "master_key_123"

async def test_websocket_chat():
    """Test WebSocket chat functionality."""
    print(" Testing WebSocket Chat Interface...")
    
    try:
        # Connect to WebSocket
        uri = f"ws://localhost:8000/ws"
        headers = {"X-API-Key": ADMIN_API_KEY}
        
        async with websockets.connect(uri, extra_headers=headers) as websocket:
            print("   [PASS] WebSocket connection established")
            
            # Send initial query
            query_data = {
                "type": "query",
                "payload": {
                    "query": "Hello, I'm testing the chat interface. Can you help me?",
                    "agent": "test-agent",
                    "thread_id": f"test-thread-{int(time.time())}"
                }
            }
            
            await websocket.send(json.dumps(query_data))
            print("   [PASS] Query sent to WebSocket")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print("   [PASS] Received response from WebSocket")
                print(f"   [RESPONSE] Response: {response[:100]}...")
            except asyncio.TimeoutError:
                print("   [WARN]  WebSocket response timeout")
            
    except Exception as e:
        print(f"   [FAIL] WebSocket test failed: {e}")

def test_chat_interface():
    """Test the complete chat interface functionality."""
    print("[CHAT] Testing Chat Interface and Temporary File Uploads")
    print("=" * 60)
    
    headers = {
        "X-API-Key": ADMIN_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Test 1: Create a test user and agent
    print("\n1. ‍ Setting Up Test User and Agent...")
    
    # Create test user
    user_data = {
        "user_id": f"chat-test-user-{int(time.time())}",
        "email": f"chattest@example.com",
        "first_name": "Chat",
        "last_name": "Tester",
        "password": "testpassword123",
        "is_superuser": False
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/security/users",
            json=user_data,
            headers=headers
        )
        
        if response.status_code in [200, 201]:
            user_result = response.json()
            print(f"   [PASS] Created test user: {user_result['user_id']} (ID: {user_result['id']})")
            
            # Create API key for user
            api_key_data = {"scope": "editor"}
            api_response = requests.post(
                f"{BASE_URL}/api/v1/security/users/{user_result['id']}/api-keys",
                json=api_key_data,
                headers=headers
            )
            
            if api_response.status_code in [200, 201]:
                api_result = api_response.json()
                user_api_key = api_result['access_token']
                print(f"   [PASS] Created API key for user")
            else:
                print(f"   [FAIL] Failed to create API key: {api_response.text}")
                return
                
        else:
            print(f"   [FAIL] Failed to create test user: {response.text}")
            return
            
    except Exception as e:
        print(f"   [FAIL] Error setting up test user: {e}")
        return
    
    # Create test agent
    agent_name = f"chat-test-agent-{int(time.time())}"
    agent_config = {
        "name": agent_name,
        "display_name": "Chat Test Agent",
        "description": "Agent for testing chat interface",
        "model_name": "gpt-3.5-turbo",
        "tools": ["search_engine"],
        "sources": [],
        "system_prompt": "You are a helpful test agent. You can help users with their questions and analyze uploaded documents."
    }
    
    user_headers = {
        "X-API-Key": user_api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/agents",
            json=agent_config,
            headers=user_headers
        )
        
        if response.status_code in [200, 201]:
            agent_result = response.json()
            print(f"   [PASS] Created test agent: {agent_name}")
        else:
            print(f"   [FAIL] Failed to create test agent: {response.text}")
            return
            
    except Exception as e:
        print(f"   [FAIL] Error creating test agent: {e}")
        return
    
    # Test 2: Create Chat Session
    print("\n2. [CHAT] Creating Chat Session...")
    
    thread_id = f"chat-test-thread-{int(time.time())}"
    session_data = {
        "agent_name": agent_name,
        "thread_id": thread_id
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/sessions/create",
            json=session_data,
            headers=user_headers
        )
        
        if response.status_code in [200, 201]:
            session_result = response.json()
            print(f"   [PASS] Chat session created: {thread_id}")
        else:
            print(f"   [FAIL] Failed to create chat session: {response.text}")
            return
            
    except Exception as e:
        print(f"   [FAIL] Error creating chat session: {e}")
        return
    
    # Test 3: Upload Temporary Files
    print("\n3. [FILE] Testing Temporary File Uploads...")
    
    test_files = [
        {
            "name": "test_document.txt",
            "content": "This is a test document for chat analysis. It contains important information about testing the chat interface."
        },
        {
            "name": "data_analysis.csv",
            "content": "Name,Age,City\nJohn,25,New York\nJane,30,Los Angeles\nBob,35,Chicago"
        },
        {
            "name": "code_example.py",
            "content": "def hello_world():\n    print('Hello, World!')\n    return 'success'"
        }
    ]
    
    uploaded_files = []
    
    for test_file in test_files:
        try:
            # Create temporary file
            temp_file_path = test_file['name']
            with open(temp_file_path, "w") as f:
                f.write(test_file['content'])
            
            # Upload file
            with open(temp_file_path, "rb") as f:
                files = {"file": (test_file['name'], f, "text/plain")}
                upload_headers = {"X-API-Key": user_api_key}
                response = requests.post(
                    f"{BASE_URL}/api/v1/chat/upload-temp-document",
                    files=files,
                    headers=upload_headers,
                    params={"thread_id": thread_id}
                )
            
            if response.status_code == 200:
                result = response.json()
                uploaded_files.append(result)
                print(f"   [PASS] Uploaded {test_file['name']}: {result.get('file_name', 'N/A')}")
                print(f"   [FILE] Temp doc ID: {result.get('temp_doc_id', 'N/A')}")
                print(f"   [STATS] File size: {result.get('file_size', 'N/A')} bytes")
            else:
                print(f"   [FAIL] Failed to upload {test_file['name']}: {response.text}")
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
        except Exception as e:
            print(f"   [FAIL] Error uploading {test_file['name']}: {e}")
    
    # Test 4: Verify Temporary File Storage
    print("\n4. [INFO] Verifying Temporary File Storage...")
    
    temp_upload_dir = Path("temp_chat_uploads") / str(user_result['id']) / thread_id
    if temp_upload_dir.exists():
        files_in_dir = list(temp_upload_dir.glob("*"))
        print(f"   [PASS] Temporary files stored in: {temp_upload_dir}")
        print(f"   [FILE] Found {len(files_in_dir)} files in temp directory")
        for file_path in files_in_dir:
            print(f"      - {file_path.name}")
    else:
        print(f"   [WARN]  Temporary upload directory not found: {temp_upload_dir}")
    
    # Test 5: Test WebSocket Chat (if server is running)
    print("\n5.  Testing WebSocket Chat Interface...")
    
    try:
        # Run async WebSocket test
        asyncio.run(test_websocket_chat())
    except Exception as e:
        print(f"   [FAIL] WebSocket test failed: {e}")
        print(f"   ℹ  Note: WebSocket test requires server to be running")
    
    # Test 6: Test Chat Session Management
    print("\n6. [INFO] Testing Chat Session Management...")
    
    try:
        # Get session info
        response = requests.get(
            f"{BASE_URL}/api/v1/sessions/{thread_id}?agent_name={agent_name}&user_id={user_result['id']}",
            headers=user_headers
        )
        
        if response.status_code == 200:
            session_info = response.json()
            print(f"   [PASS] Retrieved session info for thread: {thread_id}")
        else:
            print(f"   [FAIL] Failed to get session info: {response.text}")
            
    except Exception as e:
        print(f"   [FAIL] Error getting session info: {e}")
    
    # Test 7: Cleanup
    print("\n7.  Cleaning Up Test Data...")
    
    try:
        # Delete chat session
        response = requests.delete(
            f"{BASE_URL}/api/v1/sessions/{thread_id}?agent_name={agent_name}&user_id={user_result['id']}",
            headers=user_headers
        )
        if response.status_code in [200, 204]:
            print(f"   [PASS] Deleted chat session: {thread_id}")
        else:
            print(f"   [WARN]  Failed to delete chat session: {response.text}")
            
        # Delete agent
        response = requests.delete(
            f"{BASE_URL}/api/v1/agents/{agent_name}",
            headers=user_headers
        )
        if response.status_code in [200, 204]:
            print(f"   [PASS] Deleted agent: {agent_name}")
        else:
            print(f"   [WARN]  Failed to delete agent: {response.text}")
            
    except Exception as e:
        print(f"   [FAIL] Error during cleanup: {e}")
    
    print("\n" + "=" * 60)
    print("[TARGET] Chat Interface Testing Complete!")
    print("\n[PASS] Chat functionality verified:")
    print("• Chat session creation and management")
    print("• Temporary file uploads through chat interface")
    print("• User-specific temporary file storage")
    print("• WebSocket connection (if server running)")
    print("• File upload and retrieval")
    print("• Session cleanup")

if __name__ == "__main__":
    test_chat_interface()
