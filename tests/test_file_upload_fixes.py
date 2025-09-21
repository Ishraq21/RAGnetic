#!/usr/bin/env python3
"""
Test file upload fixes for user-specific system.
Tests regular file upload, training dataset upload, and temporary file upload.
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
ADMIN_API_KEY = "master_key_123"  # Master API key

def test_file_upload_fixes():
    """Test all file upload endpoints to ensure they work correctly."""
    print("[TEST] Testing File Upload Fixes")
    print("=" * 50)
    
    headers = {
        "X-API-Key": ADMIN_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Test 1: Regular file upload
    print("\n1. Testing Regular File Upload...")
    try:
        # Create a test file
        test_file_content = "This is a test file for upload testing."
        test_file_path = "test_upload.txt"
        with open(test_file_path, "w") as f:
            f.write(test_file_content)
        
        # Upload the file
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_upload.txt", f, "text/plain")}
            upload_headers = {"X-API-Key": ADMIN_API_KEY}
            response = requests.post(f"{BASE_URL}/api/v1/agents/upload-file", files=files, headers=upload_headers)
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   [PASS] File uploaded successfully")
            print(f"   [FILE] File path: {result.get('file_path', 'N/A')}")
            
            # Verify file exists in user-specific directory
            file_path = Path(result.get('file_path', ''))
            if file_path.exists():
                print(f"   [PASS] File exists at: {file_path}")
                # Check if it's in user-specific directory
                if "users/1" in str(file_path):
                    print(f"   [PASS] File is in user-specific directory")
                else:
                    print(f"   [WARN]  File is not in user-specific directory: {file_path}")
            else:
                print(f"   [FAIL] File not found at: {file_path}")
        else:
            print(f"   [FAIL] Upload failed: {response.text}")
        
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
    
    # Test 2: Training dataset upload
    print("\n2. Testing Training Dataset Upload...")
    try:
        # Create a test JSONL file
        test_data = [
            {"text": "This is a test document for training.", "label": "test"},
            {"text": "Another test document.", "label": "test"}
        ]
        test_jsonl_path = "test_training.jsonl"
        with open(test_jsonl_path, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
        
        # Upload the training dataset
        with open(test_jsonl_path, "rb") as f:
            files = {"file": ("test_training.jsonl", f, "application/json")}
            upload_headers = {"X-API-Key": ADMIN_API_KEY}
            response = requests.post(f"{BASE_URL}/api/v1/training/upload-dataset", files=files, headers=upload_headers)
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   [PASS] Training dataset uploaded successfully")
            print(f"   [FILE] File path: {result.get('file_path', 'N/A')}")
            
            # Verify file exists in user-specific directory
            file_path = Path(result.get('file_path', ''))
            if file_path.exists():
                print(f"   [PASS] File exists at: {file_path}")
                # Check if it's in user-specific directory
                if "users/1" in str(file_path):
                    print(f"   [PASS] File is in user-specific directory")
                else:
                    print(f"   [WARN]  File is not in user-specific directory: {file_path}")
            else:
                print(f"   [FAIL] File not found at: {file_path}")
        else:
            print(f"   [FAIL] Upload failed: {response.text}")
        
        # Clean up test file
        if os.path.exists(test_jsonl_path):
            os.remove(test_jsonl_path)
            
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
    
    # Test 3: Create a test agent for temporary file upload
    print("\n3. Testing Temporary File Upload...")
    try:
        # First, create a test agent
        agent_name = f"test-agent-{int(time.time())}"
        agent_config = {
            "name": agent_name,
            "display_name": "Test Agent for Upload",
            "description": "Test agent for file upload testing",
            "model_name": "gpt-3.5-turbo",
            "tools": [],
            "sources": [],
            "system_prompt": "You are a test agent."
        }
        
        create_response = requests.post(
            f"{BASE_URL}/api/v1/agents",
            json=agent_config,
            headers=headers
        )
        
        if create_response.status_code in [200, 201]:
            print(f"   [PASS] Test agent created: {agent_name}")
            
            # Create a chat session
            session_data = {
                "agent_name": agent_name,
                "thread_id": f"test-thread-{int(time.time())}"
            }
            
            session_response = requests.post(
                f"{BASE_URL}/api/v1/sessions",
                json=session_data,
                headers=headers
            )
            
            if session_response.status_code in [200, 201]:
                session_result = session_response.json()
                thread_id = session_result.get('thread_id', session_data['thread_id'])
                print(f"   [PASS] Chat session created: {thread_id}")
                
                # Test temporary file upload
                test_temp_content = "This is a test temporary file."
                test_temp_path = "test_temp_upload.txt"
                with open(test_temp_path, "w") as f:
                    f.write(test_temp_content)
                
                with open(test_temp_path, "rb") as f:
                    files = {"file": ("test_temp_upload.txt", f, "text/plain")}
                    upload_headers = {"X-API-Key": ADMIN_API_KEY}
                    temp_response = requests.post(
                        f"{BASE_URL}/api/v1/upload-temp-doc",
                        files=files,
                        headers=upload_headers,
                        params={"thread_id": thread_id}
                    )
                
                print(f"   Status: {temp_response.status_code}")
                if temp_response.status_code == 200:
                    result = temp_response.json()
                    print(f"   [PASS] Temporary file uploaded successfully")
                    print(f"   [FILE] File name: {result.get('file_name', 'N/A')}")
                    print(f"   [FILE] File size: {result.get('file_size', 'N/A')}")
                    print(f"   [FILE] Temp doc ID: {result.get('temp_doc_id', 'N/A')}")
                else:
                    print(f"   [FAIL] Temporary upload failed: {temp_response.text}")
                
                # Clean up test file
                if os.path.exists(test_temp_path):
                    os.remove(test_temp_path)
                
                # Clean up session
                requests.delete(f"{BASE_URL}/api/v1/sessions/{thread_id}?agent_name={agent_name}&user_id=1", headers=headers)
                
            else:
                print(f"   [FAIL] Failed to create session: {session_response.text}")
            
            # Clean up agent
            requests.delete(f"{BASE_URL}/api/v1/agents/{agent_name}", headers=headers)
            
        else:
            print(f"   [FAIL] Failed to create test agent: {create_response.text}")
            
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
    
    print("\n" + "=" * 50)
    print("[TARGET] File Upload Fixes Test Complete!")

if __name__ == "__main__":
    test_file_upload_fixes()
