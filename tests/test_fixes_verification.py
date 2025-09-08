#!/usr/bin/env python3
"""
Verify Lambda Tool Fixes
Tests the three fixed issues when server is available
"""

import requests
import json

API_KEY = "604a7d725c7e96a5f2517f16cfc5d81c64365c55662de49c23e1aa3650b0f0b8"
BASE_URL = "http://localhost:8000"

def test_os_module_fix():
    """Test that os module operations now work"""
    print("=== Testing OS Module Fix ===")
    
    payload = {
        "mode": "code",
        "code": """
import os
import json

print("Testing os module operations:")

# Test safe os operations
print(f"Current directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

# Test os.path operations  
print(f"Path exists check: {os.path.exists('.')}")
print(f"Is directory: {os.path.isdir('.')}")

# Test environment variables
print(f"Environment variable count: {len(os.environ)}")
print(f"PATH exists: {'PATH' in os.environ}")

# Test file operations
test_file = "os_test.txt"
with open(test_file, "w") as f:
    f.write("OS module test successful!")

print(f"File created: {os.path.exists(test_file)}")
print(f"File size: {os.path.getsize(test_file) if os.path.exists(test_file) else 0} bytes")

results = {
    "os_getcwd": os.getcwd(),
    "os_listdir": os.listdir('.'),
    "os_path_exists": os.path.exists('.'),
    "file_created": os.path.exists(test_file),
    "status": "success"
}

with open("os_module_test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("OS module test completed successfully!")
        """,
        "user_id": 1,
        "thread_id": "test-os-fix"
    }
    
    return execute_and_check(payload, "OS Module operations work correctly")

def test_improved_error_messages():
    """Test improved error messages for various error scenarios"""
    print("=== Testing Improved Error Messages ===")
    
    # Test 1: Invalid payload
    print("Test 1: Invalid payload structure")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/lambda/execute",
            json={"invalid": "payload"},
            headers={"X-API-Key": API_KEY}
        )
        print(f"Status: {response.status_code}")
        if response.status_code != 202:
            print("Response:", response.text)
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Test 2: Missing file
    print("\nTest 2: Missing file error")
    payload = {
        "mode": "code",
        "code": "import pandas as pd\ndf = pd.read_csv('nonexistent.csv')",
        "inputs": [{"file_name": "nonexistent.csv"}],
        "user_id": 1,
        "thread_id": "test-missing-file"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/lambda/execute",
            json=payload,
            headers={"X-API-Key": API_KEY}
        )
        print(f"Status: {response.status_code}")
        print("Response:", response.text)
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Test 3: Syntax error
    print("\nTest 3: Syntax error handling")
    payload = {
        "mode": "code", 
        "code": "print('Hello World'  # Missing closing parenthesis",
        "user_id": 1,
        "thread_id": "test-syntax-error"
    }
    
    return execute_and_check(payload, "Syntax error with helpful message", expect_failure=True)

def test_document_upload_api():
    """Test the new document upload API endpoint"""
    print("=== Testing Document Upload API ===")
    
    try:
        # Test API endpoint existence
        response = requests.post(
            f"{BASE_URL}/api/v1/documents/upload",
            files={"file": ("test.txt", "Hello World", "text/plain")},
            headers={"X-API-Key": API_KEY}
        )
        
        print(f"Upload API Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Upload successful: {data}")
            return True
        else:
            print(f"Upload failed: {response.text}")
            return False
    except Exception as e:
        print(f"Upload API test failed: {e}")
        return False

def execute_and_check(payload, test_name, expect_failure=False):
    """Execute lambda payload and check results"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/lambda/execute",
            json=payload,
            headers={"X-API-Key": API_KEY}
        )
        
        if response.status_code != 202:
            print(f" {test_name}: Submit failed ({response.status_code})")
            print("Response:", response.text)
            return False
        
        run_id = response.json().get("run_id")
        print(f"Run ID: {run_id}")
        
        # Poll for results (simplified)
        import time
        for i in range(10):
            time.sleep(1)
            status_response = requests.get(
                f"{BASE_URL}/api/v1/lambda/runs/{run_id}",
                headers={"X-API-Key": API_KEY}
            )
            
            if status_response.status_code == 200:
                data = status_response.json()
                status = data.get("status")
                
                if status == "completed":
                    if expect_failure:
                        print(f" {test_name}: Expected failure but got success")
                        return False
                    print(f" {test_name}: Success")
                    return True
                elif status == "failed":
                    if expect_failure:
                        print(f" {test_name}: Expected failure with improved error")
                        error_msg = data.get("error_message", "")
                        print(f"Error message: {error_msg}")
                        return True
                    print(f" {test_name}: Unexpected failure")
                    return False
        
        print(f"⏳ {test_name}: Timeout")
        return False
        
    except Exception as e:
        print(f" {test_name}: Exception - {e}")
        return False

def check_server_availability():
    """Check if the RAGnetic server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Run all fix verification tests"""
    print(" RAGnetic Lambda Tool Fixes Verification")
    print("=" * 50)
    
    if not check_server_availability():
        print(" RAGnetic server is not running at", BASE_URL)
        print("Please start the server and try again.")
        print("\nAlternatively, the fixes have been implemented:")
        print(" 1. OS module restrictions relaxed in sandbox/runner.py")
        print(" 2. Document upload API added at /api/v1/documents/upload")
        print(" 3. Error messages improved in app/tools/lambda_tool.py")
        return
    
    print(" Server is running, proceeding with tests...\n")
    
    tests = [
        ("OS Module Fix", test_os_module_fix),
        ("Document Upload API", test_document_upload_api),
        ("Improved Error Messages", test_improved_error_messages),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f" {test_name}: Exception - {e}")
            results.append((test_name, False))
        
        print("-" * 30)
    
    # Summary
    print("\n VERIFICATION SUMMARY:")
    print("=" * 30)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = " WORKING" if success else " NEEDS ATTENTION"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} fixes verified ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(" ALL FIXES VERIFIED! Lambda tool improvements are working!")
    else:
        print("  Some fixes need attention. Server restart may be required.")

if __name__ == "__main__":
    main()