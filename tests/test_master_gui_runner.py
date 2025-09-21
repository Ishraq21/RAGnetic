#!/usr/bin/env python3
"""
Master GUI Test Runner
Executes all GUI and frontend tests to verify the complete user-specific system.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_test(test_file, test_name):
    """Run a specific test file and return the result."""
    print(f"\n{'='*60}")
    print(f"[TEST] Running {test_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"[PASS] {test_name} PASSED")
            return True
        else:
            print(f"[FAIL] {test_name} FAILED")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} TIMED OUT")
        return False
    except Exception as e:
        print(f" {test_name} ERROR: {e}")
        return False

def check_server_status():
    """Check if the server is running."""
    import requests
    
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("[PASS] Server is running and accessible")
            return True
        else:
            print(f"[FAIL] Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Server is not accessible: {e}")
        return False

def main():
    """Run all GUI tests."""
    print("[SUCCESS] Master GUI Test Runner")
    print("=" * 60)
    print("This will run comprehensive tests of the RAGnetic GUI and frontend")
    print("to verify the complete user-specific system functionality.")
    
    # Check if server is running
    print("\n1. [INFO] Checking Server Status...")
    if not check_server_status():
        print("\n[FAIL] Server is not running. Please start the server first:")
        print("   cd /Users/ishraq21/ragnetic")
        print("   RAGNETIC_API_KEYS='master_key_123' python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return
    
    # Define tests to run
    tests = [
        ("tests/test_gui_end_to_end.py", "Complete GUI End-to-End Test"),
        ("tests/test_chat_interface.py", "Chat Interface and Temporary File Upload Test"),
        ("tests/test_gui_dashboard.py", "GUI Dashboard Functionality Test"),
        ("tests/test_file_upload_summary.py", "File Upload Fixes Summary")
    ]
    
    # Run tests
    results = []
    start_time = time.time()
    
    for test_file, test_name in tests:
        if Path(test_file).exists():
            success = run_test(test_file, test_name)
            results.append((test_name, success))
        else:
            print(f"[WARN]  Test file not found: {test_file}")
            results.append((test_name, False))
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("[STATS] TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Duration: {duration:.2f} seconds")
    
    print(f"\nDetailed Results:")
    for test_name, success in results:
        status = "[PASS] PASSED" if success else "[FAIL] FAILED"
        print(f"  {status} - {test_name}")
    
    if passed == total:
        print(f"\n[COMPLETE] ALL TESTS PASSED! The GUI and frontend system is working correctly.")
        print(f"[PASS] User-specific system verified:")
        print(f"  • User creation and directory setup")
        print(f"  • Agent creation with user isolation")
        print(f"  • File uploads (regular and temporary)")
        print(f"  • Chat interface with file support")
        print(f"  • Database synchronization")
        print(f"  • GUI dashboard functionality")
        print(f"  • Superuser access and permissions")
    else:
        print(f"\n[WARN]  Some tests failed. Please review the output above for details.")
        print(f"Common issues:")
        print(f"  • Server not running or accessible")
        print(f"  • Database not initialized")
        print(f"  • API key authentication issues")
        print(f"  • File permission problems")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
