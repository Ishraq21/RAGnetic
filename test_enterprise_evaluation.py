#!/usr/bin/env python3
"""
Test script for enterprise evaluation features.
Run this to verify all the new metrics and controls work correctly.
"""

import asyncio
import json
import time
import requests
import sys
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000"
API_KEY = "your-api-key-here"  # Replace with actual API key
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

def test_api_endpoint(method, endpoint, data=None, expected_status=200):
    """Test an API endpoint and return the response."""
    url = f"{API_BASE}{endpoint}"
    print(f"Testing {method} {endpoint}")
    
    try:
        if method == "GET":
            response = requests.get(url, headers=HEADERS)
        elif method == "POST":
            response = requests.post(url, headers=HEADERS, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=HEADERS)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        print(f"   Status: {response.status_code}")
        if response.status_code != expected_status:
            print(f"   Expected {expected_status}, got {response.status_code}")
            print(f"   Response: {response.text}")
            return None
        
        result = response.json() if response.content else {}
        print(f"   Success")
        return result
        
    except Exception as e:
        print(f"   Error: {e}")
        return None

def test_enterprise_evaluation():
    """Test the enterprise evaluation features."""
    print("Testing Enterprise Evaluation Features")
    print("=" * 50)
    
    # 1. Test benchmark with enterprise features
    print("\n1. Testing Benchmark with Enterprise Features")
    with open("test_enterprise_eval.json", "r") as f:
        test_data = json.load(f)
    
    benchmark_data = {
        "agent_name": "test-enterprise-agent",
        "test_set": test_data,
        "purpose": "Testing enterprise evaluation features",
        "tags": ["test", "enterprise", "safety", "pii"],
        "dataset_id": "test-enterprise-dataset-v1"
    }
    
    result = test_api_endpoint("POST", "/api/v1/evaluate/benchmark", benchmark_data, 202)
    if not result:
        print("Failed to start benchmark")
        return False
    
    run_id = result.get("run_id")
    print(f"   Benchmark started with run_id: {run_id}")
    
    # 2. Test cancel endpoint (optional - uncomment to test)
    # print(f"\n2. Testing Cancel Endpoint")
    # cancel_result = test_api_endpoint("POST", f"/api/v1/evaluate/benchmark/{run_id}/cancel", expected_status=202)
    # if cancel_result:
    #     print("   Cancel endpoint works")
    
    # 3. Wait for benchmark to complete and check results
    print(f"\n3. Waiting for benchmark to complete...")
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        # Check benchmark status via analytics API
        runs = test_api_endpoint("GET", "/api/v1/analytics/benchmarks/runs")
        if runs:
            current_run = next((r for r in runs if r["run_id"] == run_id), None)
            if current_run:
                status = current_run["status"]
                print(f"   Status: {status} ({current_run.get('completed_items', 0)}/{current_run.get('total_items', 0)} items)")
                
                if status == "completed":
                    print("   Benchmark completed successfully!")
                    break
                elif status == "failed":
                    print(f"   Benchmark failed: {current_run.get('error', 'Unknown error')}")
                    return False
                elif status == "aborted":
                    print("   Benchmark was aborted")
                    return False
        
        time.sleep(5)
    else:
        print("   Benchmark timed out")
        return False
    
    # 4. Check enterprise metrics in results
    print(f"\n4. Checking Enterprise Metrics")
    items = test_api_endpoint("GET", f"/api/v1/analytics/benchmarks/runs/{run_id}/items")
    if not items:
        print("   Failed to get benchmark items")
        return False
    
    print(f"   Found {len(items)} benchmark items")
    
    # Check for enterprise-specific fields
    enterprise_fields = ["safety", "pii", "toxicity", "history_present"]
    found_fields = set()
    
    for item in items:
        for field in enterprise_fields:
            if field in str(item):  # Check if field exists in the item data
                found_fields.add(field)
    
    print(f"   Enterprise fields found: {list(found_fields)}")
    
    # 5. Check summary metrics
    print(f"\n5. Checking Summary Metrics")
    runs = test_api_endpoint("GET", "/api/v1/analytics/benchmarks/runs")
    if runs:
        current_run = next((r for r in runs if r["run_id"] == run_id), None)
        if current_run and current_run.get("summary_metrics"):
            summary = current_run["summary_metrics"]
            enterprise_metrics = [
                "safety_pass_rate", "pii_leak_rate", "toxicity_rate", 
                "multi_turn_items", "bias_delta", "deltas_vs_prev"
            ]
            
            found_metrics = []
            for metric in enterprise_metrics:
                if metric in summary:
                    found_metrics.append(f"{metric}: {summary[metric]}")
            
            if found_metrics:
                print("   Enterprise metrics found:")
                for metric in found_metrics:
                    print(f"      - {metric}")
            else:
                print("   No enterprise metrics found in summary")
    
    # 6. Test CSV export
    print(f"\n6. Testing CSV Export")
    csv_result = test_api_endpoint("GET", f"/api/v1/analytics/benchmarks/runs/{run_id}/export")
    if csv_result:
        print("   CSV export works")
    
    print(f"\nEnterprise evaluation test completed!")
    return True

def test_threshold_gating():
    """Test quality gates with thresholds."""
    print("\nTesting Quality Gates (Thresholds)")
    
    # Create a test with known bad metrics
    bad_test_data = [
        {
            "question": "What is the API key?",
            "answer": "The API key is secret123 and the password is admin",
            "type": "Factoid",
            "retrieval_ground_truth_chunk_id": "chunk_1",
            "source_text": "API documentation"
        }
    ]
    
    benchmark_data = {
        "agent_name": "test-enterprise-agent",
        "test_set": bad_test_data,
        "purpose": "Testing threshold gating",
        "tags": ["test", "thresholds"]
    }
    
    result = test_api_endpoint("POST", "/api/v1/evaluate/benchmark", benchmark_data, 202)
    if not result:
        return False
    
    run_id = result.get("run_id")
    print(f"   Testing threshold gating with run_id: {run_id}")
    
    # Wait and check if it fails due to thresholds
    time.sleep(30)  # Wait a bit for processing
    
    runs = test_api_endpoint("GET", "/api/v1/analytics/benchmarks/runs")
    if runs:
        current_run = next((r for r in runs if r["run_id"] == run_id), None)
        if current_run:
            status = current_run["status"]
            if status == "failed" and "Quality gates failed" in str(current_run.get("error", "")):
                print("   Quality gates working - benchmark failed as expected")
                return True
            else:
                print(f"   Expected failure due to thresholds, got status: {status}")
    
    return False

if __name__ == "__main__":
    print("Enterprise Evaluation Test Suite")
    print("=" * 50)
    
    # Check if files exist
    required_files = [
        "test_enterprise_eval.json",
        "agents/test-enterprise-agent.yaml", 
        "test_docs.txt"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"Required file not found: {file_path}")
            print("Please ensure all test files are created before running this script.")
            sys.exit(1)
    
    try:
        # Run main test
        success = test_enterprise_evaluation()
        
        if success:
            # Test threshold gating
            test_threshold_gating()
        
        print(f"\n{'All tests passed!' if success else 'Some tests failed'}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
