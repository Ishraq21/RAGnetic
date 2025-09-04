#!/usr/bin/env python3
"""
Test RAGnetic Lambda Tool with File Workflow
Tests lambda tool functionality without uploads, using direct file references
"""

import requests
import json
import time

API_KEY = "604a7d725c7e96a5f2517f16cfc5d81c64365c55662de49c23e1aa3650b0f0b8"
BASE_URL = "http://localhost:8000"

def test_code_only_execution():
    """Test lambda tool with code-only execution (no file inputs)"""
    print("=== Testing Code-Only Execution ===")
    
    payload = {
        "mode": "code",
        "code": """
# Test mathematical operations and data structures
import json
import math

# Mathematical calculations
calculations = {
    "fibonacci": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
    "factorials": [math.factorial(i) for i in range(1, 8)],
    "squares": [i**2 for i in range(1, 11)],
    "primes": [n for n in range(2, 50) if all(n % i != 0 for i in range(2, int(n**0.5) + 1))]
}

print("Mathematical Calculations:")
for calc_type, values in calculations.items():
    print(f"{calc_type}: {values[:5]}...")

# Data processing simulation
sample_data = {
    "users": [
        {"id": 1, "name": "Alice", "age": 25, "city": "NYC"},
        {"id": 2, "name": "Bob", "age": 30, "city": "SF"},
        {"id": 3, "name": "Charlie", "age": 35, "city": "LA"}
    ]
}

# Process the data
total_age = sum(user["age"] for user in sample_data["users"])
average_age = total_age / len(sample_data["users"])
cities = [user["city"] for user in sample_data["users"]]

results = {
    "total_users": len(sample_data["users"]),
    "average_age": average_age,
    "cities": cities,
    "calculations": calculations
}

print(f"\\nData Processing Results:")
print(f"Total users: {results['total_users']}")
print(f"Average age: {results['average_age']}")
print(f"Cities: {results['cities']}")

# Save results
with open("code_execution_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\\nResults saved to code_execution_results.json")
        """,
        "user_id": 1,
        "thread_id": "test-code-only"
    }
    
    return execute_lambda(payload)

def test_multi_step_analysis():
    """Test multi-step data analysis and processing"""
    print("=== Testing Multi-Step Analysis ===")
    
    payload = {
        "mode": "code",
        "code": """
import json
import random
import datetime

print("=== Multi-Step Data Analysis Workflow ===")

# Step 1: Generate synthetic dataset
random.seed(42)
dataset = []
for i in range(100):
    record = {
        "id": i + 1,
        "timestamp": (datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 365))).isoformat(),
        "value": random.uniform(10, 100),
        "category": random.choice(["A", "B", "C", "D"]),
        "status": random.choice(["active", "inactive", "pending"])
    }
    dataset.append(record)

print(f"Step 1: Generated {len(dataset)} records")

# Step 2: Data cleaning and validation
valid_records = []
for record in dataset:
    if record["value"] > 0 and record["category"] and record["status"]:
        valid_records.append(record)

print(f"Step 2: Validated {len(valid_records)} records ({len(valid_records)/len(dataset)*100:.1f}%)")

# Step 3: Statistical analysis
values = [r["value"] for r in valid_records]
categories = [r["category"] for r in valid_records]
statuses = [r["status"] for r in valid_records]

stats = {
    "count": len(values),
    "min": min(values),
    "max": max(values),
    "mean": sum(values) / len(values),
    "median": sorted(values)[len(values)//2]
}

# Category analysis
category_counts = {}
category_values = {}
for record in valid_records:
    cat = record["category"]
    if cat not in category_counts:
        category_counts[cat] = 0
        category_values[cat] = []
    category_counts[cat] += 1
    category_values[cat].append(record["value"])

category_analysis = {}
for cat in category_counts:
    vals = category_values[cat]
    category_analysis[cat] = {
        "count": category_counts[cat],
        "avg_value": sum(vals) / len(vals),
        "percentage": category_counts[cat] / len(valid_records) * 100
    }

print(f"Step 3: Statistical analysis completed")
print(f"Overall stats: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")

# Step 4: Generate insights and reports
insights = []
if stats["mean"] > 50:
    insights.append("Average value is above median threshold")
if category_counts:
    top_category = max(category_counts, key=category_counts.get)
    insights.append(f"Category '{top_category}' has the most records ({category_counts[top_category]})")

final_report = {
    "analysis_timestamp": datetime.datetime.now().isoformat(),
    "dataset_size": len(dataset),
    "valid_records": len(valid_records),
    "overall_statistics": stats,
    "category_analysis": category_analysis,
    "key_insights": insights,
    "data_quality_score": len(valid_records) / len(dataset) * 100
}

print(f"Step 4: Generated {len(insights)} insights")
for insight in insights:
    print(f"  - {insight}")

# Step 5: Export results
with open("multi_step_analysis_report.json", "w") as f:
    json.dump(final_report, f, indent=2)

print(f"\\nStep 5: Multi-step analysis complete!")
print(f"Data quality score: {final_report['data_quality_score']:.1f}%")
print("Report saved to multi_step_analysis_report.json")
        """,
        "user_id": 1,
        "thread_id": "test-multi-step"
    }
    
    return execute_lambda(payload)

def test_file_operations():
    """Test file creation and manipulation"""
    print("=== Testing File Operations ===")
    
    payload = {
        "mode": "code",
        "code": """
import json
import csv
from io import StringIO

print("=== File Operations Test ===")

# Test 1: Create and write multiple file formats
print("Test 1: Creating multiple file formats")

# JSON file creation
json_data = {
    "test_info": {
        "test_name": "File Operations",
        "timestamp": "2024-01-01T12:00:00Z",
        "version": "1.0"
    },
    "results": [
        {"operation": "json_write", "status": "success"},
        {"operation": "csv_write", "status": "pending"},
        {"operation": "text_write", "status": "pending"}
    ]
}

with open("test_output.json", "w") as f:
    json.dump(json_data, f, indent=2)
print("  - JSON file created: test_output.json")

# CSV file creation
csv_data = [
    ["ID", "Name", "Score", "Grade"],
    [1, "Alice", 95, "A"],
    [2, "Bob", 87, "B"],
    [3, "Charlie", 92, "A"],
    [4, "Diana", 78, "C"]
]

with open("test_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
print("  - CSV file created: test_output.csv")

# Text file creation
text_content = '''File Operations Test Report
==========================

This is a test of file writing capabilities.

Operations performed:
1. JSON file creation - Success
2. CSV file creation - Success  
3. Text file creation - In progress

Test completed successfully.
'''

with open("test_output.txt", "w") as f:
    f.write(text_content)
print("  - Text file created: test_output.txt")

# Test 2: Read and verify files
print("\\nTest 2: Reading and verifying created files")

# Verify JSON
try:
    with open("test_output.json", "r") as f:
        loaded_json = json.load(f)
    print(f"  - JSON verification: {len(loaded_json)} top-level keys")
except Exception as e:
    print(f"  - JSON verification failed: {e}")

# Verify CSV
try:
    with open("test_output.csv", "r") as f:
        csv_reader = csv.reader(f)
        rows = list(csv_reader)
    print(f"  - CSV verification: {len(rows)} rows, {len(rows[0])} columns")
except Exception as e:
    print(f"  - CSV verification failed: {e}")

# Verify Text
try:
    with open("test_output.txt", "r") as f:
        text_lines = f.readlines()
    print(f"  - Text verification: {len(text_lines)} lines")
except Exception as e:
    print(f"  - Text verification failed: {e}")

# Test 3: File manipulation summary
print("\\nTest 3: File manipulation summary")
import os
created_files = ["test_output.json", "test_output.csv", "test_output.txt"]
summary = {"files_created": [], "total_size": 0}

for filename in created_files:
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        summary["files_created"].append({
            "name": filename,
            "size_bytes": size
        })
        summary["total_size"] += size

summary["success_count"] = len(summary["files_created"])
summary["success_rate"] = (summary["success_count"] / len(created_files)) * 100

with open("file_operations_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"  - Created {summary['success_count']}/{len(created_files)} files")
print(f"  - Total size: {summary['total_size']} bytes")
print(f"  - Success rate: {summary['success_rate']}%")
print("\\nFile operations test completed successfully!")
        """,
        "user_id": 1,
        "thread_id": "test-file-ops"
    }
    
    return execute_lambda(payload)

def execute_lambda(payload):
    """Execute a lambda request and poll for results"""
    response = requests.post(
        f"{BASE_URL}/api/v1/lambda/execute",
        json=payload,
        headers={"X-API-Key": API_KEY}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code != 202:
        print(f"Error: {response.text}")
        return False
        
    response_data = response.json()
    run_id = response_data.get("run_id")
    print(f"Run ID: {run_id}")
    
    if not run_id:
        return False
    
    print("Polling for results...")
    
    # Poll for results
    for i in range(30):  # 30 attempts
        time.sleep(2)
        
        status_response = requests.get(
            f"{BASE_URL}/api/v1/lambda/runs/{run_id}",
            headers={"X-API-Key": API_KEY}
        )
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            status = status_data.get("status")
            print(f"Status check {i+1}: {status}")
            
            if status == "completed":
                print("SUCCESS! Execution completed.")
                final_state = status_data.get("final_state", {})
                output = final_state.get("output", "")
                if output:
                    print("Output:")
                    print(output)
                
                artifacts = final_state.get("artifacts") or final_state.get("result_files", [])
                if artifacts:
                    print(f"Artifacts created: {len(artifacts)} files")
                    for artifact in artifacts[:3]:  # Show first 3
                        print(f"  - {artifact.get('file_name', 'Unknown')}")
                
                return True
                
            elif status == "failed":
                print("FAILED! Execution failed.")
                print(f"Error: {status_data.get('error_message', 'Unknown error')}")
                final_state = status_data.get("final_state", {})
                if final_state.get("traceback"):
                    print("Traceback:")
                    print(final_state.get("traceback"))
                return False
        else:
            print(f"Status check failed: {status_response.status_code}")
            
    print("Timeout waiting for results")
    return False

def main():
    """Run file workflow tests"""
    print("Starting RAGnetic Lambda File Workflow Tests")
    print("=" * 50)
    
    tests = [
        ("Code-Only Execution", test_code_only_execution),
        ("Multi-Step Analysis", test_multi_step_analysis), 
        ("File Operations", test_file_operations),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\\nRunning: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"Result: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((test_name, False))
        
        print("-" * 30)
    
    # Summary
    print("\\nSUMMARY:")
    print("=" * 30)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("ALL TESTS PASSED! Lambda tool file workflows working perfectly!")
    else:
        print("Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()