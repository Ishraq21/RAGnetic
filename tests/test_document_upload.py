#!/usr/bin/env python3
"""
Test RAGnetic Temporary Document Upload and Processing
"""

import requests
import json
import time
import os

API_KEY = "604a7d725c7e96a5f2517f16cfc5d81c64365c55662de49c23e1aa3650b0f0b8"
BASE_URL = "http://localhost:8000"

def test_document_upload_and_analysis():
    """Test uploading documents and analyzing them with lambda tool"""
    print("=== Testing Document Upload and Analysis ===")
    
    # Test with our existing data files
    test_files = [
        ("data/sales_data.csv", "text/csv"),
        ("data/sensor_readings.json", "application/json"),
        ("data/financial_data.txt", "text/plain")
    ]
    
    uploaded_files = []
    
    for file_path, content_type in test_files:
        if os.path.exists(file_path):
            print(f"Uploading {file_path}...")
            
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, content_type)}
                response = requests.post(
                    f"{BASE_URL}/api/v1/documents/upload",
                    files=files,
                    headers={"X-API-Key": API_KEY}
                )
            
            print(f"Upload status: {response.status_code}")
            if response.status_code == 200:
                doc_data = response.json()
                uploaded_files.append(doc_data)
                print(f"Document ID: {doc_data.get('id')}")
            else:
                print(f"Upload failed: {response.text}")
    
    if not uploaded_files:
        print("No files uploaded successfully")
        return False
    
    # Now test analysis with uploaded files
    print(f"\nAnalyzing {len(uploaded_files)} uploaded files...")
    
    # Create analysis code that works with the uploaded files
    analysis_code = f"""
import pandas as pd
import json
import os

print("Available files in workspace:")
for file in os.listdir('.'):
    if os.path.isfile(file):
        print(f"- {{file}} ({{os.path.getsize(file)}} bytes)")

results = {{"analysis_results": []}}

# Try to analyze CSV files
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        analysis = {{
            "file": csv_file,
            "type": "CSV",
            "rows": len(df),
            "columns": list(df.columns),
            "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include=['number']).empty else "No numeric columns"
        }}
        results["analysis_results"].append(analysis)
        print(f"Analyzed CSV: {{csv_file}} - {{len(df)}} rows, {{len(df.columns)}} columns")
    except Exception as e:
        print(f"Failed to analyze {{csv_file}}: {{e}}")

# Try to analyze JSON files  
json_files = [f for f in os.listdir('.') if f.endswith('.json')]
for json_file in json_files:
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        analysis = {{
            "file": json_file,
            "type": "JSON",
            "structure": type(data).__name__,
            "keys": list(data.keys()) if isinstance(data, dict) else "Not a dictionary",
            "size": len(data) if isinstance(data, (dict, list)) else "Unknown"
        }}
        results["analysis_results"].append(analysis)
        print(f"Analyzed JSON: {{json_file}} - {{type(data).__name__}} with {{len(data) if isinstance(data, (dict, list)) else '?'}} items")
    except Exception as e:
        print(f"Failed to analyze {{json_file}}: {{e}}")

# Save comprehensive results
with open("upload_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\\nAnalysis complete! Processed {{len(results['analysis_results'])}} files")
"""
    
    # Include the uploaded files as inputs
    input_files = [{"file_name": doc["filename"]} for doc in uploaded_files]
    
    payload = {
        "mode": "code",
        "code": analysis_code,
        "inputs": input_files,
        "user_id": 1,
        "thread_id": "test-document-upload"
    }
    
    return execute_lambda(payload)

def test_end_to_end_workflow():
    """Test complete workflow with file staging and processing"""
    print("=== Testing End-to-End Workflow ===")
    
    # Create a more complex analysis that combines multiple data sources
    workflow_code = """
import pandas as pd
import json
import numpy as np
from datetime import datetime

print("=== End-to-End Data Processing Workflow ===")

# Initialize results container
workflow_results = {
    "timestamp": datetime.now().isoformat(),
    "processed_files": [],
    "insights": {}
}

# Process sales data if available
try:
    sales_df = pd.read_csv("sales_data.csv")
    
    # Sales analysis
    sales_insights = {
        "total_sales": float(sales_df["sales"].sum()),
        "avg_sales": float(sales_df["sales"].mean()),
        "top_region": sales_df.groupby("region")["sales"].sum().idxmax(),
        "top_product_category": sales_df.groupby("category")["sales"].sum().idxmax(),
        "date_range": {
            "start": sales_df["date"].min(),
            "end": sales_df["date"].max()
        }
    }
    
    workflow_results["insights"]["sales"] = sales_insights
    workflow_results["processed_files"].append("sales_data.csv")
    print("Sales data processed successfully")
    
except Exception as e:
    print(f"Sales data processing failed: {e}")

# Process sensor data if available  
try:
    with open("sensor_readings.json", "r") as f:
        sensor_data = json.load(f)
    
    # Sensor analysis
    if isinstance(sensor_data, list) and len(sensor_data) > 0:
        temps = [reading.get("temperature", 0) for reading in sensor_data]
        humidity = [reading.get("humidity", 0) for reading in sensor_data]
        
        sensor_insights = {
            "reading_count": len(sensor_data),
            "temperature_stats": {
                "avg": float(np.mean(temps)),
                "min": float(np.min(temps)),
                "max": float(np.max(temps))
            },
            "humidity_stats": {
                "avg": float(np.mean(humidity)),
                "min": float(np.min(humidity)),
                "max": float(np.max(humidity))
            }
        }
        
        workflow_results["insights"]["sensors"] = sensor_insights
        workflow_results["processed_files"].append("sensor_readings.json")
        print("Sensor data processed successfully")
    
except Exception as e:
    print(f"Sensor data processing failed: {e}")

# Generate comprehensive report
print("\\n=== WORKFLOW SUMMARY ===")
print(f"Processed files: {workflow_results['processed_files']}")

for category, insights in workflow_results["insights"].items():
    print(f"\\n{category.upper()} INSIGHTS:")
    for key, value in insights.items():
        print(f"  {key}: {value}")

# Save final workflow results
with open("workflow_results.json", "w") as f:
    json.dump(workflow_results, f, indent=2, default=str)

print("\\n=== WORKFLOW COMPLETE ===")
print(f"Results saved to workflow_results.json")
"""
    
    # Use our test data files as inputs
    payload = {
        "mode": "code", 
        "code": workflow_code,
        "inputs": [
            {"file_name": "sales_data.csv"},
            {"file_name": "sensor_readings.json"}
        ],
        "user_id": 1,
        "thread_id": "test-end-to-end-workflow"
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
                    print(f"Artifacts created: {artifacts}")
                
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
    """Run document upload and workflow tests"""
    print("Starting RAGnetic Document and Workflow Tests")
    print("=" * 50)
    
    tests = [
        ("Document Upload and Analysis", test_document_upload_and_analysis),
        ("End-to-End Workflow", test_end_to_end_workflow),
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
        print("ALL TESTS PASSED! Document processing and workflows working perfectly!")
    else:
        print("Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()