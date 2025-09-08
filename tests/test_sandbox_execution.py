#!/usr/bin/env python3
"""
Test RAGnetic Lambda Sandbox Execution
"""

import requests
import json
import time

API_KEY = "your-api-key-here"  # Replace with actual API key
BASE_URL = "http://localhost:8000"

def test_basic_data_analysis():
    """Test basic data analysis in sandbox"""
    print("=== Testing Basic Data Analysis ===")
    
    payload = {
        "mode": "code",
        "code": """
import pandas as pd
import json
import numpy as np

# Simple calculation test
result = {
    "message": "Hello from RAGnetic Lambda Sandbox!",
    "calculation": 42 * 1337,
    "numpy_test": float(np.mean([1, 2, 3, 4, 5])),
    "pandas_version": pd.__version__
}

print("Sandbox Test Results:")
for key, value in result.items():
    print(f"{key}: {value}")

# Test pandas functionality
data = {
    "A": [1, 2, 3, 4, 5],
    "B": [10, 20, 30, 40, 50],
    "C": ["a", "b", "c", "d", "e"]
}

df = pd.DataFrame(data)
print("\\nDataFrame created:")
print(df)

# Calculate statistics
stats = {
    "mean_A": float(df["A"].mean()),
    "sum_B": int(df["B"].sum()),
    "count": len(df)
}

print("\\nStatistics:")
print(stats)

# Save results to JSON
result.update(stats)
result["dataframe_shape"] = list(df.shape)

with open("analysis_results.json", "w") as f:
    json.dump(result, f, indent=2)

print("\\nResults saved successfully!")
        """,
        "user_id": 1,
        "thread_id": "test-basic-analysis"
    }
    
    return execute_lambda(payload)

def test_function_mode():
    """Test function execution mode"""
    print("=== Testing Function Execution Mode ===")
    
    payload = {
        "mode": "function",
        "function_name": "analyze_numbers",
        "function_args": {
            "numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "operation": "statistics"
        },
        "function_source": """
def analyze_numbers(numbers, operation="basic"):
    import json
    
    if operation == "statistics":
        result = {
            "count": len(numbers),
            "sum": sum(numbers),
            "mean": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "even_count": len([n for n in numbers if n % 2 == 0]),
            "odd_count": len([n for n in numbers if n % 2 != 0])
        }
    else:
        result = {"numbers": numbers, "length": len(numbers)}
    
    print(f"Analysis complete for {len(numbers)} numbers")
    print(f"Operation: {operation}")
    print(f"Results: {result}")
    
    # Save results
    with open("function_results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result
        """,
        "user_id": 1,
        "thread_id": "test-function-mode"
    }
    
    return execute_lambda(payload)

def test_complex_analysis():
    """Test complex data analysis"""
    print("=== Testing Complex Data Analysis ===")
    
    payload = {
        "mode": "code",
        "code": """
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
sales_data = {
    'date': dates,
    'sales': np.random.normal(1000, 200, 100),
    'customers': np.random.poisson(50, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
}

df = pd.DataFrame(sales_data)

# Complex analysis
analysis = {
    'total_sales': float(df['sales'].sum()),
    'average_daily_sales': float(df['sales'].mean()),
    'best_sales_day': df.loc[df['sales'].idxmax(), 'date'].strftime('%Y-%m-%d'),
    'worst_sales_day': df.loc[df['sales'].idxmin(), 'date'].strftime('%Y-%m-%d'),
    'regional_performance': df.groupby('region')['sales'].agg(['sum', 'mean']).to_dict(),
    'total_customers': int(df['customers'].sum()),
    'correlation_sales_customers': float(df['sales'].corr(df['customers']))
}

print("Complex Analysis Results:")
for key, value in analysis.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for sub_key, sub_value in value.items():
            print(f"  {sub_key}: {sub_value}")
    else:
        print(f"{key}: {value}")

# Save detailed analysis
with open("complex_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2, default=str)

print("\\nComplex analysis completed and saved!")
        """,
        "user_id": 1,
        "thread_id": "test-complex-analysis"
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
                print(" SUCCESS! Execution completed.")
                final_state = status_data.get("final_state", {})
                output = final_state.get("output", "")
                if output:
                    print(" Output:")
                    print(output)
                
                artifacts = final_state.get("artifacts") or final_state.get("result_files", [])
                if artifacts:
                    print(f" Artifacts created: {artifacts}")
                
                return True
                
            elif status == "failed":
                print(" FAILED! Execution failed.")
                print(f"Error: {status_data.get('error_message', 'Unknown error')}")
                final_state = status_data.get("final_state", {})
                if final_state.get("traceback"):
                    print("Traceback:")
                    print(final_state.get("traceback"))
                return False
        else:
            print(f"Status check failed: {status_response.status_code}")
            
    print(" Timeout waiting for results")
    return False

def main():
    """Run all sandbox tests"""
    print(" Starting RAGnetic Lambda Sandbox Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Data Analysis", test_basic_data_analysis),
        ("Function Mode", test_function_mode),
        ("Complex Analysis", test_complex_analysis),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\\n Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"Result: {' PASS' if success else ' FAIL'}")
        except Exception as e:
            print(f" ERROR: {e}")
            results.append((test_name, False))
        
        print("-" * 30)
    
    # Summary
    print("\\n SUMMARY:")
    print("=" * 30)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = " PASS" if success else " FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(" ALL TESTS PASSED! Lambda tool is working perfectly!")
    else:
        print("  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()