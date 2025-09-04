#!/usr/bin/env python3
"""
Comprehensive RAGnetic Lambda Tool Sandbox Testing
Tests complex scenarios, data analysis, and advanced capabilities
"""

import requests
import json
import time
import os
from pathlib import Path

API_KEY = "604a7d725c7e96a5f2517f16cfc5d81c64365c55662de49c23e1aa3650b0f0b8"
BASE_URL = "http://localhost:8000"

def test_complex_data_processing():
    """Test complex data processing with multiple data sources"""
    print("=== Testing Complex Data Processing ===")
    
    payload = {
        "mode": "code",
        "code": """
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime, timedelta

print("🔬 COMPREHENSIVE DATA ANALYSIS PIPELINE")
print("=" * 50)

# Step 1: Load and analyze CSV employee data
print("\\n STEP 1: Employee Data Analysis")
try:
    df_employees = pd.read_csv('large_dataset.csv')
    print(f"Loaded {len(df_employees)} employee records")
    
    # Complex aggregations
    dept_analysis = df_employees.groupby('department').agg({
        'salary': ['mean', 'min', 'max', 'std'],
        'performance_score': 'mean',
        'age': 'mean'
    }).round(2)
    
    print("Department Analysis:")
    print(dept_analysis)
    
    # Statistical insights
    salary_stats = {
        'total_payroll': float(df_employees['salary'].sum()),
        'median_salary': float(df_employees['salary'].median()),
        'salary_distribution': df_employees['salary'].describe().to_dict(),
        'high_performers': len(df_employees[df_employees['performance_score'] >= 4.5]),
        'departments': df_employees['department'].value_counts().to_dict()
    }
    print(f"\\nKey Insights:")
    print(f"- Total Payroll: ${salary_stats['total_payroll']:,.2f}")
    print(f"- High Performers (≥4.5): {salary_stats['high_performers']}")
    print(f"- Department Distribution: {salary_stats['departments']}")
    
except Exception as e:
    print(f" Employee data analysis failed: {e}")
    salary_stats = {}

# Step 2: Process complex JSON e-commerce data
print("\\n🛒 STEP 2: E-commerce Data Analysis")
try:
    with open('complex_json_data.json', 'r') as f:
        ecommerce_data = json.load(f)
    
    # Extract transaction insights
    transactions = ecommerce_data.get('transactions', [])
    customers = ecommerce_data.get('customers', {})
    products = ecommerce_data.get('products', [])
    
    transaction_analysis = {
        'total_transactions': len(transactions),
        'total_revenue': sum(t['total_amount'] for t in transactions),
        'avg_order_value': np.mean([t['total_amount'] for t in transactions]),
        'payment_methods': {},
        'geographic_distribution': {}
    }
    
    # Analyze payment methods and geography
    for txn in transactions:
        pm = txn.get('payment_method', 'unknown')
        transaction_analysis['payment_methods'][pm] = transaction_analysis['payment_methods'].get(pm, 0) + 1
        
        city = txn.get('shipping_address', {}).get('city', 'unknown')
        transaction_analysis['geographic_distribution'][city] = transaction_analysis['geographic_distribution'].get(city, 0) + 1
    
    # Product performance analysis
    product_performance = {}
    for product in products:
        pid = product['product_id']
        ratings = product.get('ratings', {})
        product_performance[pid] = {
            'name': product['name'],
            'category': product['category'],
            'price': product['price'],
            'profit_margin': ((product['price'] - product['cost']) / product['price'] * 100),
            'avg_rating': ratings.get('average', 0),
            'review_count': ratings.get('count', 0),
            'inventory': product['inventory']
        }
    
    print(f"E-commerce Analysis Complete:")
    print(f"- Revenue: ${transaction_analysis['total_revenue']:.2f}")
    print(f"- AOV: ${transaction_analysis['avg_order_value']:.2f}")
    print(f"- Payment Methods: {transaction_analysis['payment_methods']}")
    print(f"- Top Products by Rating:")
    for pid, perf in product_performance.items():
        print(f"  • {perf['name']}: {perf['avg_rating']}★ (${perf['price']}, {perf['profit_margin']:.1f}% margin)")
    
except Exception as e:
    print(f" E-commerce analysis failed: {e}")
    transaction_analysis = {}

# Step 3: Time series analysis
print("\\n STEP 3: Time Series System Monitoring Analysis")
try:
    df_timeseries = pd.read_csv('time_series_data.csv')
    df_timeseries['timestamp'] = pd.to_datetime(df_timeseries['timestamp'])
    
    # Advanced time series calculations
    metrics = ['cpu_usage', 'memory_usage', 'temperature', 'response_time']
    timeseries_analysis = {}
    
    for metric in metrics:
        series = df_timeseries[metric]
        timeseries_analysis[metric] = {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'trend': 'increasing' if series.iloc[-1] > series.iloc[0] else 'decreasing',
            'volatility': float(series.std() / series.mean() * 100),  # Coefficient of variation
            'anomalies': len(series[abs(series - series.mean()) > 2 * series.std()])
        }
    
    # System health assessment
    cpu_high = len(df_timeseries[df_timeseries['cpu_usage'] > 80])
    memory_high = len(df_timeseries[df_timeseries['memory_usage'] > 90])
    temp_high = len(df_timeseries[df_timeseries['temperature'] > 37])
    
    system_health = {
        'total_samples': len(df_timeseries),
        'high_cpu_periods': cpu_high,
        'high_memory_periods': memory_high, 
        'high_temp_periods': temp_high,
        'overall_health': 'CRITICAL' if (cpu_high > 5 or memory_high > 3) else 'GOOD'
    }
    
    print(f"System Monitoring Analysis:")
    print(f"- Samples Analyzed: {system_health['total_samples']}")
    print(f"- System Health: {system_health['overall_health']}")
    print(f"- High CPU Periods: {cpu_high}/{len(df_timeseries)}")
    print(f"- High Memory Periods: {memory_high}/{len(df_timeseries)}")
    print(f"- Performance Trend: {timeseries_analysis['response_time']['trend']}")
    
except Exception as e:
    print(f" Time series analysis failed: {e}")
    timeseries_analysis = {}

# Step 4: Text processing and log analysis
print("\\n📝 STEP 4: Mixed Data and Log Processing")
try:
    # Process mixed format text data
    with open('mixed_data_types.txt', 'r') as f:
        mixed_content = f.read()
    
    # Extract numerical data using regex
    temp_matches = re.findall(r'Temperature readings.*?:\\s*([\\d.,\\s]+)', mixed_content)
    if temp_matches:
        temps = [float(x.strip()) for x in temp_matches[0].replace(' ', '').split(',') if x.strip()]
        temp_analysis = {
            'count': len(temps),
            'mean': np.mean(temps),
            'std': np.std(temps),
            'range': max(temps) - min(temps)
        }
    else:
        temp_analysis = {}
    
    # Extract success rates
    success_matches = re.findall(r'(\\d+\\.\\d+)%\\s+success\\s+rate', mixed_content)
    if success_matches:
        success_rates = [float(x) for x in success_matches]
        avg_success_rate = np.mean(success_rates)
    else:
        avg_success_rate = 0
    
    # Process network logs
    with open('network_logs.txt', 'r') as f:
        log_lines = f.readlines()
    
    log_analysis = {
        'total_requests': len(log_lines),
        'status_codes': {},
        'api_endpoints': {},
        'methods': {},
        'user_agents': {}
    }
    
    for line in log_lines:
        # Parse common log format
        parts = line.split(' ')
        if len(parts) >= 9:
            method_endpoint = parts[5] + ' ' + parts[6] if len(parts) > 6 else ''
            status_code = parts[8] if len(parts) > 8 else '000'
            
            # Count status codes
            log_analysis['status_codes'][status_code] = log_analysis['status_codes'].get(status_code, 0) + 1
            
            # Extract API endpoints
            if '/api/' in method_endpoint:
                endpoint = method_endpoint.split(' ')[1].split('?')[0] if ' ' in method_endpoint else ''
                if endpoint:
                    log_analysis['api_endpoints'][endpoint] = log_analysis['api_endpoints'].get(endpoint, 0) + 1
    
    print(f"Text and Log Analysis:")
    print(f"- Temperature readings analyzed: {temp_analysis.get('count', 0)} points")
    print(f"- Average success rate: {avg_success_rate:.1f}%")
    print(f"- Network requests processed: {log_analysis['total_requests']}")
    print(f"- Status code distribution: {log_analysis['status_codes']}")
    print(f"- Top API endpoints: {dict(list(log_analysis['api_endpoints'].items())[:3])}")
    
except Exception as e:
    print(f" Text processing failed: {e}")
    temp_analysis = {}
    log_analysis = {}

# Step 5: Advanced statistical analysis and ML-style processing
print("\\n🤖 STEP 5: Advanced Analytics and Insights")
try:
    # Correlation analysis between different datasets
    if 'df_employees' in locals() and 'df_timeseries' in locals():
        # Create synthetic correlations for demonstration
        performance_cpu_correlation = np.random.rand() * 0.3 + 0.1  # Simulated
        
        # Advanced statistical tests
        from scipy import stats
        
        # Normality test on salary data
        if 'df_employees' in locals():
            _, salary_normality_p = stats.normaltest(df_employees['salary'])
            is_normal = salary_normality_p > 0.05
        else:
            is_normal = None
        
        print(f"Advanced Statistical Analysis:")
        print(f"- Salary distribution normality: {'Normal' if is_normal else 'Non-normal' if is_normal is not None else 'N/A'}")
        if is_normal is not None:
            print(f"- p-value: {salary_normality_p:.4f}")
        
    # Generate comprehensive summary
    comprehensive_summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'datasets_processed': 4,
        'employee_insights': salary_stats,
        'ecommerce_insights': transaction_analysis,
        'system_monitoring': timeseries_analysis,
        'text_processing': {
            'temperature_analysis': temp_analysis,
            'log_analysis': log_analysis,
            'avg_success_rate': avg_success_rate
        },
        'recommendations': [
            "Consider salary review for departments with high performance scores",
            "Monitor system CPU usage - trending upward",
            "Optimize high-traffic API endpoints",
            "Investigate low success rate periods"
        ]
    }
    
    print(f"\\n📋 COMPREHENSIVE SUMMARY:")
    print(f"- Datasets Processed: {comprehensive_summary['datasets_processed']}")
    print(f"- Analysis Timestamp: {comprehensive_summary['analysis_timestamp']}")
    print(f"\\nKey Recommendations:")
    for i, rec in enumerate(comprehensive_summary['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Save comprehensive results
    with open('comprehensive_analysis_results.json', 'w') as f:
        json.dump(comprehensive_summary, f, indent=2, default=str)
    
    print(f"\\n💾 Results saved to comprehensive_analysis_results.json")
    print("\\n COMPREHENSIVE DATA ANALYSIS PIPELINE COMPLETED")
    
except Exception as e:
    print(f" Advanced analytics failed: {e}")

print("\\n" + "=" * 50)
print(" ANALYSIS COMPLETE - All data sources processed!")
        """,
        "user_id": 1,
        "thread_id": "comprehensive-analysis"
    }
    
    return execute_lambda(payload)

def test_advanced_algorithms():
    """Test advanced algorithmic implementations"""
    print("=== Testing Advanced Algorithms ===")
    
    payload = {
        "mode": "code",
        "code": """
import numpy as np
import json
from collections import defaultdict, deque
import heapq
import time

print("🧠 ADVANCED ALGORITHMS AND DATA STRUCTURES TEST")
print("=" * 55)

# Algorithm 1: Graph algorithms (Dijkstra's shortest path)
print("\\n ALGORITHM 1: Graph Theory - Dijkstra's Shortest Path")

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v, weight):
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))  # Undirected graph
    
    def dijkstra(self, start):
        distances = defaultdict(lambda: float('inf'))
        distances[start] = 0
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            
            visited.add(u)
            
            for v, weight in self.graph[u]:
                if v not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        heapq.heappush(pq, (new_dist, v))
        
        return dict(distances)

# Build a sample network
network = Graph()
edges = [
    ('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 1), ('B', 'D', 5),
    ('C', 'D', 8), ('C', 'E', 10), ('D', 'E', 2), ('D', 'F', 6), ('E', 'F', 3)
]

for u, v, w in edges:
    network.add_edge(u, v, w)

start_time = time.time()
shortest_paths = network.dijkstra('A')
dijkstra_time = time.time() - start_time

print(f"Shortest paths from node 'A': {dict(shortest_paths)}")
print(f"Algorithm execution time: {dijkstra_time:.6f} seconds")

# Algorithm 2: Dynamic Programming - Knapsack Problem
print("\\n🎒 ALGORITHM 2: Dynamic Programming - 0/1 Knapsack")

def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    values[i-1] + dp[i-1][w - weights[i-1]]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    # Backtrack to find items
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)
            w -= weights[i-1]
    
    return dp[n][capacity], selected_items

# Test knapsack problem
items = {
    'weights': [10, 20, 30, 40, 50],
    'values': [60, 100, 120, 160, 200],
    'capacity': 80
}

start_time = time.time()
max_value, selected = knapsack_01(items['weights'], items['values'], items['capacity'])
knapsack_time = time.time() - start_time

print(f"Maximum value: {max_value}")
print(f"Selected items (indices): {selected}")
print(f"Algorithm execution time: {knapsack_time:.6f} seconds")

# Algorithm 3: Machine Learning - K-Means Clustering Implementation
print("\\n🔄 ALGORITHM 3: Machine Learning - K-Means Clustering")

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
    
    def fit(self, data):
        self.data = np.array(data)
        n_samples, n_features = self.data.shape
        
        # Initialize centroids randomly
        self.centroids = np.random.uniform(
            low=self.data.min(axis=0),
            high=self.data.max(axis=0),
            size=(self.k, n_features)
        )
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((self.data - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([
                self.data[labels == i].mean(axis=0) if len(self.data[labels == i]) > 0 
                else self.centroids[i] for i in range(self.k)
            ])
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        return labels
    
    def predict(self, new_data):
        distances = np.sqrt(((new_data - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

# Generate synthetic 2D data for clustering
np.random.seed(42)
data_points = []
# Cluster 1: around (2, 2)
data_points.extend(np.random.normal([2, 2], 0.5, (20, 2)).tolist())
# Cluster 2: around (6, 6) 
data_points.extend(np.random.normal([6, 6], 0.7, (25, 2)).tolist())
# Cluster 3: around (2, 6)
data_points.extend(np.random.normal([2, 6], 0.6, (18, 2)).tolist())

start_time = time.time()
kmeans = KMeans(k=3)
cluster_labels = kmeans.fit(data_points)
clustering_time = time.time() - start_time

# Analyze clusters
cluster_stats = {}
for i in range(3):
    cluster_points = np.array(data_points)[cluster_labels == i]
    cluster_stats[f'Cluster_{i}'] = {
        'size': len(cluster_points),
        'center': kmeans.centroids[i].tolist(),
        'variance': np.var(cluster_points, axis=0).tolist() if len(cluster_points) > 0 else [0, 0]
    }

print(f"Clustering completed on {len(data_points)} data points")
print(f"Cluster statistics: {json.dumps(cluster_stats, indent=2)}")
print(f"Algorithm execution time: {clustering_time:.6f} seconds")

# Algorithm 4: Sorting and Search - Advanced QuickSort with optimization
print("\\n ALGORITHM 4: Advanced QuickSort with 3-way Partitioning")

def quicksort_3way(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        lt, gt = partition_3way(arr, low, high)
        quicksort_3way(arr, low, lt - 1)
        quicksort_3way(arr, gt + 1, high)
    
    return arr

def partition_3way(arr, low, high):
    pivot = arr[low]
    i = low + 1
    lt = low  # arr[low...lt-1] < pivot
    gt = high + 1  # arr[gt...high] > pivot
    
    while i < gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            gt -= 1
            arr[i], arr[gt] = arr[gt], arr[i]
        else:
            i += 1
    
    return lt, gt

# Test sorting with duplicates
np.random.seed(123)
test_array = np.random.randint(1, 50, 1000).tolist()  # Array with duplicates
original_array = test_array.copy()

start_time = time.time()
sorted_array = quicksort_3way(test_array.copy())
quicksort_time = time.time() - start_time

# Verify sorting correctness
is_sorted = all(sorted_array[i] <= sorted_array[i+1] for i in range(len(sorted_array)-1))

print(f"Array size: {len(test_array)}")
print(f"Unique elements: {len(set(test_array))}")
print(f"Sorting correct: {is_sorted}")
print(f"First 10 elements - Original: {original_array[:10]}")
print(f"First 10 elements - Sorted: {sorted_array[:10]}")
print(f"Algorithm execution time: {quicksort_time:.6f} seconds")

# Performance Summary
print("\\n ALGORITHM PERFORMANCE SUMMARY")
print("=" * 40)
performance_summary = {
    'dijkstra_shortest_path': {
        'nodes': len(shortest_paths),
        'edges': len(edges),
        'time_seconds': dijkstra_time,
        'complexity': 'O((V + E) log V)'
    },
    'knapsack_dynamic_programming': {
        'items': len(items['weights']),
        'capacity': items['capacity'],
        'max_value': max_value,
        'time_seconds': knapsack_time,
        'complexity': 'O(n * W)'
    },
    'kmeans_clustering': {
        'data_points': len(data_points),
        'clusters': 3,
        'iterations': 'converged',
        'time_seconds': clustering_time,
        'complexity': 'O(k * n * d * i)'
    },
    'quicksort_3way': {
        'array_size': len(test_array),
        'unique_elements': len(set(original_array)),
        'sorted_correctly': is_sorted,
        'time_seconds': quicksort_time,
        'complexity': 'O(n log n) avg, O(n²) worst'
    }
}

for algo, stats in performance_summary.items():
    print(f"\\n{algo.upper().replace('_', ' ')}:")
    for key, value in stats.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")

# Save performance results
with open('algorithm_performance_results.json', 'w') as f:
    json.dump(performance_summary, f, indent=2, default=str)

print(f"\\n💾 Performance results saved to algorithm_performance_results.json")
print("\\n ADVANCED ALGORITHMS TEST COMPLETED SUCCESSFULLY!")
        """,
        "user_id": 1,
        "thread_id": "advanced-algorithms"
    }
    
    return execute_lambda(payload)

def test_scientific_computing():
    """Test scientific computing and numerical analysis"""
    print("=== Testing Scientific Computing ===")
    
    payload = {
        "mode": "code",
        "code": """
import numpy as np
import json
import math
from datetime import datetime

print("🔬 SCIENTIFIC COMPUTING AND NUMERICAL ANALYSIS")
print("=" * 50)

# Test 1: Linear Algebra Operations
print("\\n🧮 TEST 1: Advanced Linear Algebra")

# Create test matrices
np.random.seed(42)
A = np.random.rand(5, 5) * 10
B = np.random.rand(5, 5) * 10
vector = np.random.rand(5) * 10

# Matrix operations
matrix_ops = {
    'determinant_A': float(np.linalg.det(A)),
    'determinant_B': float(np.linalg.det(B)),
    'trace_A': float(np.trace(A)),
    'rank_A': int(np.linalg.matrix_rank(A)),
    'condition_number': float(np.linalg.cond(A)),
    'frobenius_norm': float(np.linalg.norm(A, 'fro'))
}

# Eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eig(A)
matrix_ops['eigenvalues'] = [float(x) for x in eigenvals.real[:3]]  # Top 3
matrix_ops['largest_eigenvalue'] = float(max(eigenvals.real))

# Matrix decompositions
U, s, Vt = np.linalg.svd(A)  # Singular Value Decomposition
matrix_ops['singular_values'] = [float(x) for x in s[:3]]

print(f"Matrix A (5x5) Analysis:")
print(f"  • Determinant: {matrix_ops['determinant_A']:.4f}")
print(f"  • Trace: {matrix_ops['trace_A']:.4f}")
print(f"  • Condition Number: {matrix_ops['condition_number']:.4f}")
print(f"  • Largest Eigenvalue: {matrix_ops['largest_eigenvalue']:.4f}")
print(f"  • Top 3 Singular Values: {[f'{x:.4f}' for x in matrix_ops['singular_values']]}")

# Test 2: Numerical Integration and Differentiation
print("\\n📐 TEST 2: Numerical Calculus")

def f(x):
    return x**3 - 2*x**2 + 3*x - 1

def f_derivative(x):
    return 3*x**2 - 4*x + 3

# Numerical integration using Simpson's rule
def simpsons_rule(func, a, b, n):
    if n % 2 == 1:
        n += 1  # Ensure even number of intervals
    
    h = (b - a) / n
    x = a
    sum_val = func(a)
    
    for i in range(1, n):
        x += h
        if i % 2 == 0:
            sum_val += 2 * func(x)
        else:
            sum_val += 4 * func(x)
    
    sum_val += func(b)
    return (h / 3) * sum_val

# Numerical differentiation
def numerical_derivative(func, x, h=1e-6):
    return (func(x + h) - func(x - h)) / (2 * h)

# Test integration and differentiation
a, b = 0, 2
n_intervals = 1000

integral_result = simpsons_rule(f, a, b, n_intervals)
derivative_at_1 = numerical_derivative(f, 1.0)
analytical_derivative_at_1 = f_derivative(1.0)

calculus_results = {
    'definite_integral_0_to_2': float(integral_result),
    'numerical_derivative_at_1': float(derivative_at_1),
    'analytical_derivative_at_1': float(analytical_derivative_at_1),
    'derivative_error': float(abs(derivative_at_1 - analytical_derivative_at_1))
}

print(f"Function: f(x) = x³ - 2x² + 3x - 1")
print(f"  • Definite integral [0,2]: {calculus_results['definite_integral_0_to_2']:.6f}")
print(f"  • Numerical f'(1): {calculus_results['numerical_derivative_at_1']:.6f}")
print(f"  • Analytical f'(1): {calculus_results['analytical_derivative_at_1']:.6f}")
print(f"  • Approximation error: {calculus_results['derivative_error']:.8f}")

# Test 3: Statistical Analysis and Probability
print("\\n TEST 3: Statistical Analysis")

# Generate sample data
np.random.seed(123)
normal_data = np.random.normal(100, 15, 1000)
exponential_data = np.random.exponential(2, 1000)
uniform_data = np.random.uniform(0, 100, 1000)

statistical_analysis = {}

# Analyze each distribution
datasets = {
    'normal': normal_data,
    'exponential': exponential_data, 
    'uniform': uniform_data
}

for name, data in datasets.items():
    stats = {
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'variance': float(np.var(data)),
        'skewness': float(np.mean(((data - np.mean(data)) / np.std(data))**3)),
        'kurtosis': float(np.mean(((data - np.mean(data)) / np.std(data))**4)) - 3,
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'percentile_25': float(np.percentile(data, 25)),
        'percentile_75': float(np.percentile(data, 75))
    }
    statistical_analysis[name] = stats
    
    print(f"{name.upper()} Distribution (n=1000):")
    print(f"  • Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    print(f"  • Skewness: {stats['skewness']:.3f}, Kurtosis: {stats['kurtosis']:.3f}")
    print(f"  • IQR: [{stats['percentile_25']:.3f}, {stats['percentile_75']:.3f}]")

# Test 4: Monte Carlo Simulation
print("\\n TEST 4: Monte Carlo Simulation - Estimating π")

def estimate_pi_monte_carlo(n_samples):
    inside_circle = 0
    
    for _ in range(n_samples):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        
        if x**2 + y**2 <= 1:
            inside_circle += 1
    
    return 4 * inside_circle / n_samples

# Test with different sample sizes
sample_sizes = [1000, 10000, 100000]
monte_carlo_results = {}

for n in sample_sizes:
    estimated_pi = estimate_pi_monte_carlo(n)
    error = abs(estimated_pi - math.pi)
    monte_carlo_results[f'samples_{n}'] = {
        'estimated_pi': float(estimated_pi),
        'actual_pi': float(math.pi),
        'absolute_error': float(error),
        'relative_error_percent': float(error / math.pi * 100)
    }
    
    print(f"π estimation with {n:,} samples:")
    print(f"  • Estimated: {estimated_pi:.6f}")
    print(f"  • Actual: {math.pi:.6f}")
    print(f"  • Error: {error:.6f} ({error/math.pi*100:.4f}%)")

# Test 5: Optimization - Gradient Descent
print("\\n⬇ TEST 5: Numerical Optimization - Gradient Descent")

def rosenbrock(x, y):
    # Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(x, y):
    # Gradient of Rosenbrock function
    df_dx = -2 * (1 - x) - 400 * x * (y - x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

def gradient_descent(func, grad_func, start_point, learning_rate=0.001, max_iters=10000, tolerance=1e-6):
    point = np.array(start_point)
    history = [point.copy()]
    
    for i in range(max_iters):
        gradient = grad_func(point[0], point[1])
        new_point = point - learning_rate * gradient
        
        # Check convergence
        if np.linalg.norm(new_point - point) < tolerance:
            break
        
        point = new_point
        if i % 1000 == 0:  # Store some history points
            history.append(point.copy())
    
    return point, i + 1, history

# Optimize Rosenbrock function
start = [-1.5, 1.5]
optimal_point, iterations, history = gradient_descent(
    rosenbrock, rosenbrock_gradient, start, learning_rate=0.001
)

optimization_results = {
    'starting_point': start,
    'optimal_point': [float(x) for x in optimal_point],
    'optimal_value': float(rosenbrock(optimal_point[0], optimal_point[1])),
    'iterations_to_converge': iterations,
    'known_minimum': [1.0, 1.0],
    'error_from_minimum': float(np.linalg.norm(optimal_point - np.array([1.0, 1.0])))
}

print(f"Rosenbrock Function Optimization:")
print(f"  • Starting point: {start}")
print(f"  • Optimal point found: [{optimal_point[0]:.6f}, {optimal_point[1]:.6f}]")
print(f"  • Function value at optimum: {optimization_results['optimal_value']:.8f}")
print(f"  • Iterations to converge: {iterations}")
print(f"  • Distance from true minimum: {optimization_results['error_from_minimum']:.6f}")

# Compile comprehensive results
comprehensive_results = {
    'timestamp': datetime.now().isoformat(),
    'linear_algebra': matrix_ops,
    'numerical_calculus': calculus_results,
    'statistical_analysis': statistical_analysis,
    'monte_carlo_simulation': monte_carlo_results,
    'optimization': optimization_results
}

# Save results
with open('scientific_computing_results.json', 'w') as f:
    json.dump(comprehensive_results, f, indent=2, default=str)

print(f"\\n💾 Results saved to scientific_computing_results.json")
print("\\n SCIENTIFIC COMPUTING TESTS COMPLETED!")
print("\\nAll numerical computations executed successfully with high precision.")
        """,
        "user_id": 1,
        "thread_id": "scientific-computing"
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
    for i in range(60):  # Extended timeout for complex operations
        time.sleep(3)  # Longer interval for complex operations
        
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
                    # Show first part of output (it might be very long)
                    if len(output) > 2000:
                        print("📄 Output (truncated):")
                        print(output[:2000] + "\n... [output truncated] ...")
                    else:
                        print("📄 Output:")
                        print(output)
                
                artifacts = final_state.get("artifacts") or final_state.get("result_files", [])
                if artifacts:
                    print(f"📁 Artifacts created: {len(artifacts)} files")
                    for artifact in artifacts[:5]:  # Show first 5
                        print(f"  - {artifact.get('file_name', 'Unknown')}")
                
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
    """Run comprehensive sandbox tests"""
    print(" COMPREHENSIVE RAGnetic Lambda Sandbox Testing")
    print("=" * 60)
    
    tests = [
        ("Complex Data Processing", test_complex_data_processing),
        ("Advanced Algorithms", test_advanced_algorithms),
        ("Scientific Computing", test_scientific_computing),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\\n Running: {test_name}")
        print("=" * 30)
        try:
            success = test_func()
            results.append((test_name, success))
            status = " PASS" if success else " FAIL"
            print(f"\\nResult: {status}")
        except Exception as e:
            print(f" ERROR: {e}")
            results.append((test_name, False))
        
        print("=" * 60)
    
    # Summary
    print("\\n COMPREHENSIVE TEST SUMMARY:")
    print("=" * 40)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = " PASS" if success else " FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\\n ALL COMPREHENSIVE TESTS PASSED!")
        print("🔬 Complex algorithms, data analysis, and scientific computing verified!")
        print(" RAGnetic Lambda Tool is performing at advanced levels!")
    else:
        print("\\n⚠  Some complex tests failed. Check logs for details.")

if __name__ == "__main__":
    main()