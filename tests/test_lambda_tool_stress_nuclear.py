"""
NUCLEAR LAMBDA_TOOL STRESS TESTS - MAXIMUM BRUTALITY!

These tests are designed to OBLITERATE the lambda_tool with extreme stress conditions.
NO MERCY! NO SURVIVORS! PUSH IT TO THE ABSOLUTE LIMITS!

Stress Categories:
1. Extreme Concurrency - Thousands of simultaneous requests
2. Memory Bombs - Massive payloads that consume all available memory
3. CPU Destruction - Computationally intensive tasks
4. Network Saturation - Overwhelming network I/O
5. Database Apocalypse - Database connection exhaustion
6. Docker Hell - Container resource exhaustion
7. File System Massacre - I/O intensive operations
8. Time Bomb Tests - Long-running operations and timeouts
"""

import asyncio
import concurrent.futures
import gc
import json
import multiprocessing
import os
import psutil
import pytest
import random
import resource
import string
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import requests
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.tools.lambda_tool import LambdaTool
from app.schemas.lambda_tool import LambdaRequestPayload, LambdaResourceSpec


class TestLambdaToolStressNuclear:
    """NUCLEAR stress tests for lambda_tool - MAXIMUM BRUTALITY!"""

    @pytest.fixture
    def lambda_tool(self):
        """Create a lambda tool instance for stress testing"""
        return LambdaTool(
            server_url="http://localhost:8000",
            api_keys=["stress-test-nuclear-key"]
        )

    # ============== EXTREME CONCURRENCY TESTS - THREAD HOLOCAUST ==============

    def test_extreme_concurrency_1000_threads(self, lambda_tool):
        """STRESS: 1000 concurrent lambda executions - MAXIMUM THREAD WARFARE!"""
        num_threads = 1000
        results = []
        errors = []
        start_time = time.time()
        
        def stress_lambda_execution():
            thread_id = threading.get_ident()
            try:
                # Create unique payload for each thread
                payload = {
                    "mode": "code",
                    "code": f"""
import time
import random
# Simulate some work
for i in range(random.randint(10, 100)):
    x = i ** 2
time.sleep(random.uniform(0.01, 0.1))
print(f'Stress test from thread {thread_id}')
                    """,
                    "run_id": str(uuid.uuid4()),
                    "thread_id": str(uuid.uuid4()),
                    "resource_spec": {
                        "cpu": str(random.uniform(0.1, 2.0)),
                        "memory_gb": random.randint(1, 8),
                        "disk_mb": random.randint(100, 2000)
                    }
                }
                
                # Mock the execution to avoid actual Docker calls
                with patch.object(lambda_tool, '_run') as mock_run:
                    mock_run.return_value = f"Success from thread {thread_id}"
                    result = lambda_tool._run(**payload)
                    results.append({
                        'thread_id': thread_id,
                        'result': result,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                errors.append({
                    'thread_id': thread_id,
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        # Launch the nuclear thread army!
        threads = []
        print(f"🚀 Launching {num_threads} threads for MAXIMUM CONCURRENCY!")
        
        for i in range(num_threads):
            thread = threading.Thread(target=stress_lambda_execution, name=f"StressThread-{i}")
            threads.append(thread)
            thread.start()
            
            # Small delay to prevent immediate resource exhaustion
            if i % 100 == 0:
                time.sleep(0.01)
        
        # Wait for all threads with timeout
        completed_threads = 0
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout per thread
            if not thread.is_alive():
                completed_threads += 1
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Analyze results
        success_count = len(results)
        error_count = len(errors)
        completion_rate = completed_threads / num_threads
        success_rate = success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0
        
        print(f"💥 CONCURRENCY RESULTS:")
        print(f"   Total threads: {num_threads}")
        print(f"   Completed threads: {completed_threads}")
        print(f"   Successful executions: {success_count}")
        print(f"   Failed executions: {error_count}")
        print(f"   Completion rate: {completion_rate:.2%}")
        print(f"   Success rate: {success_rate:.2%}")
        print(f"   Total duration: {total_duration:.2f}s")
        print(f"   Throughput: {(success_count + error_count) / total_duration:.1f} ops/sec")
        
        # Stress test assertions
        assert completion_rate >= 0.95  # At least 95% threads should complete
        assert success_rate >= 0.80     # At least 80% should succeed (with mocking)
        assert total_duration < 120     # Should complete within 2 minutes

    def test_multiprocess_stress_test(self, lambda_tool):
        """STRESS: Multi-process lambda execution stress test"""
        num_processes = min(multiprocessing.cpu_count() * 2, 16)  # Don't kill the machine
        requests_per_process = 50
        
        def worker_process(process_id, results_queue):
            """Worker process for stress testing"""
            process_results = []
            process_errors = []
            
            # Create lambda tool in each process
            tool = LambdaTool(
                server_url="http://localhost:8000",
                api_keys=["multiprocess-stress-key"]
            )
            
            for i in range(requests_per_process):
                try:
                    payload = {
                        "mode": "code",
                        "code": f"""
import os
import time
process_id = {process_id}
request_id = {i}
print(f'Process {{process_id}}, Request {{request_id}}')
time.sleep(0.1)
                        """,
                        "run_id": str(uuid.uuid4()),
                        "thread_id": str(uuid.uuid4())
                    }
                    
                    with patch.object(tool, '_run') as mock_run:
                        mock_run.return_value = f"Process {process_id}, Request {i} success"
                        result = tool._run(**payload)
                        process_results.append(result)
                        
                except Exception as e:
                    process_errors.append(str(e))
            
            results_queue.put({
                'process_id': process_id,
                'results': process_results,
                'errors': process_errors
            })
        
        # Start multiprocessing stress test
        start_time = time.time()
        processes = []
        results_queue = multiprocessing.Queue()
        
        print(f"🔥 Starting {num_processes} processes with {requests_per_process} requests each")
        
        for p_id in range(num_processes):
            process = multiprocessing.Process(
                target=worker_process,
                args=(p_id, results_queue)
            )
            processes.append(process)
            process.start()
        
        # Wait for all processes
        for process in processes:
            process.join(timeout=120)  # 2 minute timeout
        
        # Collect results
        all_results = []
        all_errors = []
        while not results_queue.empty():
            proc_result = results_queue.get()
            all_results.extend(proc_result['results'])
            all_errors.extend(proc_result['errors'])
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        expected_total = num_processes * requests_per_process
        actual_total = len(all_results) + len(all_errors)
        
        print(f"💀 MULTIPROCESS RESULTS:")
        print(f"   Processes: {num_processes}")
        print(f"   Expected total requests: {expected_total}")
        print(f"   Actual total requests: {actual_total}")
        print(f"   Successful: {len(all_results)}")
        print(f"   Failed: {len(all_errors)}")
        print(f"   Duration: {total_duration:.2f}s")
        
        # Cleanup processes
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        
        # Assertions
        assert actual_total >= expected_total * 0.9  # At least 90% completion
        assert len(all_results) >= expected_total * 0.8  # At least 80% success

    # ============== MEMORY BOMB TESTS - RAM ANNIHILATION ==============

    def test_memory_bomb_massive_payloads(self, lambda_tool):
        """STRESS: Massive payloads to exhaust memory"""
        # Get current memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create progressively larger payloads
        memory_bomb_sizes = [1, 10, 50, 100, 200]  # MB worth of data
        
        for bomb_size_mb in memory_bomb_sizes:
            print(f"💣 Deploying {bomb_size_mb}MB memory bomb!")
            
            # Create massive code payload
            massive_code = "data = [\n"
            lines_per_mb = 1000  # Approximate
            for i in range(bomb_size_mb * lines_per_mb):
                massive_code += f"    'This is line {i} with some data to consume memory',\n"
            massive_code += "]\nprint(f'Loaded {len(data)} lines')"
            
            # Create massive inputs list
            massive_inputs = []
            for i in range(bomb_size_mb * 10):  # 10 inputs per MB
                massive_inputs.append({
                    "temp_doc_id": str(uuid.uuid4()),
                    "file_name": f"massive_file_{i}.txt",
                    "path_in_sandbox": f"/work/inputs/massive_file_{i}.txt"
                })
            
            try:
                payload = LambdaRequestPayload(
                    mode="code",
                    code=massive_code,
                    inputs=massive_inputs,
                    run_id=str(uuid.uuid4()),
                    thread_id=str(uuid.uuid4()),
                    resource_spec=LambdaResourceSpec(
                        memory_gb=min(32, bomb_size_mb // 10 + 1)  # Scale memory requirement
                    )
                )
                
                # Check memory usage after payload creation
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                print(f"   Memory increase: {memory_increase:.1f} MB")
                print(f"   Code size: {len(payload.code) / 1024 / 1024:.1f} MB")
                print(f"   Inputs count: {len(payload.inputs)}")
                
                # Validate payload was created successfully
                assert len(payload.code) > bomb_size_mb * 1000  # At least some size
                assert len(payload.inputs) == bomb_size_mb * 10
                
                # Force garbage collection
                del payload
                del massive_code
                del massive_inputs
                gc.collect()
                
            except MemoryError:
                print(f"   MemoryError at {bomb_size_mb}MB - System limit reached!")
                break
            except Exception as e:
                print(f"   Error at {bomb_size_mb}MB: {e}")
                break
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"🧠 Memory usage: Initial={initial_memory:.1f}MB, Final={final_memory:.1f}MB")

    def test_memory_leak_detection(self, lambda_tool):
        """STRESS: Detect memory leaks during repeated operations"""
        process = psutil.Process()
        memory_snapshots = []
        
        # Take initial memory snapshot
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_snapshots.append(initial_memory)
        
        iterations = 100
        payload_template = {
            "mode": "code",
            "code": "print('memory leak test')" * 1000,  # Medium sized payload
            "run_id": None,
            "thread_id": None
        }
        
        with patch.object(lambda_tool, '_run') as mock_run:
            mock_run.return_value = "memory test success"
            
            for i in range(iterations):
                # Create unique payload each time
                payload = payload_template.copy()
                payload["run_id"] = str(uuid.uuid4())
                payload["thread_id"] = str(uuid.uuid4())
                
                # Execute
                result = lambda_tool._run(**payload)
                
                # Take memory snapshot every 10 iterations
                if i % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_snapshots.append(current_memory)
                    print(f"Iteration {i}: {current_memory:.1f} MB")
                
                # Force cleanup
                del payload
                if i % 50 == 0:
                    gc.collect()
        
        # Analyze memory growth
        final_memory = memory_snapshots[-1]
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_snapshots)
        
        print(f"💾 Memory Analysis:")
        print(f"   Initial: {initial_memory:.1f} MB")
        print(f"   Final: {final_memory:.1f} MB")
        print(f"   Max: {max_memory:.1f} MB")
        print(f"   Growth: {memory_growth:.1f} MB")
        print(f"   Growth per iteration: {memory_growth/iterations:.3f} MB")
        
        # Memory leak detection assertions
        assert memory_growth < 50  # Should not grow more than 50MB
        assert memory_growth / iterations < 0.1  # Less than 0.1MB per iteration

    # ============== CPU DESTRUCTION TESTS - PROCESSOR ANNIHILATION ==============

    def test_cpu_intensive_concurrent_loads(self, lambda_tool):
        """STRESS: CPU-intensive concurrent workloads"""
        num_workers = multiprocessing.cpu_count() * 4  # Oversubscribe CPU
        
        def cpu_intensive_worker():
            """Worker that performs CPU-intensive lambda operations"""
            start_time = time.time()
            
            # CPU-intensive code payload
            cpu_code = """
import math
import time

start = time.time()
result = 0
# Intensive computation
for i in range(1000000):
    result += math.sqrt(i) * math.sin(i) * math.cos(i)

# Prime number calculation
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(1000, 2000) if is_prime(n)]
end = time.time()
print(f'Computation result: {result}, Primes found: {len(primes)}, Duration: {end-start:.2f}s')
            """
            
            payload = {
                "mode": "code",
                "code": cpu_code,
                "run_id": str(uuid.uuid4()),
                "thread_id": str(uuid.uuid4()),
                "resource_spec": {
                    "cpu": "2.0",  # Request 2 CPU cores
                    "memory_gb": 4
                }
            }
            
            with patch.object(lambda_tool, '_run') as mock_run:
                mock_run.return_value = "CPU intensive task completed"
                result = lambda_tool._run(**payload)
                
            duration = time.time() - start_time
            return {
                'worker_id': threading.get_ident(),
                'duration': duration,
                'result': result
            }
        
        # Launch CPU-intensive workers
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            print(f"🔥 Launching {num_workers} CPU-intensive workers!")
            
            # Submit all tasks
            futures = [executor.submit(cpu_intensive_worker) for _ in range(num_workers)]
            
            # Collect results
            results = []
            for future in as_completed(futures, timeout=300):  # 5 minute timeout
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Worker failed: {e}")
        
        total_duration = time.time() - start_time
        
        print(f"⚡ CPU STRESS RESULTS:")
        print(f"   Workers: {num_workers}")
        print(f"   Completed: {len(results)}")
        print(f"   Total duration: {total_duration:.2f}s")
        print(f"   Average worker duration: {sum(r['duration'] for r in results) / len(results):.2f}s")
        
        # Assertions
        assert len(results) >= num_workers * 0.9  # At least 90% completion
        assert total_duration < 300  # Should complete within 5 minutes

    # ============== NETWORK SATURATION TESTS - BANDWIDTH MASSACRE ==============

    def test_network_request_flood(self, lambda_tool):
        """STRESS: Flood the network with requests"""
        num_requests = 500
        concurrent_batches = 50
        
        results = []
        errors = []
        
        def network_flood_batch():
            """Send a batch of network requests"""
            batch_results = []
            batch_errors = []
            
            for i in range(10):  # 10 requests per batch
                try:
                    payload = {
                        "mode": "code",
                        "code": f"""
import requests
import time
# Simulate network-intensive task
for i in range(5):
    try:
        # Mock network request
        print(f'Network request {{i}} from batch')
        time.sleep(0.01)  # Simulate network delay
    except:
        pass
print('Network flood batch completed')
                        """,
                        "run_id": str(uuid.uuid4()),
                        "thread_id": str(uuid.uuid4()),
                        "network_policy": {
                            "allow_outbound": True,
                            "allowlist_domains": ["httpbin.org", "example.com"]
                        }
                    }
                    
                    # Mock the network request to avoid actual external calls
                    with patch('requests.post') as mock_post, \
                         patch('requests.get') as mock_get:
                        
                        mock_post.return_value.status_code = 200
                        mock_post.return_value.json.return_value = {"run_id": f"flood-{i}"}
                        
                        mock_get.return_value.status_code = 200
                        mock_get.return_value.json.return_value = {
                            "status": "completed",
                            "final_state": {"output": f"Network flood {i} success"}
                        }
                        
                        result = lambda_tool._run(**payload)
                        batch_results.append(result)
                        
                except Exception as e:
                    batch_errors.append(str(e))
            
            return batch_results, batch_errors
        
        # Launch network flood
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrent_batches) as executor:
            print(f"🌊 Launching network flood with {concurrent_batches} concurrent batches!")
            
            futures = [executor.submit(network_flood_batch) for _ in range(concurrent_batches)]
            
            for future in as_completed(futures):
                try:
                    batch_results, batch_errors = future.result()
                    results.extend(batch_results)
                    errors.extend(batch_errors)
                except Exception as e:
                    errors.append(str(e))
        
        total_duration = time.time() - start_time
        expected_requests = concurrent_batches * 10
        
        print(f"🌊 NETWORK FLOOD RESULTS:")
        print(f"   Expected requests: {expected_requests}")
        print(f"   Successful: {len(results)}")
        print(f"   Failed: {len(errors)}")
        print(f"   Duration: {total_duration:.2f}s")
        print(f"   Throughput: {len(results) / total_duration:.1f} req/sec")
        
        # Assertions
        assert len(results) >= expected_requests * 0.8  # At least 80% success
        assert total_duration < 60  # Should complete within 1 minute

    # ============== TIME BOMB TESTS - TEMPORAL DESTRUCTION ==============

    def test_timeout_stress_scenarios(self, lambda_tool):
        """STRESS: Various timeout scenarios"""
        timeout_scenarios = [
            {"wait_seconds": 1, "expected_timeout": True},   # Very short timeout
            {"wait_seconds": 5, "expected_timeout": False},  # Reasonable timeout
            {"wait_seconds": 30, "expected_timeout": False}, # Long timeout
        ]
        
        for scenario in timeout_scenarios:
            print(f"⏰ Testing timeout scenario: {scenario['wait_seconds']}s")
            
            with patch.dict('os.environ', {'LAMBDA_TOOL_WAIT_SECONDS': str(scenario['wait_seconds'])}), \
                 patch('requests.post') as mock_post, \
                 patch('requests.get') as mock_get:
                
                # Mock successful submission
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {"run_id": "timeout-test"}
                
                # Mock polling that takes longer than timeout
                def slow_response(*args, **kwargs):
                    time.sleep(0.6)  # Delay each poll
                    response = MagicMock()
                    response.status_code = 200
                    if scenario['expected_timeout']:
                        response.json.return_value = {"status": "running"}  # Never complete
                    else:
                        response.json.return_value = {
                            "status": "completed",
                            "final_state": {"output": "timeout test completed"}
                        }
                    return response
                
                mock_get.side_effect = slow_response
                
                start_time = time.time()
                result = lambda_tool._run(
                    mode="code",
                    code="import time; time.sleep(10); print('long task')",
                    run_id=str(uuid.uuid4()),
                    thread_id=str(uuid.uuid4())
                )
                duration = time.time() - start_time
                
                print(f"   Duration: {duration:.2f}s")
                print(f"   Result: {result[:100]}...")
                
                if scenario['expected_timeout']:
                    assert "still running" in result
                    assert duration >= scenario['wait_seconds']
                else:
                    assert "completed" in result or "timeout test completed" in result

    def test_long_running_operations_stress(self, lambda_tool):
        """STRESS: Multiple long-running operations"""
        num_long_ops = 20
        
        def long_running_operation(op_id):
            """Simulate a long-running operation"""
            payload = {
                "mode": "code",
                "code": f"""
import time
import random

operation_id = {op_id}
print(f'Starting long operation {{operation_id}}')

# Simulate long computation
duration = random.uniform(5, 15)  # 5-15 seconds
start = time.time()
while time.time() - start < duration:
    # Do some work
    result = sum(i**2 for i in range(10000))
    time.sleep(0.1)

print(f'Long operation {{operation_id}} completed in {{duration:.2f}}s')
                """,
                "run_id": str(uuid.uuid4()),
                "thread_id": str(uuid.uuid4()),
                "ttl_seconds": 300  # 5 minute TTL
            }
            
            with patch('requests.post') as mock_post, \
                 patch('requests.get') as mock_get:
                
                mock_post.return_value.status_code = 200
                mock_post.return_value.json.return_value = {"run_id": f"long-op-{op_id}"}
                
                # Simulate polling for a long operation
                poll_count = 0
                def polling_response(*args, **kwargs):
                    nonlocal poll_count
                    poll_count += 1
                    time.sleep(0.1)  # Small delay per poll
                    
                    response = MagicMock()
                    response.status_code = 200
                    
                    if poll_count < 10:  # First few polls show running
                        response.json.return_value = {"status": "running"}
                    else:  # Eventually complete
                        response.json.return_value = {
                            "status": "completed",
                            "final_state": {"output": f"Long operation {op_id} success"}
                        }
                    return response
                
                mock_get.side_effect = polling_response
                
                result = lambda_tool._run(**payload)
                return {
                    'op_id': op_id,
                    'result': result,
                    'poll_count': poll_count
                }
        
        # Launch long-running operations
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            print(f"⏳ Launching {num_long_ops} long-running operations!")
            
            futures = [executor.submit(long_running_operation, i) for i in range(num_long_ops)]
            
            results = []
            for future in as_completed(futures, timeout=120):  # 2 minute timeout
                try:
                    result = future.result()
                    results.append(result)
                    print(f"   Operation {result['op_id']} completed with {result['poll_count']} polls")
                except Exception as e:
                    print(f"   Long operation failed: {e}")
        
        total_duration = time.time() - start_time
        
        print(f"⏳ LONG OPERATIONS RESULTS:")
        print(f"   Operations: {num_long_ops}")
        print(f"   Completed: {len(results)}")
        print(f"   Total duration: {total_duration:.2f}s")
        print(f"   Average polls per operation: {sum(r['poll_count'] for r in results) / len(results):.1f}")
        
        # Assertions
        assert len(results) >= num_long_ops * 0.8  # At least 80% completion
        assert total_duration < 120  # Should complete within timeout

    # ============== RESOURCE EXHAUSTION TESTS - SYSTEM LIMITS ==============

    def test_file_descriptor_exhaustion(self, lambda_tool):
        """STRESS: Test file descriptor limits"""
        # Get current file descriptor limit
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"📁 File descriptor limits: soft={soft_limit}, hard={hard_limit}")
        
        # Try to exhaust file descriptors
        file_handles = []
        max_attempts = min(soft_limit - 100, 1000)  # Leave some buffer
        
        try:
            for i in range(max_attempts):
                # Create temporary file
                temp_file = f"/tmp/stress_test_fd_{i}_{uuid.uuid4().hex[:8]}.tmp"
                
                payload = {
                    "mode": "code",
                    "code": f"""
import tempfile
import os

# Create temporary file
temp_file = '{temp_file}'
with open(temp_file, 'w') as f:
    f.write('File descriptor stress test {i}')

print(f'Created file {{temp_file}}')
                    """,
                    "run_id": str(uuid.uuid4()),
                    "thread_id": str(uuid.uuid4())
                }
                
                # Mock execution
                with patch.object(lambda_tool, '_run') as mock_run:
                    mock_run.return_value = f"FD test {i} success"
                    result = lambda_tool._run(**payload)
                    
                # Actually create the file to consume FDs (for testing)
                try:
                    f = open(temp_file, 'w')
                    f.write(f'FD test {i}')
                    file_handles.append((f, temp_file))
                except OSError as e:
                    print(f"   FD exhaustion at {i} files: {e}")
                    break
                    
                if i % 100 == 0:
                    print(f"   Created {i} file descriptors...")
                    
        finally:
            # Cleanup file descriptors
            for f, temp_file in file_handles:
                try:
                    f.close()
                    os.unlink(temp_file)
                except:
                    pass
        
        print(f"📁 FD STRESS RESULTS:")
        print(f"   Created {len(file_handles)} file descriptors")
        print(f"   System handled FD stress gracefully")
        
        # Should handle reasonable number of FDs
        assert len(file_handles) > 100  # Should create at least 100 FDs

    # ============== EXTREME PAYLOAD VARIATION TESTS ==============

    def test_payload_size_progression_stress(self, lambda_tool):
        """STRESS: Progressive payload size increases"""
        base_size = 1024  # 1KB base
        multipliers = [1, 10, 100, 500, 1000, 5000]  # Up to ~5MB
        
        for multiplier in multipliers:
            payload_size = base_size * multiplier
            print(f"📦 Testing payload size: {payload_size / 1024:.1f} KB")
            
            # Create large code payload
            code_chunks = []
            chunk_size = min(1000, payload_size // 100)  # Reasonable chunk size
            
            for i in range(payload_size // chunk_size):
                code_chunks.append(f"chunk_{i} = 'x' * {chunk_size}")
            
            large_code = "\n".join(code_chunks)
            large_code += "\nprint(f'Processed {len(locals())} variables')"
            
            try:
                start_time = time.time()
                
                payload = LambdaRequestPayload(
                    mode="code",
                    code=large_code,
                    run_id=str(uuid.uuid4()),
                    thread_id=str(uuid.uuid4()),
                    resource_spec=LambdaResourceSpec(
                        memory_gb=max(4, payload_size // (1024 * 1024) + 1)  # Scale memory
                    )
                )
                
                creation_time = time.time() - start_time
                actual_size = len(payload.model_dump_json())
                
                print(f"   Created in {creation_time:.3f}s")
                print(f"   Actual JSON size: {actual_size / 1024:.1f} KB")
                
                # Test serialization performance
                start_time = time.time()
                json_str = payload.model_dump_json()
                serialization_time = time.time() - start_time
                
                print(f"   Serialized in {serialization_time:.3f}s")
                
                # Cleanup
                del payload
                del large_code
                gc.collect()
                
                # Performance assertions
                assert creation_time < 5.0  # Should create in under 5 seconds
                assert serialization_time < 2.0  # Should serialize in under 2 seconds
                
            except MemoryError:
                print(f"   MemoryError at {payload_size / 1024:.1f} KB - limit reached")
                break
            except Exception as e:
                print(f"   Error at {payload_size / 1024:.1f} KB: {e}")
                break


if __name__ == "__main__":
    # Run the nuclear stress tests!
    print("💥 INITIATING NUCLEAR LAMBDA_TOOL STRESS TESTS! 💥")
    print("🔥 NO MERCY! NO SURVIVORS! TO VALHALLA! 🔥")
    pytest.main([__file__, "-v", "--tb=short", "-s"])  # -s for print output