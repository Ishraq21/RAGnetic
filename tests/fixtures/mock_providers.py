# Mock GPU Providers for Testing
# Comprehensive mocks for all GPU providers to avoid real API calls and costs

import pytest
from unittest.mock import Mock, patch, MagicMock
from httpx import Response, Request, MockTransport
import json
import time
from typing import Dict, Any

class MockRunPodTransport(MockTransport):
    """Mock transport for RunPod API calls."""
    
    def __init__(self):
        self.pods = {}
        self.call_count = 0
        super().__init__(self._handler)
    
    def _handler(self, request: Request) -> Response:
        self.call_count += 1
        path = request.url.path
        method = request.method
        
        # Create pod
        if path.endswith("/pods") and method == "POST":
            body = json.loads(request.content)
            pod_id = f"pod-{self.call_count}"
            
            self.pods[pod_id] = {
                "id": pod_id,
                "name": body.get("name", "test-pod"),
                "status": "RUNNING",
                "gpu_type": body.get("gpu_type", "RTX4090"),
                "price": 2.0,
                "created_at": time.time()
            }
            
            return Response(200, json=self.pods[pod_id])
        
        # Get pod status
        elif "/pods/" in path and method == "GET":
            pod_id = path.split("/pods/")[1].split("/")[0]
            if pod_id in self.pods:
                return Response(200, json=self.pods[pod_id])
            return Response(404, json={"error": "Pod not found"})
        
        # Stop pod
        elif "/pods/" in path and "stop" in path and method == "POST":
            pod_id = path.split("/pods/")[1].split("/")[0]
            if pod_id in self.pods:
                self.pods[pod_id]["status"] = "STOPPED"
                return Response(200, json={"message": "Pod stopped"})
            return Response(404, json={"error": "Pod not found"})
        
        # Get pod logs
        elif "/pods/" in path and "logs" in path and method == "GET":
            pod_id = path.split("/pods/")[1].split("/")[0]
            if pod_id in self.pods:
                logs = f"""
Starting training for pod {pod_id}...
Loading model...
Training step 1/100: loss=0.5
Training step 2/100: loss=0.4
Training step 3/100: loss=0.3
Training completed successfully!
Model saved to /workspace/output/
"""
                return Response(200, text=logs.strip())
            return Response(404, json={"error": "Pod not found"})
        
        return Response(404, json={"error": "Endpoint not found"})

class MockCoreWeaveTransport(MockTransport):
    """Mock transport for CoreWeave API calls."""
    
    def __init__(self):
        self.instances = {}
        self.call_count = 0
        super().__init__(self._handler)
    
    def _handler(self, request: Request) -> Response:
        self.call_count += 1
        path = request.url.path
        method = request.method
        
        # Create instance
        if path.endswith("/instances") and method == "POST":
            body = json.loads(request.content)
            instance_id = f"cw-{self.call_count}"
            
            self.instances[instance_id] = {
                "id": instance_id,
                "name": body.get("name", "test-instance"),
                "status": "running",
                "gpu_type": body.get("gpu_type", "A100"),
                "price": 3.0,
                "created_at": time.time()
            }
            
            return Response(201, json=self.instances[instance_id])
        
        # Get instance
        elif "/instances/" in path and method == "GET":
            instance_id = path.split("/instances/")[1]
            if instance_id in self.instances:
                return Response(200, json=self.instances[instance_id])
            return Response(404, json={"error": "Instance not found"})
        
        # Delete instance
        elif "/instances/" in path and method == "DELETE":
            instance_id = path.split("/instances/")[1]
            if instance_id in self.instances:
                self.instances[instance_id]["status"] = "terminated"
                return Response(204)
            return Response(404, json={"error": "Instance not found"})
        
        return Response(404, json={"error": "Endpoint not found"})

class MockVastTransport(MockTransport):
    """Mock transport for Vast.ai API calls."""
    
    def __init__(self):
        self.instances = {}
        self.call_count = 0
        super().__init__(self._handler)
    
    def _handler(self, request: Request) -> Response:
        self.call_count += 1
        path = request.url.path
        method = request.method
        
        # Create instance
        if path.endswith("/instances/") and method == "PUT":
            body = json.loads(request.content)
            instance_id = f"vast-{self.call_count}"
            
            self.instances[instance_id] = {
                "id": instance_id,
                "status": "running",
                "gpu_name": body.get("gpu_name", "RTX3090"),
                "price": 1.5,
                "created_at": time.time()
            }
            
            return Response(200, json={"success": True, "new_contract": instance_id})
        
        # Get instance
        elif "/instances/" in path and method == "GET":
            instance_id = path.split("/instances/")[1].rstrip("/")
            if instance_id in self.instances:
                return Response(200, json=self.instances[instance_id])
            return Response(404, json={"error": "Instance not found"})
        
        # Destroy instance
        elif "/instances/" in path and method == "DELETE":
            instance_id = path.split("/instances/")[1].rstrip("/")
            if instance_id in self.instances:
                self.instances[instance_id]["status"] = "exited"
                return Response(200, json={"success": True})
            return Response(404, json={"error": "Instance not found"})
        
        return Response(404, json={"error": "Endpoint not found"})

@pytest.fixture
def mock_runpod_transport():
    """Fixture for RunPod mock transport."""
    return MockRunPodTransport()

@pytest.fixture
def mock_coreweave_transport():
    """Fixture for CoreWeave mock transport."""
    return MockCoreWeaveTransport()

@pytest.fixture
def mock_vast_transport():
    """Fixture for Vast.ai mock transport."""
    return MockVastTransport()

@pytest.fixture
def mock_all_providers(mock_runpod_transport, mock_coreweave_transport, mock_vast_transport):
    """Mock all GPU providers."""
    with patch('app.services.gpu_providers.runpod.httpx.Client') as mock_runpod_client, \
         patch('app.services.gpu_providers.coreweave.httpx.Client') as mock_coreweave_client, \
         patch('app.services.gpu_providers.vast.httpx.Client') as mock_vast_client:
        
        # Configure RunPod
        mock_runpod_instance = Mock()
        mock_runpod_client.return_value = mock_runpod_instance
        mock_runpod_instance.request.side_effect = mock_runpod_transport.handle_request
        
        # Configure CoreWeave
        mock_coreweave_instance = Mock()
        mock_coreweave_client.return_value = mock_coreweave_instance
        mock_coreweave_instance.request.side_effect = mock_coreweave_transport.handle_request
        
        # Configure Vast.ai
        mock_vast_instance = Mock()
        mock_vast_client.return_value = mock_vast_instance
        mock_vast_instance.request.side_effect = mock_vast_transport.handle_request
        
        yield {
            "runpod": mock_runpod_instance,
            "coreweave": mock_coreweave_instance,
            "vast": mock_vast_instance
        }

@pytest.fixture
def mock_provider_failures():
    """Mock provider failures for testing error handling."""
    def _make_failing_provider(failure_type="timeout"):
        mock_client = Mock()
        
        if failure_type == "timeout":
            mock_client.request.side_effect = TimeoutError("Request timed out")
        elif failure_type == "500":
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"error": "Internal server error"}
            mock_client.request.return_value = mock_response
        elif failure_type == "rate_limit":
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.json.return_value = {"error": "Rate limit exceeded"}
            mock_response.headers = {"Retry-After": "60"}
            mock_client.request.return_value = mock_response
        elif failure_type == "auth":
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.json.return_value = {"error": "Unauthorized"}
            mock_client.request.return_value = mock_response
        
        return mock_client
    
    return _make_failing_provider

@pytest.fixture
def mock_provider_with_preemption():
    """Mock provider that simulates spot instance preemption."""
    def _preemptible_provider():
        mock_client = Mock()
        call_count = 0
        
        def _request_handler(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # First few calls succeed
            if call_count <= 3:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "id": "preempt-pod-123",
                    "status": "RUNNING"
                }
                return mock_response
            
            # Then simulate preemption
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "preempt-pod-123",
                "status": "INTERRUPTED",
                "reason": "Spot instance preempted"
            }
            return mock_response
        
        mock_client.request.side_effect = _request_handler
        return mock_client
    
    return _preemptible_provider

@pytest.fixture
def mock_slow_provider():
    """Mock provider that responds slowly for performance testing."""
    def _slow_provider(delay_seconds=2.0):
        mock_client = Mock()
        
        def _slow_request(*args, **kwargs):
            time.sleep(delay_seconds)
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": "slow-pod-123", "status": "RUNNING"}
            return mock_response
        
        mock_client.request.side_effect = _slow_request
        return mock_client
    
    return _slow_provider

# Utility functions for test assertions
def assert_provider_called_correctly(mock_client, expected_calls=1):
    """Assert that provider was called the expected number of times."""
    assert mock_client.request.call_count == expected_calls

def assert_provider_called_with_auth(mock_client, expected_api_key):
    """Assert that provider was called with correct authentication."""
    calls = mock_client.request.call_args_list
    for call in calls:
        args, kwargs = call
        headers = kwargs.get('headers', {})
        auth_header = headers.get('Authorization', '')
        assert expected_api_key in auth_header or 'Bearer' in auth_header

def get_provider_call_history(mock_client):
    """Get history of all calls made to provider."""
    return [
        {
            "args": call[0],
            "kwargs": call[1],
            "method": call[1].get("method", "GET"),
            "url": str(call[0][0]) if call[0] else None
        }
        for call in mock_client.request.call_args_list
    ]
