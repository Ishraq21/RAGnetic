# Integration tests for GPU management API endpoints
import pytest
from httpx import AsyncClient
from unittest.mock import patch


class TestGPUAPI:
    """Test GPU management API endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_available_gpus(self, client: AsyncClient, test_user, mock_all_providers):
        """Test listing available GPU types and providers."""
        response = await client.get("/api/v1/gpu/available")
        
        assert response.status_code == 200
        result = response.json()
        assert "providers" in result
        assert len(result["providers"]) > 0
        
        # Check provider structure
        for provider in result["providers"]:
            assert "name" in provider
            assert "gpu_types" in provider
            assert "availability" in provider
    
    @pytest.mark.asyncio
    async def test_get_gpu_pricing(self, client: AsyncClient, test_user, mock_all_providers):
        """Test getting GPU pricing information."""
        response = await client.get("/api/v1/gpu/pricing")
        
        assert response.status_code == 200
        result = response.json()
        assert "pricing" in result
        
        # Check pricing structure
        for price_info in result["pricing"]:
            assert "provider" in price_info
            assert "gpu_type" in price_info
            assert "cost_per_hour" in price_info
            assert price_info["cost_per_hour"] > 0
    
    @pytest.mark.asyncio
    async def test_provision_gpu_instance(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test provisioning a GPU instance."""
        provision_data = {
            "gpu_type": "A100",
            "provider": "runpod",
            "max_hours": 2.0,
            "project_id": test_project["id"],
            "purpose": "training"
        }
        
        response = await client.post("/api/v1/gpu/provision", json=provision_data)
        
        assert response.status_code == 201
        result = response.json()
        assert "instance_id" in result
        assert "status" in result
        assert "estimated_cost" in result
        assert result["gpu_type"] == "A100"
        assert result["provider"] == "runpod"
    
    @pytest.mark.asyncio
    async def test_provision_gpu_insufficient_credits(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test GPU provisioning with insufficient credits."""
        # Don't provide test_credits fixture
        provision_data = {
            "gpu_type": "H100",
            "provider": "coreweave",
            "max_hours": 10.0,  # Expensive
            "project_id": test_project["id"],
            "purpose": "training"
        }
        
        response = await client.post("/api/v1/gpu/provision", json=provision_data)
        
        assert response.status_code == 400
        result = response.json()
        assert "insufficient credits" in result["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_provision_gpu_invalid_type(self, client: AsyncClient, test_user, test_project, test_credits):
        """Test GPU provisioning with invalid GPU type."""
        provision_data = {
            "gpu_type": "INVALID_GPU",
            "provider": "runpod",
            "max_hours": 1.0,
            "project_id": test_project["id"],
            "purpose": "training"
        }
        
        response = await client.post("/api/v1/gpu/provision", json=provision_data)
        
        assert response.status_code == 400
        result = response.json()
        assert "invalid" in result["detail"].lower() or "not found" in result["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_get_gpu_instance_status(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test getting GPU instance status."""
        # Provision instance first
        provision_data = {
            "gpu_type": "RTX4090",
            "provider": "runpod",
            "max_hours": 1.0,
            "project_id": test_project["id"],
            "purpose": "inference"
        }
        
        provision_response = await client.post("/api/v1/gpu/provision", json=provision_data)
        instance_id = provision_response.json()["instance_id"]
        
        # Get status
        response = await client.get(f"/api/v1/gpu/instances/{instance_id}")
        
        assert response.status_code == 200
        result = response.json()
        assert result["instance_id"] == instance_id
        assert "status" in result
        assert "uptime" in result
        assert "cost_so_far" in result
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_instance(self, client: AsyncClient, test_user):
        """Test getting status of non-existent instance."""
        response = await client.get("/api/v1/gpu/instances/nonexistent-instance")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_stop_gpu_instance(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test stopping a GPU instance."""
        # Provision instance first
        provision_data = {
            "gpu_type": "A100",
            "provider": "runpod",
            "max_hours": 1.0,
            "project_id": test_project["id"],
            "purpose": "training"
        }
        
        provision_response = await client.post("/api/v1/gpu/provision", json=provision_data)
        instance_id = provision_response.json()["instance_id"]
        
        # Stop instance
        response = await client.post(f"/api/v1/gpu/instances/{instance_id}/stop")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["stopping", "stopped"]
        assert "final_cost" in result
    
    @pytest.mark.asyncio
    async def test_list_user_gpu_instances(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test listing user's GPU instances."""
        # Provision multiple instances
        instance_configs = [
            {"gpu_type": "RTX4090", "provider": "runpod", "max_hours": 1.0},
            {"gpu_type": "A100", "provider": "coreweave", "max_hours": 2.0}
        ]
        
        for config in instance_configs:
            config["project_id"] = test_project["id"]
            config["purpose"] = "testing"
            await client.post("/api/v1/gpu/provision", json=config)
        
        # List instances
        response = await client.get("/api/v1/gpu/instances")
        
        assert response.status_code == 200
        result = response.json()
        assert "instances" in result
        assert len(result["instances"]) >= 2
        
        # Check instance structure
        for instance in result["instances"]:
            assert "instance_id" in instance
            assert "gpu_type" in instance
            assert "provider" in instance
            assert "status" in instance
    
    @pytest.mark.asyncio
    async def test_list_instances_with_filters(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test listing instances with filters."""
        # Provision instances with different providers
        await client.post("/api/v1/gpu/provision", json={
            "gpu_type": "A100",
            "provider": "runpod",
            "max_hours": 1.0,
            "project_id": test_project["id"],
            "purpose": "training"
        })
        
        await client.post("/api/v1/gpu/provision", json={
            "gpu_type": "A100",
            "provider": "coreweave",
            "max_hours": 1.0,
            "project_id": test_project["id"],
            "purpose": "inference"
        })
        
        # Filter by provider
        response = await client.get("/api/v1/gpu/instances?provider=runpod")
        
        assert response.status_code == 200
        result = response.json()
        
        # All returned instances should be from runpod
        for instance in result["instances"]:
            assert instance["provider"] == "runpod"
        
        # Filter by purpose
        response = await client.get("/api/v1/gpu/instances?purpose=training")
        
        assert response.status_code == 200
        result = response.json()
        
        # All returned instances should be for training
        for instance in result["instances"]:
            assert instance["purpose"] == "training"
    
    @pytest.mark.asyncio
    async def test_get_gpu_instance_logs(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test getting GPU instance logs."""
        # Provision instance
        provision_data = {
            "gpu_type": "A100",
            "provider": "runpod",
            "max_hours": 1.0,
            "project_id": test_project["id"],
            "purpose": "training"
        }
        
        provision_response = await client.post("/api/v1/gpu/provision", json=provision_data)
        instance_id = provision_response.json()["instance_id"]
        
        # Get logs
        response = await client.get(f"/api/v1/gpu/instances/{instance_id}/logs")
        
        assert response.status_code == 200
        # Should return logs as text
        assert isinstance(response.text, str)
        assert len(response.text) > 0
    
    @pytest.mark.asyncio
    async def test_get_gpu_usage_metrics(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test getting GPU usage metrics."""
        # Provision instance
        provision_data = {
            "gpu_type": "A100",
            "provider": "runpod",
            "max_hours": 1.0,
            "project_id": test_project["id"],
            "purpose": "training"
        }
        
        provision_response = await client.post("/api/v1/gpu/provision", json=provision_data)
        instance_id = provision_response.json()["instance_id"]
        
        # Get metrics
        response = await client.get(f"/api/v1/gpu/instances/{instance_id}/metrics")
        
        assert response.status_code == 200
        result = response.json()
        assert "metrics" in result
        
        # Check for common GPU metrics
        metrics = result["metrics"]
        if metrics:  # Might be empty for new instance
            assert any(key in metrics for key in ["gpu_utilization", "memory_usage", "temperature"])


class TestGPUEstimation:
    """Test GPU cost estimation endpoints."""
    
    @pytest.mark.asyncio
    async def test_estimate_training_cost(self, client: AsyncClient, test_user):
        """Test training cost estimation."""
        estimation_data = {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "dataset_size_mb": 100,
            "epochs": 3,
            "gpu_type": "A100",
            "provider": "runpod"
        }
        
        response = await client.post("/api/v1/gpu/estimate-training-cost", json=estimation_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "estimated_cost" in result
        assert "estimated_hours" in result
        assert "cost_breakdown" in result
        assert result["estimated_cost"] > 0
        assert result["estimated_hours"] > 0
    
    @pytest.mark.asyncio
    async def test_estimate_inference_cost(self, client: AsyncClient, test_user):
        """Test inference cost estimation."""
        estimation_data = {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "expected_queries_per_hour": 100,
            "gpu_type": "RTX4090",
            "provider": "vast"
        }
        
        response = await client.post("/api/v1/gpu/estimate-inference-cost", json=estimation_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "cost_per_hour" in result
        assert "cost_per_query" in result
        assert "recommended_gpu" in result
        assert result["cost_per_hour"] > 0
    
    @pytest.mark.asyncio
    async def test_compare_providers(self, client: AsyncClient, test_user, mock_all_providers):
        """Test comparing providers for cost optimization."""
        comparison_data = {
            "gpu_type": "A100",
            "duration_hours": 4.0,
            "purpose": "training"
        }
        
        response = await client.post("/api/v1/gpu/compare-providers", json=comparison_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "comparisons" in result
        assert "recommended" in result
        
        # Should have multiple provider options
        assert len(result["comparisons"]) > 1
        
        # Each comparison should have cost info
        for comparison in result["comparisons"]:
            assert "provider" in comparison
            assert "total_cost" in comparison
            assert "cost_per_hour" in comparison
            assert "availability" in comparison
    
    @pytest.mark.asyncio
    async def test_get_cheapest_provider(self, client: AsyncClient, test_user, mock_all_providers):
        """Test finding the cheapest provider."""
        response = await client.get("/api/v1/gpu/cheapest?gpu_type=A100&duration_hours=2")
        
        assert response.status_code == 200
        result = response.json()
        assert "provider" in result
        assert "cost_per_hour" in result
        assert "total_cost" in result
        assert "savings" in result


class TestGPUQuotas:
    """Test GPU quotas and limits."""
    
    @pytest.mark.asyncio
    async def test_get_user_gpu_quota(self, client: AsyncClient, test_user):
        """Test getting user's GPU quota."""
        response = await client.get("/api/v1/gpu/quota")
        
        assert response.status_code == 200
        result = response.json()
        assert "max_concurrent_instances" in result
        assert "max_hours_per_day" in result
        assert "current_usage" in result
        assert "remaining_quota" in result
    
    @pytest.mark.asyncio
    async def test_provision_exceeding_quota(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test provisioning that would exceed quota."""
        # First, provision multiple instances to approach quota
        for i in range(5):  # Assume quota is less than 5
            provision_data = {
                "gpu_type": "RTX4090",
                "provider": "runpod",
                "max_hours": 0.5,  # Short duration
                "project_id": test_project["id"],
                "purpose": f"quota_test_{i}"
            }
            await client.post("/api/v1/gpu/provision", json=provision_data)
        
        # Try to provision one more (should fail if quota exceeded)
        provision_data = {
            "gpu_type": "A100",
            "provider": "runpod",
            "max_hours": 1.0,
            "project_id": test_project["id"],
            "purpose": "quota_exceed_test"
        }
        
        response = await client.post("/api/v1/gpu/provision", json=provision_data)
        
        # Might succeed if quota is high, or fail if quota exceeded
        if response.status_code == 400:
            result = response.json()
            assert "quota" in result["detail"].lower() or "limit" in result["detail"].lower()
        else:
            assert response.status_code == 201
    
    @pytest.mark.asyncio
    async def test_update_gpu_quota(self, client: AsyncClient, test_user):
        """Test updating user's GPU quota (admin only)."""
        quota_data = {
            "max_concurrent_instances": 10,
            "max_hours_per_day": 24.0
        }
        
        response = await client.patch("/api/v1/gpu/quota", json=quota_data)
        
        # Should succeed if user is admin, or fail with 403 if not
        if response.status_code == 200:
            result = response.json()
            assert result["max_concurrent_instances"] == 10
            assert result["max_hours_per_day"] == 24.0
        else:
            assert response.status_code == 403  # Forbidden for non-admin users


class TestGPUMonitoring:
    """Test GPU monitoring and alerts."""
    
    @pytest.mark.asyncio
    async def test_get_gpu_usage_summary(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test getting GPU usage summary."""
        # Provision some instances first
        provision_data = {
            "gpu_type": "A100",
            "provider": "runpod",
            "max_hours": 1.0,
            "project_id": test_project["id"],
            "purpose": "monitoring_test"
        }
        await client.post("/api/v1/gpu/provision", json=provision_data)
        
        # Get usage summary
        response = await client.get("/api/v1/gpu/usage-summary")
        
        assert response.status_code == 200
        result = response.json()
        assert "total_instances" in result
        assert "total_cost_today" in result
        assert "total_hours_today" in result
        assert "by_provider" in result
        assert "by_gpu_type" in result
    
    @pytest.mark.asyncio
    async def test_get_gpu_alerts(self, client: AsyncClient, test_user):
        """Test getting GPU-related alerts."""
        response = await client.get("/api/v1/gpu/alerts")
        
        assert response.status_code == 200
        result = response.json()
        assert "alerts" in result
        assert isinstance(result["alerts"], list)
        
        # Check alert structure if any exist
        for alert in result["alerts"]:
            assert "alert_id" in alert
            assert "type" in alert
            assert "message" in alert
            assert "severity" in alert
            assert "created_at" in alert
    
    @pytest.mark.asyncio
    async def test_create_gpu_alert(self, client: AsyncClient, test_user):
        """Test creating a GPU usage alert."""
        alert_data = {
            "type": "cost_threshold",
            "threshold": 50.0,  # Alert when daily cost exceeds $50
            "notification_method": "email"
        }
        
        response = await client.post("/api/v1/gpu/alerts", json=alert_data)
        
        assert response.status_code == 201
        result = response.json()
        assert "alert_id" in result
        assert result["type"] == "cost_threshold"
        assert result["threshold"] == 50.0
    
    @pytest.mark.asyncio
    async def test_delete_gpu_alert(self, client: AsyncClient, test_user):
        """Test deleting a GPU alert."""
        # Create alert first
        alert_data = {
            "type": "instance_count",
            "threshold": 5,
            "notification_method": "webhook"
        }
        
        create_response = await client.post("/api/v1/gpu/alerts", json=alert_data)
        alert_id = create_response.json()["alert_id"]
        
        # Delete alert
        response = await client.delete(f"/api/v1/gpu/alerts/{alert_id}")
        
        assert response.status_code == 204
        
        # Verify alert is deleted
        get_response = await client.get("/api/v1/gpu/alerts")
        alerts = get_response.json()["alerts"]
        alert_ids = [alert["alert_id"] for alert in alerts]
        assert alert_id not in alert_ids


@pytest.mark.performance
class TestGPUAPIPerformance:
    """Performance tests for GPU API."""
    
    @pytest.mark.asyncio
    async def test_gpu_provision_performance(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test GPU provisioning performance."""
        import time
        
        provision_data = {
            "gpu_type": "RTX4090",
            "provider": "runpod",
            "max_hours": 1.0,
            "project_id": test_project["id"],
            "purpose": "benchmark"
        }
        
        # Measure provisioning time
        start_time = time.time()
        response = await client.post("/api/v1/gpu/provision", json=provision_data)
        end_time = time.time()
        
        # Should complete within reasonable time (5 seconds)
        assert end_time - start_time < 5.0
        assert response.status_code == 201
    
    @pytest.mark.asyncio
    async def test_concurrent_gpu_operations(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test concurrent GPU operations."""
        import asyncio
        
        async def provision_and_check(i):
            # Provision
            provision_data = {
                "gpu_type": "RTX4090",
                "provider": "runpod",
                "max_hours": 0.5,
                "project_id": test_project["id"],
                "purpose": f"concurrent_test_{i}"
            }
            
            provision_response = await client.post("/api/v1/gpu/provision", json=provision_data)
            if provision_response.status_code != 201:
                return False
            
            instance_id = provision_response.json()["instance_id"]
            
            # Check status
            status_response = await client.get(f"/api/v1/gpu/instances/{instance_id}")
            return status_response.status_code == 200
        
        # Run 5 concurrent provision+check operations
        tasks = [provision_and_check(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most should succeed (some might fail due to quota limits)
        successful = sum(1 for r in results if r is True)
        assert successful >= 1  # At least one should succeed
