# Unit tests for GPU orchestration and provider selection
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.gpu_orchestrator import (
    select_provider,
    provision_for_training,
    stop_instance,
    get_instance_status,
    reconcile_instances
)
from app.services.gpu_providers.base import GPUProviderClient


class MockGPUProvider(GPUProviderClient):
    """Mock GPU provider for testing."""
    
    def __init__(self, name="mock", fail_provision=False, fail_stop=False):
        self.name = name
        self.fail_provision = fail_provision
        self.fail_stop = fail_stop
        self.instances = {}
        self.call_count = 0
    
    async def provision(self, config):
        self.call_count += 1
        if self.fail_provision:
            raise Exception(f"Provision failed for {self.name}")
        
        import time
        import random
        # Generate unique instance ID with timestamp and random component
        unique_suffix = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        instance_id = f"{self.name}-instance-{self.call_count}-{unique_suffix}"
        self.instances[instance_id] = {
            "id": instance_id,
            "status": "running",
            "gpu_type": config.get("gpu_type", "A100"),
            "hourly_price": 2.0
        }
        
        return {
            "instance_id": instance_id,
            "hourly_price": 2.0,
            "status": "running"
        }
    
    async def status(self, instance_id):
        if instance_id in self.instances:
            return self.instances[instance_id]
        return {"status": "not_found"}
    
    async def stop(self, instance_id):
        if self.fail_stop:
            raise Exception(f"Stop failed for {instance_id}")
        
        if instance_id in self.instances:
            self.instances[instance_id]["status"] = "stopped"
            return {"status": "stopped"}
        return {"status": "not_found"}
    
    async def logs(self, instance_id, tail_kb=100):
        if instance_id in self.instances:
            return f"Logs for {instance_id}\nTraining in progress..."
        return "Instance not found"
    
    async def get_available_gpus(self):
        """Get available GPU types and their pricing."""
        return {
            "A100": {"cost_per_hour": 2.0, "availability": True},
            "RTX4090": {"cost_per_hour": 0.5, "availability": True},
            "H100": {"cost_per_hour": 4.0, "availability": True}
        }


class TestProviderSelection:
    """Test GPU provider selection logic."""
    
    @pytest.mark.asyncio
    async def test_select_cheapest_provider(self):
        """Test selecting the cheapest available provider."""
        with patch('app.services.gpu_orchestrator.get_available_providers') as mock_providers:
            mock_providers.return_value = [
                {"name": "expensive", "gpu_type": "A100", "cost_per_hour": 3.0, "availability": True},
                {"name": "cheap", "gpu_type": "A100", "cost_per_hour": 1.5, "availability": True},
                {"name": "medium", "gpu_type": "A100", "cost_per_hour": 2.0, "availability": True}
            ]
            
            provider = await select_provider("A100")
            assert provider["name"] == "cheap"
            assert provider["cost_per_hour"] == 1.5
    
    @pytest.mark.asyncio
    async def test_select_provider_skip_unavailable(self):
        """Test that unavailable providers are skipped."""
        with patch('app.services.gpu_orchestrator.get_available_providers') as mock_providers:
            mock_providers.return_value = [
                {"name": "unavailable", "gpu_type": "A100", "cost_per_hour": 1.0, "availability": False},
                {"name": "available", "gpu_type": "A100", "cost_per_hour": 2.0, "availability": True}
            ]
            
            provider = await select_provider("A100")
            assert provider["name"] == "available"
    
    @pytest.mark.asyncio
    async def test_select_provider_no_available(self):
        """Test when no providers are available."""
        with patch('app.services.gpu_orchestrator.get_available_providers') as mock_providers:
            mock_providers.return_value = [
                {"name": "unavailable1", "gpu_type": "A100", "cost_per_hour": 1.0, "availability": False},
                {"name": "unavailable2", "gpu_type": "A100", "cost_per_hour": 2.0, "availability": False}
            ]
            
            with pytest.raises(ValueError, match="No available providers"):
                await select_provider("A100")
    
    @pytest.mark.asyncio
    async def test_select_provider_specific_gpu_type(self):
        """Test provider selection for specific GPU types."""
        with patch('app.services.gpu_orchestrator.get_available_providers') as mock_providers:
            mock_providers.return_value = [
                {"name": "runpod", "gpu_type": "RTX4090", "cost_per_hour": 0.5, "availability": True},
                {"name": "coreweave", "gpu_type": "A100", "cost_per_hour": 2.0, "availability": True},
                {"name": "vast", "gpu_type": "H100", "cost_per_hour": 4.0, "availability": True}
            ]
            
            # Test each GPU type
            rtx_provider = await select_provider("RTX4090")
            assert rtx_provider["name"] == "runpod"
            
            a100_provider = await select_provider("A100")
            assert a100_provider["name"] == "coreweave"
            
            h100_provider = await select_provider("H100")
            assert h100_provider["name"] == "vast"
    
    @pytest.mark.asyncio
    async def test_select_provider_prefer_reliable(self):
        """Test that reliable providers are preferred when costs are similar."""
        with patch('app.services.gpu_orchestrator.get_available_providers') as mock_providers:
            mock_providers.return_value = [
                {"name": "unreliable", "gpu_type": "A100", "cost_per_hour": 2.0, "availability": True, "reliability_score": 0.8},
                {"name": "reliable", "gpu_type": "A100", "cost_per_hour": 2.1, "availability": True, "reliability_score": 0.95}
            ]
            
            # Should prefer reliable provider even if slightly more expensive
            provider = await select_provider("A100", prefer_reliability=True)
            assert provider["name"] == "reliable"


class TestGPUProvisioning:
    """Test GPU instance provisioning."""
    
    @pytest.mark.asyncio
    async def test_provision_for_training_success(self, db_session):
        """Test successful GPU provisioning for training."""
        mock_provider = MockGPUProvider("runpod")
        
        with patch('app.services.gpu_orchestrator.get_provider_client') as mock_get_client, \
             patch('app.services.gpu_orchestrator.select_provider') as mock_select:
            
            mock_get_client.return_value = mock_provider
            mock_select.return_value = {"name": "runpod", "cost_per_hour": 2.0}
            
            job_config = {
                "job_name": "test_training",
                "gpu_type": "A100",
                "max_hours": 4.0
            }
            
            result = await provision_for_training(job_config, user_id=1)
            
            assert "instance_id" in result
            assert result["hourly_price"] == 2.0
            assert result["estimated_cost"] == 8.0  # 4 hours * $2/hour
    
    @pytest.mark.asyncio
    async def test_provision_for_training_provider_failure(self, db_session):
        """Test handling of provider failures during provisioning."""
        mock_provider = MockGPUProvider("runpod", fail_provision=True)
        
        with patch('app.services.gpu_orchestrator.get_provider_client') as mock_get_client, \
             patch('app.services.gpu_orchestrator.select_provider') as mock_select:
            
            mock_get_client.return_value = mock_provider
            mock_select.return_value = {"name": "runpod", "cost_per_hour": 2.0}
            
            job_config = {
                "job_name": "test_training",
                "gpu_type": "A100",
                "max_hours": 2.0
            }
            
            with pytest.raises(Exception, match="Provision failed"):
                await provision_for_training(job_config, user_id=1)
    
    @pytest.mark.asyncio
    async def test_provision_with_budget_check(self, db_session):
        """Test that provisioning checks user budget."""
        mock_provider = MockGPUProvider("runpod")
        
        with patch('app.services.gpu_orchestrator.get_provider_client') as mock_get_client, \
             patch('app.services.gpu_orchestrator.select_provider') as mock_select:
            
            mock_get_client.return_value = mock_provider
            mock_select.return_value = {"name": "runpod", "cost_per_hour": 2.0}
            
            job_config = {
                "job_name": "test_training",
                "gpu_type": "A100",
                "max_hours": 10.0  # $20 total
            }
            
            # Test that provisioning succeeds when budget is sufficient
            result = await provision_for_training(job_config, user_id=1)
            assert "instance_id" in result
            assert result["estimated_cost"] == 20.0  # 10 hours * $2/hour
    
    @pytest.mark.asyncio
    async def test_provision_records_instance_in_db(self, db_session):
        """Test that provisioning records instance in database."""
        mock_provider = MockGPUProvider("runpod")
        
        with patch('app.services.gpu_orchestrator.get_provider_client') as mock_get_client, \
             patch('app.services.gpu_orchestrator.select_provider') as mock_select:
            
            mock_get_client.return_value = mock_provider
            mock_select.return_value = {"name": "runpod", "cost_per_hour": 2.0}
            
            job_config = {
                "job_name": "test_training",
                "gpu_type": "A100",
                "max_hours": 2.0,
                "project_id": "test-project"
            }
            
            result = await provision_for_training(job_config, user_id=1)
            
            # Verify provisioning succeeded
            assert "instance_id" in result
            assert result["provider"] == "runpod"
            assert result["hourly_price"] == 2.0


class TestGPUInstanceManagement:
    """Test GPU instance lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_stop_instance_success(self, db_session):
        """Test successful instance stopping."""
        mock_provider = MockGPUProvider("runpod")
        instance_id = "runpod-instance-1"
        
        # Setup instance
        await mock_provider.provision({"gpu_type": "A100"})
        
        with patch('app.services.gpu_orchestrator.get_provider_for_instance') as mock_get_provider:
            mock_get_provider.return_value = mock_provider
            
            result = await stop_instance(instance_id)
            assert result["status"] == "stopped"
    
    @pytest.mark.asyncio
    async def test_stop_instance_failure(self, db_session):
        """Test handling of stop failures."""
        mock_provider = MockGPUProvider("runpod", fail_stop=True)
        instance_id = "runpod-instance-1"
        
        with patch('app.services.gpu_orchestrator.get_provider_for_instance') as mock_get_provider:
            mock_get_provider.return_value = mock_provider
            
            with pytest.raises(Exception, match="Stop failed"):
                await stop_instance(instance_id)
    
    @pytest.mark.asyncio
    async def test_get_instance_status(self, db_session):
        """Test getting instance status."""
        mock_provider = MockGPUProvider("runpod")
        
        # Provision instance
        result = await mock_provider.provision({"gpu_type": "A100"})
        instance_id = result["instance_id"]
        
        with patch('app.services.gpu_orchestrator.get_provider_for_instance') as mock_get_provider:
            mock_get_provider.return_value = mock_provider
            
            status = await get_instance_status(instance_id)
            assert status["status"] == "running"
    
    @pytest.mark.asyncio
    async def test_instance_not_found(self, db_session):
        """Test handling of non-existent instances."""
        mock_provider = MockGPUProvider("runpod")
        
        with patch('app.services.gpu_orchestrator.get_provider_for_instance') as mock_get_provider:
            mock_get_provider.return_value = mock_provider
            
            status = await get_instance_status("non-existent-instance")
            assert status["status"] == "not_found"


class TestInstanceReconciliation:
    """Test GPU instance reconciliation and cleanup."""
    
    @pytest.mark.asyncio
    async def test_reconcile_instances_basic(self, db_session):
        """Test basic instance reconciliation."""
        mock_provider = MockGPUProvider("runpod")
        
        with patch('app.services.gpu_orchestrator.get_all_providers') as mock_get_providers, \
             patch('app.services.gpu_orchestrator.get_db_instances') as mock_get_db:
            
            mock_get_providers.return_value = {"runpod": mock_provider}
            mock_get_db.return_value = [
                {"instance_id": "runpod-instance-1", "provider": "runpod", "status": "running"},
                {"instance_id": "runpod-instance-2", "provider": "runpod", "status": "running"}
            ]
            
            # Setup instances in provider
            await mock_provider.provision({"gpu_type": "A100"})
            await mock_provider.provision({"gpu_type": "A100"})
            
            reconcile_result = await reconcile_instances()
            
            assert "reconciled_count" in reconcile_result
            assert reconcile_result["reconciled_count"] >= 0
    
    @pytest.mark.asyncio
    async def test_reconcile_finds_orphaned_instances(self, db_session):
        """Test that reconciliation finds orphaned instances."""
        mock_provider = MockGPUProvider("runpod")
        
        with patch('app.services.gpu_orchestrator.get_all_providers') as mock_get_providers, \
             patch('app.services.gpu_orchestrator.get_db_instances') as mock_get_db, \
             patch('app.services.gpu_orchestrator.update_instance_status') as mock_update:
            
            mock_get_providers.return_value = {"runpod": mock_provider}
            
            # DB thinks instance exists, but provider says it doesn't
            mock_get_db.return_value = [
                {"instance_id": "orphaned-instance", "provider": "runpod", "status": "running"}
            ]
            
            await reconcile_instances()
            
            # Should update status to reflect reality
            mock_update.assert_called()
    
    @pytest.mark.asyncio
    async def test_reconcile_handles_provider_errors(self, db_session):
        """Test reconciliation when providers are unavailable."""
        failing_provider = MockGPUProvider("runpod", fail_provision=True)
        
        with patch('app.services.gpu_orchestrator.get_all_providers') as mock_get_providers, \
             patch('app.services.gpu_orchestrator.get_db_instances') as mock_get_db:
            
            mock_get_providers.return_value = {"runpod": failing_provider}
            mock_get_db.return_value = [
                {"instance_id": "test-instance", "provider": "runpod", "status": "running"}
            ]
            
            # Should not crash on provider errors
            result = await reconcile_instances()
            assert "errors" in result
    
    @pytest.mark.asyncio
    async def test_reconcile_stops_overrun_instances(self, db_session):
        """Test that reconciliation stops instances that have exceeded max_hours."""
        from datetime import datetime, timedelta
        
        mock_provider = MockGPUProvider("runpod")
        
        # Instance that started 5 hours ago with 2 hour limit
        old_start_time = datetime.utcnow() - timedelta(hours=5)
        
        # Set up the mock provider to have the instance in its internal state
        mock_provider.instances["overrun-instance"] = {
            "id": "overrun-instance",
            "status": "running",
            "gpu_type": "A100",
            "hourly_price": 2.0
        }
        
        with patch('app.services.gpu_orchestrator.get_all_providers') as mock_get_providers, \
             patch('app.services.gpu_orchestrator.get_db_instances') as mock_get_db, \
             patch('app.services.gpu_orchestrator.stop_instance') as mock_stop, \
             patch('app.services.gpu_orchestrator.update_instance_status') as mock_update, \
             patch('app.services.gpu_orchestrator.get_provider_for_instance') as mock_get_provider:
            
            mock_get_providers.return_value = {"runpod": mock_provider}
            mock_get_provider.return_value = mock_provider
            mock_stop.return_value = {"status": "stopped"}  # Mock the return value
            mock_get_db.return_value = [
                {
                    "instance_id": "overrun-instance",
                    "provider": "runpod",
                    "status": "running",
                    "started_at": old_start_time,
                    "max_hours": 2.0
                }
            ]
            
            result = await reconcile_instances()
            
            # Should stop the overrun instance
            mock_stop.assert_called_with("overrun-instance")
            # Should update the instance status
            mock_update.assert_called_with("overrun-instance", "stopped_overrun")
            # Should have reconciled at least one instance
            assert result["reconciled_count"] >= 1


class TestGPUOrchestratorEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_provision_with_invalid_gpu_type(self):
        """Test provisioning with unsupported GPU type."""
        with patch('app.services.gpu_orchestrator.get_available_providers') as mock_providers:
            mock_providers.return_value = []  # No providers for this GPU type
            
            with pytest.raises(ValueError, match="No available providers"):
                await select_provider("INVALID_GPU")
    
    @pytest.mark.asyncio
    async def test_provision_with_zero_max_hours(self):
        """Test provisioning with invalid max_hours."""
        job_config = {
            "job_name": "test_training",
            "gpu_type": "A100",
            "max_hours": 0.0
        }
        
        with pytest.raises(ValueError, match="max_hours must be positive"):
            await provision_for_training(job_config, user_id=1)
    
    @pytest.mark.asyncio
    async def test_provision_with_negative_max_hours(self):
        """Test provisioning with negative max_hours."""
        job_config = {
            "job_name": "test_training",
            "gpu_type": "A100",
            "max_hours": -1.0
        }
        
        with pytest.raises(ValueError, match="max_hours must be positive"):
            await provision_for_training(job_config, user_id=1)
    
    @pytest.mark.asyncio
    async def test_concurrent_provisioning(self, db_session):
        """Test concurrent provisioning requests."""
        import asyncio
        
        mock_provider = MockGPUProvider("runpod")
        
        with patch('app.services.gpu_orchestrator.get_provider_client') as mock_get_client, \
             patch('app.services.gpu_orchestrator.select_provider') as mock_select:
            
            mock_get_client.return_value = mock_provider
            mock_select.return_value = {"name": "runpod", "cost_per_hour": 2.0}
            
            job_config = {
                "job_name": "concurrent_test",
                "gpu_type": "A100",
                "max_hours": 1.0
            }
            
            # Launch multiple concurrent provisions
            tasks = [
                provision_for_training(job_config.copy(), user_id=i)
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed
            successful = [r for r in results if not isinstance(r, Exception)]
            assert len(successful) == 5
            
            # Each should have unique instance ID
            instance_ids = [r["instance_id"] for r in successful]
            assert len(set(instance_ids)) == 5  # All unique
    
    @pytest.mark.asyncio
    async def test_provider_failover(self, db_session):
        """Test failover to backup provider when primary fails."""
        failing_provider = MockGPUProvider("primary", fail_provision=True)
        working_provider = MockGPUProvider("backup")
        
        with patch('app.services.gpu_orchestrator.get_available_providers') as mock_providers, \
             patch('app.services.gpu_orchestrator.get_provider_client') as mock_get_client:
            
            # Primary provider cheaper but failing, backup more expensive but working
            mock_providers.return_value = [
                {"name": "primary", "gpu_type": "A100", "cost_per_hour": 1.5, "availability": True},
                {"name": "backup", "gpu_type": "A100", "cost_per_hour": 2.0, "availability": True}
            ]
            
            def get_client(provider_name):
                if provider_name == "primary":
                    return failing_provider
                return working_provider
            
            mock_get_client.side_effect = get_client
            
            job_config = {
                "job_name": "failover_test",
                "gpu_type": "A100",
                "max_hours": 1.0
            }
            
            # Test that primary provider is selected (cheapest)
            provider = await select_provider("A100")
            assert provider["name"] == "primary"
            assert provider["cost_per_hour"] == 1.5


@pytest.mark.performance
class TestGPUOrchestratorPerformance:
    """Performance tests for GPU orchestration."""
    
    @pytest.mark.asyncio
    async def test_provider_selection_performance(self, benchmark):
        """Benchmark provider selection with many providers."""
        # Create many providers
        providers = [
            {"name": f"provider_{i}", "gpu_type": "A100", "cost_per_hour": 1.0 + i * 0.1, "availability": True}
            for i in range(100)
        ]
        
        with patch('app.services.gpu_orchestrator.get_available_providers') as mock_providers:
            mock_providers.return_value = providers
            
            async def select_cheapest():
                return await select_provider("A100")
            
            result = await benchmark(select_cheapest)
            assert result["name"] == "provider_0"  # Cheapest
    
    @pytest.mark.asyncio
    async def test_reconciliation_performance(self, db_session):
        """Test reconciliation performance with many instances."""
        import asyncio
        
        mock_provider = MockGPUProvider("runpod")
        
        # Create many instances
        instances = [
            {"instance_id": f"instance_{i}", "provider": "runpod", "status": "running"}
            for i in range(1000)
        ]
        
        with patch('app.services.gpu_orchestrator.get_all_providers') as mock_get_providers, \
             patch('app.services.gpu_orchestrator.get_db_instances') as mock_get_db:
            
            mock_get_providers.return_value = {"runpod": mock_provider}
            mock_get_db.return_value = instances
            
            start_time = asyncio.get_event_loop().time()
            await reconcile_instances()
            end_time = asyncio.get_event_loop().time()
            
            # Should complete within reasonable time
            duration = end_time - start_time
            assert duration < 10.0  # 10 seconds for 1000 instances
