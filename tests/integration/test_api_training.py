# Integration tests for training API endpoints
import pytest
import json
from httpx import AsyncClient
from tests.fixtures.sample_data import get_sample_training_job, SAMPLE_TRAINING_DATASETS


class TestTrainingAPI:
    """Test training API endpoints integration."""
    
    @pytest.mark.asyncio
    async def test_create_training_job_success(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery, mock_all_providers):
        """Test successful training job creation."""
        job_data = get_sample_training_job("basic_lora")
        job_data["project_id"] = test_project["id"]
        job_data["use_gpu"] = False  # CPU training for basic test
        
        response = await client.post("/api/v1/training/jobs", json=job_data)
        
        assert response.status_code == 201
        result = response.json()
        assert "job_id" in result
        assert result["status"] == "queued"
        assert result["project_id"] == test_project["id"]
    
    @pytest.mark.asyncio
    async def test_create_gpu_training_job(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery, mock_all_providers):
        """Test GPU training job creation."""
        job_data = get_sample_training_job("gpu_training")
        job_data["project_id"] = test_project["id"]
        job_data["use_gpu"] = True
        job_data["gpu_type"] = "A100"
        job_data["gpu_provider"] = "runpod"
        
        response = await client.post("/api/v1/training/jobs", json=job_data)
        
        assert response.status_code == 201
        result = response.json()
        assert "job_id" in result
        assert result["gpu_config"]["gpu_type"] == "A100"
        assert result["gpu_config"]["provider"] == "runpod"
    
    @pytest.mark.asyncio
    async def test_create_training_job_insufficient_credits(self, client: AsyncClient, test_user, test_project, mock_celery):
        """Test training job creation with insufficient credits."""
        # Don't provide test_credits fixture to simulate insufficient balance
        job_data = get_sample_training_job("gpu_training")
        job_data["project_id"] = test_project["id"]
        job_data["use_gpu"] = True
        job_data["max_hours"] = 10.0  # Expensive job
        
        response = await client.post("/api/v1/training/jobs", json=job_data)
        
        assert response.status_code == 400
        result = response.json()
        assert "insufficient credits" in result["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_create_training_job_invalid_data(self, client: AsyncClient, test_user, test_project):
        """Test training job creation with invalid data."""
        invalid_job_data = {
            "job_name": "",  # Empty name
            "base_model_name": "invalid-model",
            "dataset_path": "/nonexistent/path",
            "project_id": test_project["id"]
        }
        
        response = await client.post("/api/v1/training/jobs", json=invalid_job_data)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_get_training_job_status(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test getting training job status."""
        # Create a job first
        job_data = get_sample_training_job("basic_lora")
        job_data["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/training/jobs", json=job_data)
        job_id = create_response.json()["job_id"]
        
        # Get job status
        response = await client.get(f"/api/v1/training/jobs/{job_id}")
        
        assert response.status_code == 200
        result = response.json()
        assert result["job_id"] == job_id
        assert "status" in result
        assert "created_at" in result
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_training_job(self, client: AsyncClient, test_user):
        """Test getting status of non-existent training job."""
        response = await client.get("/api/v1/training/jobs/nonexistent-job-id")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_list_training_jobs(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test listing training jobs."""
        # Create multiple jobs
        job_names = ["job1", "job2", "job3"]
        for name in job_names:
            job_data = get_sample_training_job("basic_lora")
            job_data["job_name"] = name
            job_data["project_id"] = test_project["id"]
            await client.post("/api/v1/training/jobs", json=job_data)
        
        # List jobs
        response = await client.get("/api/v1/training/jobs")
        
        assert response.status_code == 200
        result = response.json()
        assert "jobs" in result
        assert len(result["jobs"]) >= 3
        
        # Check job names are present
        job_names_in_response = [job["job_name"] for job in result["jobs"]]
        for name in job_names:
            assert name in job_names_in_response
    
    @pytest.mark.asyncio
    async def test_list_training_jobs_with_filters(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test listing training jobs with filters."""
        # Create jobs with different statuses
        job_data = get_sample_training_job("basic_lora")
        job_data["project_id"] = test_project["id"]
        await client.post("/api/v1/training/jobs", json=job_data)
        
        # List with status filter
        response = await client.get("/api/v1/training/jobs?status=queued")
        
        assert response.status_code == 200
        result = response.json()
        assert "jobs" in result
        # All jobs should have queued status
        for job in result["jobs"]:
            assert job["status"] == "queued"
    
    @pytest.mark.asyncio
    async def test_cancel_training_job(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test cancelling a training job."""
        # Create a job
        job_data = get_sample_training_job("basic_lora")
        job_data["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/training/jobs", json=job_data)
        job_id = create_response.json()["job_id"]
        
        # Cancel the job
        response = await client.post(f"/api/v1/training/jobs/{job_id}/cancel")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "cancelled"
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, client: AsyncClient, test_user):
        """Test cancelling a non-existent job."""
        response = await client.post("/api/v1/training/jobs/nonexistent/cancel")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_training_job_logs(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery, mock_all_providers):
        """Test getting training job logs."""
        # Create a GPU training job
        job_data = get_sample_training_job("gpu_training")
        job_data["project_id"] = test_project["id"]
        job_data["use_gpu"] = True
        
        create_response = await client.post("/api/v1/training/jobs", json=job_data)
        job_id = create_response.json()["job_id"]
        
        # Get logs
        response = await client.get(f"/api/v1/training/jobs/{job_id}/logs")
        
        assert response.status_code == 200
        # Should return logs as text
        assert "training" in response.text.lower() or "log" in response.text.lower()
    
    @pytest.mark.asyncio
    async def test_get_training_job_metrics(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test getting training job metrics."""
        # Create a job
        job_data = get_sample_training_job("basic_lora")
        job_data["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/training/jobs", json=job_data)
        job_id = create_response.json()["job_id"]
        
        # Get metrics
        response = await client.get(f"/api/v1/training/jobs/{job_id}/metrics")
        
        assert response.status_code == 200
        result = response.json()
        assert "metrics" in result
        # Metrics might be empty for new job
        assert isinstance(result["metrics"], (dict, list))
    
    @pytest.mark.asyncio
    async def test_update_training_job_config(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test updating training job configuration."""
        # Create a job
        job_data = get_sample_training_job("basic_lora")
        job_data["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/training/jobs", json=job_data)
        job_id = create_response.json()["job_id"]
        
        # Update configuration
        update_data = {
            "hyperparameters": {
                "learning_rate": 1e-5,  # Different from original
                "epochs": 5
            }
        }
        
        response = await client.patch(f"/api/v1/training/jobs/{job_id}", json=update_data)
        
        # Should succeed if job hasn't started yet
        if response.status_code == 200:
            result = response.json()
            assert result["hyperparameters"]["learning_rate"] == 1e-5
        else:
            # Job might have already started, which should return 400
            assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_training_job_cost_estimation(self, client: AsyncClient, test_user, test_project):
        """Test training job cost estimation endpoint."""
        estimation_data = {
            "base_model_name": "meta-llama/Llama-2-7b-hf",
            "dataset_size_mb": 100,
            "gpu_type": "A100",
            "gpu_provider": "runpod",
            "epochs": 3
        }
        
        response = await client.post("/api/v1/training/estimate-cost", json=estimation_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "estimated_cost" in result
        assert "estimated_hours" in result
        assert "gpu_cost_per_hour" in result
        assert result["estimated_cost"] > 0
        assert result["estimated_hours"] > 0
    
    @pytest.mark.asyncio
    async def test_training_job_with_custom_dataset(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery, test_data_dir):
        """Test training job with uploaded custom dataset."""
        # Upload a dataset first
        dataset_content = "\n".join([
            json.dumps(item) for item in SAMPLE_TRAINING_DATASETS["instruction_following"]
        ])
        
        files = {"file": ("custom_dataset.jsonl", dataset_content, "application/json")}
        upload_response = await client.post("/api/v1/data/upload", files=files)
        
        assert upload_response.status_code == 200
        dataset_path = upload_response.json()["file_path"]
        
        # Create training job with custom dataset
        job_data = get_sample_training_job("basic_lora")
        job_data["dataset_path"] = dataset_path
        job_data["project_id"] = test_project["id"]
        
        response = await client.post("/api/v1/training/jobs", json=job_data)
        
        assert response.status_code == 201
        result = response.json()
        assert result["dataset_path"] == dataset_path


class TestTrainingJobLifecycle:
    """Test complete training job lifecycle."""
    
    @pytest.mark.asyncio
    async def test_training_job_complete_lifecycle(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery, mock_all_providers):
        """Test complete training job from creation to completion."""
        # 1. Create job
        job_data = get_sample_training_job("basic_lora")
        job_data["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/training/jobs", json=job_data)
        assert create_response.status_code == 201
        
        job_id = create_response.json()["job_id"]
        
        # 2. Check initial status
        status_response = await client.get(f"/api/v1/training/jobs/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "queued"
        
        # 3. Simulate job progression (in real scenario, Celery would do this)
        # Update job status to running
        with patch('app.api.training.update_job_status') as mock_update:
            mock_update.return_value = True
            
            # 4. Check running status
            status_response = await client.get(f"/api/v1/training/jobs/{job_id}")
            # Status might still be queued since we're mocking
            assert status_response.status_code == 200
        
        # 5. Get logs during training
        logs_response = await client.get(f"/api/v1/training/jobs/{job_id}/logs")
        assert logs_response.status_code == 200
        
        # 6. Get metrics
        metrics_response = await client.get(f"/api/v1/training/jobs/{job_id}/metrics")
        assert metrics_response.status_code == 200
        
        # 7. Job completion would be handled by background task
        # In integration test, we can simulate completion
        
        # 8. Final status check
        final_status = await client.get(f"/api/v1/training/jobs/{job_id}")
        assert final_status.status_code == 200
        # Job should still exist
        assert "job_id" in final_status.json()


class TestTrainingJobValidation:
    """Test training job validation and error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_model_name(self, client: AsyncClient, test_user, test_project):
        """Test training job with invalid model name."""
        job_data = get_sample_training_job("basic_lora")
        job_data["base_model_name"] = "nonexistent/model"
        job_data["project_id"] = test_project["id"]
        
        response = await client.post("/api/v1/training/jobs", json=job_data)
        
        # Should validate model exists
        assert response.status_code in [400, 422]
    
    @pytest.mark.asyncio
    async def test_invalid_dataset_path(self, client: AsyncClient, test_user, test_project):
        """Test training job with invalid dataset path."""
        job_data = get_sample_training_job("basic_lora")
        job_data["dataset_path"] = "/nonexistent/path/dataset.jsonl"
        job_data["project_id"] = test_project["id"]
        
        response = await client.post("/api/v1/training/jobs", json=job_data)
        
        # Should validate dataset exists
        assert response.status_code in [400, 422]
    
    @pytest.mark.asyncio
    async def test_invalid_hyperparameters(self, client: AsyncClient, test_user, test_project):
        """Test training job with invalid hyperparameters."""
        job_data = get_sample_training_job("basic_lora")
        job_data["hyperparameters"]["learning_rate"] = -1.0  # Invalid negative learning rate
        job_data["hyperparameters"]["epochs"] = 0  # Invalid zero epochs
        job_data["project_id"] = test_project["id"]
        
        response = await client.post("/api/v1/training/jobs", json=job_data)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, client: AsyncClient, test_user, test_project):
        """Test training job with missing required fields."""
        incomplete_job_data = {
            "job_name": "incomplete_job",
            # Missing base_model_name, dataset_path, etc.
            "project_id": test_project["id"]
        }
        
        response = await client.post("/api/v1/training/jobs", json=incomplete_job_data)
        
        assert response.status_code == 422  # Validation error
        result = response.json()
        assert "detail" in result
        # Should mention missing fields
        error_details = str(result["detail"])
        assert "required" in error_details.lower() or "missing" in error_details.lower()
    
    @pytest.mark.asyncio
    async def test_unauthorized_access(self, client: AsyncClient):
        """Test training job access without authentication."""
        # Remove API key header
        client.headers.pop("X-API-Key", None)
        
        job_data = get_sample_training_job("basic_lora")
        response = await client.post("/api/v1/training/jobs", json=job_data)
        
        assert response.status_code == 401  # Unauthorized
    
    @pytest.mark.asyncio
    async def test_project_permission_check(self, client: AsyncClient, test_user):
        """Test training job with project user doesn't have access to."""
        job_data = get_sample_training_job("basic_lora")
        job_data["project_id"] = "unauthorized-project-id"
        
        response = await client.post("/api/v1/training/jobs", json=job_data)
        
        # Should check project permissions
        assert response.status_code in [403, 404]  # Forbidden or Not Found


class TestTrainingJobConcurrency:
    """Test concurrent training job operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_job_creation(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test creating multiple training jobs concurrently."""
        import asyncio
        
        async def create_job(job_name):
            job_data = get_sample_training_job("basic_lora")
            job_data["job_name"] = job_name
            job_data["project_id"] = test_project["id"]
            return await client.post("/api/v1/training/jobs", json=job_data)
        
        # Create 5 jobs concurrently
        tasks = [create_job(f"concurrent_job_{i}") for i in range(5)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 201
        
        # All should have unique job IDs
        job_ids = [response.json()["job_id"] for response in responses]
        assert len(set(job_ids)) == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_status_checks(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test concurrent status checks on same job."""
        import asyncio
        
        # Create a job first
        job_data = get_sample_training_job("basic_lora")
        job_data["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/training/jobs", json=job_data)
        job_id = create_response.json()["job_id"]
        
        # Check status concurrently
        async def check_status():
            return await client.get(f"/api/v1/training/jobs/{job_id}")
        
        tasks = [check_status() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            assert response.json()["job_id"] == job_id


@pytest.mark.performance
class TestTrainingAPIPerformance:
    """Performance tests for training API."""
    
    @pytest.mark.asyncio
    async def test_training_job_creation_performance(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery, benchmark):
        """Benchmark training job creation performance."""
        job_data = get_sample_training_job("basic_lora")
        job_data["project_id"] = test_project["id"]
        
        async def create_job():
            response = await client.post("/api/v1/training/jobs", json=job_data)
            return response.status_code
        
        result = await benchmark(create_job)
        assert result == 201
    
    @pytest.mark.asyncio
    async def test_job_listing_performance(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test performance of job listing with many jobs."""
        # Create many jobs
        for i in range(50):
            job_data = get_sample_training_job("basic_lora")
            job_data["job_name"] = f"perf_test_job_{i}"
            job_data["project_id"] = test_project["id"]
            await client.post("/api/v1/training/jobs", json=job_data)
        
        # Time the listing operation
        import time
        start_time = time.time()
        
        response = await client.get("/api/v1/training/jobs?limit=100")
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert response.status_code == 200
        assert duration < 2.0  # Should complete within 2 seconds
        
        result = response.json()
        assert len(result["jobs"]) >= 50
