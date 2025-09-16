# End-to-end tests for complete user workflows
import pytest
import time
from httpx import AsyncClient
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from tests.fixtures.sample_data import get_sample_training_job, SAMPLE_AGENTS


@pytest.fixture(scope="session")
def browser():
    """Create browser instance for E2E tests."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(
        service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    
    yield driver
    driver.quit()


class TestCompleteUserJourneys:
    """Test complete end-to-end user journeys."""
    
    @pytest.mark.asyncio
    async def test_agent_creation_to_deployment_workflow(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers, mock_celery):
        """Test complete workflow: Create agent -> Train -> Evaluate -> Deploy."""
        
        # Step 1: Create Agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        agent_response = await client.post("/api/v1/agents", json=agent_config)
        assert agent_response.status_code == 201
        agent_id = agent_response.json()["agent_id"]
        
        # Step 2: Upload training data
        training_data = [
            {"instruction": "What is a contract?", "output": "A contract is a legally binding agreement."},
            {"instruction": "Define tort law", "output": "Tort law deals with civil wrongs and provides remedies."}
        ]
        
        import json
        dataset_content = "\n".join([json.dumps(item) for item in training_data])
        files = {"file": ("legal_training.jsonl", dataset_content, "application/json")}
        
        upload_response = await client.post("/api/v1/data/upload", files=files)
        assert upload_response.status_code == 200
        dataset_path = upload_response.json()["file_path"]
        
        # Step 3: Create training job
        job_data = get_sample_training_job("gpu_training")
        job_data["project_id"] = test_project["id"]
        job_data["dataset_path"] = dataset_path
        job_data["use_gpu"] = True
        job_data["gpu_type"] = "A100"
        
        training_response = await client.post("/api/v1/training/jobs", json=job_data)
        assert training_response.status_code == 201
        job_id = training_response.json()["job_id"]
        
        # Step 4: Monitor training progress
        status_response = await client.get(f"/api/v1/training/jobs/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] in ["queued", "running"]
        
        # Step 5: Test agent query
        query_data = {
            "message": "What is a contract?",
            "session_id": "e2e-test-session"
        }
        
        query_response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
        assert query_response.status_code == 200
        assert len(query_response.json()["response"]) > 0
        
        # Step 6: Create evaluation
        eval_data = {
            "name": "Legal Agent E2E Evaluation",
            "test_set": [
                {"question": "What is a contract?", "expected_answer": "A legally binding agreement"},
                {"question": "Define negligence", "expected_answer": "Failure to exercise reasonable care"}
            ],
            "metrics": ["accuracy", "response_time"]
        }
        
        eval_response = await client.post(f"/api/v1/agents/{agent_id}/evaluate", json=eval_data)
        assert eval_response.status_code == 202
        evaluation_id = eval_response.json()["evaluation_id"]
        
        # Step 7: Deploy agent
        deployment_data = {
            "name": "Legal Agent Deployment",
            "agent_id": agent_id,
            "deployment_type": "api",
            "config": {
                "max_queries_per_hour": 100,
                "timeout_seconds": 30
            }
        }
        
        deploy_response = await client.post("/api/v1/deployments", json=deployment_data)
        assert deploy_response.status_code == 201
        deployment_id = deploy_response.json()["deployment_id"]
        
        # Step 8: Test deployed agent
        deployed_query_data = {
            "message": "What is tort law?",
            "session_id": "deployed-test"
        }
        
        deployed_response = await client.post(f"/api/v1/deployments/{deployment_id}/query", json=deployed_query_data)
        assert deployed_response.status_code == 200
        assert len(deployed_response.json()["response"]) > 0
        
        # Step 9: Check deployment metrics
        metrics_response = await client.get(f"/api/v1/deployments/{deployment_id}/metrics")
        assert metrics_response.status_code == 200
        metrics = metrics_response.json()
        assert "total_queries" in metrics
        assert "avg_response_time" in metrics
    
    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers):
        """Test workflow with multiple coordinated agents."""
        
        # Create Research Agent
        research_agent_config = SAMPLE_AGENTS["code_agent"].copy()
        research_agent_config["name"] = "research_agent"
        research_agent_config["display_name"] = "Research Agent"
        research_agent_config["project_id"] = test_project["id"]
        research_agent_config["tools"] = ["search", "retriever"]
        
        research_response = await client.post("/api/v1/agents", json=research_agent_config)
        assert research_response.status_code == 201
        research_agent_id = research_response.json()["agent_id"]
        
        # Create Summarizer Agent
        summarizer_config = SAMPLE_AGENTS["legal_agent"].copy()
        summarizer_config["name"] = "summarizer_agent"
        summarizer_config["display_name"] = "Summarizer Agent"
        summarizer_config["project_id"] = test_project["id"]
        summarizer_config["persona_prompt"] = "You are an expert at summarizing research findings."
        
        summarizer_response = await client.post("/api/v1/agents", json=summarizer_config)
        assert summarizer_response.status_code == 201
        summarizer_agent_id = summarizer_response.json()["agent_id"]
        
        # Create multi-agent workflow
        workflow_data = {
            "name": "Research and Summarize Workflow",
            "project_id": test_project["id"],
            "agents": [
                {
                    "agent_id": research_agent_id,
                    "role": "researcher",
                    "order": 1
                },
                {
                    "agent_id": summarizer_agent_id,
                    "role": "summarizer",
                    "order": 2
                }
            ],
            "flow": {
                "type": "sequential",
                "steps": [
                    {
                        "agent_id": research_agent_id,
                        "action": "research",
                        "output_to": "summarizer"
                    },
                    {
                        "agent_id": summarizer_agent_id,
                        "action": "summarize",
                        "input_from": "researcher"
                    }
                ]
            }
        }
        
        workflow_response = await client.post("/api/v1/workflows", json=workflow_data)
        assert workflow_response.status_code == 201
        workflow_id = workflow_response.json()["workflow_id"]
        
        # Execute workflow
        execution_data = {
            "input": "Research the latest developments in AI safety and provide a summary",
            "session_id": "multi-agent-test"
        }
        
        execution_response = await client.post(f"/api/v1/workflows/{workflow_id}/execute", json=execution_data)
        assert execution_response.status_code == 200
        
        result = execution_response.json()
        assert "execution_id" in result
        assert "status" in result
        assert "output" in result or "partial_output" in result
    
    @pytest.mark.asyncio
    async def test_gpu_training_complete_lifecycle(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers, mock_celery):
        """Test complete GPU training lifecycle from provisioning to cleanup."""
        
        # Step 1: Check available GPUs
        gpu_response = await client.get("/api/v1/gpu/available")
        assert gpu_response.status_code == 200
        available_gpus = gpu_response.json()["providers"]
        assert len(available_gpus) > 0
        
        # Step 2: Get cost estimation
        estimation_data = {
            "model_name": "microsoft/DialoGPT-small",
            "dataset_size_mb": 50,
            "epochs": 2,
            "gpu_type": "A100",
            "provider": "runpod"
        }
        
        estimate_response = await client.post("/api/v1/gpu/estimate-training-cost", json=estimation_data)
        assert estimate_response.status_code == 200
        estimated_cost = estimate_response.json()["estimated_cost"]
        
        # Step 3: Create training job with GPU
        job_data = get_sample_training_job("gpu_training")
        job_data["project_id"] = test_project["id"]
        job_data["use_gpu"] = True
        job_data["gpu_type"] = "A100"
        job_data["gpu_provider"] = "runpod"
        job_data["max_hours"] = 2.0
        
        training_response = await client.post("/api/v1/training/jobs", json=job_data)
        assert training_response.status_code == 201
        job_id = training_response.json()["job_id"]
        gpu_instance_id = training_response.json().get("gpu_instance_id")
        
        # Step 4: Monitor GPU instance
        if gpu_instance_id:
            gpu_status_response = await client.get(f"/api/v1/gpu/instances/{gpu_instance_id}")
            assert gpu_status_response.status_code == 200
            gpu_status = gpu_status_response.json()
            assert gpu_status["status"] in ["running", "provisioning"]
        
        # Step 5: Monitor training progress
        job_status_response = await client.get(f"/api/v1/training/jobs/{job_id}")
        assert job_status_response.status_code == 200
        job_status = job_status_response.json()["status"]
        assert job_status in ["queued", "running", "provisioning"]
        
        # Step 6: Get training logs
        logs_response = await client.get(f"/api/v1/training/jobs/{job_id}/logs")
        assert logs_response.status_code == 200
        assert len(logs_response.text) > 0
        
        # Step 7: Get training metrics
        metrics_response = await client.get(f"/api/v1/training/jobs/{job_id}/metrics")
        assert metrics_response.status_code == 200
        
        # Step 8: Check cost tracking
        cost_response = await client.get(f"/api/v1/training/jobs/{job_id}/cost")
        assert cost_response.status_code == 200
        cost_data = cost_response.json()
        assert "estimated_cost" in cost_data
        assert "actual_cost" in cost_data
        
        # Step 9: List all user's GPU instances
        instances_response = await client.get("/api/v1/gpu/instances")
        assert instances_response.status_code == 200
        instances = instances_response.json()["instances"]
        assert len(instances) > 0
        
        # Step 10: Get usage summary
        usage_response = await client.get("/api/v1/gpu/usage-summary")
        assert usage_response.status_code == 200
        usage = usage_response.json()
        assert "total_cost_today" in usage
        assert usage["total_cost_today"] >= 0
    
    @pytest.mark.asyncio
    async def test_project_management_workflow(self, client: AsyncClient, test_user, test_credits):
        """Test complete project management workflow."""
        
        # Step 1: Create project
        project_data = {
            "name": "AI Research Project",
            "description": "A comprehensive AI research project",
            "settings": {
                "budget_limit": 500.0,
                "auto_stop_on_budget": True
            }
        }
        
        project_response = await client.post("/api/v1/projects", json=project_data)
        assert project_response.status_code == 201
        project_id = project_response.json()["project_id"]
        
        # Step 2: Create multiple agents in project
        agent_configs = [
            SAMPLE_AGENTS["legal_agent"].copy(),
            SAMPLE_AGENTS["code_agent"].copy(),
            SAMPLE_AGENTS["customer_support"].copy()
        ]
        
        agent_ids = []
        for i, config in enumerate(agent_configs):
            config["name"] = f"project_agent_{i}"
            config["project_id"] = project_id
            
            agent_response = await client.post("/api/v1/agents", json=config)
            assert agent_response.status_code == 201
            agent_ids.append(agent_response.json()["agent_id"])
        
        # Step 3: Create training jobs for agents
        training_job_ids = []
        for i, agent_id in enumerate(agent_ids):
            job_data = get_sample_training_job("basic_lora")
            job_data["job_name"] = f"project_training_{i}"
            job_data["project_id"] = project_id
            job_data["use_gpu"] = False  # Keep costs low
            
            training_response = await client.post("/api/v1/training/jobs", json=job_data)
            assert training_response.status_code == 201
            training_job_ids.append(training_response.json()["job_id"])
        
        # Step 4: Monitor project costs
        project_cost_response = await client.get(f"/api/v1/projects/{project_id}/costs")
        assert project_cost_response.status_code == 200
        cost_data = project_cost_response.json()
        assert "total_spent" in cost_data
        assert "budget_remaining" in cost_data
        
        # Step 5: Get project analytics
        analytics_response = await client.get(f"/api/v1/projects/{project_id}/analytics")
        assert analytics_response.status_code == 200
        analytics = analytics_response.json()
        assert "agent_count" in analytics
        assert "training_jobs_count" in analytics
        assert analytics["agent_count"] == len(agent_ids)
        assert analytics["training_jobs_count"] == len(training_job_ids)
        
        # Step 6: Export project data
        export_response = await client.post(f"/api/v1/projects/{project_id}/export")
        assert export_response.status_code == 200
        export_data = export_response.json()
        assert "export_id" in export_data
        assert "status" in export_data
        
        # Step 7: Update project settings
        update_data = {
            "description": "Updated AI research project description",
            "settings": {
                "budget_limit": 750.0
            }
        }
        
        update_response = await client.patch(f"/api/v1/projects/{project_id}", json=update_data)
        assert update_response.status_code == 200
        updated_project = update_response.json()
        assert updated_project["settings"]["budget_limit"] == 750.0


class TestWebUIWorkflows:
    """Test workflows through the web interface."""
    
    @pytest.mark.skip(reason="Requires running web server")
    def test_dashboard_navigation(self, browser):
        """Test basic dashboard navigation."""
        # Navigate to dashboard
        browser.get("http://localhost:8000")
        
        # Wait for page to load
        wait = WebDriverWait(browser, 10)
        
        # Check if dashboard loads
        dashboard_title = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "dashboard-title"))
        )
        assert "RAGnetic" in dashboard_title.text
        
        # Test navigation tabs
        tabs = ["overview", "agents", "training", "gpu", "projects", "billing"]
        
        for tab in tabs:
            tab_element = browser.find_element(By.CSS_SELECTOR, f'[data-view="{tab}"]')
            tab_element.click()
            
            # Wait for tab content to load
            time.sleep(1)
            
            # Verify tab is active
            assert "active" in tab_element.get_attribute("class")
    
    @pytest.mark.skip(reason="Requires running web server")
    def test_agent_creation_ui(self, browser):
        """Test agent creation through UI."""
        browser.get("http://localhost:8000")
        wait = WebDriverWait(browser, 10)
        
        # Navigate to agents tab
        agents_tab = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-view="agents"]'))
        )
        agents_tab.click()
        
        # Click create agent button
        create_button = wait.until(
            EC.element_to_be_clickable((By.ID, "create-agent-btn"))
        )
        create_button.click()
        
        # Fill agent form
        name_input = wait.until(
            EC.presence_of_element_located((By.ID, "agent-name"))
        )
        name_input.send_keys("Test Agent")
        
        description_input = browser.find_element(By.ID, "agent-description")
        description_input.send_keys("A test agent created through UI")
        
        # Select model
        model_select = browser.find_element(By.ID, "llm-model")
        model_select.click()
        
        gpt4_option = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '[value="gpt-4o-mini"]'))
        )
        gpt4_option.click()
        
        # Submit form
        submit_button = browser.find_element(By.ID, "create-agent-submit")
        submit_button.click()
        
        # Wait for success message
        success_message = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "success-message"))
        )
        assert "created successfully" in success_message.text.lower()
    
    @pytest.mark.skip(reason="Requires running web server")
    def test_training_job_ui(self, browser):
        """Test training job creation through UI."""
        browser.get("http://localhost:8000")
        wait = WebDriverWait(browser, 10)
        
        # Navigate to training tab
        training_tab = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-view="training"]'))
        )
        training_tab.click()
        
        # Click create training job button
        create_button = wait.until(
            EC.element_to_be_clickable((By.ID, "create-training-job-btn"))
        )
        create_button.click()
        
        # Fill training job form
        job_name_input = wait.until(
            EC.presence_of_element_located((By.ID, "job-name"))
        )
        job_name_input.send_keys("UI Test Training Job")
        
        # Select base model
        model_select = browser.find_element(By.ID, "base-model")
        model_select.click()
        
        model_option = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '[value="microsoft/DialoGPT-small"]'))
        )
        model_option.click()
        
        # Configure GPU settings
        use_gpu_checkbox = browser.find_element(By.ID, "use-gpu")
        use_gpu_checkbox.click()
        
        gpu_type_select = browser.find_element(By.ID, "gpu-type")
        gpu_type_select.click()
        
        a100_option = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '[value="A100"]'))
        )
        a100_option.click()
        
        # Set max hours
        max_hours_input = browser.find_element(By.ID, "max-hours")
        max_hours_input.clear()
        max_hours_input.send_keys("1.0")
        
        # Submit form
        submit_button = browser.find_element(By.ID, "create-training-submit")
        submit_button.click()
        
        # Wait for job creation confirmation
        confirmation = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "training-job-created"))
        )
        assert "training job created" in confirmation.text.lower()


class TestErrorRecoveryWorkflows:
    """Test error handling and recovery in complete workflows."""
    
    @pytest.mark.asyncio
    async def test_training_job_failure_recovery(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers, mock_celery):
        """Test recovery from training job failures."""
        
        # Create a job that will fail
        job_data = get_sample_training_job("basic_lora")
        job_data["project_id"] = test_project["id"]
        job_data["dataset_path"] = "/nonexistent/dataset.jsonl"  # Invalid path
        
        training_response = await client.post("/api/v1/training/jobs", json=job_data)
        
        # Job creation might fail immediately or succeed and fail during execution
        if training_response.status_code == 201:
            job_id = training_response.json()["job_id"]
            
            # Wait a moment for job to potentially fail
            time.sleep(1)
            
            # Check job status
            status_response = await client.get(f"/api/v1/training/jobs/{job_id}")
            assert status_response.status_code == 200
            
            # Job might be failed or still queued
            status = status_response.json()["status"]
            
            if status == "failed":
                # Test retry mechanism
                retry_response = await client.post(f"/api/v1/training/jobs/{job_id}/retry")
                
                # Retry might succeed or fail depending on the error
                if retry_response.status_code == 200:
                    retry_result = retry_response.json()
                    assert "new_job_id" in retry_result
                else:
                    # Some errors might not be retryable
                    assert retry_response.status_code in [400, 409]
        else:
            # Job creation failed, which is also acceptable
            assert training_response.status_code in [400, 422]
    
    @pytest.mark.asyncio
    async def test_gpu_preemption_recovery(self, client: AsyncClient, test_user, test_project, test_credits, mock_provider_with_preemption, mock_celery):
        """Test recovery from GPU spot instance preemption."""
        
        # Create training job with spot instance
        job_data = get_sample_training_job("gpu_training")
        job_data["project_id"] = test_project["id"]
        job_data["use_gpu"] = True
        job_data["gpu_type"] = "A100"
        job_data["gpu_provider"] = "runpod"
        job_data["use_spot_instances"] = True  # Use cheaper spot instances
        
        training_response = await client.post("/api/v1/training/jobs", json=job_data)
        assert training_response.status_code == 201
        job_id = training_response.json()["job_id"]
        
        # Simulate preemption (mock provider will simulate this)
        with mock_provider_with_preemption():
            # Check if job handles preemption gracefully
            status_response = await client.get(f"/api/v1/training/jobs/{job_id}")
            assert status_response.status_code == 200
            
            # Job should either be retrying or failed with preemption reason
            status = status_response.json()["status"]
            assert status in ["retrying", "failed", "running", "queued"]
            
            if status == "failed":
                error_details = status_response.json().get("error_details", "")
                assert "preempt" in error_details.lower() or "interrupt" in error_details.lower()
    
    @pytest.mark.asyncio
    async def test_credit_exhaustion_recovery(self, client: AsyncClient, test_user, test_project, mock_all_providers, mock_celery):
        """Test handling of credit exhaustion during operations."""
        
        # Start with very low credits
        from app.services.credit_service import top_up
        await top_up(test_user.id, 1.0)  # Only $1
        
        # Try to create expensive training job
        job_data = get_sample_training_job("gpu_training")
        job_data["project_id"] = test_project["id"]
        job_data["use_gpu"] = True
        job_data["gpu_type"] = "H100"  # Expensive GPU
        job_data["max_hours"] = 10.0  # Long duration
        
        training_response = await client.post("/api/v1/training/jobs", json=job_data)
        
        # Should fail due to insufficient credits
        assert training_response.status_code == 400
        result = training_response.json()
        assert "insufficient credits" in result["detail"].lower()
        
        # Top up credits and retry
        await top_up(test_user.id, 200.0)  # Add more credits
        
        retry_response = await client.post("/api/v1/training/jobs", json=job_data)
        
        # Should now succeed
        assert retry_response.status_code == 201
        job_id = retry_response.json()["job_id"]
        
        # Verify job was created
        status_response = await client.get(f"/api/v1/training/jobs/{job_id}")
        assert status_response.status_code == 200


@pytest.mark.performance
class TestWorkflowPerformance:
    """Performance tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_concurrent_user_workflows(self, client: AsyncClient, user_factory, project_factory, mock_all_providers, mock_celery):
        """Test multiple users running workflows concurrently."""
        import asyncio
        
        async def user_workflow(user_id):
            """Simulate a complete user workflow."""
            try:
                # Create project
                project_data = project_factory()
                project_data["user_id"] = user_id
                project_response = await client.post("/api/v1/projects", json=project_data)
                if project_response.status_code != 201:
                    return False
                
                project_id = project_response.json()["project_id"]
                
                # Create agent
                agent_config = SAMPLE_AGENTS["legal_agent"].copy()
                agent_config["name"] = f"concurrent_agent_{user_id}"
                agent_config["project_id"] = project_id
                
                agent_response = await client.post("/api/v1/agents", json=agent_config)
                if agent_response.status_code != 201:
                    return False
                
                agent_id = agent_response.json()["agent_id"]
                
                # Query agent
                query_data = {
                    "message": "Test query",
                    "session_id": f"concurrent_session_{user_id}"
                }
                
                query_response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
                return query_response.status_code == 200
                
            except Exception as e:
                print(f"Workflow failed for user {user_id}: {e}")
                return False
        
        # Run workflows for 10 concurrent users
        tasks = [user_workflow(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful workflows
        successful = sum(1 for r in results if r is True)
        
        # At least 70% should succeed
        assert successful >= 7
    
    @pytest.mark.asyncio
    async def test_large_scale_training_workflow(self, client: AsyncClient, test_user, test_project, test_credits, mock_all_providers, mock_celery):
        """Test workflow with many training jobs."""
        import asyncio
        
        async def create_training_job(job_index):
            job_data = get_sample_training_job("basic_lora")
            job_data["job_name"] = f"scale_test_job_{job_index}"
            job_data["project_id"] = test_project["id"]
            job_data["use_gpu"] = False  # Keep it fast and cheap
            
            response = await client.post("/api/v1/training/jobs", json=job_data)
            return response.status_code == 201
        
        # Create 20 training jobs concurrently
        start_time = time.time()
        
        tasks = [create_training_job(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # All should succeed
        assert all(results)
        
        # Should complete within reasonable time
        assert duration < 30.0  # 30 seconds for 20 jobs
        
        # Verify all jobs were created
        list_response = await client.get("/api/v1/training/jobs")
        assert list_response.status_code == 200
        
        jobs = list_response.json()["jobs"]
        scale_test_jobs = [job for job in jobs if job["job_name"].startswith("scale_test_job_")]
        assert len(scale_test_jobs) == 20
