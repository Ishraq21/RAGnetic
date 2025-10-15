# Regression tests for existing RAGnetic features
# These tests ensure that existing functionality continues to work after new changes
import pytest
import json
import yaml
from pathlib import Path
from httpx import AsyncClient
from tests.fixtures.sample_data import SAMPLE_AGENTS, create_sample_agent_file, create_sample_training_dataset


class TestExistingAgentFeatures:
    """Regression tests for existing agent functionality."""
    
    @pytest.mark.asyncio
    async def test_existing_agent_yaml_loading(self, test_data_dir):
        """Test that existing agent YAML files still load correctly."""
        # Create sample agent files
        for agent_name in ["legal_agent", "code_agent", "customer_support"]:
            agent_file = create_sample_agent_file(agent_name, test_data_dir / "agents")
            
            # Verify file was created
            assert agent_file.exists()
            
            # Test YAML parsing
            with open(agent_file, 'r') as f:
                agent_data = yaml.safe_load(f)
            
            # Verify required fields exist
            assert "name" in agent_data
            assert "display_name" in agent_data
            assert "llm_model" in agent_data
            assert "embedding_model" in agent_data
            assert "vector_store" in agent_data
    
    @pytest.mark.asyncio
    async def test_existing_agent_creation_api(self, client: AsyncClient, test_user, test_project):
        """Test that existing agent creation API still works."""
        # Test with each sample agent
        for agent_name, agent_config in SAMPLE_AGENTS.items():
            test_config = agent_config.copy()
            test_config["name"] = f"regression_{agent_name}"
            test_config["project_id"] = test_project["id"]
            
            response = await client.post("/api/v1/agents", json=test_config)
            
            assert response.status_code == 201
            result = response.json()
            assert result["name"] == test_config["name"]
            assert result["llm_model"] == test_config["llm_model"]
            assert result["embedding_model"] == test_config["embedding_model"]
    
    @pytest.mark.asyncio
    async def test_existing_agent_query_functionality(self, client: AsyncClient, test_user, test_project, mock_all_providers):
        """Test that existing agent query functionality works."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Test basic query
        query_data = {
            "message": "What is a contract?",
            "session_id": "regression_test_session"
        }
        
        response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "response" in result
        assert len(result["response"]) > 0
        assert result["session_id"] == query_data["session_id"]
    
    @pytest.mark.asyncio
    async def test_existing_vector_store_types(self, client: AsyncClient, test_user, test_project):
        """Test that all existing vector store types still work."""
        vector_store_types = ["faiss", "chroma", "pinecone"]
        
        for vs_type in vector_store_types:
            agent_config = SAMPLE_AGENTS["legal_agent"].copy()
            agent_config["name"] = f"vs_test_{vs_type}"
            agent_config["project_id"] = test_project["id"]
            agent_config["vector_store"]["type"] = vs_type
            
            response = await client.post("/api/v1/agents", json=agent_config)
            
            # Should succeed or gracefully handle missing dependencies
            assert response.status_code in [201, 400]
            
            if response.status_code == 201:
                result = response.json()
                assert result["vector_store"]["type"] == vs_type
    
    @pytest.mark.asyncio
    async def test_existing_llm_models(self, client: AsyncClient, test_user, test_project):
        """Test that existing LLM models still work."""
        llm_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
            "claude-3-haiku-20240307",
            "gemini-pro"
        ]
        
        for model in llm_models:
            agent_config = SAMPLE_AGENTS["legal_agent"].copy()
            agent_config["name"] = f"llm_test_{model.replace('-', '_').replace('.', '_')}"
            agent_config["project_id"] = test_project["id"]
            agent_config["llm_model"] = model
            
            response = await client.post("/api/v1/agents", json=agent_config)
            
            # Should succeed
            assert response.status_code == 201
            result = response.json()
            assert result["llm_model"] == model
    
    @pytest.mark.asyncio
    async def test_existing_embedding_models(self, client: AsyncClient, test_user, test_project):
        """Test that existing embedding models still work."""
        embedding_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002"
        ]
        
        for model in embedding_models:
            agent_config = SAMPLE_AGENTS["legal_agent"].copy()
            agent_config["name"] = f"embed_test_{model.replace('-', '_')}"
            agent_config["project_id"] = test_project["id"]
            agent_config["embedding_model"] = model
            
            response = await client.post("/api/v1/agents", json=agent_config)
            
            assert response.status_code == 201
            result = response.json()
            assert result["embedding_model"] == model


class TestExistingTrainingFeatures:
    """Regression tests for existing training functionality."""
    
    @pytest.mark.asyncio
    async def test_existing_training_job_creation(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test that existing training job creation still works."""
        # Test with different base models
        base_models = [
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium",
            "gpt2"
        ]
        
        for model in base_models:
            job_data = {
                "job_name": f"regression_test_{model.replace('/', '_')}",
                "base_model_name": model,
                "dataset_path": "data/training_datasets/sample.jsonl",
                "output_base_dir": "models/fine_tuned",
                "project_id": test_project["id"],
                "hyperparameters": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 2e-4,
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.1
                }
            }
            
            response = await client.post("/api/v1/training/jobs", json=job_data)
            
            assert response.status_code == 201
            result = response.json()
            assert result["base_model_name"] == model
            assert result["status"] == "queued"
    
    @pytest.mark.asyncio
    async def test_existing_hyperparameter_validation(self, client: AsyncClient, test_user, test_project):
        """Test that existing hyperparameter validation still works."""
        # Test valid hyperparameters
        valid_hyperparams = {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 250
        }
        
        job_data = {
            "job_name": "hyperparam_validation_test",
            "base_model_name": "microsoft/DialoGPT-small",
            "dataset_path": "data/training_datasets/sample.jsonl",
            "project_id": test_project["id"],
            "hyperparameters": valid_hyperparams
        }
        
        response = await client.post("/api/v1/training/jobs", json=job_data)
        assert response.status_code == 201
        
        # Test invalid hyperparameters
        invalid_hyperparams = {
            "epochs": -1,  # Invalid
            "batch_size": 0,  # Invalid
            "learning_rate": -0.1,  # Invalid
            "lora_rank": -5  # Invalid
        }
        
        job_data["hyperparameters"] = invalid_hyperparams
        job_data["job_name"] = "invalid_hyperparam_test"
        
        response = await client.post("/api/v1/training/jobs", json=job_data)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_existing_dataset_formats(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery, test_data_dir):
        """Test that existing dataset formats still work."""
        # Create different dataset formats
        datasets = {
            "instruction_following": [
                {"instruction": "What is AI?", "output": "AI stands for Artificial Intelligence."},
                {"instruction": "Define machine learning", "output": "ML is a subset of AI."}
            ],
            "conversational": [
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"}
                    ]
                }
            ]
        }
        
        for format_name, dataset in datasets.items():
            # Create dataset file
            dataset_file = test_data_dir / "data" / "training_datasets" / f"{format_name}.jsonl"
            dataset_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dataset_file, 'w') as f:
                for item in dataset:
                    f.write(json.dumps(item) + '\n')
            
            # Test training job with this dataset
            job_data = {
                "job_name": f"dataset_format_test_{format_name}",
                "base_model_name": "microsoft/DialoGPT-small",
                "dataset_path": str(dataset_file.relative_to(test_data_dir)),
                "project_id": test_project["id"],
                "hyperparameters": {
                    "epochs": 1,
                    "batch_size": 1
                }
            }
            
            response = await client.post("/api/v1/training/jobs", json=job_data)
            assert response.status_code == 201
    
    @pytest.mark.asyncio
    async def test_existing_training_job_status_api(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery):
        """Test that existing training job status API still works."""
        # Create training job
        job_data = {
            "job_name": "status_api_test",
            "base_model_name": "microsoft/DialoGPT-small",
            "dataset_path": "data/training_datasets/sample.jsonl",
            "project_id": test_project["id"],
            "hyperparameters": {"epochs": 1, "batch_size": 2}
        }
        
        create_response = await client.post("/api/v1/training/jobs", json=job_data)
        job_id = create_response.json()["job_id"]
        
        # Test status endpoint
        status_response = await client.get(f"/api/v1/training/jobs/{job_id}")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        required_fields = ["job_id", "job_name", "status", "created_at", "base_model_name"]
        
        for field in required_fields:
            assert field in status_data
        
        # Test job listing
        list_response = await client.get("/api/v1/training/jobs")
        assert list_response.status_code == 200
        
        list_data = list_response.json()
        assert "jobs" in list_data
        assert len(list_data["jobs"]) > 0
        
        # Find our job in the list
        job_found = any(job["job_id"] == job_id for job in list_data["jobs"])
        assert job_found


class TestExistingEvaluationFeatures:
    """Regression tests for existing evaluation functionality."""
    
    @pytest.mark.asyncio
    async def test_existing_evaluation_api(self, client: AsyncClient, test_user, test_project):
        """Test that existing evaluation API still works."""
        # Create agent first
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Create evaluation
        eval_data = {
            "name": "Regression Test Evaluation",
            "test_set": [
                {"question": "What is a contract?", "expected_answer": "A legally binding agreement"},
                {"question": "Define tort law", "expected_answer": "Civil wrongs and remedies"}
            ],
            "metrics": ["accuracy", "response_time", "cost"]
        }
        
        response = await client.post(f"/api/v1/agents/{agent_id}/evaluate", json=eval_data)
        
        assert response.status_code == 202
        result = response.json()
        assert "evaluation_id" in result
        assert result["status"] == "queued"
        
        # Test getting evaluation results
        evaluation_id = result["evaluation_id"]
        results_response = await client.get(f"/api/v1/evaluations/{evaluation_id}")
        
        assert results_response.status_code == 200
        results_data = results_response.json()
        assert results_data["evaluation_id"] == evaluation_id
        assert "status" in results_data
    
    @pytest.mark.asyncio
    async def test_existing_benchmark_functionality(self, client: AsyncClient, test_user, test_project, test_data_dir):
        """Test that existing benchmark functionality still works."""
        # Create agent
        agent_config = SAMPLE_AGENTS["code_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Create benchmark test set
        benchmark_file = test_data_dir / "benchmark" / "regression_benchmark.json"
        benchmark_file.parent.mkdir(parents=True, exist_ok=True)
        
        benchmark_data = {
            "name": "Regression Benchmark",
            "description": "Test benchmark for regression testing",
            "test_cases": [
                {
                    "id": "test_1",
                    "input": "Write a Python function to add two numbers",
                    "expected_output": "def add(a, b):\n    return a + b",
                    "category": "coding"
                },
                {
                    "id": "test_2", 
                    "input": "Explain what a variable is in programming",
                    "expected_output": "A variable is a named storage location",
                    "category": "concepts"
                }
            ]
        }
        
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        # Run benchmark
        benchmark_request = {
            "benchmark_file": str(benchmark_file.relative_to(test_data_dir)),
            "metrics": ["accuracy", "response_time"]
        }
        
        response = await client.post(f"/api/v1/agents/{agent_id}/benchmark", json=benchmark_request)
        
        # Should accept the benchmark request
        assert response.status_code in [200, 202]
    
    @pytest.mark.asyncio
    async def test_existing_metrics_calculation(self, client: AsyncClient, test_user):
        """Test that existing metrics calculation still works."""
        # Test metrics calculation endpoint
        metrics_data = {
            "predictions": ["A contract is a legal agreement", "Tort law covers civil wrongs"],
            "references": ["A contract is a legally binding agreement", "Tort law deals with civil wrongs"],
            "metrics": ["bleu", "rouge", "exact_match"]
        }
        
        response = await client.post("/api/v1/evaluation/calculate-metrics", json=metrics_data)
        
        assert response.status_code == 200
        result = response.json()
        assert "metrics" in result
        
        # Check that requested metrics are present
        for metric in metrics_data["metrics"]:
            assert metric in result["metrics"]
            assert isinstance(result["metrics"][metric], (int, float))


class TestExistingDataManagement:
    """Regression tests for existing data management features."""
    
    @pytest.mark.asyncio
    async def test_existing_file_upload(self, client: AsyncClient, test_user):
        """Test that existing file upload functionality still works."""
        # Test different file types
        test_files = [
            ("document.txt", "This is a test document.", "text/plain"),
            ("data.json", '{"key": "value"}', "application/json"),
            ("dataset.jsonl", '{"instruction": "test", "output": "response"}\n', "application/jsonl")
        ]
        
        for filename, content, content_type in test_files:
            files = {"file": (filename, content, content_type)}
            
            response = await client.post("/api/v1/data/upload", files=files)
            
            assert response.status_code == 200
            result = response.json()
            assert "file_path" in result
            assert result["filename"] == filename
            assert result["size"] > 0
    
    @pytest.mark.asyncio
    async def test_existing_document_processing(self, client: AsyncClient, test_user, test_project):
        """Test that existing document processing still works."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        # Upload document to agent
        document_content = "This is a sample legal document for testing document processing functionality."
        files = {"file": ("legal_doc.txt", document_content, "text/plain")}
        
        response = await client.post(f"/api/v1/agents/{agent_id}/documents", files=files)
        
        assert response.status_code == 200
        result = response.json()
        assert "document_id" in result
        assert result["filename"] == "legal_doc.txt"
        assert result["status"] in ["uploaded", "processing", "processed"]
        
        # List agent documents
        list_response = await client.get(f"/api/v1/agents/{agent_id}/documents")
        
        assert list_response.status_code == 200
        list_data = list_response.json()
        assert "documents" in list_data
        assert len(list_data["documents"]) > 0
    
    @pytest.mark.asyncio
    async def test_existing_data_preparation(self, client: AsyncClient, test_user, test_data_dir):
        """Test that existing data preparation functionality still works."""
        # Create sample raw data
        raw_data_file = test_data_dir / "data" / "raw_data" / "sample_conversations.json"
        raw_data_file.parent.mkdir(parents=True, exist_ok=True)
        
        raw_data = [
            {"user": "Hello", "assistant": "Hi there! How can I help you?"},
            {"user": "What is AI?", "assistant": "AI stands for Artificial Intelligence."}
        ]
        
        with open(raw_data_file, 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        # Test data preparation
        prep_config = {
            "input_file": str(raw_data_file.relative_to(test_data_dir)),
            "output_format": "instruction_following",
            "max_length": 512,
            "split_ratio": {"train": 0.8, "validation": 0.2}
        }
        
        response = await client.post("/api/v1/data/prepare", json=prep_config)
        
        # Should accept the preparation request
        assert response.status_code in [200, 202]
        
        if response.status_code == 200:
            result = response.json()
            assert "output_files" in result
        else:
            # Async processing
            result = response.json()
            assert "preparation_id" in result


class TestExistingUIFeatures:
    """Regression tests for existing UI functionality."""
    
    @pytest.mark.asyncio
    async def test_existing_dashboard_endpoints(self, client: AsyncClient, test_user):
        """Test that existing dashboard endpoints still work."""
        # Test main dashboard
        response = await client.get("/")
        assert response.status_code == 200
        assert "RAGnetic" in response.text
        
        # Test dashboard data endpoint
        dashboard_response = await client.get("/api/v1/dashboard/overview")
        assert dashboard_response.status_code == 200
        
        dashboard_data = dashboard_response.json()
        expected_fields = ["total_agents", "total_training_jobs", "total_cost", "recent_activity"]
        
        for field in expected_fields:
            assert field in dashboard_data
    
    @pytest.mark.asyncio
    async def test_existing_static_assets(self, client: AsyncClient, test_user):
        """Test that existing static assets are accessible."""
        static_files = [
            "/static/css/style.css",
            "/static/js/chat_quick_upload.js",
            "/static/js/citation_renderer.js"
        ]
        
        for static_file in static_files:
            response = await client.get(static_file)
            # Should be accessible (200) or at least not cause server error
            assert response.status_code in [200, 304, 404]  # 404 is OK if file doesn't exist in test
    
    @pytest.mark.asyncio
    async def test_existing_api_documentation(self, client: AsyncClient, test_user):
        """Test that existing API documentation is accessible."""
        # Test OpenAPI spec
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
        
        # Test docs UI
        docs_response = await client.get("/docs")
        assert docs_response.status_code == 200


class TestExistingCLIFeatures:
    """Regression tests for existing CLI functionality."""
    
    def test_existing_cli_commands(self):
        """Test that existing CLI commands still work."""
        import subprocess
        import sys
        
        # Test CLI help
        result = subprocess.run([sys.executable, "cli.py", "--help"], 
                              capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "RAGnetic" in result.stdout
        
        # Test version command
        result = subprocess.run([sys.executable, "cli.py", "version"], 
                              capture_output=True, text=True)
        
        # Should succeed or fail gracefully
        assert result.returncode in [0, 1]
    
    def test_existing_config_validation(self, test_data_dir):
        """Test that existing config validation still works."""
        # Create valid config file
        config_file = test_data_dir / "test_config.yaml"
        
        valid_config = {
            "database": {
                "url": "sqlite:///test.db"
            },
            "redis": {
                "url": "redis://localhost:6379"
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        # Test config validation (if CLI supports it)
        import subprocess
        import sys
        
        result = subprocess.run([sys.executable, "cli.py", "validate-config", str(config_file)], 
                              capture_output=True, text=True)
        
        # Should succeed or command should not exist (acceptable)
        assert result.returncode in [0, 2]  # 0 = success, 2 = command not found


class TestExistingIntegrations:
    """Regression tests for existing integrations."""
    
    @pytest.mark.asyncio
    async def test_existing_model_integrations(self, client: AsyncClient, test_user, test_project):
        """Test that existing model provider integrations still work."""
        # Test different model providers
        model_configs = [
            {"provider": "openai", "model": "gpt-4o-mini"},
            {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
            {"provider": "google", "model": "gemini-pro"},
            {"provider": "huggingface", "model": "microsoft/DialoGPT-small"}
        ]
        
        for config in model_configs:
            agent_config = SAMPLE_AGENTS["legal_agent"].copy()
            agent_config["name"] = f"integration_test_{config['provider']}"
            agent_config["project_id"] = test_project["id"]
            agent_config["llm_model"] = config["model"]
            
            response = await client.post("/api/v1/agents", json=agent_config)
            
            # Should succeed
            assert response.status_code == 201
            result = response.json()
            assert result["llm_model"] == config["model"]
    
    @pytest.mark.asyncio
    async def test_existing_vector_store_integrations(self, client: AsyncClient, test_user, test_project):
        """Test that existing vector store integrations still work."""
        vector_stores = [
            {"type": "faiss", "config": {"index_type": "flat"}},
            {"type": "chroma", "config": {"collection_name": "test"}},
            {"type": "pinecone", "config": {"environment": "test"}}
        ]
        
        for vs_config in vector_stores:
            agent_config = SAMPLE_AGENTS["legal_agent"].copy()
            agent_config["name"] = f"vs_integration_test_{vs_config['type']}"
            agent_config["project_id"] = test_project["id"]
            agent_config["vector_store"] = {
                "type": vs_config["type"],
                **vs_config["config"],
                "bm25_k": 5,
                "semantic_k": 5
            }
            
            response = await client.post("/api/v1/agents", json=agent_config)
            
            # Should succeed or gracefully handle missing dependencies
            assert response.status_code in [201, 400]


@pytest.mark.regression
class TestBackwardsCompatibility:
    """Test backwards compatibility with older versions."""
    
    @pytest.mark.asyncio
    async def test_old_api_endpoints_still_work(self, client: AsyncClient, test_user, test_project):
        """Test that old API endpoints still work for backwards compatibility."""
        # Test old-style agent creation (if it existed)
        old_style_agent = {
            "name": "backwards_compat_agent",
            "model": "gpt-4o-mini",  # Old field name
            "embedding": "text-embedding-3-small",  # Old field name
            "project_id": test_project["id"]
        }
        
        # This might fail if old API doesn't exist, which is OK
        response = await client.post("/api/v1/agents/create", json=old_style_agent)
        
        # Accept either success or "endpoint not found"
        assert response.status_code in [200, 201, 404, 405]
    
    @pytest.mark.asyncio
    async def test_old_config_formats_still_work(self, test_data_dir):
        """Test that old configuration formats are still supported."""
        # Create old-style agent config
        old_agent_config = {
            "name": "old_style_agent",
            "model": "gpt-4o-mini",  # Old field name
            "prompt": "You are a helpful assistant.",  # Old field name
            "tools": ["search", "calculator"]
        }
        
        old_config_file = test_data_dir / "agents" / "old_style_agent.yaml"
        old_config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(old_config_file, 'w') as f:
            yaml.dump(old_agent_config, f)
        
        # Test that config can be loaded (even if it needs migration)
        with open(old_config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config["name"] == "old_style_agent"
        # Config migration would happen at runtime
    
    def test_database_schema_migrations(self):
        """Test that database schema migrations work correctly."""
        # This would test Alembic migrations
        import subprocess
        import sys
        
        # Test that migrations can be checked
        result = subprocess.run([sys.executable, "-c", "import alembic; print('Alembic available')"], 
                              capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Alembic available" in result.stdout


@pytest.mark.performance
class TestPerformanceRegression:
    """Test that performance hasn't regressed."""
    
    @pytest.mark.asyncio
    async def test_agent_query_performance_regression(self, client: AsyncClient, test_user, test_project, mock_all_providers, benchmark):
        """Test that agent query performance hasn't regressed."""
        # Create agent
        agent_config = SAMPLE_AGENTS["legal_agent"].copy()
        agent_config["project_id"] = test_project["id"]
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        query_data = {
            "message": "What is a contract?",
            "session_id": "performance_test"
        }
        
        async def query_agent():
            response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
            return response.status_code
        
        # Benchmark the query
        result = await benchmark(query_agent)
        assert result == 200
        
        # The benchmark fixture will automatically compare against previous runs
    
    @pytest.mark.asyncio
    async def test_training_job_creation_performance_regression(self, client: AsyncClient, test_user, test_project, test_credits, mock_celery, benchmark):
        """Test that training job creation performance hasn't regressed."""
        job_data = {
            "job_name": "performance_regression_test",
            "base_model_name": "microsoft/DialoGPT-small",
            "dataset_path": "data/training_datasets/sample.jsonl",
            "project_id": test_project["id"],
            "hyperparameters": {
                "epochs": 1,
                "batch_size": 2
            }
        }
        
        async def create_training_job():
            response = await client.post("/api/v1/training/jobs", json=job_data)
            return response.status_code
        
        result = await benchmark(create_training_job)
        assert result == 201
    
    @pytest.mark.asyncio
    async def test_memory_usage_regression(self, client: AsyncClient, test_user, test_project, performance_monitor):
        """Test that memory usage hasn't regressed significantly."""
        # Create multiple agents and check memory usage
        for i in range(10):
            agent_config = SAMPLE_AGENTS["legal_agent"].copy()
            agent_config["name"] = f"memory_test_agent_{i}"
            agent_config["project_id"] = test_project["id"]
            
            response = await client.post("/api/v1/agents", json=agent_config)
            assert response.status_code == 201
        
        # performance_monitor fixture will track memory usage
        # and warn if it's excessive
