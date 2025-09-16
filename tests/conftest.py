# RAGnetic Comprehensive Test Configuration
# This file bootstraps the entire test environment with all necessary fixtures

import asyncio
import os
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from faker import Faker
import factory
import pytest_asyncio

# Ensure provider mocks are registered as fixtures
from tests.fixtures.mock_providers import (
    mock_runpod_transport,
    mock_coreweave_transport,
    mock_vast_transport,
    mock_all_providers,
    mock_provider_failures,
    mock_provider_with_preemption,
    mock_slow_provider,
)

# Import RAGnetic components
from app.main import app
from app.db import get_db
from app.db.models import metadata, users_table, projects_table, user_credits_table
from app.schemas.security import User
from app.core.config import get_path_settings

fake = Faker()

# Test environment setup
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session", autouse=True)
def test_env():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Core test settings
    os.environ["RAGNETIC_ENV"] = "test"
    os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"
    os.environ["CELERY_TASK_EAGER_PROPAGATES"] = "true"
    
    # Provider API keys (mock)
    os.environ["RUNPOD_API_KEY"] = "test-runpod-key-12345"
    os.environ["COREWEAVE_API_KEY"] = "test-coreweave-key-12345"
    os.environ["VAST_API_KEY"] = "test-vast-key-12345"
    
    # Storage
    os.environ["S3_BUCKET"] = "ragnetic-test-bucket"
    os.environ["S3_ACCESS_KEY"] = "test-access-key"
    os.environ["S3_SECRET_KEY"] = "test-secret-key"
    os.environ["S3_REGION"] = "us-east-1"
    
    # Billing
    os.environ["STRIPE_SECRET_KEY"] = "sk_test_12345"
    os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_test_12345"
    
    # Security
    os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"
    
    # Disable external services
    os.environ["DISABLE_EXTERNAL_APIS"] = "true"
    os.environ["MOCK_PROVIDERS"] = "true"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ragnetic_test_"))
    
    # Create necessary subdirectories
    (temp_dir / "agents").mkdir()
    (temp_dir / "data" / "training_datasets").mkdir(parents=True)
    (temp_dir / "data" / "uploads").mkdir()
    (temp_dir / "models" / "fine_tuned").mkdir(parents=True)
    (temp_dir / "benchmark").mkdir()
    
    # Create sample files
    sample_agent = """
name: test_agent
display_name: Test Agent
description: A test agent for testing purposes
persona_prompt: You are a helpful test assistant.
embedding_model: text-embedding-3-small
llm_model: gpt-4o-mini
sources: []
tools: []
vector_store:
  type: faiss
  bm25_k: 5
  semantic_k: 5
"""
    (temp_dir / "agents" / "test_agent.yaml").write_text(sample_agent)
    
    # Sample training dataset
    sample_training = '''{"instruction": "What is AI?", "output": "AI stands for Artificial Intelligence."}
{"instruction": "How does machine learning work?", "output": "Machine learning uses algorithms to learn from data."}
'''
    (temp_dir / "data" / "training_datasets" / "sample.jsonl").write_text(sample_training)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest_asyncio.fixture(scope="session")
async def test_db_engine():
    """Create test database engine with in-memory SQLite."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        future=True,
        echo=False  # Set to True for SQL debugging
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    
    yield engine
    await engine.dispose()

@pytest_asyncio.fixture
async def db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for each test."""
    async_session = sessionmaker(
        bind=test_db_engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest_asyncio.fixture
async def test_user(db_session) -> User:
    """Create a test user."""
    from app.db.dao import create_user
    from app.schemas.security import UserCreate
    
    unique = fake.uuid4()
    user_data = UserCreate(
        username=f"testuser_{unique}",
        email=f"test_{unique}@example.com",
        password="testpassword123",
        first_name="Test",
        last_name="User",
        is_active=True,
        is_superuser=True,
        roles=["admin"]
    )
    
    user_dict = await create_user(db_session, user_data)
    return User(**user_dict)

@pytest_asyncio.fixture
async def test_project(db_session, test_user):
    """Create a test project."""
    from sqlalchemy import insert
    
    project_data = {
        "id": f"test-project-{fake.uuid4()}",
        "name": f"Test Project {fake.company()}",
        "description": "A test project for testing",
        "user_id": test_user.id
    }
    
    stmt = insert(projects_table).values(**project_data)
    await db_session.execute(stmt)
    await db_session.commit()
    
    return project_data

@pytest_asyncio.fixture
async def test_credits(db_session, test_user):
    """Give test user credits."""
    from app.services.credit_service import top_up
    await top_up(test_user.id, 100.0)  # $100 test credits
    return 100.0

@pytest.fixture(autouse=True)
def override_dependencies(db_session, test_user, test_data_dir):
    """Override FastAPI dependencies for testing."""
    
    # Override database dependency
    async def _get_test_db():
        yield db_session
    
    # Override auth dependency - always return test user
    async def _get_test_user():
        return test_user
    
    # Override path settings
    def _get_test_paths():
        return {
            "PROJECT_ROOT": test_data_dir,
            "AGENTS_DIR": test_data_dir / "agents",
            "DATA_DIR": test_data_dir / "data",
            "MODELS_DIR": test_data_dir / "models",
            "BENCHMARK_DIR": test_data_dir / "benchmark"
        }
    
    # Apply overrides
    from app.db import get_db
    from app.core.security import get_current_user_from_api_key
    from app.core.config import get_path_settings
    
    app.dependency_overrides[get_db] = _get_test_db
    app.dependency_overrides[get_current_user_from_api_key] = _get_test_user
    
    # Patch path settings
    with patch('app.core.config.get_path_settings', side_effect=_get_test_paths):
        yield
    
    # Clean up
    app.dependency_overrides.clear()

@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create HTTP client for API testing using ASGITransport."""
    # Initialize database connections for testing
    from app.db import initialize_db_connections
    from app.core.config import get_memory_storage_config
    
    try:
        mem_cfg = get_memory_storage_config()
        conn_name = mem_cfg.get('connection_name', 'default')
        initialize_db_connections(conn_name)
    except Exception as e:
        # If database initialization fails, continue without it
        print(f"Warning: Database initialization failed in test: {e}")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": "test-api-key"}
    ) as client:
        yield client

# Mock fixtures for external services
@pytest.fixture
def mock_runpod():
    """Mock RunPod API responses."""
    with patch('app.services.gpu_providers.runpod.httpx.Client') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        # Mock successful responses
        mock_instance.post.return_value.status_code = 200
        mock_instance.post.return_value.json.return_value = {
            "id": "test-pod-123",
            "status": "RUNNING",
            "price": 2.0
        }
        
        mock_instance.get.return_value.status_code = 200
        mock_instance.get.return_value.json.return_value = {
            "id": "test-pod-123",
            "status": "RUNNING"
        }
        
        mock_instance.get.return_value.text = "Training log line 1\nTraining log line 2\n"
        
        yield mock_instance

@pytest.fixture
def mock_stripe():
    """Mock Stripe API."""
    with patch('stripe.PaymentIntent') as mock_pi, \
         patch('stripe.Webhook') as mock_webhook:
        
        mock_pi.create.return_value = Mock(
            id="pi_test_123",
            status="succeeded",
            amount=5000,  # $50.00
            currency="usd"
        )
        
        mock_webhook.construct_event.return_value = {
            "type": "payment_intent.succeeded",
            "data": {
                "object": {
                    "id": "pi_test_123",
                    "amount": 5000,
                    "metadata": {"user_id": "1"}
                }
            }
        }
        
        yield {"payment_intent": mock_pi, "webhook": mock_webhook}

@pytest.fixture
def mock_s3():
    """Mock S3/boto3 operations."""
    with patch('boto3.client') as mock_boto:
        mock_s3_client = Mock()
        mock_boto.return_value = mock_s3_client
        
        # Mock upload
        mock_s3_client.upload_file.return_value = None
        mock_s3_client.upload_fileobj.return_value = None
        
        # Mock download
        mock_s3_client.download_file.return_value = None
        
        # Mock generate presigned URL
        mock_s3_client.generate_presigned_url.return_value = "https://s3.amazonaws.com/test-bucket/test-key?signed"
        
        yield mock_s3_client

@pytest.fixture
def mock_celery():
    """Mock Celery task execution."""
    with patch('app.training.trainer_tasks.fine_tune_llm_task') as mock_task:
        mock_task.delay.return_value = Mock(id="test-task-123")
        mock_task.apply_async.return_value = Mock(id="test-task-123")
        yield mock_task

# Factory fixtures for creating test data
class UserFactory(factory.Factory):
    class Meta:
        model = dict
    
    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@test.com")
    first_name = factory.Faker('first_name')
    last_name = factory.Faker('last_name')
    is_active = True
    is_superuser = False

class ProjectFactory(factory.Factory):
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: f"proj-{fake.uuid4()}")
    name = factory.Faker('company')
    description = factory.Faker('text', max_nb_chars=200)

@pytest.fixture
def user_factory():
    return UserFactory

@pytest.fixture
def project_factory():
    return ProjectFactory

# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    import time
    import psutil
    import gc
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    yield
    
    gc.collect()  # Force garbage collection
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    duration = end_time - start_time
    memory_diff = end_memory - start_memory
    
    if duration > 5.0:  # Warn if test takes > 5 seconds
        print(f"!  Slow test: {duration:.2f}s")
    
    if memory_diff > 50 * 1024 * 1024:  # Warn if memory usage > 50MB
        print(f"!  High memory usage: {memory_diff / 1024 / 1024:.1f}MB")

# Async generator fixtures for streaming tests
@pytest_asyncio.fixture
async def websocket_client():
    """WebSocket client for real-time features."""
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        yield client

# Security testing fixtures
@pytest.fixture
def security_headers():
    """Security headers for testing."""
    return {
        "X-API-Key": "test-api-key",
        "User-Agent": "RAGnetic-Test/1.0",
        "X-Forwarded-For": "127.0.0.1",
        "X-Real-IP": "127.0.0.1"
    }

@pytest.fixture
def malicious_payloads():
    """Common attack payloads for security testing."""
    return {
        "sql_injection": ["'; DROP TABLE users; --", "1' OR '1'='1", "admin'--"],
        "xss": ["<script>alert('xss')</script>", "javascript:alert('xss')", "<img src=x onerror=alert('xss')>"],
        "path_traversal": ["../../../etc/passwd", "..\\..\\..\\windows\\system32\\config\\sam", "....//....//etc/passwd"],
        "command_injection": ["; ls -la", "| whoami", "&& cat /etc/passwd"],
        "large_input": "A" * 10000,
        "null_bytes": "test\x00.txt",
        "unicode_bypass": "test\u202e.txt"
    }
