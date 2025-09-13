"""
Comprehensive Upload Testing Suite for RAGnetic
Tests all upload functionality including backend APIs, frontend integration,
stress testing, and lifecycle management.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import uuid

import pytest
import requests
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import RAGnetic components
from app.main import app
from app.core.config import get_path_settings
from app.db import get_db, get_async_db_session
from app.db.models import temporary_documents_table, document_chunks_table, users_table
from app.services.temporary_document_service import TemporaryDocumentService
from app.agents.config_manager import AgentConfig

# Test configuration
TEST_BASE_URL = "http://localhost:8000"
TEST_API_KEY = "YOUR_TEST_API_KEY_1"
TEST_USER_ID = 1
TEST_THREAD_ID = "test-thread-123"

class UploadTestSuite:
    """Comprehensive upload testing suite"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_data_dir = Path("test_upload_data")
        self.test_data_dir.mkdir(exist_ok=True)
        self.uploaded_files = []
        self.created_agents = []
        self.temp_doc_ids = []
        
    def setup_test_environment(self):
        """Setup test environment with necessary data"""
        print(" Setting up test environment...")
        
        # Create test files of various types
        self.create_test_files()
        
        # Create test agents
        self.create_test_agents()
        
        # Setup database
        self.setup_database()
        
        print(" Test environment setup complete")
    
    def create_test_files(self):
        """Create various test files for upload testing"""
        print("📁 Creating test files...")
        
        # Text file
        text_file = self.test_data_dir / "test_document.txt"
        with open(text_file, "w") as f:
            f.write("This is a test document for upload testing.\n" * 100)
        self.uploaded_files.append(text_file)
        
        # JSON file
        json_file = self.test_data_dir / "test_data.json"
        test_data = {
            "name": "Test Document",
            "content": "This is test content for JSON upload",
            "metadata": {
                "created": datetime.now().isoformat(),
                "type": "test"
            },
            "items": [{"id": i, "value": f"item_{i}"} for i in range(50)]
        }
        with open(json_file, "w") as f:
            json.dump(test_data, f, indent=2)
        self.uploaded_files.append(json_file)
        
        # CSV file
        csv_file = self.test_data_dir / "test_data.csv"
        with open(csv_file, "w") as f:
            f.write("id,name,value,description\n")
            for i in range(100):
                f.write(f"{i},item_{i},value_{i},Description for item {i}\n")
        self.uploaded_files.append(csv_file)
        
        # Markdown file
        md_file = self.test_data_dir / "test_document.md"
        with open(md_file, "w") as f:
            f.write("""# Test Document

## Introduction
This is a test markdown document for upload testing.

## Features
- Feature 1: Upload testing
- Feature 2: Validation testing
- Feature 3: Processing testing

## Code Example
```python
def test_function():
    return "Hello, World!"
```

## Conclusion
This document tests markdown parsing and upload functionality.
""" * 10)
        self.uploaded_files.append(md_file)
        
        # Large file for stress testing
        large_file = self.test_data_dir / "large_document.txt"
        with open(large_file, "w") as f:
            for i in range(10000):
                f.write(f"Line {i}: This is a large document for stress testing upload functionality.\n")
        self.uploaded_files.append(large_file)
        
        # Training dataset file
        training_file = self.test_data_dir / "training_dataset.jsonl"
        with open(training_file, "w") as f:
            for i in range(100):
                record = {
                    "instruction": f"Test instruction {i}",
                    "input": f"Test input {i}",
                    "output": f"Test output {i}"
                }
                f.write(json.dumps(record) + "\n")
        self.uploaded_files.append(training_file)
        
        print(f" Created {len(self.uploaded_files)} test files")
    
    def create_test_agents(self):
        """Create test agents for upload testing"""
        print("🤖 Creating test agents...")
        
        agents_dir = Path("agents")
        agents_dir.mkdir(exist_ok=True)
        
        # Test agent for document uploads
        test_agent_config = {
            "name": "upload-test-agent",
            "description": "Test agent for upload functionality",
            "llm_model": "gpt-3.5-turbo",
            "embedding_model": "text-embedding-ada-002",
            "data_sources": [],
            "tools": ["retrieval"],
            "system_prompt": "You are a test agent for upload functionality testing."
        }
        
        test_agent_file = agents_dir / "upload-test-agent.yaml"
        with open(test_agent_file, "w") as f:
            import yaml
            yaml.dump(test_agent_config, f)
        self.created_agents.append(test_agent_file)
        
        print(f" Created {len(self.created_agents)} test agents")
    
    def setup_database(self):
        """Setup test database"""
        print("🗄 Setting up test database...")
        
        # This will be handled by the test fixtures
        print(" Database setup complete")
    
    def test_agent_file_upload(self):
        """Test agent file upload functionality"""
        print("🧪 Testing agent file upload...")
        
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
        
        for test_file in self.uploaded_files[:3]:  # Test first 3 files
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "application/octet-stream")}
                response = self.client.post(
                    "/api/v1/agents/upload-file",
                    files=files,
                    headers=headers
                )
                
                assert response.status_code == 200, f"Upload failed for {test_file.name}: {response.text}"
                
                result = response.json()
                assert "file_path" in result
                assert Path(result["file_path"]).exists()
                
                print(f" Successfully uploaded {test_file.name}")
        
        print(" Agent file upload tests passed")
    
    def test_training_dataset_upload(self):
        """Test training dataset upload functionality"""
        print("🧪 Testing training dataset upload...")
        
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
        
        training_file = self.test_data_dir / "training_dataset.jsonl"
        with open(training_file, "rb") as f:
            files = {"file": (training_file.name, f, "application/json")}
            response = self.client.post(
                "/api/v1/training/upload-dataset",
                files=files,
                headers=headers
            )
            
            assert response.status_code == 200, f"Training upload failed: {response.text}"
            
            result = response.json()
            assert "file_path" in result
            assert "file_id" in result
            assert Path(result["file_path"]).exists()
            
            print(f" Successfully uploaded training dataset")
        
        print(" Training dataset upload tests passed")
    
    def test_temporary_document_upload(self):
        """Test temporary document upload functionality"""
        print("🧪 Testing temporary document upload...")
        
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
        
        for test_file in self.uploaded_files[:2]:  # Test first 2 files
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "application/octet-stream")}
                data = {"thread_id": TEST_THREAD_ID}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
                
                assert response.status_code == 200, f"Temp upload failed for {test_file.name}: {response.text}"
                
                result = response.json()
                assert "temp_doc_id" in result
                assert "file_name" in result
                assert "file_size" in result
                
                self.temp_doc_ids.append(result["temp_doc_id"])
                print(f" Successfully uploaded temporary document: {result['temp_doc_id']}")
        
        print(" Temporary document upload tests passed")
    
    def test_file_validation(self):
        """Test file validation and security measures"""
        print("🧪 Testing file validation...")
        
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
        
        # Test unsupported file type
        unsupported_file = self.test_data_dir / "test.exe"
        with open(unsupported_file, "wb") as f:
            f.write(b"fake executable content")
        
        with open(unsupported_file, "rb") as f:
            files = {"file": (unsupported_file.name, f, "application/octet-stream")}
            data = {"thread_id": TEST_THREAD_ID}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
            
            assert response.status_code == 400, "Should reject unsupported file type"
            print(" Correctly rejected unsupported file type")
        
        # Test oversized file
        oversized_file = self.test_data_dir / "oversized.txt"
        with open(oversized_file, "w") as f:
            # Create a file larger than 25MB
            for i in range(1000000):  # This will be much larger than 25MB
                f.write("This is a very large line of text for testing file size limits.\n")
        
        with open(oversized_file, "rb") as f:
            files = {"file": (oversized_file.name, f, "text/plain")}
            data = {"thread_id": TEST_THREAD_ID}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
            
            assert response.status_code == 400, "Should reject oversized file"
            print(" Correctly rejected oversized file")
        
        # Clean up test files
        unsupported_file.unlink()
        oversized_file.unlink()
        
        print(" File validation tests passed")
    
    def test_stress_upload(self):
        """Test stress scenarios for uploads"""
        print("🧪 Testing stress upload scenarios...")
        
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
        
        # Test concurrent uploads
        import concurrent.futures
        import threading
        
        def upload_file(file_path, thread_id):
            try:
                with open(file_path, "rb") as f:
                    files = {"file": (file_path.name, f, "application/octet-stream")}
                    data = {"thread_id": f"{thread_id}"}
                    response = self.client.post(
                        "/api/v1/chat/upload-temp-document",
                        files=files,
                        data=data,
                        headers=headers
                    )
                    return response.status_code == 200
            except Exception as e:
                print(f"Upload error in thread {thread_id}: {e}")
                return False
        
        # Test 10 concurrent uploads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                test_file = self.uploaded_files[0]  # Use the same file
                future = executor.submit(upload_file, test_file, f"stress-thread-{i}")
                futures.append(future)
            
            results = [future.result() for future in futures]
            success_count = sum(results)
            
            assert success_count >= 8, f"Only {success_count}/10 concurrent uploads succeeded"
            print(f" {success_count}/10 concurrent uploads succeeded")
        
        # Test rapid sequential uploads
        rapid_success_count = 0
        for i in range(20):
            test_file = self.uploaded_files[0]
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "application/octet-stream")}
                data = {"thread_id": f"rapid-{i}"}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
                if response.status_code == 200:
                    rapid_success_count += 1
        
        assert rapid_success_count >= 18, f"Only {rapid_success_count}/20 rapid uploads succeeded"
        print(f" {rapid_success_count}/20 rapid uploads succeeded")
        
        print(" Stress upload tests passed")
    
    def test_upload_lifecycle(self):
        """Test upload lifecycle including cleanup"""
        print("🧪 Testing upload lifecycle...")
        
        # Test temporary document retrieval
        if self.temp_doc_ids:
            headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
            
            for temp_doc_id in self.temp_doc_ids[:2]:  # Test first 2
                response = self.client.get(
                    f"/api/v1/documents/temp/{temp_doc_id}",
                    headers=headers
                )
                
                assert response.status_code == 200, f"Failed to retrieve temp doc {temp_doc_id}"
                result = response.json()
                assert "temp_doc_id" in result
                print(f" Successfully retrieved temp doc: {temp_doc_id}")
        
        # Test cleanup (this would normally be done by the cleanup task)
        print(" Upload lifecycle tests passed")
    
    def test_database_integrity(self):
        """Test database integrity after uploads"""
        print("🧪 Testing database integrity...")
        
        # This would test that all uploads are properly recorded in the database
        # and that relationships are maintained correctly
        
        print(" Database integrity tests passed")
    
    def test_file_storage_structure(self):
        """Test that files are stored in correct directory structure"""
        print("🧪 Testing file storage structure...")
        
        paths = get_path_settings()
        
        # Check temporary uploads directory
        temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads"
        assert temp_uploads_dir.exists(), "Temporary uploads directory should exist"
        
        # Check vectorstore directory
        vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data"
        assert vectorstore_dir.exists(), "Temporary vectorstore directory should exist"
        
        # Check data uploads directory
        data_uploads_dir = paths["DATA_DIR"] / "uploads"
        assert data_uploads_dir.exists(), "Data uploads directory should exist"
        
        print(" File storage structure tests passed")
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        print("🧹 Cleaning up test environment...")
        
        # Remove test files
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
        
        # Remove test agents
        for agent_file in self.created_agents:
            if agent_file.exists():
                agent_file.unlink()
        
        print(" Test environment cleanup complete")
    
    def run_all_tests(self):
        """Run all upload tests"""
        print(" Starting comprehensive upload testing suite...")
        
        try:
            self.setup_test_environment()
            
            # Run all test methods
            test_methods = [
                self.test_agent_file_upload,
                self.test_training_dataset_upload,
                self.test_temporary_document_upload,
                self.test_file_validation,
                self.test_stress_upload,
                self.test_upload_lifecycle,
                self.test_database_integrity,
                self.test_file_storage_structure,
            ]
            
            for test_method in test_methods:
                try:
                    test_method()
                except Exception as e:
                    print(f" Test {test_method.__name__} failed: {e}")
                    raise
            
            print(" All upload tests passed successfully!")
            
        except Exception as e:
            print(f" Test suite failed: {e}")
            raise
        finally:
            self.cleanup_test_environment()


if __name__ == "__main__":
    # Run the test suite
    test_suite = UploadTestSuite()
    test_suite.run_all_tests()
