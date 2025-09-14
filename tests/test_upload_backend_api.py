"""
Backend API Upload Tests for RAGnetic
Tests all backend API endpoints for file uploads with comprehensive scenarios.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.config import get_path_settings
from app.db import get_db
from app.db.models import temporary_documents_table, document_chunks_table, users_table
from app.services.temporary_document_service import TemporaryDocumentService


class BackendUploadAPITests:
    """Backend API upload testing class"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method"""
        self.client = TestClient(app)
        self.test_api_key = "YOUR_TEST_API_KEY_1"
        self.test_user_id = 1
        self.test_thread_id = f"test-thread-{uuid.uuid4()}"
        self.temp_files = []
        
    def teardown_method(self):
        """Cleanup after each test method"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
    
    def create_test_file(self, filename: str, content: str, size_mb: float = None) -> Path:
        """Create a test file with specified content and size"""
        temp_file = Path(tempfile.mktemp(suffix=f"_{filename}"))
        
        if size_mb:
            # Create a file of specified size
            content_size = len(content.encode('utf-8'))
            target_size = int(size_mb * 1024 * 1024)
            repetitions = max(1, target_size // content_size)
            content = content * repetitions
        
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def test_agent_file_upload_success(self):
        """Test successful agent file upload"""
        print(" Testing agent file upload success...")
        
        # Create test file
        test_content = "This is a test document for agent upload.\n" * 100
        test_file = self.create_test_file("test_agent_doc.txt", test_content)
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            response = self.client.post(
                "/api/v1/agents/upload-file",
                files=files,
                headers=headers
            )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        
        result = response.json()
        assert "file_path" in result
        assert Path(result["file_path"]).exists()
        
        # Verify file content
        with open(result["file_path"], "r") as f:
            uploaded_content = f.read()
            assert uploaded_content == test_content
        
        print("Agent file upload success test passed")
    
    def test_agent_file_upload_unauthorized(self):
        """Test agent file upload without authorization"""
        print(" Testing agent file upload unauthorized...")
        
        test_file = self.create_test_file("unauthorized_test.txt", "test content")
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            response = self.client.post(
                "/api/v1/agents/upload-file",
                files=files
            )
        
        assert response.status_code == 401, "Should require authorization"
        print("Agent file upload unauthorized test passed")
    
    def test_agent_file_upload_invalid_file(self):
        """Test agent file upload with invalid file"""
        print(" Testing agent file upload invalid file...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test with no file
        response = self.client.post(
            "/api/v1/agents/upload-file",
            headers=headers
        )
        assert response.status_code == 422, "Should require file"
        
        # Test with empty file
        empty_file = self.create_test_file("empty.txt", "")
        with open(empty_file, "rb") as f:
            files = {"file": (empty_file.name, f, "text/plain")}
            response = self.client.post(
                "/api/v1/agents/upload-file",
                files=files,
                headers=headers
            )
        assert response.status_code == 200, "Empty files should be allowed for agent uploads"
        
        print("Agent file upload invalid file test passed")
    
    def test_training_dataset_upload_success(self):
        """Test successful training dataset upload"""
        print(" Testing training dataset upload success...")
        
        # Create valid JSONL training data
        training_data = []
        for i in range(10):
            record = {
                "instruction": f"Test instruction {i}",
                "input": f"Test input {i}",
                "output": f"Test output {i}"
            }
            training_data.append(json.dumps(record))
        
        test_file = self.create_test_file("training_dataset.jsonl", "\n".join(training_data))
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "application/json")}
            response = self.client.post(
                "/api/v1/training/upload-dataset",
                files=files,
                headers=headers
            )
        
        assert response.status_code == 200, f"Training upload failed: {response.text}"
        
        result = response.json()
        assert "file_path" in result
        assert "file_id" in result
        assert "filename" in result
        assert "size" in result
        assert Path(result["file_path"]).exists()
        
        print("Training dataset upload success test passed")
    
    def test_training_dataset_upload_invalid_format(self):
        """Test training dataset upload with invalid format"""
        print(" Testing training dataset upload invalid format...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test with invalid JSON
        invalid_json = '{"invalid": json content}'
        test_file = self.create_test_file("invalid_training.jsonl", invalid_json)
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "application/json")}
            response = self.client.post(
                "/api/v1/training/upload-dataset",
                files=files,
                headers=headers
            )
        
        assert response.status_code == 400, "Should reject invalid JSON format"
        print(" Training dataset upload invalid format test passed")
    
    def test_training_dataset_upload_wrong_extension(self):
        """Test training dataset upload with wrong file extension"""
        print(" Testing training dataset upload wrong extension...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test with .txt instead of .jsonl
        test_file = self.create_test_file("training_data.txt", "not jsonl content")
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            response = self.client.post(
                "/api/v1/training/upload-dataset",
                files=files,
                headers=headers
            )
        
        assert response.status_code == 400, "Should reject non-JSONL/JSON files"
        print(" Training dataset upload wrong extension test passed")
    
    def test_temporary_document_upload_success(self):
        """Test successful temporary document upload"""
        print(" Testing temporary document upload success...")
        
        # Create test document
        test_content = "# Test Document\n\nThis is a test markdown document.\n" * 50
        test_file = self.create_test_file("test_doc.md", test_content)
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/markdown")}
            data = {"thread_id": self.test_thread_id}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 200, f"Temp upload failed: {response.text}"
        
        result = response.json()
        assert "temp_doc_id" in result
        assert "file_name" in result
        assert "file_size" in result
        assert result["file_name"] == test_file.name
        
        # Verify file was stored
        paths = get_path_settings()
        temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads" / str(self.test_user_id) / self.test_thread_id
        stored_file = temp_uploads_dir / f"{result['temp_doc_id']}_{test_file.name}"
        assert stored_file.exists(), "Temporary file should be stored"
        
        # Verify vector store was created
        vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data" / result["temp_doc_id"]
        assert vectorstore_dir.exists(), "Vector store should be created"
        assert (vectorstore_dir / "index.faiss").exists(), "FAISS index should exist"
        assert (vectorstore_dir / "index.pkl").exists(), "FAISS pickle should exist"
        
        print(" Temporary document upload success test passed")
    
    def test_temporary_document_upload_missing_thread_id(self):
        """Test temporary document upload without thread_id"""
        print(" Testing temporary document upload missing thread_id...")
        
        test_file = self.create_test_file("test_doc.txt", "test content")
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                headers=headers
            )
        
        assert response.status_code == 400, "Should require thread_id"
        print(" Temporary document upload missing thread_id test passed")
    
    def test_temporary_document_upload_unsupported_type(self):
        """Test temporary document upload with unsupported file type"""
        print(" Testing temporary document upload unsupported type...")
        
        # Create executable file
        test_file = self.create_test_file("test.exe", "fake executable content")
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "application/octet-stream")}
            data = {"thread_id": self.test_thread_id}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 400, "Should reject unsupported file type"
        print(" Temporary document upload unsupported type test passed")
    
    def test_temporary_document_upload_oversized(self):
        """Test temporary document upload with oversized file"""
        print(" Testing temporary document upload oversized...")
        
        # Create file larger than 25MB
        large_content = "This is a large line of text for testing file size limits.\n"
        test_file = self.create_test_file("large_doc.txt", large_content, size_mb=26)
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            data = {"thread_id": self.test_thread_id}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 400, "Should reject oversized file"
        print(" Temporary document upload oversized test passed")
    
    def test_temporary_document_retrieval(self):
        """Test temporary document retrieval"""
        print(" Testing temporary document retrieval...")
        
        # First upload a document
        test_file = self.create_test_file("retrieval_test.txt", "test content for retrieval")
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            data = {"thread_id": self.test_thread_id}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 200
        temp_doc_id = response.json()["temp_doc_id"]
        
        # Now retrieve it
        response = self.client.get(
            f"/api/v1/documents/temp/{temp_doc_id}",
            headers=headers
        )
        
        assert response.status_code == 200, f"Retrieval failed: {response.text}"
        
        result = response.json()
        assert "temp_doc_id" in result
        assert "original_name" in result
        assert "file_size" in result
        assert result["temp_doc_id"] == temp_doc_id
        
        print(" Temporary document retrieval test passed")
    
    def test_temporary_document_retrieval_nonexistent(self):
        """Test temporary document retrieval for nonexistent document"""
        print(" Testing temporary document retrieval nonexistent...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        fake_temp_doc_id = str(uuid.uuid4())
        
        response = self.client.get(
            f"/api/v1/documents/temp/{fake_temp_doc_id}",
            headers=headers
        )
        
        assert response.status_code == 404, "Should return 404 for nonexistent document"
        print(" Temporary document retrieval nonexistent test passed")
    
    def test_concurrent_uploads(self):
        """Test concurrent uploads to the same endpoint"""
        print(" Testing concurrent uploads...")
        
        import concurrent.futures
        import threading
        
        def upload_document(thread_id):
            test_file = self.create_test_file(f"concurrent_test_{thread_id}.txt", f"content {thread_id}")
            headers = {"Authorization": f"Bearer {self.test_api_key}"}
            
            try:
                with open(test_file, "rb") as f:
                    files = {"file": (test_file.name, f, "text/plain")}
                    data = {"thread_id": f"concurrent-{thread_id}"}
                    response = self.client.post(
                        "/api/v1/chat/upload-temp-document",
                        files=files,
                        data=data,
                        headers=headers
                    )
                    return response.status_code == 200
            except Exception as e:
                print(f"Concurrent upload error in thread {thread_id}: {e}")
                return False
        
        # Test 5 concurrent uploads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(upload_document, i) for i in range(5)]
            results = [future.result() for future in futures]
        
        success_count = sum(results)
        assert success_count >= 4, f"Only {success_count}/5 concurrent uploads succeeded"
        
        print(f" {success_count}/5 concurrent uploads succeeded")
    
    def test_upload_rate_limiting(self):
        """Test upload rate limiting"""
        print(" Testing upload rate limiting...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Try rapid uploads to test rate limiting
        success_count = 0
        for i in range(25):  # Try 25 rapid uploads
            test_file = self.create_test_file(f"rate_limit_test_{i}.txt", f"content {i}")
            
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "text/plain")}
                data = {"thread_id": f"rate-limit-{i}"}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:
                    print(f"Rate limited at upload {i}")
                    break
        
        # Should allow some uploads but may rate limit after a certain number
        assert success_count > 0, "Should allow some uploads"
        print(f" Rate limiting test completed: {success_count} uploads succeeded")
    
    def run_all_backend_tests(self):
        """Run all backend API tests"""
        print(" Starting backend API upload tests...")
        
        test_methods = [
            self.test_agent_file_upload_success,
            self.test_agent_file_upload_unauthorized,
            self.test_agent_file_upload_invalid_file,
            self.test_training_dataset_upload_success,
            self.test_training_dataset_upload_invalid_format,
            self.test_training_dataset_upload_wrong_extension,
            self.test_temporary_document_upload_success,
            self.test_temporary_document_upload_missing_thread_id,
            self.test_temporary_document_upload_unsupported_type,
            self.test_temporary_document_upload_oversized,
            self.test_temporary_document_retrieval,
            self.test_temporary_document_retrieval_nonexistent,
            self.test_concurrent_uploads,
            self.test_upload_rate_limiting,
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f" Test {test_method.__name__} failed: {e}")
                raise
        
        print(" All backend API upload tests passed!")


if __name__ == "__main__":
    test_suite = BackendUploadAPITests()
    test_suite.run_all_backend_tests()
