"""
Cleanup and Lifecycle Tests for RAGnetic Upload Functionality
Tests file cleanup, expiration, and lifecycle management.
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text, select
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.config import get_path_settings
from app.db import get_db, get_async_db_session
from app.db.models import temporary_documents_table, document_chunks_table, users_table
from app.db.dao import create_temp_document, delete_temp_document_data, mark_temp_document_cleaned
from app.services.temporary_document_service import TemporaryDocumentService
from app.core.tasks import cleanup_temporary_documents


class UploadCleanupLifecycleTests:
    """Cleanup and lifecycle testing class"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_api_key = "YOUR_TEST_API_KEY_1"
        self.test_user_id = 1
        self.temp_files = []
        self.temp_doc_ids = []
        
    def create_test_file(self, filename: str, content: str) -> Path:
        """Create a test file"""
        temp_file = Path(tempfile.mktemp(suffix=f"_{filename}"))
        
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def cleanup_temp_files(self):
        """Clean up temporary test files"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
        self.temp_files.clear()
    
    def test_temporary_document_creation(self):
        """Test temporary document creation in database"""
        print(" Testing temporary document creation...")
        
        # Create test file
        test_file = self.create_test_file("lifecycle_test.txt", "Lifecycle test content")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            data = {"thread_id": "lifecycle-test"}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        
        result = response.json()
        temp_doc_id = result["temp_doc_id"]
        self.temp_doc_ids.append(temp_doc_id)
        
        # Verify database record
        with get_db() as db:
            stmt = select(temporary_documents_table).where(
                temporary_documents_table.c.temp_doc_id == temp_doc_id
            )
            result = db.execute(stmt).mappings().first()
            
            assert result is not None, "Temporary document not found in database"
            assert result["user_id"] == self.test_user_id
            assert result["thread_id"] == "lifecycle-test"
            assert result["original_name"] == test_file.name
            assert result["cleaned_up"] == False
            assert result["expires_at"] is not None
            
            # Check expiration time (should be ~7 days from now)
            expires_at = result["expires_at"]
            expected_expiry = datetime.utcnow() + timedelta(days=7)
            time_diff = abs((expires_at - expected_expiry).total_seconds())
            assert time_diff < 3600, f"Expiration time incorrect: {expires_at}"
        
        print(" Temporary document creation test passed")
    
    def test_document_chunks_creation(self):
        """Test document chunks creation"""
        print(" Testing document chunks creation...")
        
        # Create test file with substantial content
        content = "# Test Document\n\nThis is a test document with multiple paragraphs.\n\n" * 20
        test_file = self.create_test_file("chunks_test.txt", content)
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            data = {"thread_id": "chunks-test"}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        
        result = response.json()
        temp_doc_id = result["temp_doc_id"]
        self.temp_doc_ids.append(temp_doc_id)
        
        # Verify chunks were created
        with get_db() as db:
            # Get temp document ID
            stmt = select(temporary_documents_table.c.id).where(
                temporary_documents_table.c.temp_doc_id == temp_doc_id
            )
            temp_doc_db_id = db.execute(stmt).scalar_one()
            
            # Check chunks
            stmt = select(document_chunks_table).where(
                document_chunks_table.c.temp_document_id == temp_doc_db_id
            )
            chunks = db.execute(stmt).mappings().all()
            
            assert len(chunks) > 0, "No document chunks created"
            assert len(chunks) >= 1, "Should have at least one chunk"
            
            # Verify chunk content
            for chunk in chunks:
                assert chunk["content"] is not None, "Chunk content should not be None"
                assert len(chunk["content"]) > 0, "Chunk content should not be empty"
                assert chunk["document_name"].startswith("temp::"), "Chunk document name should start with temp::"
                assert chunk["chunk_index"] is not None, "Chunk index should not be None"
        
        print(" Document chunks creation test passed")
    
    def test_vector_store_creation(self):
        """Test vector store creation"""
        print(" Testing vector store creation...")
        
        # Create test file
        test_file = self.create_test_file("vectorstore_test.txt", "Vector store test content")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            data = {"thread_id": "vectorstore-test"}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        
        result = response.json()
        temp_doc_id = result["temp_doc_id"]
        self.temp_doc_ids.append(temp_doc_id)
        
        # Verify vector store files
        paths = get_path_settings()
        vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data" / temp_doc_id
        
        assert vectorstore_dir.exists(), "Vector store directory should exist"
        assert (vectorstore_dir / "index.faiss").exists(), "FAISS index file should exist"
        assert (vectorstore_dir / "index.pkl").exists(), "FAISS pickle file should exist"
        
        # Verify file sizes
        faiss_size = (vectorstore_dir / "index.faiss").stat().st_size
        pkl_size = (vectorstore_dir / "index.pkl").stat().st_size
        
        assert faiss_size > 0, "FAISS index file should not be empty"
        assert pkl_size > 0, "FAISS pickle file should not be empty"
        
        print(" Vector store creation test passed")
    
    def test_file_storage_structure(self):
        """Test file storage structure"""
        print(" Testing file storage structure...")
        
        # Create test file
        test_file = self.create_test_file("storage_test.txt", "Storage structure test content")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            data = {"thread_id": "storage-test"}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        
        result = response.json()
        temp_doc_id = result["temp_doc_id"]
        self.temp_doc_ids.append(temp_doc_id)
        
        # Verify file storage structure
        paths = get_path_settings()
        
        # Check temporary uploads directory
        temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads" / str(self.test_user_id) / "storage-test"
        stored_file = temp_uploads_dir / f"{temp_doc_id}_{test_file.name}"
        
        assert temp_uploads_dir.exists(), "Temporary uploads directory should exist"
        assert stored_file.exists(), "Stored file should exist"
        
        # Verify file content
        with open(stored_file, "r") as f:
            stored_content = f.read()
            assert stored_content == "Storage structure test content"
        
        print(" File storage structure test passed")
    
    def test_manual_cleanup(self):
        """Test manual cleanup functionality"""
        print(" Testing manual cleanup...")
        
        # Create test file
        test_file = self.create_test_file("cleanup_test.txt", "Cleanup test content")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            data = {"thread_id": "cleanup-test"}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        
        result = response.json()
        temp_doc_id = result["temp_doc_id"]
        
        # Verify files exist before cleanup
        paths = get_path_settings()
        temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads" / str(self.test_user_id) / "cleanup-test"
        vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data" / temp_doc_id
        
        stored_file = temp_uploads_dir / f"{temp_doc_id}_{test_file.name}"
        assert stored_file.exists(), "Stored file should exist before cleanup"
        assert vectorstore_dir.exists(), "Vector store should exist before cleanup"
        
        # Get database record
        with get_db() as db:
            stmt = select(temporary_documents_table).where(
                temporary_documents_table.c.temp_doc_id == temp_doc_id
            )
            temp_doc_record = db.execute(stmt).mappings().first()
            assert temp_doc_record is not None, "Temp document should exist in database"
        
        # Perform manual cleanup
        with get_db() as db:
            # Get temp document data for cleanup
            temp_doc_data = dict(temp_doc_record)
            
            # Cleanup filesystem
            TemporaryDocumentService.cleanup_fs(temp_doc_data)
            
            # Cleanup database
            delete_temp_document_data(db, temp_doc_id)
            
            # Mark as cleaned
            mark_temp_document_cleaned(db, temp_doc_record["id"])
        
        # Verify cleanup
        assert not stored_file.exists(), "Stored file should be deleted after cleanup"
        assert not vectorstore_dir.exists(), "Vector store should be deleted after cleanup"
        
        # Verify database cleanup
        with get_db() as db:
            stmt = select(temporary_documents_table).where(
                temporary_documents_table.c.temp_doc_id == temp_doc_id
            )
            temp_doc_record = db.execute(stmt).mappings().first()
            assert temp_doc_record is None, "Temp document should be deleted from database"
        
        print(" Manual cleanup test passed")
    
    def test_expiration_cleanup(self):
        """Test expiration-based cleanup"""
        print(" Testing expiration-based cleanup...")
        
        # Create test file
        test_file = self.create_test_file("expiration_test.txt", "Expiration test content")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            data = {"thread_id": "expiration-test"}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        
        result = response.json()
        temp_doc_id = result["temp_doc_id"]
        
        # Manually set expiration to past time
        with get_db() as db:
            stmt = (
                temporary_documents_table.update()
                .where(temporary_documents_table.c.temp_doc_id == temp_doc_id)
                .values(expires_at=datetime.utcnow() - timedelta(hours=1))
            )
            db.execute(stmt)
            db.commit()
        
        # Verify files exist before cleanup
        paths = get_path_settings()
        temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads" / str(self.test_user_id) / "expiration-test"
        vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data" / temp_doc_id
        
        stored_file = temp_uploads_dir / f"{temp_doc_id}_{test_file.name}"
        assert stored_file.exists(), "Stored file should exist before cleanup"
        assert vectorstore_dir.exists(), "Vector store should exist before cleanup"
        
        # Run cleanup task
        cleanup_temporary_documents()
        
        # Verify cleanup
        assert not stored_file.exists(), "Stored file should be deleted after expiration cleanup"
        assert not vectorstore_dir.exists(), "Vector store should be deleted after expiration cleanup"
        
        # Verify database cleanup
        with get_db() as db:
            stmt = select(temporary_documents_table).where(
                temporary_documents_table.c.temp_doc_id == temp_doc_id
            )
            temp_doc_record = db.execute(stmt).mappings().first()
            assert temp_doc_record is None, "Temp document should be deleted after expiration cleanup"
        
        print(" Expiration cleanup test passed")
    
    def test_cleanup_robustness(self):
        """Test cleanup robustness with missing files"""
        print(" Testing cleanup robustness...")
        
        # Create a temporary document record manually (simulating orphaned record)
        temp_doc_id = str(uuid.uuid4())
        
        with get_db() as db:
            stmt = (
                temporary_documents_table.insert().values(
                    temp_doc_id=temp_doc_id,
                    user_id=self.test_user_id,
                    thread_id="robustness-test",
                    original_name="robustness_test.txt",
                    file_size=100,
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() - timedelta(hours=1),
                    cleaned_up=False
                )
            )
            db.execute(stmt)
            db.commit()
        
        # Run cleanup task (should handle missing files gracefully)
        try:
            cleanup_temporary_documents()
            print(" Cleanup task completed without errors")
        except Exception as e:
            print(f" Cleanup task failed: {e}")
            raise
        
        # Verify database record is cleaned up
        with get_db() as db:
            stmt = select(temporary_documents_table).where(
                temporary_documents_table.c.temp_doc_id == temp_doc_id
            )
            temp_doc_record = db.execute(stmt).mappings().first()
            assert temp_doc_record is None, "Orphaned record should be cleaned up"
        
        print(" Cleanup robustness test passed")
    
    def test_cleanup_performance(self):
        """Test cleanup performance with many documents"""
        print(" Testing cleanup performance...")
        
        # Create multiple temporary documents
        num_docs = 10
        temp_doc_ids = []
        
        for i in range(num_docs):
            test_file = self.create_test_file(f"performance_test_{i}.txt", f"Performance test content {i}")
            
            headers = {"Authorization": f"Bearer {self.test_api_key}"}
            
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "text/plain")}
                data = {"thread_id": f"performance-test-{i}"}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            assert response.status_code == 200, f"Upload {i} failed: {response.text}"
            temp_doc_ids.append(response.json()["temp_doc_id"])
        
        # Set all to expired
        with get_db() as db:
            for temp_doc_id in temp_doc_ids:
                stmt = (
                    temporary_documents_table.update()
                    .where(temporary_documents_table.c.temp_doc_id == temp_doc_id)
                    .values(expires_at=datetime.utcnow() - timedelta(hours=1))
                )
                db.execute(stmt)
            db.commit()
        
        # Measure cleanup performance
        start_time = time.time()
        cleanup_temporary_documents()
        cleanup_time = time.time() - start_time
        
        print(f" Cleanup Performance Results:")
        print(f"   Documents cleaned: {num_docs}")
        print(f"   Cleanup time: {cleanup_time:.2f}s")
        print(f"   Documents per second: {num_docs / cleanup_time:.2f}")
        
        # Verify all documents are cleaned up
        with get_db() as db:
            for temp_doc_id in temp_doc_ids:
                stmt = select(temporary_documents_table).where(
                    temporary_documents_table.c.temp_doc_id == temp_doc_id
                )
                temp_doc_record = db.execute(stmt).mappings().first()
                assert temp_doc_record is None, f"Document {temp_doc_id} should be cleaned up"
        
        # Assertions
        assert cleanup_time < 30, f"Cleanup took too long: {cleanup_time:.2f}s"
        
        print(" Cleanup performance test passed")
    
    def test_cleanup_transaction_safety(self):
        """Test cleanup transaction safety"""
        print(" Testing cleanup transaction safety...")
        
        # This test would verify that cleanup operations are atomic
        # and don't leave the system in an inconsistent state
        
        # Create test document
        test_file = self.create_test_file("transaction_test.txt", "Transaction safety test content")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        with open(test_file, "rb") as f:
            files = {"file": (test_file.name, f, "text/plain")}
            data = {"thread_id": "transaction-test"}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 200, f"Upload failed: {response.text}"
        
        result = response.json()
        temp_doc_id = result["temp_doc_id"]
        
        # Set to expired
        with get_db() as db:
            stmt = (
                temporary_documents_table.update()
                .where(temporary_documents_table.c.temp_doc_id == temp_doc_id)
                .values(expires_at=datetime.utcnow() - timedelta(hours=1))
            )
            db.execute(stmt)
            db.commit()
        
        # Run cleanup multiple times (should be idempotent)
        for i in range(3):
            cleanup_temporary_documents()
            
            # Verify system state is consistent
            with get_db() as db:
                stmt = select(temporary_documents_table).where(
                    temporary_documents_table.c.temp_doc_id == temp_doc_id
                )
                temp_doc_record = db.execute(stmt).mappings().first()
                assert temp_doc_record is None, f"Document should be cleaned up after run {i+1}"
        
        print(" Cleanup transaction safety test passed")
    
    def run_all_cleanup_lifecycle_tests(self):
        """Run all cleanup and lifecycle tests"""
        print(" Starting cleanup and lifecycle tests...")
        
        try:
            test_methods = [
                self.test_temporary_document_creation,
                self.test_document_chunks_creation,
                self.test_vector_store_creation,
                self.test_file_storage_structure,
                self.test_manual_cleanup,
                self.test_expiration_cleanup,
                self.test_cleanup_robustness,
                self.test_cleanup_performance,
                self.test_cleanup_transaction_safety,
            ]
            
            for test_method in test_methods:
                try:
                    print(f"\n{'='*50}")
                    test_method()
                    print(f"{'='*50}\n")
                except Exception as e:
                    print(f" Cleanup test {test_method.__name__} failed: {e}")
                    raise
            
            print(" All cleanup and lifecycle tests passed!")
            
        except Exception as e:
            print(f" Cleanup test suite failed: {e}")
            raise
        finally:
            self.cleanup_temp_files()


if __name__ == "__main__":
    test_suite = UploadCleanupLifecycleTests()
    test_suite.run_all_cleanup_lifecycle_tests()
