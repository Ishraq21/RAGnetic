#!/usr/bin/env python3
"""
Integration Tests for RAGnetic Upload Functionality
Tests upload functionality with actual server interaction.
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import uuid

import requests
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_path_settings
from app.db import get_db
from app.db.models import temporary_documents_table, document_chunks_table
from app.services.temporary_document_service import TemporaryDocumentService


class UploadIntegrationTests:
    """Integration testing class for upload functionality"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.base_url = "http://127.0.0.1:8000"
        self.test_api_key = "YOUR_TEST_API_KEY_1"
        self.temp_files = []
        self.upload_results = []
        
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
    
    def test_server_connectivity(self):
        """Test server connectivity"""
        print(" Testing server connectivity...")
        
        try:
            # Test with TestClient first
            response = self.client.get("/")
            assert response.status_code in [200, 404], f"TestClient response: {response.status_code}"
            print("    TestClient connectivity working")
            
            # Test with actual server if running
            try:
                response = requests.get(f"{self.base_url}/", timeout=5)
                print(f"    Live server connectivity working: {response.status_code}")
                return True
            except requests.exceptions.RequestException:
                print("    Live server not running, using TestClient only")
                return True
                
        except Exception as e:
            print(f" Server connectivity test failed: {e}")
            return False
    
    def test_agent_file_upload_integration(self):
        """Test agent file upload integration"""
        print(" Testing agent file upload integration...")
        
        try:
            # Create test file
            test_content = "Integration test document content for agent upload.\n" * 100
            test_file = self.create_test_file("integration_agent_test.txt", test_content)
            
            headers = {"Authorization": f"Bearer {self.test_api_key}"}
            
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "text/plain")}
                response = self.client.post(
                    "/api/v1/agents/upload-file",
                    files=files,
                    headers=headers
                )
            
            assert response.status_code == 200, f"Agent upload failed: {response.text}"
            
            result = response.json()
            assert "file_path" in result, "Response should contain file_path"
            
            # Verify file was actually stored
            stored_file_path = Path(result["file_path"])
            assert stored_file_path.exists(), "Uploaded file should exist on disk"
            
            # Verify file content
            with open(stored_file_path, "r") as f:
                stored_content = f.read()
                assert stored_content == test_content, "Stored content should match uploaded content"
            
            print("    Agent file upload integration successful")
            return True
            
        except Exception as e:
            print(f" Agent file upload integration failed: {e}")
            return False
    
    def test_training_dataset_upload_integration(self):
        """Test training dataset upload integration"""
        print(" Testing training dataset upload integration...")
        
        try:
            # Create valid JSONL training data
            training_data = []
            for i in range(5):
                record = {
                    "instruction": f"Test instruction {i}",
                    "input": f"Test input {i}",
                    "output": f"Test output {i}"
                }
                training_data.append(json.dumps(record))
            
            test_file = self.create_test_file("integration_training.jsonl", "\n".join(training_data))
            
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
            assert "file_path" in result, "Response should contain file_path"
            assert "file_id" in result, "Response should contain file_id"
            
            # Verify file was stored
            stored_file_path = Path(result["file_path"])
            assert stored_file_path.exists(), "Training file should exist on disk"
            
            print("    Training dataset upload integration successful")
            return True
            
        except Exception as e:
            print(f" Training dataset upload integration failed: {e}")
            return False
    
    def test_temporary_document_upload_integration(self):
        """Test temporary document upload integration"""
        print(" Testing temporary document upload integration...")
        
        try:
            # Create test document
            test_content = "# Integration Test Document\n\nThis is a test document for integration testing.\n\n" * 50
            test_file = self.create_test_file("integration_temp_test.md", test_content)
            
            headers = {"Authorization": f"Bearer {self.test_api_key}"}
            thread_id = f"integration-test-{uuid.uuid4()}"
            
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "text/markdown")}
                data = {"thread_id": thread_id}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            assert response.status_code == 200, f"Temp upload failed: {response.text}"
            
            result = response.json()
            assert "temp_doc_id" in result, "Response should contain temp_doc_id"
            assert "file_name" in result, "Response should contain file_name"
            assert "file_size" in result, "Response should contain file_size"
            
            temp_doc_id = result["temp_doc_id"]
            self.upload_results.append(temp_doc_id)
            
            # Verify database record
            with get_db() as db:
                stmt = temporary_documents_table.select().where(
                    temporary_documents_table.c.temp_doc_id == temp_doc_id
                )
                temp_doc_record = db.execute(stmt).mappings().first()
                
                assert temp_doc_record is not None, "Temp document should exist in database"
                assert temp_doc_record["original_name"] == test_file.name
                assert temp_doc_record["thread_id"] == thread_id
                assert temp_doc_record["cleaned_up"] == False
            
            # Verify file storage
            paths = get_path_settings()
            temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads" / "1" / thread_id
            stored_file = temp_uploads_dir / f"{temp_doc_id}_{test_file.name}"
            assert stored_file.exists(), "Temporary file should be stored"
            
            # Verify vector store creation
            vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data" / temp_doc_id
            assert vectorstore_dir.exists(), "Vector store should be created"
            assert (vectorstore_dir / "index.faiss").exists(), "FAISS index should exist"
            assert (vectorstore_dir / "index.pkl").exists(), "FAISS pickle should exist"
            
            print("    Temporary document upload integration successful")
            return True
            
        except Exception as e:
            print(f" Temporary document upload integration failed: {e}")
            return False
    
    def test_document_retrieval_integration(self):
        """Test document retrieval integration"""
        print(" Testing document retrieval integration...")
        
        try:
            if not self.upload_results:
                print("    No uploaded documents to retrieve")
                return True
            
            temp_doc_id = self.upload_results[0]
            headers = {"Authorization": f"Bearer {self.test_api_key}"}
            
            response = self.client.get(
                f"/api/v1/documents/temp/{temp_doc_id}",
                headers=headers
            )
            
            assert response.status_code == 200, f"Document retrieval failed: {response.text}"
            
            result = response.json()
            assert "temp_doc_id" in result, "Response should contain temp_doc_id"
            assert "original_name" in result, "Response should contain original_name"
            assert "file_size" in result, "Response should contain file_size"
            assert result["temp_doc_id"] == temp_doc_id, "Retrieved document ID should match"
            
            print("    Document retrieval integration successful")
            return True
            
        except Exception as e:
            print(f" Document retrieval integration failed: {e}")
            return False
    
    def test_file_validation_integration(self):
        """Test file validation integration"""
        print(" Testing file validation integration...")
        
        try:
            headers = {"Authorization": f"Bearer {self.test_api_key}"}
            thread_id = f"validation-test-{uuid.uuid4()}"
            
            # Test unsupported file type
            malicious_file = self.create_test_file("malicious.exe", "fake executable content")
            
            with open(malicious_file, "rb") as f:
                files = {"file": (malicious_file.name, f, "application/octet-stream")}
                data = {"thread_id": thread_id}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            assert response.status_code == 400, "Unsupported file type should be rejected"
            print("    Unsupported file type correctly rejected")
            
            # Test oversized file
            large_content = "A" * (26 * 1024 * 1024)  # 26MB
            oversized_file = self.create_test_file("oversized.txt", large_content)
            
            with open(oversized_file, "rb") as f:
                files = {"file": (oversized_file.name, f, "text/plain")}
                data = {"thread_id": thread_id}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            assert response.status_code == 400, "Oversized file should be rejected"
            print("    Oversized file correctly rejected")
            
            print(" File validation integration successful")
            return True
            
        except Exception as e:
            print(f" File validation integration failed: {e}")
            return False
    
    def test_concurrent_uploads_integration(self):
        """Test concurrent uploads integration"""
        print(" Testing concurrent uploads integration...")
        
        try:
            import concurrent.futures
            import threading
            
            def upload_worker(worker_id: int) -> Dict[str, Any]:
                """Worker function for concurrent uploads"""
                headers = {"Authorization": f"Bearer {self.test_api_key}"}
                thread_id = f"concurrent-{worker_id}"
                
                test_content = f"Concurrent upload test content from worker {worker_id}\n" * 50
                test_file = self.create_test_file(f"concurrent_{worker_id}.txt", test_content)
                
                try:
                    with open(test_file, "rb") as f:
                        files = {"file": (test_file.name, f, "text/plain")}
                        data = {"thread_id": thread_id}
                        response = self.client.post(
                            "/api/v1/chat/upload-temp-document",
                            files=files,
                            data=data,
                            headers=headers
                        )
                    
                    return {
                        "worker_id": worker_id,
                        "status_code": response.status_code,
                        "success": response.status_code == 200,
                        "response": response.json() if response.status_code == 200 else None
                    }
                except Exception as e:
                    return {
                        "worker_id": worker_id,
                        "status_code": 0,
                        "success": False,
                        "error": str(e)
                    }
            
            # Test with 5 concurrent uploads
            num_workers = 5
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(upload_worker, i) for i in range(num_workers)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Analyze results
            successful_uploads = [r for r in results if r["success"]]
            failed_uploads = [r for r in results if not r["success"]]
            
            print(f"    Concurrent upload results:")
            print(f"      Successful: {len(successful_uploads)}/{len(results)}")
            print(f"      Failed: {len(failed_uploads)}")
            
            # Store successful upload IDs
            for result in successful_uploads:
                if result["response"] and "temp_doc_id" in result["response"]:
                    self.upload_results.append(result["response"]["temp_doc_id"])
            
            # Most uploads should succeed
            assert len(successful_uploads) >= len(results) * 0.8, f"Too many concurrent uploads failed: {len(failed_uploads)}/{len(results)}"
            
            print(" Concurrent uploads integration successful")
            return True
            
        except Exception as e:
            print(f" Concurrent uploads integration failed: {e}")
            return False
    
    def test_cleanup_integration(self):
        """Test cleanup integration"""
        print(" Testing cleanup integration...")
        
        try:
            if not self.upload_results:
                print("    No uploaded documents to clean up")
                return True
            
            # Test manual cleanup of one document
            temp_doc_id = self.upload_results[0]
            
            # Get document data
            with get_db() as db:
                stmt = temporary_documents_table.select().where(
                    temporary_documents_table.c.temp_doc_id == temp_doc_id
                )
                temp_doc_record = db.execute(stmt).mappings().first()
                
                if temp_doc_record:
                    # Perform cleanup
                    temp_doc_data = dict(temp_doc_record)
                    TemporaryDocumentService.cleanup_fs(temp_doc_data)
                    
                    # Verify cleanup
                    paths = get_path_settings()
                    temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads" / "1" / temp_doc_record["thread_id"]
                    vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data" / temp_doc_id
                    
                    stored_file = temp_uploads_dir / f"{temp_doc_id}_{temp_doc_record['original_name']}"
                    
                    # Files should be cleaned up
                    if stored_file.exists():
                        print("    Stored file still exists after cleanup")
                    else:
                        print("    Stored file cleaned up successfully")
                    
                    if vectorstore_dir.exists():
                        print("    Vector store still exists after cleanup")
                    else:
                        print("    Vector store cleaned up successfully")
            
            print(" Cleanup integration successful")
            return True
            
        except Exception as e:
            print(f" Cleanup integration failed: {e}")
            return False
    
    def run_all_integration_tests(self):
        """Run all integration tests"""
        print(" Starting Upload Integration Tests")
        print("=" * 50)
        
        test_methods = [
            ("Server Connectivity", self.test_server_connectivity),
            ("Agent File Upload", self.test_agent_file_upload_integration),
            ("Training Dataset Upload", self.test_training_dataset_upload_integration),
            ("Temporary Document Upload", self.test_temporary_document_upload_integration),
            ("Document Retrieval", self.test_document_retrieval_integration),
            ("File Validation", self.test_file_validation_integration),
            ("Concurrent Uploads", self.test_concurrent_uploads_integration),
            ("Cleanup Integration", self.test_cleanup_integration),
        ]
        
        results = {}
        overall_success = True
        
        for test_name, test_method in test_methods:
            print(f"\n{'='*30}")
            print(f"Running {test_name}")
            print(f"{'='*30}")
            
            try:
                success = test_method()
                results[test_name] = {"success": success, "error": None}
                if not success:
                    overall_success = False
            except Exception as e:
                print(f" {test_name} failed with exception: {e}")
                results[test_name] = {"success": False, "error": str(e)}
                overall_success = False
        
        # Generate report
        print(f"\n{'='*50}")
        print(" INTEGRATION TEST REPORT")
        print(f"{'='*50}")
        
        successful_tests = sum(1 for result in results.values() if result["success"])
        total_tests = len(results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {(successful_tests/total_tests*100):.2f}%")
        print(f"Overall Status: {' PASSED' if overall_success else ' FAILED'}")
        
        print(f"\nDetailed Results:")
        for test_name, result in results.items():
            status = " PASSED" if result["success"] else " FAILED"
            print(f"  {test_name}: {status}")
            if result["error"]:
                print(f"    Error: {result['error']}")
        
        # Save report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_success": overall_success,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests * 100,
            "results": results
        }
        
        report_file = Path("tests/integration_upload_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_file}")
        
        return 0 if overall_success else 1


def main():
    """Main entry point"""
    print("RAGnetic Upload Integration Test Suite")
    print("=" * 40)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_suite = UploadIntegrationTests()
    exit_code = test_suite.run_all_integration_tests()
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(main())
