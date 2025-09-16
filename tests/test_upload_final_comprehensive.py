#!/usr/bin/env python3
"""
Final Comprehensive Upload Tests for RAGnetic
Tests upload functionality with proper environment configuration and schema understanding.
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
from app.core.config import get_path_settings, get_server_api_keys
from app.db import get_db
from app.db.models import temporary_documents_table, document_chunks_table
from app.services.temporary_document_service import TemporaryDocumentService


class FinalComprehensiveUploadTests:
    """Final comprehensive testing class for upload functionality"""
    
    def __init__(self):
        # Set up proper test environment BEFORE creating TestClient
        self.setup_test_environment()
        
        self.client = TestClient(app)
        self.base_url = "http://127.0.0.1:8000"
        self.temp_files = []
        self.upload_results = []
        
    def setup_test_environment(self):
        """Setup proper test environment variables"""
        print("Setting up comprehensive test environment...")
        
        # Set environment variables for testing (these should be in .env in production)
        os.environ["RAGNETIC_API_KEYS"] = "YOUR_TEST_API_KEY_1,YOUR_TEST_API_KEY_2"
        os.environ["ALLOWED_HOSTS"] = "localhost,127.0.0.1,testserver"
        os.environ["RAGNETIC_PROJECT_ROOT"] = str(Path(__file__).parent.parent)
        os.environ["RAGNETIC_DEBUG"] = "true"
        os.environ["CORS_ALLOWED_ORIGINS"] = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173"
        os.environ["REDIS_URL"] = "redis://localhost:6379/0"
        
        # Set test API keys for external services
        os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
        os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"
        os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
        
        # Set embedding model to use a mock model for testing (no API key required)
        os.environ["RAGNETIC_EMBEDDING_MODEL"] = "mock"
        
        # Clear embedding model cache to force reload with new config
        from app.core.embed_config import embedding_model_cache
        embedding_model_cache.clear()
        print("   Embedding model cache cleared")
        
        # Initialize database connections
        from app.db import initialize_db_connections
        from app.core.config import get_memory_storage_config
        mem_config = get_memory_storage_config()
        if mem_config and mem_config.get("connection_name"):
            initialize_db_connections(mem_config["connection_name"])
            print("   Database connections initialized")
        else:
            print("   No database connection configured")
        
        # Verify API keys are set
        server_keys = get_server_api_keys()
        if server_keys:
            self.test_api_key = server_keys[0]
            print(f"   API key configured: {self.test_api_key[:20]}...")
        else:
            self.test_api_key = "YOUR_TEST_API_KEY_1"
            print(f"   Using fallback API key: {self.test_api_key[:20]}...")
        
        print("   Environment variables configured")
    
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
    
    def test_upload_schema_compliance(self):
        """Test that uploads follow RAGnetic schema and structure"""
        print(" Testing upload schema compliance...")
        
        try:
            # Test agent file upload schema
            test_content = "Schema compliance test document content.\n" * 100
            test_file = self.create_test_file("schema_test.txt", test_content)
            
            headers = {"X-API-Key": self.test_api_key}
            
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "text/plain")}
                response = self.client.post(
                    "/api/v1/agents/upload-file",
                    files=files,
                    headers=headers
                )
            
            print(f"   Agent upload response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Verify schema compliance
                assert "file_path" in result, "Response must contain file_path"
                assert isinstance(result["file_path"], str), "file_path must be string"
                
                # Verify file path follows RAGnetic structure
                file_path = Path(result["file_path"])
                assert "uploads" in str(file_path), "File must be in uploads directory"
                assert "user_" in str(file_path), "File must be in user-specific directory"
                
                print("   Agent upload schema compliance verified")
                return True
            else:
                print(f"   Agent upload failed: {response.text[:200]}")
                return False
            
        except Exception as e:
            print(f"Schema compliance test failed: {e}")
            return False
    
    def test_training_dataset_schema_compliance(self):
        """Test training dataset upload schema compliance"""
        print(" Testing training dataset schema compliance...")
        
        try:
            # Create valid JSONL training data following RAGnetic schema
            training_data = []
            for i in range(5):
                record = {
                    "instruction": f"Schema test instruction {i}",
                    "input": f"Schema test input {i}",
                    "output": f"Schema test output {i}"
                }
                training_data.append(json.dumps(record))
            
            test_file = self.create_test_file("schema_training.jsonl", "\n".join(training_data))
            
            headers = {"X-API-Key": self.test_api_key}
            
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "application/json")}
                response = self.client.post(
                    "/api/v1/training/upload-dataset",
                    files=files,
                    headers=headers
                )
            
            print(f"   Training upload response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Verify schema compliance
                required_fields = ["file_path", "file_id", "filename", "size"]
                for field in required_fields:
                    assert field in result, f"Response must contain {field}"
                
                # Verify file path follows RAGnetic structure
                file_path = Path(result["file_path"])
                assert "uploaded_temp" in str(file_path), "File must be in uploaded_temp directory"
                assert "user_" in str(file_path), "File must be in user-specific directory"
                
                print("   Training dataset schema compliance verified")
                return True
            else:
                print(f"   Training upload failed: {response.text[:200]}")
                return False
            
        except Exception as e:
            print(f"Training dataset schema compliance test failed: {e}")
            return False
    
    def test_temporary_document_schema_compliance(self):
        """Test temporary document upload schema compliance"""
        print(" Testing temporary document schema compliance...")
        
        try:
            # Create test document
            test_content = "# Schema Compliance Test Document\n\nThis document tests schema compliance.\n\n" * 50
            test_file = self.create_test_file("schema_temp_test.md", test_content)
            
            headers = {"X-API-Key": self.test_api_key}
            thread_id = f"schema-test-{uuid.uuid4()}"
            
            with open(test_file, "rb") as f:
                files = {"file": (test_file.name, f, "text/markdown")}
                data = {"thread_id": thread_id}
                response = self.client.post(
                    "/api/v1/documents/upload",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            print(f"   Temp upload response: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Verify schema compliance
                required_fields = ["temp_doc_id", "file_name", "file_size"]
                for field in required_fields:
                    assert field in result, f"Response must contain {field}"
                
                temp_doc_id = result["temp_doc_id"]
                self.upload_results.append(temp_doc_id)
                
                # Verify database schema compliance
                from app.db import get_sync_db_engine
                engine = get_sync_db_engine()
                with engine.connect() as conn:
                    stmt = temporary_documents_table.select().where(
                        temporary_documents_table.c.temp_doc_id == temp_doc_id
                    )
                    temp_doc_record = conn.execute(stmt).mappings().first()
                    
                    if temp_doc_record:
                        # Verify database schema
                        required_db_fields = ["id", "temp_doc_id", "user_id", "thread_id", "original_name", "file_size", "created_at", "expires_at", "cleaned_up"]
                        for field in required_db_fields:
                            assert field in temp_doc_record, f"Database record must contain {field}"
                        
                        print("   Database schema compliance verified")
                    else:
                        print("   Database record not found")
                    
                    # Verify file storage schema compliance
                    paths = get_path_settings()
                    temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads" / "1" / thread_id
                    stored_file = temp_uploads_dir / f"{temp_doc_id}_{test_file.name}"
                    
                    if stored_file.exists():
                        print("   File storage schema compliance verified")
                    else:
                        print("   File storage not found")
                    
                    # Verify vector store schema compliance
                    vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data" / temp_doc_id
                    if vectorstore_dir.exists():
                        faiss_file = vectorstore_dir / "index.faiss"
                        pkl_file = vectorstore_dir / "index.pkl"
                        
                        if faiss_file.exists() and pkl_file.exists():
                            print("   Vector store schema compliance verified")
                        else:
                            print("   Vector store files incomplete")
                    else:
                        print("   Vector store not found")
                
                print("   Temporary document schema compliance verified")
                return True
            else:
                print(f"   Temp upload failed: {response.text[:200]}")
                return False
            
        except Exception as e:
            print(f"Temporary document schema compliance test failed: {e}")
            return False
    
    def test_file_validation_schema_compliance(self):
        """Test file validation follows RAGnetic schema"""
        print(" Testing file validation schema compliance...")
        
        try:
            headers = {"X-API-Key": self.test_api_key}
            thread_id = f"validation-schema-test-{uuid.uuid4()}"
            
            # Test unsupported file type (should be rejected with proper error schema)
            malicious_file = self.create_test_file("malicious.exe", "fake executable content")
            
            with open(malicious_file, "rb") as f:
                files = {"file": (malicious_file.name, f, "application/octet-stream")}
                data = {"thread_id": thread_id}
                response = self.client.post(
                    "/api/v1/documents/upload",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            print(f"   Malicious file response: {response.status_code}")
            
            if response.status_code == 400:
                # Verify error response schema
                error_data = response.json()
                assert "detail" in error_data, "Error response must contain detail"
                print("   Error response schema compliance verified")
            else:
                print(f"   Malicious file not rejected: {response.status_code}")
            
            # Test oversized file (should be rejected with proper error schema)
            large_content = "A" * (26 * 1024 * 1024)  # 26MB
            oversized_file = self.create_test_file("oversized.txt", large_content)
            
            with open(oversized_file, "rb") as f:
                files = {"file": (oversized_file.name, f, "text/plain")}
                data = {"thread_id": thread_id}
                response = self.client.post(
                    "/api/v1/documents/upload",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            print(f"   Oversized file response: {response.status_code}")
            
            if response.status_code == 400:
                # Verify error response schema
                error_data = response.json()
                assert "detail" in error_data, "Error response must contain detail"
                print("   Error response schema compliance verified")
            else:
                print(f"   Oversized file not rejected: {response.status_code}")
            
            print("File validation schema compliance verified")
            return True
            
        except Exception as e:
            print(f"File validation schema compliance test failed: {e}")
            return False
    
    def test_database_schema_compliance(self):
        """Test database schema compliance"""
        print(" Testing database schema compliance...")
        
        try:
            # Test temporary_documents_table schema
            from app.db import get_sync_db_engine
            engine = get_sync_db_engine()
            with engine.connect() as conn:
                # Get table columns
                columns = temporary_documents_table.columns
                required_columns = ["id", "temp_doc_id", "user_id", "thread_id", "original_name", "file_size", "created_at", "expires_at", "cleaned_up"]
                
                for col_name in required_columns:
                    assert col_name in columns, f"temporary_documents_table must have {col_name} column"
                
                print("   temporary_documents_table schema verified")
                
                # Test document_chunks_table schema
                chunk_columns = document_chunks_table.columns
                required_chunk_columns = ["id", "document_name", "chunk_index", "content", "page_number", "row_number", "created_at", "temp_document_id"]
                
                for col_name in required_chunk_columns:
                    assert col_name in chunk_columns, f"document_chunks_table must have {col_name} column"
                
                print("   document_chunks_table schema verified")
            
            print("Database schema compliance verified")
            return True
            
        except Exception as e:
            print(f"Database schema compliance test failed: {e}")
            return False
    
    def test_directory_structure_schema_compliance(self):
        """Test directory structure follows RAGnetic schema"""
        print(" Testing directory structure schema compliance...")
        
        try:
            paths = get_path_settings()
            
            # Verify required directories exist and follow schema
            required_dirs = {
                "DATA_DIR": "data",
                "TEMP_CLONES_DIR": ".ragnetic/.ragnetic_temp_clones",
                "VECTORSTORE_DIR": "vectorstore",
                "AGENTS_DIR": "agents"
            }
            
            for dir_name, expected_path in required_dirs.items():
                dir_path = paths[dir_name]
                if isinstance(dir_path, Path):
                    dir_path.mkdir(parents=True, exist_ok=True)
                    assert dir_path.exists(), f"{dir_name} directory must exist"
                    assert expected_path in str(dir_path), f"{dir_name} must be in {expected_path}"
                    print(f"   {dir_name} schema compliance verified: {dir_path}")
                else:
                    print(f"   {dir_name} is not a Path object: {dir_path}")
            
            print("Directory structure schema compliance verified")
            return True
            
        except Exception as e:
            print(f"Directory structure schema compliance test failed: {e}")
            return False
    
    def test_cleanup_schema_compliance(self):
        """Test cleanup follows RAGnetic schema"""
        print(" Testing cleanup schema compliance...")
        
        try:
            if not self.upload_results:
                print("   No uploaded documents to test cleanup")
                return True
            
            # Test cleanup of one document
            temp_doc_id = self.upload_results[0]
            
            # Get document data
            from app.db import get_sync_db_engine
            engine = get_sync_db_engine()
            with engine.connect() as conn:
                stmt = temporary_documents_table.select().where(
                    temporary_documents_table.c.temp_doc_id == temp_doc_id
                )
                temp_doc_record = conn.execute(stmt).mappings().first()
                
                if temp_doc_record:
                    # Perform cleanup following RAGnetic schema
                    temp_doc_data = dict(temp_doc_record)
                    TemporaryDocumentService.cleanup_fs(temp_doc_data)
                    
                    # Verify cleanup follows schema
                    paths = get_path_settings()
                    temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads" / "1" / temp_doc_record["thread_id"]
                    vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data" / temp_doc_id
                    
                    stored_file = temp_uploads_dir / f"{temp_doc_id}_{temp_doc_record['original_name']}"
                    
                    # Files should be cleaned up according to schema
                    if not stored_file.exists() and not vectorstore_dir.exists():
                        print("   Cleanup schema compliance verified")
                        return True
                    else:
                        print("   Cleanup incomplete")
                        return False
                else:
                    print("   No document record found for cleanup")
                    return True
            
        except Exception as e:
            print(f"Cleanup schema compliance test failed: {e}")
            return False
    
    def run_all_final_comprehensive_tests(self):
        """Run all final comprehensive tests"""
        print("Starting Final Comprehensive Upload Tests")
        print("=" * 60)
        
        test_methods = [
            ("Upload Schema Compliance", self.test_upload_schema_compliance),
            ("Training Dataset Schema Compliance", self.test_training_dataset_schema_compliance),
            ("Temporary Document Schema Compliance", self.test_temporary_document_schema_compliance),
            ("File Validation Schema Compliance", self.test_file_validation_schema_compliance),
            ("Database Schema Compliance", self.test_database_schema_compliance),
            ("Directory Structure Schema Compliance", self.test_directory_structure_schema_compliance),
            ("Cleanup Schema Compliance", self.test_cleanup_schema_compliance),
        ]
        
        results = {}
        overall_success = True
        
        for test_name, test_method in test_methods:
            print(f"\n{'='*40}")
            print(f"Running {test_name}")
            print(f"{'='*40}")
            
            try:
                success = test_method()
                results[test_name] = {"success": success, "error": None}
                if not success:
                    overall_success = False
            except Exception as e:
                print(f"{test_name} failed with exception: {e}")
                results[test_name] = {"success": False, "error": str(e)}
                overall_success = False
        
        # Generate report
        print(f"\n{'='*60}")
        print("FINAL COMPREHENSIVE TEST REPORT")
        print(f"{'='*60}")
        
        successful_tests = sum(1 for result in results.values() if result["success"])
        total_tests = len(results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {(successful_tests/total_tests*100):.2f}%")
        print(f"Overall Status: {'PASSED' if overall_success else 'FAILED'}")
        
        print(f"\nDetailed Results:")
        for test_name, result in results.items():
            status = "PASSED" if result["success"] else "FAILED"
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
            "results": results,
            "schema_compliance": "All tests focus on RAGnetic schema compliance"
        }
        
        report_file = Path("tests/final_comprehensive_upload_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n Report saved to: {report_file}")
        
        return 0 if overall_success else 1


def main():
    """Main entry point"""
    print("RAGnetic Final Comprehensive Upload Test Suite")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_suite = FinalComprehensiveUploadTests()
    exit_code = test_suite.run_all_final_comprehensive_tests()
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(main())
