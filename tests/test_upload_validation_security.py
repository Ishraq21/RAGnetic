"""
File Validation and Security Tests for RAGnetic Upload Functionality
Tests file validation, security measures, and malicious file handling.
"""

import asyncio
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Any, List
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_path_settings


class UploadValidationSecurityTests:
    """File validation and security testing class"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_api_key = "YOUR_TEST_API_KEY_1"
        self.test_user_id = 1
        self.temp_files = []
        
    def create_test_file(self, filename: str, content: str) -> Path:
        """Create a test file"""
        temp_file = Path(tempfile.mktemp(suffix=f"_{filename}"))
        
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def create_binary_file(self, filename: str, content: bytes) -> Path:
        """Create a binary test file"""
        temp_file = Path(tempfile.mktemp(suffix=f"_{filename}"))
        
        with open(temp_file, "wb") as f:
            f.write(content)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def cleanup_temp_files(self):
        """Clean up temporary test files"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
        self.temp_files.clear()
    
    def test_supported_file_types(self):
        """Test supported file types validation"""
        print(" Testing supported file types...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test all supported file types
        supported_files = [
            ("test.pdf", "PDF content", "application/pdf"),
            ("test.docx", "DOCX content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("test.txt", "Plain text content", "text/plain"),
            ("test.csv", "id,name,value\n1,test,value", "text/csv"),
            ("test.json", '{"test": "json content"}', "application/json"),
            ("test.yaml", "test: yaml content", "text/x-yaml"),
            ("test.yml", "test: yml content", "application/x-yaml"),
            ("test.hcl", "resource \"test\" \"example\" {}", "text/x-hcl"),
            ("test.tf", "resource \"test\" \"example\" {}", "text/x-terraform"),
            ("test.ipynb", '{"cells": [], "metadata": {}}', "application/x-ipynb+json"),
            ("test.md", "# Markdown content", "text/markdown"),
            ("test.log", "Log file content", "text/plain"),
            ("test.html", "<html><body>HTML content</body></html>", "text/html"),
        ]
        
        for filename, content, mime_type in supported_files:
            test_file = self.create_test_file(filename, content)
            
            with open(test_file, "rb") as f:
                files = {"file": (filename, f, mime_type)}
                data = {"thread_id": f"validation-{filename}"}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            assert response.status_code == 200, f"Supported file type {filename} should be accepted: {response.text}"
            print(f"    {filename} accepted")
        
        print(" Supported file types test passed")
    
    def test_unsupported_file_types(self):
        """Test unsupported file types rejection"""
        print(" Testing unsupported file types...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test unsupported file types
        unsupported_files = [
            ("test.exe", b"fake executable content", "application/octet-stream"),
            ("test.bat", "@echo off\necho batch file", "application/x-msdos-program"),
            ("test.sh", "#!/bin/bash\necho shell script", "application/x-sh"),
            ("test.py", "print('python script')", "text/x-python"),
            ("test.js", "console.log('javascript')", "application/javascript"),
            ("test.php", "<?php echo 'php script'; ?>", "application/x-php"),
            ("test.sql", "SELECT * FROM table;", "application/sql"),
            ("test.zip", b"fake zip content", "application/zip"),
            ("test.rar", b"fake rar content", "application/x-rar-compressed"),
            ("test.tar", b"fake tar content", "application/x-tar"),
            ("test.gz", b"fake gzip content", "application/gzip"),
            ("test.7z", b"fake 7z content", "application/x-7z-compressed"),
            ("test.iso", b"fake iso content", "application/x-iso9660-image"),
            ("test.img", b"fake image content", "application/octet-stream"),
            ("test.bin", b"fake binary content", "application/octet-stream"),
        ]
        
        for filename, content, mime_type in unsupported_files:
            if isinstance(content, str):
                test_file = self.create_test_file(filename, content)
            else:
                test_file = self.create_binary_file(filename, content)
            
            with open(test_file, "rb") as f:
                files = {"file": (filename, f, mime_type)}
                data = {"thread_id": f"validation-{filename}"}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            assert response.status_code == 400, f"Unsupported file type {filename} should be rejected: {response.text}"
            print(f"    {filename} rejected")
        
        print(" Unsupported file types test passed")
    
    def test_file_size_limits(self):
        """Test file size limits"""
        print(" Testing file size limits...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test file sizes
        size_tests = [
            (1, True),      # 1MB - should pass
            (10, True),     # 10MB - should pass
            (20, True),     # 20MB - should pass
            (24, True),     # 24MB - should pass
            (25, True),     # 25MB - should pass (at limit)
            (26, False),    # 26MB - should fail (over limit)
            (50, False),    # 50MB - should fail (way over limit)
            (100, False),   # 100MB - should fail (way over limit)
        ]
        
        for size_mb, should_pass in size_tests:
            # Create file of specified size
            content = "A" * (1024 * 1024)  # 1MB of 'A's
            test_file = self.create_test_file(f"size_test_{size_mb}mb.txt", content * size_mb)
            
            with open(test_file, "rb") as f:
                files = {"file": (f"size_test_{size_mb}mb.txt", f, "text/plain")}
                data = {"thread_id": f"size-test-{size_mb}mb"}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            if should_pass:
                assert response.status_code == 200, f"{size_mb}MB file should be accepted: {response.text}"
                print(f"    {size_mb}MB file accepted")
            else:
                assert response.status_code == 400, f"{size_mb}MB file should be rejected: {response.text}"
                print(f"    {size_mb}MB file rejected")
        
        print(" File size limits test passed")
    
    def test_filename_security(self):
        """Test filename security measures"""
        print(" Testing filename security...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test potentially malicious filenames
        malicious_filenames = [
            "../../../etc/passwd.txt",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.with.dots.txt",
            "file with unicode .txt",
            "file with special chars !@#$%^&*().txt",
            "very_long_filename_" + "a" * 200 + ".txt",
            "file\nwith\nnewlines.txt",
            "file\twith\ttabs.txt",
            "file\rwith\rcarriage.txt",
            "file with null\x00char.txt",
        ]
        
        for malicious_filename in malicious_filenames:
            # Create safe test content
            test_file = self.create_test_file("safe_content.txt", "Safe test content")
            
            with open(test_file, "rb") as f:
                files = {"file": (malicious_filename, f, "text/plain")}
                data = {"thread_id": f"filename-test-{hash(malicious_filename)}"}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            # Should either accept (with sanitized filename) or reject
            assert response.status_code in [200, 400], f"Unexpected response for filename '{malicious_filename}': {response.status_code}"
            
            if response.status_code == 200:
                result = response.json()
                # Verify the stored filename is safe
                stored_filename = result["file_name"]
                assert ".." not in stored_filename, f"Directory traversal in stored filename: {stored_filename}"
                assert "/" not in stored_filename, f"Path separator in stored filename: {stored_filename}"
                assert "\\" not in stored_filename, f"Windows path separator in stored filename: {stored_filename}"
                print(f"    Malicious filename '{malicious_filename}' sanitized to '{stored_filename}'")
            else:
                print(f"    Malicious filename '{malicious_filename}' rejected")
        
        print(" Filename security test passed")
    
    def test_mime_type_validation(self):
        """Test MIME type validation"""
        print(" Testing MIME type validation...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test MIME type spoofing
        mime_type_tests = [
            ("test.txt", "text/plain", "Plain text content", True),
            ("test.txt", "application/octet-stream", "Plain text content", True),  # Should be detected as text
            ("test.exe", "text/plain", b"fake executable content", False),  # Should be rejected
            ("test.pdf", "application/pdf", "PDF content", True),
            ("test.pdf", "text/plain", "PDF content", False),  # Wrong MIME type
            ("test.json", "application/json", '{"test": "json"}', True),
            ("test.json", "text/plain", '{"test": "json"}', True),  # Should be detected as JSON
        ]
        
        for filename, mime_type, content, should_pass in mime_type_tests:
            if isinstance(content, str):
                test_file = self.create_test_file(filename, content)
            else:
                test_file = self.create_binary_file(filename, content)
            
            with open(test_file, "rb") as f:
                files = {"file": (filename, f, mime_type)}
                data = {"thread_id": f"mime-test-{filename}"}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            if should_pass:
                assert response.status_code == 200, f"File {filename} with MIME {mime_type} should be accepted: {response.text}"
                print(f"    {filename} with MIME {mime_type} accepted")
            else:
                assert response.status_code == 400, f"File {filename} with MIME {mime_type} should be rejected: {response.text}"
                print(f"    {filename} with MIME {mime_type} rejected")
        
        print(" MIME type validation test passed")
    
    def test_malicious_content_detection(self):
        """Test malicious content detection"""
        print(" Testing malicious content detection...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test potentially malicious content
        malicious_content_tests = [
            ("script_injection.txt", "<script>alert('xss')</script>", True),  # Should be accepted as text
            ("sql_injection.txt", "'; DROP TABLE users; --", True),  # Should be accepted as text
            ("path_traversal.txt", "../../../etc/passwd", True),  # Should be accepted as text
            ("command_injection.txt", "; rm -rf /", True),  # Should be accepted as text
            ("fake_pdf.txt", "%PDF-1.4 fake pdf content", True),  # Should be accepted as text
            ("fake_zip.txt", "PK fake zip content", True),  # Should be accepted as text
        ]
        
        for filename, content, should_pass in malicious_content_tests:
            test_file = self.create_test_file(filename, content)
            
            with open(test_file, "rb") as f:
                files = {"file": (filename, f, "text/plain")}
                data = {"thread_id": f"malicious-test-{filename}"}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            if should_pass:
                assert response.status_code == 200, f"Malicious content in {filename} should be accepted as text: {response.text}"
                print(f"    {filename} with malicious content accepted as text")
            else:
                assert response.status_code == 400, f"Malicious content in {filename} should be rejected: {response.text}"
                print(f"    {filename} with malicious content rejected")
        
        print(" Malicious content detection test passed")
    
    def test_zip_bomb_protection(self):
        """Test ZIP bomb protection"""
        print(" Testing ZIP bomb protection...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Create a ZIP file (this would be rejected anyway due to file type)
        zip_file = self.create_test_file("test.zip", "fake zip content")
        
        with open(zip_file, "rb") as f:
            files = {"file": ("test.zip", f, "application/zip")}
            data = {"thread_id": "zip-bomb-test"}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 400, "ZIP files should be rejected: {response.text}"
        print("    ZIP file rejected")
        
        print(" ZIP bomb protection test passed")
    
    def test_empty_file_handling(self):
        """Test empty file handling"""
        print(" Testing empty file handling...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test empty file
        empty_file = self.create_test_file("empty.txt", "")
        
        with open(empty_file, "rb") as f:
            files = {"file": ("empty.txt", f, "text/plain")}
            data = {"thread_id": "empty-file-test"}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=headers
            )
        
        assert response.status_code == 400, "Empty files should be rejected: {response.text}"
        print("    Empty file rejected")
        
        print(" Empty file handling test passed")
    
    def test_unicode_handling(self):
        """Test Unicode and special character handling"""
        print(" Testing Unicode handling...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test various Unicode content
        unicode_tests = [
            ("unicode_basic.txt", "Hello ", True),
            ("unicode_emoji.txt", "Hello  World ", True),
            ("unicode_math.txt", "∑∞∫∂∇", True),
            ("unicode_arabic.txt", "مرحبا بالعالم", True),
            ("unicode_chinese.txt", "", True),
            ("unicode_cyrillic.txt", "Привет мир", True),
            ("unicode_control.txt", "Text with\x00null\x01control\x02chars", True),
        ]
        
        for filename, content, should_pass in unicode_tests:
            test_file = self.create_test_file(filename, content)
            
            with open(test_file, "rb") as f:
                files = {"file": (filename, f, "text/plain")}
                data = {"thread_id": f"unicode-test-{filename}"}
                response = self.client.post(
                    "/api/v1/chat/upload-temp-document",
                    files=files,
                    data=data,
                    headers=headers
                )
            
            if should_pass:
                assert response.status_code == 200, f"Unicode content in {filename} should be accepted: {response.text}"
                print(f"    {filename} with Unicode content accepted")
            else:
                assert response.status_code == 400, f"Unicode content in {filename} should be rejected: {response.text}"
                print(f"    {filename} with Unicode content rejected")
        
        print(" Unicode handling test passed")
    
    def test_concurrent_malicious_uploads(self):
        """Test concurrent malicious uploads"""
        print(" Testing concurrent malicious uploads...")
        
        import concurrent.futures
        import threading
        
        def malicious_upload_worker(worker_id: int, num_uploads: int) -> List[Dict]:
            """Worker function for malicious uploads"""
            results = []
            headers = {"Authorization": f"Bearer {self.test_api_key}"}
            
            malicious_files = [
                ("malicious.exe", b"fake executable content"),
                ("script.js", "console.log('malicious script')"),
                ("../../etc/passwd", "root:x:0:0:root:/root:/bin/bash"),
                ("large_file.txt", "A" * (26 * 1024 * 1024)),  # 26MB file
            ]
            
            for i in range(num_uploads):
                try:
                    filename, content = malicious_files[i % len(malicious_files)]
                    
                    if isinstance(content, str):
                        test_file = self.create_test_file(f"{worker_id}_{i}_{filename}", content)
                    else:
                        test_file = self.create_binary_file(f"{worker_id}_{i}_{filename}", content)
                    
                    with open(test_file, "rb") as f:
                        files = {"file": (filename, f, "application/octet-stream")}
                        data = {"thread_id": f"malicious-worker-{worker_id}-{i}"}
                        response = self.client.post(
                            "/api/v1/chat/upload-temp-document",
                            files=files,
                            data=data,
                            headers=headers
                        )
                    
                    results.append({
                        "worker_id": worker_id,
                        "upload_id": i,
                        "filename": filename,
                        "status_code": response.status_code,
                        "success": response.status_code == 200,
                    })
                    
                except Exception as e:
                    results.append({
                        "worker_id": worker_id,
                        "upload_id": i,
                        "filename": filename,
                        "status_code": 0,
                        "success": False,
                        "error": str(e)
                    })
            
            return results
        
        # Test with multiple workers
        num_workers = 5
        uploads_per_worker = 4
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(malicious_upload_worker, worker_id, uploads_per_worker)
                for worker_id in range(num_workers)
            ]
            
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    worker_results = future.result()
                    all_results.extend(worker_results)
                except Exception as e:
                    print(f"Malicious upload worker failed: {e}")
        
        # Analyze results
        total_uploads = len(all_results)
        successful_uploads = sum(1 for r in all_results if r["success"])
        failed_uploads = total_uploads - successful_uploads
        
        print(f" Concurrent Malicious Upload Test Results:")
        print(f"   Total uploads: {total_uploads}")
        print(f"   Successful: {successful_uploads}")
        print(f"   Failed: {failed_uploads}")
        print(f"   Success rate: {(successful_uploads/total_uploads*100):.2f}%")
        
        # Most malicious uploads should fail
        assert failed_uploads >= total_uploads * 0.8, f"Too many malicious uploads succeeded: {successful_uploads}/{total_uploads}"
        
        print(" Concurrent malicious uploads test passed")
    
    def test_authorization_security(self):
        """Test authorization security"""
        print(" Testing authorization security...")
        
        # Test without authorization
        test_file = self.create_test_file("auth_test.txt", "Authorization test content")
        
        with open(test_file, "rb") as f:
            files = {"file": ("auth_test.txt", f, "text/plain")}
            data = {"thread_id": "auth-test"}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data
            )
        
        assert response.status_code == 401, "Upload without authorization should be rejected"
        print("    Upload without authorization rejected")
        
        # Test with invalid authorization
        invalid_headers = {"Authorization": "Bearer invalid-api-key"}
        
        with open(test_file, "rb") as f:
            files = {"file": ("auth_test.txt", f, "text/plain")}
            data = {"thread_id": "auth-test"}
            response = self.client.post(
                "/api/v1/chat/upload-temp-document",
                files=files,
                data=data,
                headers=invalid_headers
            )
        
        assert response.status_code == 401, "Upload with invalid authorization should be rejected"
        print("    Upload with invalid authorization rejected")
        
        print(" Authorization security test passed")
    
    def run_all_validation_security_tests(self):
        """Run all validation and security tests"""
        print(" Starting validation and security tests...")
        
        try:
            test_methods = [
                self.test_supported_file_types,
                self.test_unsupported_file_types,
                self.test_file_size_limits,
                self.test_filename_security,
                self.test_mime_type_validation,
                self.test_malicious_content_detection,
                self.test_zip_bomb_protection,
                self.test_empty_file_handling,
                self.test_unicode_handling,
                self.test_concurrent_malicious_uploads,
                self.test_authorization_security,
            ]
            
            for test_method in test_methods:
                try:
                    print(f"\n{'='*50}")
                    test_method()
                    print(f"{'='*50}\n")
                except Exception as e:
                    print(f" Validation test {test_method.__name__} failed: {e}")
                    raise
            
            print(" All validation and security tests passed!")
            
        except Exception as e:
            print(f" Validation test suite failed: {e}")
            raise
        finally:
            self.cleanup_temp_files()


if __name__ == "__main__":
    test_suite = UploadValidationSecurityTests()
    test_suite.run_all_validation_security_tests()
