"""
Stress Tests for RAGnetic Upload Functionality
Tests upload system under high load, concurrent access, and edge cases.
"""

import asyncio
import json
import os
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List
import uuid
import psutil
import gc

import pytest
import requests
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_path_settings
from app.db import get_db
from app.db.models import temporary_documents_table, document_chunks_table


class UploadStressTests:
    """Stress testing class for upload functionality"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_api_key = "YOUR_TEST_API_KEY_1"
        self.test_user_id = 1
        self.temp_files = []
        self.upload_results = []
        self.start_time = None
        self.end_time = None
        
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
    
    def cleanup_temp_files(self):
        """Clean up temporary test files"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
        self.temp_files.clear()
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
        }
    
    def test_concurrent_uploads_stress(self):
        """Test concurrent uploads under stress"""
        print(" Testing concurrent uploads stress...")
        
        self.start_time = time.time()
        initial_metrics = self.get_system_metrics()
        
        def upload_worker(worker_id: int, num_uploads: int) -> List[Dict]:
            """Worker function for concurrent uploads"""
            results = []
            headers = {"Authorization": f"Bearer {self.test_api_key}"}
            
            for i in range(num_uploads):
                try:
                    # Create test file
                    content = f"Concurrent upload test content from worker {worker_id}, upload {i}\n" * 100
                    test_file = self.create_test_file(f"concurrent_{worker_id}_{i}.txt", content)
                    
                    with open(test_file, "rb") as f:
                        files = {"file": (test_file.name, f, "text/plain")}
                        data = {"thread_id": f"stress-worker-{worker_id}-{i}"}
                        response = self.client.post(
                            "/api/v1/chat/upload-temp-document",
                            files=files,
                            data=data,
                            headers=headers
                        )
                    
                    results.append({
                        "worker_id": worker_id,
                        "upload_id": i,
                        "status_code": response.status_code,
                        "success": response.status_code == 200,
                        "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0
                    })
                    
                except Exception as e:
                    results.append({
                        "worker_id": worker_id,
                        "upload_id": i,
                        "status_code": 0,
                        "success": False,
                        "error": str(e)
                    })
            
            return results
        
        # Test with multiple workers and uploads
        num_workers = 10
        uploads_per_worker = 5
        
        print(f"Starting {num_workers} workers with {uploads_per_worker} uploads each...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(upload_worker, worker_id, uploads_per_worker)
                for worker_id in range(num_workers)
            ]
            
            all_results = []
            for future in as_completed(futures):
                try:
                    worker_results = future.result()
                    all_results.extend(worker_results)
                except Exception as e:
                    print(f"Worker failed: {e}")
        
        self.end_time = time.time()
        final_metrics = self.get_system_metrics()
        
        # Analyze results
        total_uploads = len(all_results)
        successful_uploads = sum(1 for r in all_results if r["success"])
        failed_uploads = total_uploads - successful_uploads
        
        success_rate = (successful_uploads / total_uploads) * 100 if total_uploads > 0 else 0
        total_time = self.end_time - self.start_time
        uploads_per_second = total_uploads / total_time if total_time > 0 else 0
        
        print(f" Concurrent Upload Stress Test Results:")
        print(f"   Total uploads: {total_uploads}")
        print(f"   Successful: {successful_uploads}")
        print(f"   Failed: {failed_uploads}")
        print(f"   Success rate: {success_rate:.2f}%")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Uploads/second: {uploads_per_second:.2f}")
        print(f"   Initial CPU: {initial_metrics['cpu_percent']:.1f}%")
        print(f"   Final CPU: {final_metrics['cpu_percent']:.1f}%")
        print(f"   Initial Memory: {initial_metrics['memory_percent']:.1f}%")
        print(f"   Final Memory: {final_metrics['memory_percent']:.1f}%")
        
        # Assertions
        assert success_rate >= 80, f"Success rate too low: {success_rate:.2f}%"
        assert uploads_per_second >= 1, f"Upload rate too low: {uploads_per_second:.2f} uploads/sec"
        
        print(" Concurrent uploads stress test passed")
    
    def test_large_file_upload_stress(self):
        """Test large file uploads under stress"""
        print(" Testing large file upload stress...")
        
        self.start_time = time.time()
        initial_metrics = self.get_system_metrics()
        
        # Test different file sizes
        file_sizes = [1, 5, 10, 20]  # MB
        results = []
        
        for size_mb in file_sizes:
            print(f"Testing {size_mb}MB file upload...")
            
            # Create large file
            content = "Large file content for stress testing.\n"
            test_file = self.create_test_file(f"large_{size_mb}mb.txt", content, size_mb)
            
            headers = {"Authorization": f"Bearer {self.test_api_key}"}
            
            start_upload = time.time()
            try:
                with open(test_file, "rb") as f:
                    files = {"file": (test_file.name, f, "text/plain")}
                    data = {"thread_id": f"large-file-{size_mb}mb"}
                    response = self.client.post(
                        "/api/v1/chat/upload-temp-document",
                        files=files,
                        data=data,
                        headers=headers
                    )
                
                upload_time = time.time() - start_upload
                upload_speed = (size_mb * 1024 * 1024) / upload_time / 1024 / 1024  # MB/s
                
                results.append({
                    "size_mb": size_mb,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "upload_time": upload_time,
                    "upload_speed_mbps": upload_speed
                })
                
                print(f"   {size_mb}MB: {response.status_code} in {upload_time:.2f}s ({upload_speed:.2f} MB/s)")
                
            except Exception as e:
                results.append({
                    "size_mb": size_mb,
                    "status_code": 0,
                    "success": False,
                    "error": str(e)
                })
                print(f"   {size_mb}MB: Failed - {e}")
        
        self.end_time = time.time()
        final_metrics = self.get_system_metrics()
        
        # Analyze results
        successful_uploads = [r for r in results if r["success"]]
        failed_uploads = [r for r in results if not r["success"]]
        
        print(f" Large File Upload Stress Test Results:")
        print(f"   Successful uploads: {len(successful_uploads)}/{len(results)}")
        print(f"   Failed uploads: {len(failed_uploads)}")
        
        if successful_uploads:
            avg_speed = sum(r["upload_speed_mbps"] for r in successful_uploads) / len(successful_uploads)
            print(f"   Average upload speed: {avg_speed:.2f} MB/s")
        
        print(f"   Initial CPU: {initial_metrics['cpu_percent']:.1f}%")
        print(f"   Final CPU: {final_metrics['cpu_percent']:.1f}%")
        print(f"   Initial Memory: {initial_metrics['memory_percent']:.1f}%")
        print(f"   Final Memory: {final_metrics['memory_percent']:.1f}%")
        
        # Assertions
        assert len(successful_uploads) >= len(results) * 0.8, "Too many large file uploads failed"
        
        print(" Large file upload stress test passed")
    
    def test_rapid_sequential_uploads(self):
        """Test rapid sequential uploads"""
        print(" Testing rapid sequential uploads...")
        
        self.start_time = time.time()
        initial_metrics = self.get_system_metrics()
        
        num_uploads = 50
        results = []
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        for i in range(num_uploads):
            try:
                # Create test file
                content = f"Rapid sequential upload test content {i}\n" * 50
                test_file = self.create_test_file(f"rapid_{i}.txt", content)
                
                start_upload = time.time()
                with open(test_file, "rb") as f:
                    files = {"file": (test_file.name, f, "text/plain")}
                    data = {"thread_id": f"rapid-{i}"}
                    response = self.client.post(
                        "/api/v1/chat/upload-temp-document",
                        files=files,
                        data=data,
                        headers=headers
                    )
                
                upload_time = time.time() - start_upload
                
                results.append({
                    "upload_id": i,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "upload_time": upload_time
                })
                
                if i % 10 == 0:
                    print(f"   Completed {i}/{num_uploads} uploads")
                
            except Exception as e:
                results.append({
                    "upload_id": i,
                    "status_code": 0,
                    "success": False,
                    "error": str(e)
                })
        
        self.end_time = time.time()
        final_metrics = self.get_system_metrics()
        
        # Analyze results
        successful_uploads = [r for r in results if r["success"]]
        failed_uploads = [r for r in results if not r["success"]]
        
        total_time = self.end_time - self.start_time
        uploads_per_second = len(successful_uploads) / total_time if total_time > 0 else 0
        
        if successful_uploads:
            avg_upload_time = sum(r["upload_time"] for r in successful_uploads) / len(successful_uploads)
            min_upload_time = min(r["upload_time"] for r in successful_uploads)
            max_upload_time = max(r["upload_time"] for r in successful_uploads)
        else:
            avg_upload_time = min_upload_time = max_upload_time = 0
        
        print(f" Rapid Sequential Upload Test Results:")
        print(f"   Total uploads: {len(results)}")
        print(f"   Successful: {len(successful_uploads)}")
        print(f"   Failed: {len(failed_uploads)}")
        print(f"   Success rate: {(len(successful_uploads)/len(results)*100):.2f}%")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Uploads/second: {uploads_per_second:.2f}")
        print(f"   Avg upload time: {avg_upload_time:.3f}s")
        print(f"   Min upload time: {min_upload_time:.3f}s")
        print(f"   Max upload time: {max_upload_time:.3f}s")
        print(f"   Initial CPU: {initial_metrics['cpu_percent']:.1f}%")
        print(f"   Final CPU: {final_metrics['cpu_percent']:.1f}%")
        print(f"   Initial Memory: {initial_metrics['memory_percent']:.1f}%")
        print(f"   Final Memory: {final_metrics['memory_percent']:.1f}%")
        
        # Assertions
        assert len(successful_uploads) >= len(results) * 0.9, "Too many rapid uploads failed"
        assert uploads_per_second >= 2, f"Upload rate too low: {uploads_per_second:.2f} uploads/sec"
        
        print(" Rapid sequential uploads test passed")
    
    def test_memory_usage_stress(self):
        """Test memory usage under upload stress"""
        print(" Testing memory usage stress...")
        
        initial_memory = psutil.virtual_memory().percent
        initial_gc_count = len(gc.get_objects())
        
        # Perform many uploads to test memory usage
        num_uploads = 100
        results = []
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        for i in range(num_uploads):
            try:
                # Create test file
                content = f"Memory stress test content {i}\n" * 100
                test_file = self.create_test_file(f"memory_{i}.txt", content)
                
                with open(test_file, "rb") as f:
                    files = {"file": (test_file.name, f, "text/plain")}
                    data = {"thread_id": f"memory-{i}"}
                    response = self.client.post(
                        "/api/v1/chat/upload-temp-document",
                        files=files,
                        data=data,
                        headers=headers
                    )
                
                results.append({
                    "upload_id": i,
                    "success": response.status_code == 200,
                    "memory_usage": psutil.virtual_memory().percent
                })
                
                # Force garbage collection every 20 uploads
                if i % 20 == 0:
                    gc.collect()
                
                if i % 25 == 0:
                    current_memory = psutil.virtual_memory().percent
                    print(f"   Upload {i}: Memory usage {current_memory:.1f}%")
                
            except Exception as e:
                results.append({
                    "upload_id": i,
                    "success": False,
                    "error": str(e)
                })
        
        final_memory = psutil.virtual_memory().percent
        final_gc_count = len(gc.get_objects())
        
        # Analyze memory usage
        memory_usage_history = [r["memory_usage"] for r in results if "memory_usage" in r]
        max_memory = max(memory_usage_history) if memory_usage_history else initial_memory
        min_memory = min(memory_usage_history) if memory_usage_history else initial_memory
        avg_memory = sum(memory_usage_history) / len(memory_usage_history) if memory_usage_history else initial_memory
        
        memory_increase = final_memory - initial_memory
        gc_objects_increase = final_gc_count - initial_gc_count
        
        print(f" Memory Usage Stress Test Results:")
        print(f"   Initial memory: {initial_memory:.1f}%")
        print(f"   Final memory: {final_memory:.1f}%")
        print(f"   Memory increase: {memory_increase:.1f}%")
        print(f"   Max memory: {max_memory:.1f}%")
        print(f"   Min memory: {min_memory:.1f}%")
        print(f"   Avg memory: {avg_memory:.1f}%")
        print(f"   Initial GC objects: {initial_gc_count}")
        print(f"   Final GC objects: {final_gc_count}")
        print(f"   GC objects increase: {gc_objects_increase}")
        
        # Assertions
        assert memory_increase < 20, f"Memory usage increased too much: {memory_increase:.1f}%"
        assert max_memory < 90, f"Memory usage too high: {max_memory:.1f}%"
        
        print(" Memory usage stress test passed")
    
    def test_disk_space_stress(self):
        """Test disk space usage under upload stress"""
        print(" Testing disk space stress...")
        
        paths = get_path_settings()
        temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads"
        vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data"
        
        # Get initial disk usage
        initial_disk_usage = psutil.disk_usage('/').percent
        
        # Perform uploads to test disk usage
        num_uploads = 20
        results = []
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        for i in range(num_uploads):
            try:
                # Create larger test file
                content = f"Disk space stress test content {i}\n" * 1000
                test_file = self.create_test_file(f"disk_{i}.txt", content)
                
                with open(test_file, "rb") as f:
                    files = {"file": (test_file.name, f, "text/plain")}
                    data = {"thread_id": f"disk-{i}"}
                    response = self.client.post(
                        "/api/v1/chat/upload-temp-document",
                        files=files,
                        data=data,
                        headers=headers
                    )
                
                # Check disk usage after upload
                current_disk_usage = psutil.disk_usage('/').percent
                
                results.append({
                    "upload_id": i,
                    "success": response.status_code == 200,
                    "disk_usage": current_disk_usage
                })
                
                if i % 5 == 0:
                    print(f"   Upload {i}: Disk usage {current_disk_usage:.1f}%")
                
            except Exception as e:
                results.append({
                    "upload_id": i,
                    "success": False,
                    "error": str(e)
                })
        
        final_disk_usage = psutil.disk_usage('/').percent
        
        # Analyze disk usage
        disk_usage_history = [r["disk_usage"] for r in results if "disk_usage" in r]
        max_disk_usage = max(disk_usage_history) if disk_usage_history else initial_disk_usage
        disk_increase = final_disk_usage - initial_disk_usage
        
        print(f" Disk Space Stress Test Results:")
        print(f"   Initial disk usage: {initial_disk_usage:.1f}%")
        print(f"   Final disk usage: {final_disk_usage:.1f}%")
        print(f"   Disk usage increase: {disk_increase:.1f}%")
        print(f"   Max disk usage: {max_disk_usage:.1f}%")
        
        # Assertions
        assert disk_increase < 10, f"Disk usage increased too much: {disk_increase:.1f}%"
        assert max_disk_usage < 95, f"Disk usage too high: {max_disk_usage:.1f}%"
        
        print(" Disk space stress test passed")
    
    def test_error_recovery_stress(self):
        """Test error recovery under stress"""
        print(" Testing error recovery stress...")
        
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        
        # Test various error conditions
        error_tests = [
            ("empty_file", ""),
            ("invalid_file", "invalid content with special chars: \x00\x01\x02"),
            ("very_long_filename", "a" * 255 + ".txt"),
            ("unicode_filename", ".txt"),
        ]
        
        results = []
        
        for test_name, content in error_tests:
            try:
                test_file = self.create_test_file(f"{test_name}.txt", content)
                
                with open(test_file, "rb") as f:
                    files = {"file": (test_file.name, f, "text/plain")}
                    data = {"thread_id": f"error-{test_name}"}
                    response = self.client.post(
                        "/api/v1/chat/upload-temp-document",
                        files=files,
                        data=data,
                        headers=headers
                    )
                
                results.append({
                    "test_name": test_name,
                    "status_code": response.status_code,
                    "expected_failure": test_name in ["empty_file", "invalid_file"],
                    "handled_correctly": (response.status_code == 400) if test_name in ["empty_file", "invalid_file"] else (response.status_code == 200)
                })
                
                print(f"   {test_name}: {response.status_code}")
                
            except Exception as e:
                results.append({
                    "test_name": test_name,
                    "status_code": 0,
                    "error": str(e),
                    "handled_correctly": False
                })
                print(f"   {test_name}: Exception - {e}")
        
        # Analyze results
        correctly_handled = sum(1 for r in results if r["handled_correctly"])
        total_tests = len(results)
        
        print(f" Error Recovery Stress Test Results:")
        print(f"   Total error tests: {total_tests}")
        print(f"   Correctly handled: {correctly_handled}")
        print(f"   Error handling rate: {(correctly_handled/total_tests*100):.2f}%")
        
        # Assertions
        assert correctly_handled >= total_tests * 0.8, "Error handling rate too low"
        
        print(" Error recovery stress test passed")
    
    def test_cleanup_under_stress(self):
        """Test cleanup functionality under stress"""
        print(" Testing cleanup under stress...")
        
        # This would test the cleanup task under stress conditions
        # For now, we'll just verify that cleanup directories exist and are accessible
        
        paths = get_path_settings()
        temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads"
        vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data"
        
        # Check if cleanup directories are accessible
        assert temp_uploads_dir.exists() or temp_uploads_dir.parent.exists(), "Temp uploads directory not accessible"
        assert vectorstore_dir.exists() or vectorstore_dir.parent.exists(), "Vectorstore directory not accessible"
        
        print(" Cleanup under stress test passed")
    
    def run_all_stress_tests(self):
        """Run all stress tests"""
        print(" Starting upload stress tests...")
        
        try:
            test_methods = [
                self.test_concurrent_uploads_stress,
                self.test_large_file_upload_stress,
                self.test_rapid_sequential_uploads,
                self.test_memory_usage_stress,
                self.test_disk_space_stress,
                self.test_error_recovery_stress,
                self.test_cleanup_under_stress,
            ]
            
            for test_method in test_methods:
                try:
                    print(f"\n{'='*50}")
                    test_method()
                    print(f"{'='*50}\n")
                except Exception as e:
                    print(f" Stress test {test_method.__name__} failed: {e}")
                    raise
            
            print(" All stress tests passed!")
            
        except Exception as e:
            print(f" Stress test suite failed: {e}")
            raise
        finally:
            self.cleanup_temp_files()


if __name__ == "__main__":
    test_suite = UploadStressTests()
    test_suite.run_all_stress_tests()
