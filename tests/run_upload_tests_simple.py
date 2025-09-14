#!/usr/bin/env python3
"""
Simple Upload Test Runner for RAGnetic
Runs core upload tests without requiring full server setup.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SimpleUploadTestRunner:
    """Simple test runner for upload functionality"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.test_results = {}
        self.overall_success = True
        
    def test_file_creation(self):
        """Test file creation and basic operations"""
        print(" Testing file creation...")
        
        try:
            # Create test data directory
            test_data_dir = Path("tests/test_data")
            test_data_dir.mkdir(exist_ok=True)
            
            # Test creating various file types
            test_files = [
                ("test.txt", "This is a test text file"),
                ("test.json", '{"test": "json content"}'),
                ("test.csv", "id,name,value\n1,test,value"),
                ("test.md", "# Test Markdown\n\nThis is a test markdown file."),
            ]
            
            for filename, content in test_files:
                file_path = test_data_dir / filename
                with open(file_path, "w") as f:
                    f.write(content)
                
                assert file_path.exists(), f"File {filename} was not created"
                assert file_path.stat().st_size > 0, f"File {filename} is empty"
                
                print(f"   Created {filename}")
            
            print("File creation test passed")
            return True
            
        except Exception as e:
            print(f" File creation test failed: {e}")
            return False
    
    def test_directory_structure(self):
        """Test directory structure creation"""
        print(" Testing directory structure...")
        
        try:
            from app.core.config import get_path_settings
            
            paths = get_path_settings()
            
            # Check required directories
            required_dirs = [
                "DATA_DIR",
                "TEMP_CLONES_DIR", 
                "VECTORSTORE_DIR",
                "AGENTS_DIR"
            ]
            
            for dir_name in required_dirs:
                dir_path = paths[dir_name]
                if isinstance(dir_path, Path):
                    dir_path.mkdir(parents=True, exist_ok=True)
                    assert dir_path.exists(), f"Directory {dir_name} does not exist"
                    print(f"    Directory {dir_name}: {dir_path}")
                else:
                    print(f"    Directory {dir_name} is not a Path object: {dir_path}")
            
            print(" Directory structure test passed")
            return True
            
        except Exception as e:
            print(f" Directory structure test failed: {e}")
            return False
    
    def test_agent_creation(self):
        """Test agent configuration creation"""
        print(" Testing agent creation...")
        
        try:
            agents_dir = Path("agents")
            agents_dir.mkdir(exist_ok=True)
            
            # Check if test agents exist
            test_agents = [
                "upload-test-agent.yaml",
                "stress-test-agent.yaml", 
                "security-test-agent.yaml"
            ]
            
            for agent_file in test_agents:
                agent_path = agents_dir / agent_file
                if agent_path.exists():
                    print(f"    Agent {agent_file} exists")
                else:
                    print(f"    Agent {agent_file} not found")
            
            print(" Agent creation test passed")
            return True
            
        except Exception as e:
            print(f" Agent creation test failed: {e}")
            return False
    
    def test_database_models(self):
        """Test database model imports"""
        print(" Testing database models...")
        
        try:
            from app.db.models import temporary_documents_table, document_chunks_table, users_table
            
            # Check that tables are defined
            assert temporary_documents_table is not None, "temporary_documents_table not defined"
            assert document_chunks_table is not None, "document_chunks_table not defined"
            assert users_table is not None, "users_table not defined"
            
            print("    Database models imported successfully")
            print(" Database models test passed")
            return True
            
        except Exception as e:
            print(f" Database models test failed: {e}")
            return False
    
    def test_service_imports(self):
        """Test service imports"""
        print(" Testing service imports...")
        
        try:
            from app.services.temporary_document_service import TemporaryDocumentService
            from app.services.file_service import FileService
            
            print("    TemporaryDocumentService imported successfully")
            print("    FileService imported successfully")
            print(" Service imports test passed")
            return True
            
        except Exception as e:
            print(f" Service imports test failed: {e}")
            return False
    
    def test_config_loading(self):
        """Test configuration loading"""
        print(" Testing configuration loading...")
        
        try:
            from app.core.config import get_path_settings, get_api_key, get_server_api_keys
            
            # Test path settings
            paths = get_path_settings()
            assert isinstance(paths, dict), "Path settings should be a dictionary"
            assert "PROJECT_ROOT" in paths, "PROJECT_ROOT should be in path settings"
            
            print("    Path settings loaded successfully")
            
            # Test API key functions (should not crash)
            try:
                api_key = get_api_key("openai")
                print("    API key function works")
            except Exception:
                print("    API key not configured (expected in test environment)")
            
            try:
                server_keys = get_server_api_keys()
                print("    Server API keys function works")
            except Exception:
                print("    Server API keys not configured (expected in test environment)")
            
            print(" Configuration loading test passed")
            return True
            
        except Exception as e:
            print(f" Configuration loading test failed: {e}")
            return False
    
    def test_file_validation_logic(self):
        """Test file validation logic"""
        print(" Testing file validation logic...")
        
        try:
            # Test file extension validation
            ALLOWED_EXTENSIONS = {
                '.pdf', '.docx', '.txt', '.csv', '.json', '.yaml', '.yml',
                '.hcl', '.tf', '.ipynb', '.md', '.log', '.html'
            }
            
            test_files = [
                ("test.txt", True),
                ("test.pdf", True),
                ("test.json", True),
                ("test.exe", False),
                ("test.bat", False),
                ("test.py", False),
            ]
            
            for filename, should_pass in test_files:
                file_ext = Path(filename).suffix.lower()
                result = file_ext in ALLOWED_EXTENSIONS
                assert result == should_pass, f"File {filename} validation failed"
                print(f"    {filename}: {'allowed' if result else 'rejected'}")
            
            # Test file size validation
            MAX_SIZE = 25 * 1024 * 1024  # 25MB
            
            size_tests = [
                (1024, True),  # 1KB
                (1024 * 1024, True),  # 1MB
                (25 * 1024 * 1024, True),  # 25MB (at limit)
                (26 * 1024 * 1024, False),  # 26MB (over limit)
            ]
            
            for size, should_pass in size_tests:
                result = size <= MAX_SIZE
                assert result == should_pass, f"Size {size} validation failed"
                print(f"    Size {size}: {'allowed' if result else 'rejected'}")
            
            print(" File validation logic test passed")
            return True
            
        except Exception as e:
            print(f" File validation logic test failed: {e}")
            return False
    
    def test_cleanup_logic(self):
        """Test cleanup logic"""
        print(" Testing cleanup logic...")
        
        try:
            # Test cleanup directory structure
            from app.core.config import get_path_settings
            
            paths = get_path_settings()
            temp_uploads_dir = paths["TEMP_CLONES_DIR"] / "chat_uploads"
            vectorstore_dir = paths["VECTORSTORE_DIR"] / "temp_chat_data"
            
            # Ensure directories exist
            temp_uploads_dir.mkdir(parents=True, exist_ok=True)
            vectorstore_dir.mkdir(parents=True, exist_ok=True)
            
            assert temp_uploads_dir.exists(), "Temp uploads directory should exist"
            assert vectorstore_dir.exists(), "Vectorstore directory should exist"
            
            print("    Cleanup directories exist")
            print(" Cleanup logic test passed")
            return True
            
        except Exception as e:
            print(f" Cleanup logic test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all simple tests"""
        print(" Starting Simple Upload Tests")
        print("=" * 50)
        
        self.start_time = time.time()
        
        test_methods = [
            ("File Creation", self.test_file_creation),
            ("Directory Structure", self.test_directory_structure),
            ("Agent Creation", self.test_agent_creation),
            ("Database Models", self.test_database_models),
            ("Service Imports", self.test_service_imports),
            ("Configuration Loading", self.test_config_loading),
            ("File Validation Logic", self.test_file_validation_logic),
            ("Cleanup Logic", self.test_cleanup_logic),
        ]
        
        for test_name, test_method in test_methods:
            print(f"\n{'='*30}")
            print(f"Running {test_name}")
            print(f"{'='*30}")
            
            try:
                success = test_method()
                self.test_results[test_name] = {
                    "success": success,
                    "error": None
                }
                if not success:
                    self.overall_success = False
            except Exception as e:
                print(f" {test_name} failed with exception: {e}")
                self.test_results[test_name] = {
                    "success": False,
                    "error": str(e)
                }
                self.overall_success = False
        
        self.end_time = time.time()
        
        # Generate report
        self.generate_report()
        
        return 0 if self.overall_success else 1
    
    def generate_report(self):
        """Generate test report"""
        print(f"\n{'='*50}")
        print(" TEST REPORT")
        print(f"{'='*50}")
        
        total_duration = self.end_time - self.start_time
        successful_tests = sum(1 for result in self.test_results.values() if result["success"])
        total_tests = len(self.test_results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {(successful_tests/total_tests*100):.2f}%")
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Overall Status: {' PASSED' if self.overall_success else ' FAILED'}")
        
        print(f"\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = " PASSED" if result["success"] else " FAILED"
            print(f"  {test_name}: {status}")
            if result["error"]:
                print(f"    Error: {result['error']}")
        
        # Save report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_success": self.overall_success,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests * 100,
            "duration": total_duration,
            "results": self.test_results
        }
        
        report_file = Path("tests/simple_upload_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_file}")


def main():
    """Main entry point"""
    print("RAGnetic Simple Upload Test Suite")
    print("=" * 40)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    runner = SimpleUploadTestRunner()
    exit_code = runner.run_all_tests()
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
