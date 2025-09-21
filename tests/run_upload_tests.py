#!/usr/bin/env python3
"""
Master Test Runner for RAGnetic Upload Functionality
Runs all upload tests in sequence with comprehensive reporting.
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

from test_upload_comprehensive import UploadTestSuite
from test_upload_backend_api import BackendUploadAPITests
from test_upload_frontend_integration import FrontendUploadIntegrationTests
from test_upload_stress import UploadStressTests
from test_upload_cleanup_lifecycle import UploadCleanupLifecycleTests
from test_upload_validation_security import UploadValidationSecurityTests


class MasterUploadTestRunner:
    """Master test runner for all upload functionality tests"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.test_results = {}
        self.overall_success = True
        
    def setup_test_environment(self):
        """Setup test environment"""
        print("Setting up test environment...")
        
        # Ensure test data directory exists
        test_data_dir = Path("tests/test_data")
        test_data_dir.mkdir(exist_ok=True)
        
        # Ensure agents directory exists
        agents_dir = Path("agents")
        agents_dir.mkdir(exist_ok=True)
        
        print("Test environment setup complete")
    
    def run_test_suite(self, test_suite_class, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite"""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")
        
        suite_start_time = time.time()
        success = False
        error_message = None
        
        try:
            if suite_name == "Comprehensive Upload Tests":
                test_suite = test_suite_class()
                test_suite.run_all_tests()
            elif suite_name == "Backend API Tests":
                test_suite = test_suite_class()
                test_suite.run_all_backend_tests()
            elif suite_name == "Frontend Integration Tests":
                test_suite = test_suite_class()
                test_suite.run_all_frontend_tests()
            elif suite_name == "Stress Tests":
                test_suite = test_suite_class()
                test_suite.run_all_stress_tests()
            elif suite_name == "Cleanup & Lifecycle Tests":
                test_suite = test_suite_class()
                test_suite.run_all_cleanup_lifecycle_tests()
            elif suite_name == "Validation & Security Tests":
                test_suite = test_suite_class()
                test_suite.run_all_validation_security_tests()
            
            success = True
            print(f"{suite_name} completed successfully")
            
        except Exception as e:
            error_message = str(e)
            print(f"{suite_name} failed: {error_message}")
            self.overall_success = False
        
        suite_end_time = time.time()
        suite_duration = suite_end_time - suite_start_time
        
        return {
            "suite_name": suite_name,
            "success": success,
            "duration": suite_duration,
            "error_message": error_message,
            "start_time": suite_start_time,
            "end_time": suite_end_time
        }
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print(f"\n{'='*60}")
        print("GENERATING TEST REPORT")
        print(f"{'='*60}")
        
        total_duration = self.end_time - self.start_time
        successful_suites = sum(1 for result in self.test_results.values() if result["success"])
        total_suites = len(self.test_results)
        
        print(f" Overall Results:")
        print(f"   Total Test Suites: {total_suites}")
        print(f"   Successful: {successful_suites}")
        print(f"   Failed: {total_suites - successful_suites}")
        print(f"   Success Rate: {(successful_suites/total_suites*100):.2f}%")
        print(f"   Total Duration: {total_duration:.2f} seconds")
        print(f"   Overall Status: {'PASSED' if self.overall_success else 'FAILED'}")
        
        print(f"\n Detailed Results:")
        for suite_name, result in self.test_results.items():
            status = "PASSED" if result["success"] else "FAILED"
            duration = result["duration"]
            print(f"   {suite_name}: {status} ({duration:.2f}s)")
            if result["error_message"]:
                print(f"      Error: {result['error_message']}")
        
        # Generate JSON report
        report_data = {
            "test_run": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_duration": total_duration,
                "overall_success": self.overall_success,
                "successful_suites": successful_suites,
                "total_suites": total_suites,
                "success_rate": successful_suites / total_suites * 100
            },
            "test_suites": self.test_results
        }
        
        # Save report to file
        report_file = Path("tests/upload_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n Detailed report saved to: {report_file}")
        
        return report_data
    
    def run_all_tests(self):
        """Run all upload test suites"""
        print("Starting RAGnetic Upload Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        try:
            self.setup_test_environment()
            
            # Define test suites to run
            test_suites = [
                (UploadTestSuite, "Comprehensive Upload Tests"),
                (BackendUploadAPITests, "Backend API Tests"),
                (FrontendUploadIntegrationTests, "Frontend Integration Tests"),
                (UploadStressTests, "Stress Tests"),
                (UploadCleanupLifecycleTests, "Cleanup & Lifecycle Tests"),
                (UploadValidationSecurityTests, "Validation & Security Tests"),
            ]
            
            # Run each test suite
            for test_suite_class, suite_name in test_suites:
                result = self.run_test_suite(test_suite_class, suite_name)
                self.test_results[suite_name] = result
            
            self.end_time = time.time()
            
            # Generate report
            report = self.generate_test_report()
            
            # Final status
            if self.overall_success:
                print(f"\nALL TESTS PASSED! Upload functionality is working correctly.")
                return 0
            else:
                print(f"\nSOME TESTS FAILED! Please review the errors above.")
                return 1
                
        except Exception as e:
            print(f"\n CRITICAL ERROR: Test runner failed: {e}")
            return 2


def main():
    """Main entry point"""
    print("RAGnetic Upload Test Suite")
    print("=" * 40)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    runner = MasterUploadTestRunner()
    exit_code = runner.run_all_tests()
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exit code: {exit_code}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
