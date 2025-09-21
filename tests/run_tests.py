#!/usr/bin/env python3
"""
RAGnetic Test Runner
Comprehensive test execution script with different test suites and reporting.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any


class RAGneticTestRunner:
    """Test runner for RAGnetic with multiple test suites."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = Path(__file__).parent
        
    def run_command(self, cmd: List[str], description: str) -> Dict[str, Any]:
        """Run a command and return results."""
        print(f"\n {description}")
        print(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f" {description} - PASSED ({duration:.1f}s)")
                return {
                    "success": True,
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                print(f" {description} - FAILED ({duration:.1f}s)")
                print(f"Exit code: {result.returncode}")
                if result.stdout:
                    print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
                if result.stderr:
                    print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
                
                return {
                    "success": False,
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} - TIMEOUT (30 minutes)")
            return {
                "success": False,
                "duration": 1800,
                "error": "Timeout"
            }
        except Exception as e:
            print(f" {description} - ERROR: {e}")
            return {
                "success": False,
                "duration": 0,
                "error": str(e)
            }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/unit/",
            "-m", "unit",
            "--cov=app",
            "--cov-report=html:htmlcov/unit",
            "--cov-report=term-missing",
            "--junit-xml=test-reports/unit-results.xml",
            "-v"
        ]
        
        return self.run_command(cmd, "Unit Tests")
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/integration/",
            "-m", "integration",
            "--junit-xml=test-reports/integration-results.xml",
            "-v"
        ]
        
        return self.run_command(cmd, "Integration Tests")
    
    def run_e2e_tests(self) -> Dict[str, Any]:
        """Run end-to-end tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/e2e/",
            "-m", "e2e",
            "--junit-xml=test-reports/e2e-results.xml",
            "-v"
        ]
        
        return self.run_command(cmd, "End-to-End Tests")
    
    def run_regression_tests(self) -> Dict[str, Any]:
        """Run regression tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/regression/",
            "-m", "regression",
            "--junit-xml=test-reports/regression-results.xml",
            "-v"
        ]
        
        return self.run_command(cmd, "Regression Tests")
    
    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/security/",
            "-m", "security",
            "--junit-xml=test-reports/security-results.xml",
            "-v"
        ]
        
        return self.run_command(cmd, "Security Tests")
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/performance/",
            "-m", "performance",
            "--junit-xml=test-reports/performance-results.xml",
            "--benchmark-json=test-reports/benchmark-results.json",
            "-v"
        ]
        
        return self.run_command(cmd, "Performance Tests")
    
    def run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/performance/",
            "-m", "stress",
            "--junit-xml=test-reports/stress-results.xml",
            "-v"
        ]
        
        return self.run_command(cmd, "Stress Tests")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "--cov=app",
            "--cov-report=html:htmlcov/all",
            "--cov-report=term-missing",
            "--junit-xml=test-reports/all-results.xml",
            "--benchmark-json=test-reports/all-benchmark-results.json",
            "-v"
        ]
        
        return self.run_command(cmd, "All Tests")
    
    def run_quick_tests(self) -> Dict[str, Any]:
        """Run quick tests (unit + fast integration)."""
        cmd = [
            "python", "-m", "pytest",
            "tests/unit/",
            "tests/integration/",
            "-m", "not slow and not stress",
            "--cov=app",
            "--cov-report=term-missing",
            "--junit-xml=test-reports/quick-results.xml",
            "-x",  # Stop on first failure
            "-v"
        ]
        
        return self.run_command(cmd, "Quick Tests")
    
    def run_code_quality_checks(self) -> List[Dict[str, Any]]:
        """Run code quality checks."""
        results = []
        
        # Ruff linting
        cmd = ["ruff", "check", "app/", "tests/", "--output-format=json"]
        results.append(self.run_command(cmd, "Ruff Linting"))
        
        # Type checking with mypy
        cmd = ["mypy", "app/", "--ignore-missing-imports"]
        results.append(self.run_command(cmd, "MyPy Type Checking"))
        
        # Security scanning with bandit
        cmd = ["bandit", "-r", "app/", "-f", "json", "-o", "test-reports/bandit-results.json"]
        results.append(self.run_command(cmd, "Bandit Security Scan"))
        
        # Safety check for vulnerabilities
        cmd = ["safety", "check", "--json", "--output", "test-reports/safety-results.json"]
        results.append(self.run_command(cmd, "Safety Vulnerability Check"))
        
        return results
    
    def setup_test_environment(self) -> bool:
        """Set up test environment."""
        print(" Setting up test environment...")
        
        # Create test reports directory
        reports_dir = self.project_root / "test-reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Create htmlcov directory
        htmlcov_dir = self.project_root / "htmlcov"
        htmlcov_dir.mkdir(exist_ok=True)
        
        # Install test dependencies
        cmd = ["pip", "install", "-r", "requirements-dev.txt"]
        result = self.run_command(cmd, "Installing test dependencies")
        
        if not result["success"]:
            print(" Failed to install test dependencies")
            return False
        
        # Set environment variables
        os.environ["RAGNETIC_ENV"] = "test"
        os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"
        os.environ["DISABLE_EXTERNAL_APIS"] = "true"
        os.environ["MOCK_PROVIDERS"] = "true"
        
        print(" Test environment setup complete")
        return True
    
    def generate_test_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print(" TEST EXECUTION SUMMARY")
        print("="*80)
        
        total_duration = 0
        passed_suites = 0
        failed_suites = 0
        
        for suite_name, result in results.items():
            status = " PASSED" if result["success"] else " FAILED"
            duration = result.get("duration", 0)
            total_duration += duration
            
            if result["success"]:
                passed_suites += 1
            else:
                failed_suites += 1
            
            print(f"{suite_name:<25} {status:<10} ({duration:.1f}s)")
        
        print("-" * 80)
        print(f"Total Suites: {len(results)}")
        print(f"Passed: {passed_suites}")
        print(f"Failed: {failed_suites}")
        print(f"Success Rate: {passed_suites/len(results)*100:.1f}%")
        print(f"Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        
        # Save detailed report
        report_file = self.project_root / "test-reports" / "test-summary.txt"
        with open(report_file, "w") as f:
            f.write("RAGnetic Test Execution Summary\n")
            f.write("=" * 40 + "\n\n")
            
            for suite_name, result in results.items():
                f.write(f"{suite_name}:\n")
                f.write(f"  Status: {'PASSED' if result['success'] else 'FAILED'}\n")
                f.write(f"  Duration: {result.get('duration', 0):.1f}s\n")
                
                if not result["success"]:
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
                    if "stderr" in result:
                        f.write(f"  stderr: {result['stderr'][-500:]}\n")  # Last 500 chars
                
                f.write("\n")
        
        print(f"\n[RESPONSE] Detailed report saved to: {report_file}")
        
        # Print coverage info if available
        coverage_file = self.project_root / "htmlcov" / "index.html"
        if coverage_file.exists():
            print(f" Coverage report available at: {coverage_file}")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="RAGnetic Test Runner")
    parser.add_argument(
        "suite",
        nargs="?",
        choices=[
            "unit", "integration", "e2e", "regression", 
            "security", "performance", "stress", 
            "all", "quick", "quality"
        ],
        default="quick",
        help="Test suite to run (default: quick)"
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Skip test environment setup"
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running other test suites even if one fails"
    )
    
    args = parser.parse_args()
    
    runner = RAGneticTestRunner()
    
    # Setup test environment
    if not args.no_setup:
        if not runner.setup_test_environment():
            sys.exit(1)
    
    print(f"\n Starting RAGnetic test execution: {args.suite}")
    print(f"Project root: {runner.project_root}")
    
    results = {}
    
    try:
        if args.suite == "unit":
            results["Unit Tests"] = runner.run_unit_tests()
            
        elif args.suite == "integration":
            results["Integration Tests"] = runner.run_integration_tests()
            
        elif args.suite == "e2e":
            results["E2E Tests"] = runner.run_e2e_tests()
            
        elif args.suite == "regression":
            results["Regression Tests"] = runner.run_regression_tests()
            
        elif args.suite == "security":
            results["Security Tests"] = runner.run_security_tests()
            
        elif args.suite == "performance":
            results["Performance Tests"] = runner.run_performance_tests()
            
        elif args.suite == "stress":
            results["Stress Tests"] = runner.run_stress_tests()
            
        elif args.suite == "quality":
            quality_results = runner.run_code_quality_checks()
            for i, result in enumerate(quality_results):
                results[f"Quality Check {i+1}"] = result
                
        elif args.suite == "quick":
            results["Quick Tests"] = runner.run_quick_tests()
            
        elif args.suite == "all":
            # Run all test suites
            test_suites = [
                ("Unit Tests", runner.run_unit_tests),
                ("Integration Tests", runner.run_integration_tests),
                ("E2E Tests", runner.run_e2e_tests),
                ("Regression Tests", runner.run_regression_tests),
                ("Security Tests", runner.run_security_tests),
                ("Performance Tests", runner.run_performance_tests),
            ]
            
            for suite_name, suite_func in test_suites:
                result = suite_func()
                results[suite_name] = result
                
                # Stop on failure unless continue-on-failure is set
                if not result["success"] and not args.continue_on_failure:
                    print(f"\n Stopping due to failure in {suite_name}")
                    break
            
            # Run quality checks
            quality_results = runner.run_code_quality_checks()
            for i, result in enumerate(quality_results):
                results[f"Quality Check {i+1}"] = result
    
    except KeyboardInterrupt:
        print("\n  Test execution interrupted by user")
        sys.exit(1)
    
    # Generate report
    runner.generate_test_report(results)
    
    # Exit with appropriate code
    all_passed = all(result["success"] for result in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
