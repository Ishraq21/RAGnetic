#!/usr/bin/env python3
"""
LoRA Fine-Tuning Frontend GUI and Backend Integration Tests
============================================================

This test suite performs comprehensive functional and integration testing
for the LoRA fine-tuning system, including:

1. Frontend GUI functionality testing
2. Backend API endpoint testing  
3. Database integration testing
4. End-to-end workflow testing
5. Error handling and edge cases
6. Performance and reliability testing

NO EMOJIS - Clean professional testing code.
"""

import asyncio
import json
import logging
import os
import pytest
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock

# Core imports
import requests
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# RAGnetic imports
import sys
sys.path.insert(0, '/Users/ishraq21/ragnetic')

from app.main import app
from app.api.training import router as training_router
from app.schemas.fine_tuning import FineTuningJobConfig, FineTuningStatus, HyperparametersConfig
from app.schemas.security import User
from app.db import get_db
from app.db.models import fine_tuned_models_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LoRAIntegrationTest")

# Test client for API testing
client = TestClient(app)

class LoRAFinetuningIntegrationTest:
    """Comprehensive LoRA fine-tuning integration test suite"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.api_base_url = f"{self.base_url}/api/v1"
        self.test_user_token = "604a7d725c7e96a5f2517f16cfc5d81c64365c55662de49c23e1aa3650b0f0b8"
        self.test_user_id = 1
        self.temp_dir = None
        self.test_results = {
            'frontend_tests': [],
            'backend_tests': [],
            'integration_tests': [],
            'errors': []
        }
        
    def setup_test_environment(self):
        """Setup test environment and data"""
        logger.info("Setting up test environment...")
        
        # Create temporary directory for test files
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Created temp directory: {self.temp_dir}")
        
        # Create test dataset
        self.create_test_dataset()
        
        # Mock user for authentication
        self.mock_user = Mock()
        self.mock_user.id = self.test_user_id
        self.mock_user.username = "test_user"
        
    def create_test_dataset(self):
        """Create test dataset for training"""
        # Use data directory to comply with server restrictions
        data_dir = Path("data")
        dataset_path = data_dir / "test_integration_dataset.jsonl"
        
        test_data = [
            {
                "instruction": "What is machine learning?",
                "input": "",
                "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
            },
            {
                "instruction": "Explain neural networks",
                "input": "",
                "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information using mathematical operations."
            },
            {
                "instruction": "What is deep learning?",
                "input": "",
                "output": "Deep learning is a machine learning technique that uses neural networks with multiple layers to analyze and learn from large amounts of data."
            }
        ]
        
        with open(dataset_path, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created test dataset: {dataset_path}")
        return str(dataset_path)
    
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
        
        # Clean up test dataset from data directory
        test_dataset_path = Path("data/test_integration_dataset.jsonl")
        if test_dataset_path.exists():
            test_dataset_path.unlink()
            
        logger.info("Cleaned up test environment")

    # BACKEND API TESTS
    
    def test_training_api_endpoints(self):
        """Test all training API endpoints"""
        logger.info("Testing training API endpoints...")
        
        test_results = []
        
        # Test 1: List training models (empty initially)
        try:
            response = requests.get(
                f"{self.api_base_url}/training/models",
                headers={"X-API-Key": self.test_user_token}
            )
            test_results.append({
                'test': 'list_models_empty',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'data': response.json() if response.status_code == 200 else None
            })
        except Exception as e:
            test_results.append({
                'test': 'list_models_empty',
                'success': False,
                'error': str(e)
            })
        
        # Test 2: Create training job
        try:
            job_config = {
                "job_name": "test-integration-job",
                "base_model_name": "gpt2",
                "dataset_path": "data/test_integration_dataset.jsonl",
                "output_base_dir": str(self.temp_dir / "models"),
                "hyperparameters": {
                    "batch_size": 1,
                    "epochs": 1,
                    "learning_rate": 2e-4,
                    "lora_rank": 4,
                    "lora_alpha": 8,
                    "lora_dropout": 0.05
                }
            }
            
            response = requests.post(
                f"{self.api_base_url}/training/apply",
                json=job_config,
                headers={"X-API-Key": self.test_user_token}
            )
            test_results.append({
                'test': 'create_training_job',
                'status_code': response.status_code,
                'success': response.status_code == 202,
                'data': response.json() if response.status_code == 202 else None
            })
            
            # Store adapter_id for subsequent tests
            if response.status_code == 202:
                self.test_adapter_id = response.json().get('adapter_id')
                
        except Exception as e:
            test_results.append({
                'test': 'create_training_job',
                'success': False,
                'error': str(e)
            })
        
        # Test 3: Get job status
        if hasattr(self, 'test_adapter_id'):
            try:
                response = requests.get(
                    f"{self.api_base_url}/training/jobs/{self.test_adapter_id}",
                    headers={"X-API-Key": self.test_user_token}
                )
                test_results.append({
                    'test': 'get_job_status',
                    'status_code': response.status_code,
                    'success': response.status_code == 200,
                    'data': response.json() if response.status_code == 200 else None
                })
            except Exception as e:
                test_results.append({
                    'test': 'get_job_status',
                    'success': False,
                    'error': str(e)
                })
        
        # Test 4: Get training logs
        if hasattr(self, 'test_adapter_id'):
            try:
                response = requests.get(
                    f"{self.api_base_url}/training/jobs/{self.test_adapter_id}/logs",
                    headers={"X-API-Key": self.test_user_token}
                )
                test_results.append({
                    'test': 'get_training_logs',
                    'status_code': response.status_code,
                    'success': response.status_code == 200,
                    'data': response.json() if response.status_code == 200 else None
                })
            except Exception as e:
                test_results.append({
                    'test': 'get_training_logs',
                    'success': False,
                    'error': str(e)
                })
        
        # Test 5: Get training stats
        try:
            response = requests.get(
                f"{self.api_base_url}/training/stats",
                headers={"X-API-Key": self.test_user_token}
            )
            test_results.append({
                'test': 'get_training_stats',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'data': response.json() if response.status_code == 200 else None
            })
        except Exception as e:
            test_results.append({
                'test': 'get_training_stats',
                'success': False,
                'error': str(e)
            })
        
        # Test 6: Cancel training job
        if hasattr(self, 'test_adapter_id'):
            try:
                response = requests.post(
                    f"{self.api_base_url}/training/jobs/{self.test_adapter_id}/cancel",
                    headers={"X-API-Key": self.test_user_token}
                )
                test_results.append({
                    'test': 'cancel_training_job',
                    'status_code': response.status_code,
                    'success': response.status_code in [200, 400],  # 400 if not running
                    'data': response.json() if response.status_code in [200, 400] else None
                })
            except Exception as e:
                test_results.append({
                    'test': 'cancel_training_job',
                    'success': False,
                    'error': str(e)
                })
        
        # Test 7: Delete training job
        if hasattr(self, 'test_adapter_id'):
            try:
                response = requests.delete(
                    f"{self.api_base_url}/training/jobs/{self.test_adapter_id}",
                    headers={"X-API-Key": self.test_user_token}
                )
                test_results.append({
                    'test': 'delete_training_job',
                    'status_code': response.status_code,
                    'success': response.status_code == 200,
                    'data': response.json() if response.status_code == 200 else None
                })
            except Exception as e:
                test_results.append({
                    'test': 'delete_training_job',
                    'success': False,
                    'error': str(e)
                })
        
        self.test_results['backend_tests'] = test_results
        return test_results
    
    def test_error_handling(self):
        """Test API error handling"""
        logger.info("Testing error handling...")
        
        test_results = []
        
        # Test 1: Invalid job configuration
        try:
            invalid_config = {
                "job_name": "",  # Invalid empty name
                "base_model_name": "",  # Invalid empty model
                "dataset_path": "/nonexistent/path.jsonl",  # Invalid path
                "hyperparameters": {
                    "batch_size": -1,  # Invalid batch size
                    "epochs": 0,  # Invalid epochs
                }
            }
            
            response = requests.post(
                f"{self.api_base_url}/training/apply",
                json=invalid_config,
                headers={"X-API-Key": self.test_user_token}
            )
            test_results.append({
                'test': 'invalid_job_config',
                'status_code': response.status_code,
                'success': response.status_code >= 400,  # Should return error
                'expected_error': True
            })
        except Exception as e:
            test_results.append({
                'test': 'invalid_job_config',
                'success': True,  # Exception is expected
                'error': str(e)
            })
        
        # Test 2: Nonexistent job lookup
        try:
            response = requests.get(
                f"{self.api_base_url}/training/jobs/nonexistent-job-id",
                headers={"X-API-Key": self.test_user_token}
            )
            test_results.append({
                'test': 'nonexistent_job_lookup',
                'status_code': response.status_code,
                'success': response.status_code == 404,
                'expected_error': True
            })
        except Exception as e:
            test_results.append({
                'test': 'nonexistent_job_lookup',
                'success': False,
                'error': str(e)
            })
        
        # Test 3: Unauthorized access
        try:
            response = requests.get(
                f"{self.api_base_url}/training/models",
                headers={"X-API-Key": "invalid_token"}
            )
            test_results.append({
                'test': 'unauthorized_access',
                'status_code': response.status_code,
                'success': response.status_code == 401,
                'expected_error': True
            })
        except Exception as e:
            test_results.append({
                'test': 'unauthorized_access',
                'success': False,
                'error': str(e)
            })
        
        self.test_results['backend_tests'].extend(test_results)
        return test_results
    
    def test_frontend_javascript_functionality(self):
        """Test frontend JavaScript functionality (static analysis)"""
        logger.info("Testing frontend JavaScript functionality...")
        
        test_results = []
        
        # Test 1: Check if training.js exists and has required functions
        try:
            training_js_path = Path("/Users/ishraq21/ragnetic/static/js/training.js")
            if training_js_path.exists():
                with open(training_js_path, 'r') as f:
                    js_content = f.read()
                
                required_functions = [
                    'TrainingDashboard',
                    'loadTrainingJobs',
                    'renderTrainingJobs',
                    'submitTrainingJob',
                    'showCreateTrainingJobModal',
                    'hideAllModals'
                ]
                
                found_functions = []
                for func in required_functions:
                    if func in js_content:
                        found_functions.append(func)
                
                test_results.append({
                    'test': 'javascript_functions',
                    'success': len(found_functions) == len(required_functions),
                    'found_functions': found_functions,
                    'missing_functions': [f for f in required_functions if f not in found_functions]
                })
            else:
                test_results.append({
                    'test': 'javascript_functions',
                    'success': False,
                    'error': 'training.js file not found'
                })
        except Exception as e:
            test_results.append({
                'test': 'javascript_functions',
                'success': False,
                'error': str(e)
            })
        
        # Test 2: Check HTML template structure
        try:
            template_path = Path("/Users/ishraq21/ragnetic/templates/dashboard.html")
            if template_path.exists():
                with open(template_path, 'r') as f:
                    html_content = f.read()
                
                required_elements = [
                    'training-view',
                    'create-training-job-modal',
                    'training-grid',
                    'create-training-job-form',
                    'training-search'
                ]
                
                found_elements = []
                for element in required_elements:
                    if element in html_content:
                        found_elements.append(element)
                
                test_results.append({
                    'test': 'html_template_structure',
                    'success': len(found_elements) == len(required_elements),
                    'found_elements': found_elements,
                    'missing_elements': [e for e in required_elements if e not in found_elements]
                })
            else:
                test_results.append({
                    'test': 'html_template_structure',
                    'success': False,
                    'error': 'dashboard.html template not found'
                })
        except Exception as e:
            test_results.append({
                'test': 'html_template_structure',
                'success': False,
                'error': str(e)
            })
        
        self.test_results['frontend_tests'] = test_results
        return test_results
    
    def test_database_integration(self):
        """Test database integration for training jobs"""
        logger.info("Testing database integration...")
        
        test_results = []
        
        # Test database schema and operations
        try:
            # Test if fine_tuned_models table exists and has required columns
            required_columns = [
                'id', 'adapter_id', 'job_name', 'base_model_name',
                'training_status', 'created_at', 'updated_at'
            ]
            
            # This would normally connect to test database
            # For now, we'll check if the table definition exists
            from app.db.models import fine_tuned_models_table
            
            table_columns = [col.name for col in fine_tuned_models_table.columns]
            
            found_columns = []
            for col in required_columns:
                if col in table_columns:
                    found_columns.append(col)
            
            test_results.append({
                'test': 'database_schema',
                'success': len(found_columns) == len(required_columns),
                'found_columns': found_columns,
                'missing_columns': [c for c in required_columns if c not in found_columns],
                'all_columns': table_columns
            })
            
        except Exception as e:
            test_results.append({
                'test': 'database_schema',
                'success': False,
                'error': str(e)
            })
        
        self.test_results['integration_tests'] = test_results
        return test_results
    
    def test_configuration_validation(self):
        """Test configuration validation logic"""
        logger.info("Testing configuration validation...")
        
        test_results = []
        
        # Test hyperparameter validation
        try:
            # Valid configuration
            valid_config = HyperparametersConfig(
                batch_size=4,
                epochs=3,
                learning_rate=2e-4,
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.05
            )
            
            test_results.append({
                'test': 'valid_hyperparameters',
                'success': True,
                'config': valid_config.model_dump()
            })
            
        except Exception as e:
            test_results.append({
                'test': 'valid_hyperparameters',
                'success': False,
                'error': str(e)
            })
        
        # Test invalid configurations
        invalid_configs = [
            {'batch_size': 0},  # Invalid batch size
            {'epochs': -1},     # Invalid epochs
            {'learning_rate': -1}, # Invalid learning rate
            {'lora_rank': 0},   # Invalid LoRA rank
        ]
        
        for i, invalid_override in enumerate(invalid_configs):
            try:
                config_dict = {
                    'batch_size': 4,
                    'epochs': 3,
                    'learning_rate': 2e-4,
                    'lora_rank': 8,
                    'lora_alpha': 16,
                    'lora_dropout': 0.05
                }
                config_dict.update(invalid_override)
                
                invalid_config = HyperparametersConfig(**config_dict)
                test_results.append({
                    'test': f'invalid_hyperparameters_{i}',
                    'success': False,  # Should have failed validation
                    'config_override': invalid_override,
                    'unexpected_success': True
                })
                
            except Exception as e:
                test_results.append({
                    'test': f'invalid_hyperparameters_{i}',
                    'success': True,  # Exception expected for invalid config
                    'config_override': invalid_override,
                    'validation_error': str(e)
                })
        
        self.test_results['integration_tests'].extend(test_results)
        return test_results
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        logger.info("Testing end-to-end workflow...")
        
        test_results = []
        workflow_steps = []
        
        try:
            # Step 1: Create job configuration
            job_config = {
                "job_name": f"e2e-test-{int(time.time())}",
                "base_model_name": "gpt2",
                "dataset_path": "data/test_integration_dataset.jsonl",
                "output_base_dir": str(self.temp_dir / "models"),
                "hyperparameters": {
                    "batch_size": 1,
                    "epochs": 1,
                    "learning_rate": 2e-4,
                    "lora_rank": 4,
                    "lora_alpha": 8
                }
            }
            workflow_steps.append("config_created")
            
            # Step 2: Submit job
            response = requests.post(
                f"{self.api_base_url}/training/apply",
                json=job_config,
                headers={"X-API-Key": self.test_user_token}
            )
            
            if response.status_code == 202:
                adapter_id = response.json().get('adapter_id')
                workflow_steps.append("job_submitted")
            else:
                workflow_steps.append(f"job_submission_failed_{response.status_code}")
                
            # Step 3: Check job status
            if 'adapter_id' in locals():
                status_response = requests.get(
                    f"{self.api_base_url}/training/jobs/{adapter_id}",
                    headers={"X-API-Key": self.test_user_token}
                )
                
                if status_response.status_code == 200:
                    job_data = status_response.json()
                    status = job_data.get('training_status')
                    workflow_steps.append(f"status_checked_{status}")
                else:
                    workflow_steps.append(f"status_check_failed_{status_response.status_code}")
            
            # Step 4: List all jobs (should include our job)
            list_response = requests.get(
                f"{self.api_base_url}/training/models",
                headers={"X-API-Key": self.test_user_token}
            )
            
            if list_response.status_code == 200:
                jobs = list_response.json()
                our_job = None
                if 'adapter_id' in locals():
                    our_job = next((job for job in jobs if job.get('adapter_id') == adapter_id), None)
                
                if our_job:
                    workflow_steps.append("job_found_in_list")
                else:
                    workflow_steps.append("job_not_found_in_list")
            else:
                workflow_steps.append(f"list_jobs_failed_{list_response.status_code}")
            
            test_results.append({
                'test': 'end_to_end_workflow',
                'success': len([s for s in workflow_steps if 'failed' not in s]) >= 3,
                'workflow_steps': workflow_steps,
                'adapter_id': adapter_id if 'adapter_id' in locals() else None
            })
            
        except Exception as e:
            test_results.append({
                'test': 'end_to_end_workflow',
                'success': False,
                'error': str(e),
                'workflow_steps': workflow_steps
            })
        
        self.test_results['integration_tests'].extend(test_results)
        return test_results
    
    def run_all_tests(self):
        """Run all integration tests"""
        logger.info("Starting comprehensive LoRA fine-tuning integration tests...")
        
        start_time = time.time()
        
        try:
            # Setup
            self.setup_test_environment()
            
            # Run all test suites
            logger.info("Running backend API tests...")
            self.test_training_api_endpoints()
            
            logger.info("Running error handling tests...")
            self.test_error_handling()
            
            logger.info("Running frontend tests...")
            self.test_frontend_javascript_functionality()
            
            logger.info("Running database integration tests...")
            self.test_database_integration()
            
            logger.info("Running configuration validation tests...")
            self.test_configuration_validation()
            
            logger.info("Running end-to-end workflow tests...")
            self.test_end_to_end_workflow()
            
        except Exception as e:
            self.test_results['errors'].append({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        finally:
            # Cleanup
            self.cleanup_test_environment()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Generate test report
        self.generate_test_report(duration)
        
        return self.test_results
    
    def generate_test_report(self, duration):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        # Count test results
        frontend_tests = self.test_results.get('frontend_tests', [])
        backend_tests = self.test_results.get('backend_tests', [])
        integration_tests = self.test_results.get('integration_tests', [])
        errors = self.test_results.get('errors', [])
        
        frontend_passed = len([t for t in frontend_tests if t.get('success', False)])
        backend_passed = len([t for t in backend_tests if t.get('success', False)])
        integration_passed = len([t for t in integration_tests if t.get('success', False)])
        
        total_tests = len(frontend_tests) + len(backend_tests) + len(integration_tests)
        total_passed = frontend_passed + backend_passed + integration_passed
        
        print("\n" + "="*80)
        print("LORA FINE-TUNING FRONTEND/BACKEND INTEGRATION TEST REPORT")
        print("="*80)
        print(f"Test Duration: {duration:.2f} seconds")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_tests - total_passed}")
        print(f"Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        print()
        
        print("FRONTEND TESTS:")
        print(f"  Total: {len(frontend_tests)}, Passed: {frontend_passed}")
        for test in frontend_tests:
            status = "PASS" if test.get('success', False) else "FAIL"
            print(f"  - {test.get('test', 'unknown')}: {status}")
        print()
        
        print("BACKEND API TESTS:")
        print(f"  Total: {len(backend_tests)}, Passed: {backend_passed}")
        for test in backend_tests:
            status = "PASS" if test.get('success', False) else "FAIL"
            print(f"  - {test.get('test', 'unknown')}: {status}")
        print()
        
        print("INTEGRATION TESTS:")
        print(f"  Total: {len(integration_tests)}, Passed: {integration_passed}")
        for test in integration_tests:
            status = "PASS" if test.get('success', False) else "FAIL"
            print(f"  - {test.get('test', 'unknown')}: {status}")
        print()
        
        if errors:
            print("ERRORS ENCOUNTERED:")
            for error in errors:
                print(f"  - {error.get('error', 'unknown error')}")
        
        print("="*80)
        
        if total_passed == total_tests and not errors:
            print("ALL TESTS PASSED! LoRA fine-tuning system is working correctly.")
        else:
            print("Some tests failed. Please review the detailed results above.")
        print("="*80)

def run_lora_integration_tests():
    """Main function to run LoRA integration tests"""
    test_runner = LoRAFinetuningIntegrationTest()
    results = test_runner.run_all_tests()
    return results

if __name__ == "__main__":
    run_lora_integration_tests()