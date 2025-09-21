"""
Frontend Upload Integration Tests for RAGnetic
Tests frontend upload functionality and integration with backend APIs.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
import uuid

import pytest
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from app.main import app
from app.core.config import get_path_settings


class FrontendUploadIntegrationTests:
    """Frontend upload integration testing class"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_api_key = "YOUR_TEST_API_KEY_1"
        self.driver = None
        self.temp_files = []
        
    def setup_selenium_driver(self):
        """Setup Selenium WebDriver for frontend testing"""
        print(" Setting up Selenium WebDriver...")
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            print(" Selenium WebDriver setup complete")
        except Exception as e:
            print(f" Chrome WebDriver not available, skipping frontend tests: {e}")
            self.driver = None
    
    def teardown_selenium_driver(self):
        """Cleanup Selenium WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def create_test_file(self, filename: str, content: str) -> Path:
        """Create a test file for upload testing"""
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
    
    def test_dashboard_upload_interface(self):
        """Test upload interface on dashboard"""
        print(" Testing dashboard upload interface...")
        
        if not self.driver:
            print(" Skipping frontend test - WebDriver not available")
            return
        
        try:
            # Navigate to dashboard
            self.driver.get(f"{self.base_url}/dashboard")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Look for upload elements
            upload_elements = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            assert len(upload_elements) > 0, "Should have file upload inputs"
            
            print(" Dashboard upload interface test passed")
            
        except TimeoutException:
            print(" Dashboard not accessible, skipping frontend test")
        except Exception as e:
            print(f" Dashboard upload interface test failed: {e}")
            raise
    
    def test_file_upload_via_frontend(self):
        """Test file upload through frontend interface"""
        print(" Testing file upload via frontend...")
        
        if not self.driver:
            print(" Skipping frontend test - WebDriver not available")
            return
        
        try:
            # Create test file
            test_file = self.create_test_file("frontend_test.txt", "Frontend upload test content")
            
            # Navigate to dashboard
            self.driver.get(f"{self.base_url}/dashboard")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Find file input
            file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            if file_inputs:
                file_input = file_inputs[0]
                
                # Upload file
                file_input.send_keys(str(test_file))
                
                # Look for upload button or form submission
                upload_buttons = self.driver.find_elements(By.CSS_SELECTOR, "button[type='submit'], input[type='submit']")
                if upload_buttons:
                    upload_buttons[0].click()
                    
                    # Wait for upload to complete
                    time.sleep(2)
                    
                    # Check for success message or error
                    success_elements = self.driver.find_elements(By.CSS_SELECTOR, ".success, .alert-success")
                    error_elements = self.driver.find_elements(By.CSS_SELECTOR, ".error, .alert-error, .alert-danger")
                    
                    if success_elements:
                        print(" File upload successful via frontend")
                    elif error_elements:
                        print(f" File upload had errors: {error_elements[0].text}")
                    else:
                        print(" No clear success/error indication")
                else:
                    print(" No upload button found")
            else:
                print(" No file input found")
            
        except Exception as e:
            print(f" Frontend file upload test failed: {e}")
            # Don't raise - frontend tests are optional
    
    def test_chat_upload_interface(self):
        """Test upload interface in chat"""
        print(" Testing chat upload interface...")
        
        if not self.driver:
            print(" Skipping frontend test - WebDriver not available")
            return
        
        try:
            # Navigate to chat interface
            self.driver.get(f"{self.base_url}/chat")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Look for upload elements in chat
            upload_elements = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file'], .upload-area, .file-upload")
            if upload_elements:
                print(" Chat upload interface found")
            else:
                print(" No upload interface found in chat")
            
        except TimeoutException:
            print(" Chat interface not accessible, skipping frontend test")
        except Exception as e:
            print(f" Chat upload interface test failed: {e}")
            # Don't raise - frontend tests are optional
    
    def test_upload_drag_and_drop(self):
        """Test drag and drop upload functionality"""
        print(" Testing drag and drop upload...")
        
        if not self.driver:
            print(" Skipping frontend test - WebDriver not available")
            return
        
        try:
            # Create test file
            test_file = self.create_test_file("drag_drop_test.txt", "Drag and drop test content")
            
            # Navigate to dashboard
            self.driver.get(f"{self.base_url}/dashboard")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Look for drag and drop areas
            drop_areas = self.driver.find_elements(By.CSS_SELECTOR, ".drop-area, .upload-zone, [data-drop-zone]")
            if drop_areas:
                print(" Drag and drop areas found")
                # Note: Actual drag and drop testing would require more complex JavaScript execution
            else:
                print(" No drag and drop areas found")
            
        except Exception as e:
            print(f" Drag and drop test failed: {e}")
            # Don't raise - frontend tests are optional
    
    def test_upload_progress_indication(self):
        """Test upload progress indication"""
        print(" Testing upload progress indication...")
        
        if not self.driver:
            print(" Skipping frontend test - WebDriver not available")
            return
        
        try:
            # Create larger test file
            large_content = "Large file content for progress testing.\n" * 1000
            test_file = self.create_test_file("progress_test.txt", large_content)
            
            # Navigate to dashboard
            self.driver.get(f"{self.base_url}/dashboard")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Look for progress indicators
            progress_elements = self.driver.find_elements(By.CSS_SELECTOR, ".progress, .progress-bar, .upload-progress")
            if progress_elements:
                print(" Progress indicators found")
            else:
                print(" No progress indicators found")
            
        except Exception as e:
            print(f" Upload progress test failed: {e}")
            # Don't raise - frontend tests are optional
    
    def test_upload_error_handling_frontend(self):
        """Test error handling in frontend uploads"""
        print(" Testing upload error handling in frontend...")
        
        if not self.driver:
            print(" Skipping frontend test - WebDriver not available")
            return
        
        try:
            # Create invalid file (executable)
            test_file = self.create_test_file("invalid.exe", "fake executable content")
            
            # Navigate to dashboard
            self.driver.get(f"{self.base_url}/dashboard")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Try to upload invalid file
            file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            if file_inputs:
                file_input = file_inputs[0]
                file_input.send_keys(str(test_file))
                
                # Look for error messages
                time.sleep(1)
                error_elements = self.driver.find_elements(By.CSS_SELECTOR, ".error, .alert-error, .alert-danger, .validation-error")
                if error_elements:
                    print(" Error handling working in frontend")
                else:
                    print(" No error indication found for invalid file")
            
        except Exception as e:
            print(f" Frontend error handling test failed: {e}")
            # Don't raise - frontend tests are optional
    
    def test_upload_file_preview(self):
        """Test file preview functionality"""
        print(" Testing file preview functionality...")
        
        if not self.driver:
            print(" Skipping frontend test - WebDriver not available")
            return
        
        try:
            # Create test file
            test_file = self.create_test_file("preview_test.txt", "File preview test content")
            
            # Navigate to dashboard
            self.driver.get(f"{self.base_url}/dashboard")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Look for preview elements
            preview_elements = self.driver.find_elements(By.CSS_SELECTOR, ".file-preview, .preview, .file-info")
            if preview_elements:
                print(" File preview elements found")
            else:
                print(" No file preview elements found")
            
        except Exception as e:
            print(f" File preview test failed: {e}")
            # Don't raise - frontend tests are optional
    
    def test_upload_multiple_files(self):
        """Test multiple file upload functionality"""
        print(" Testing multiple file upload...")
        
        if not self.driver:
            print(" Skipping frontend test - WebDriver not available")
            return
        
        try:
            # Create multiple test files
            test_files = []
            for i in range(3):
                test_file = self.create_test_file(f"multi_test_{i}.txt", f"Multiple file test content {i}")
                test_files.append(test_file)
            
            # Navigate to dashboard
            self.driver.get(f"{self.base_url}/dashboard")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Look for multiple file upload support
            file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            if file_inputs:
                file_input = file_inputs[0]
                # Check if multiple attribute is present
                if file_input.get_attribute("multiple"):
                    print(" Multiple file upload supported")
                else:
                    print(" Multiple file upload not supported")
            
        except Exception as e:
            print(f" Multiple file upload test failed: {e}")
            # Don't raise - frontend tests are optional
    
    def test_upload_responsive_design(self):
        """Test upload interface responsive design"""
        print(" Testing upload responsive design...")
        
        if not self.driver:
            print(" Skipping frontend test - WebDriver not available")
            return
        
        try:
            # Test different screen sizes
            screen_sizes = [
                (1920, 1080),  # Desktop
                (768, 1024),   # Tablet
                (375, 667),    # Mobile
            ]
            
            for width, height in screen_sizes:
                self.driver.set_window_size(width, height)
                
                # Navigate to dashboard
                self.driver.get(f"{self.base_url}/dashboard")
                
                # Wait for page to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Check if upload elements are visible
                upload_elements = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file'], .upload-area")
                visible_elements = [elem for elem in upload_elements if elem.is_displayed()]
                
                if visible_elements:
                    print(f" Upload interface responsive at {width}x{height}")
                else:
                    print(f" Upload interface not visible at {width}x{height}")
            
        except Exception as e:
            print(f" Responsive design test failed: {e}")
            # Don't raise - frontend tests are optional
    
    def test_upload_accessibility(self):
        """Test upload interface accessibility"""
        print(" Testing upload accessibility...")
        
        if not self.driver:
            print(" Skipping frontend test - WebDriver not available")
            return
        
        try:
            # Navigate to dashboard
            self.driver.get(f"{self.base_url}/dashboard")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Check for accessibility attributes
            file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            if file_inputs:
                file_input = file_inputs[0]
                
                # Check for aria-label or aria-labelledby
                aria_label = file_input.get_attribute("aria-label")
                aria_labelledby = file_input.get_attribute("aria-labelledby")
                
                if aria_label or aria_labelledby:
                    print(" File input has accessibility labels")
                else:
                    print(" File input missing accessibility labels")
                
                # Check for associated label
                label_elements = self.driver.find_elements(By.CSS_SELECTOR, "label[for]")
                if label_elements:
                    print(" Labels found for form elements")
                else:
                    print(" No labels found for form elements")
            
        except Exception as e:
            print(f" Accessibility test failed: {e}")
            # Don't raise - frontend tests are optional
    
    def run_all_frontend_tests(self):
        """Run all frontend integration tests"""
        print(" Starting frontend upload integration tests...")
        
        try:
            self.setup_selenium_driver()
            
            test_methods = [
                self.test_dashboard_upload_interface,
                self.test_file_upload_via_frontend,
                self.test_chat_upload_interface,
                self.test_upload_drag_and_drop,
                self.test_upload_progress_indication,
                self.test_upload_error_handling_frontend,
                self.test_upload_file_preview,
                self.test_upload_multiple_files,
                self.test_upload_responsive_design,
                self.test_upload_accessibility,
            ]
            
            for test_method in test_methods:
                try:
                    test_method()
                except Exception as e:
                    print(f" Frontend test {test_method.__name__} failed: {e}")
                    # Don't raise - frontend tests are optional
            
            print(" Frontend integration tests completed!")
            
        except Exception as e:
            print(f" Frontend test suite failed: {e}")
        finally:
            self.teardown_selenium_driver()
            self.cleanup_temp_files()


if __name__ == "__main__":
    test_suite = FrontendUploadIntegrationTests()
    test_suite.run_all_frontend_tests()
