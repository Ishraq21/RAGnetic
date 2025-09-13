#!/usr/bin/env python3
"""
Test Server Startup Script for RAGnetic
Starts the server with proper test configuration.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_test_environment():
    """Setup test environment variables"""
    print("Setting up test environment...")
    
    # Set environment variables for testing
    os.environ["RAGNETIC_API_KEYS"] = "YOUR_TEST_API_KEY_1,YOUR_TEST_API_KEY_2"
    os.environ["ALLOWED_HOSTS"] = "localhost,127.0.0.1,testserver"
    os.environ["RAGNETIC_PROJECT_ROOT"] = str(project_root)
    
    # Set debug mode for testing
    os.environ["RAGNETIC_DEBUG"] = "true"
    
    print("   Environment variables set")
    print(f"   Project root: {project_root}")
    print(f"   API keys configured")
    print(f"   Allowed hosts configured")


def start_server():
    """Start the RAGnetic server"""
    print("Starting RAGnetic server...")
    
    try:
        # Start server using uvicorn directly
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        
        # Start server in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root
        )
        
        print(f"   Server started with PID: {process.pid}")
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            import requests
            response = requests.get("http://127.0.0.1:8000/", timeout=5)
            print(f"   Server is responding: {response.status_code}")
            return process
        except Exception as e:
            print(f"   Server not responding yet: {e}")
            return process
            
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None


def stop_server(process):
    """Stop the server process"""
    if process:
        print("🛑 Stopping server...")
        process.terminate()
        process.wait()
        print("   Server stopped")


def main():
    """Main entry point"""
    print("RAGnetic Test Server Startup")
    print("=" * 40)
    
    setup_test_environment()
    server_process = start_server()
    
    if server_process:
        try:
            print("\nServer is running!")
            print("   URL: http://127.0.0.1:8000")
            print("   Press Ctrl+C to stop")
            
            # Keep running until interrupted
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            stop_server(server_process)
    else:
        print("Failed to start server")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
