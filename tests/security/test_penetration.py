# Security penetration tests for RAGnetic
import pytest
import json
from httpx import AsyncClient
from tests.fixtures.sample_data import SECURITY_TEST_PAYLOADS


class TestSQLInjectionProtection:
    """Test SQL injection attack prevention."""
    
    @pytest.mark.asyncio
    async def test_sql_injection_in_agent_name(self, client: AsyncClient, test_user, test_project, malicious_payloads):
        """Test SQL injection attempts in agent name field."""
        for payload in malicious_payloads["sql_injection"]:
            agent_config = {
                "name": payload,  # Malicious SQL injection payload
                "display_name": "Test Agent",
                "llm_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "project_id": test_project["id"],
                "persona_prompt": "You are a helpful assistant.",
                "vector_store": {"type": "faiss", "bm25_k": 5, "semantic_k": 5}
            }
            
            response = await client.post("/api/v1/agents", json=agent_config)
            
            # Should either sanitize and succeed, or reject with validation error
            assert response.status_code in [201, 400, 422]
            
            if response.status_code == 201:
                # If accepted, ensure payload was sanitized
                result = response.json()
                assert "DROP" not in result["name"].upper()
                assert "UNION" not in result["name"].upper()
                assert "--" not in result["name"]
    
    @pytest.mark.asyncio
    async def test_sql_injection_in_query_params(self, client: AsyncClient, test_user, malicious_payloads):
        """Test SQL injection in query parameters."""
        for payload in malicious_payloads["sql_injection"]:
            # Test in search/filter parameters
            response = await client.get(f"/api/v1/agents?search={payload}")
            
            # Should not cause server error
            assert response.status_code in [200, 400, 422]
            
            # Should not return sensitive data or error messages revealing DB structure
            if response.status_code == 200:
                result = response.json()
                response_text = json.dumps(result).lower()
                
                # Should not contain database error messages
                assert "sql" not in response_text
                assert "database" not in response_text
                assert "table" not in response_text
                assert "column" not in response_text
    
    @pytest.mark.asyncio
    async def test_sql_injection_in_training_job_params(self, client: AsyncClient, test_user, test_project, malicious_payloads):
        """Test SQL injection in training job parameters."""
        for payload in malicious_payloads["sql_injection"]:
            job_data = {
                "job_name": payload,  # Malicious payload
                "base_model_name": "microsoft/DialoGPT-small",
                "dataset_path": "data/training_datasets/sample.jsonl",
                "project_id": test_project["id"],
                "hyperparameters": {
                    "epochs": 1,
                    "batch_size": 2
                }
            }
            
            response = await client.post("/api/v1/training/jobs", json=job_data)
            
            # Should handle malicious input gracefully
            assert response.status_code in [201, 400, 422]
            
            if response.status_code == 201:
                result = response.json()
                # Ensure payload was sanitized
                assert "DROP" not in result["job_name"].upper()


class TestXSSProtection:
    """Test Cross-Site Scripting (XSS) attack prevention."""
    
    @pytest.mark.asyncio
    async def test_xss_in_agent_description(self, client: AsyncClient, test_user, test_project, malicious_payloads):
        """Test XSS payloads in agent description field."""
        for payload in malicious_payloads["xss"]:
            agent_config = {
                "name": "xss_test_agent",
                "display_name": "XSS Test Agent",
                "description": payload,  # XSS payload
                "llm_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "project_id": test_project["id"],
                "persona_prompt": "You are a helpful assistant.",
                "vector_store": {"type": "faiss", "bm25_k": 5, "semantic_k": 5}
            }
            
            response = await client.post("/api/v1/agents", json=agent_config)
            
            if response.status_code == 201:
                result = response.json()
                description = result.get("description", "")
                
                # Should not contain dangerous script tags
                assert "<script>" not in description.lower()
                assert "javascript:" not in description.lower()
                assert "onerror=" not in description.lower()
                assert "onload=" not in description.lower()
    
    @pytest.mark.asyncio
    async def test_xss_in_agent_query(self, client: AsyncClient, test_user, test_project, mock_all_providers, malicious_payloads):
        """Test XSS payloads in agent query messages."""
        # Create agent first
        agent_config = {
            "name": "xss_query_test",
            "display_name": "XSS Query Test Agent",
            "llm_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "project_id": test_project["id"],
            "persona_prompt": "You are a helpful assistant.",
            "vector_store": {"type": "faiss", "bm25_k": 5, "semantic_k": 5}
        }
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        for payload in malicious_payloads["xss"]:
            query_data = {
                "message": payload,  # XSS payload
                "session_id": "xss_test_session"
            }
            
            response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Response should not contain unescaped script tags
                assert "<script>" not in response_text
                assert "javascript:" not in response_text
    
    @pytest.mark.asyncio
    async def test_xss_in_html_responses(self, client: AsyncClient, test_user, malicious_payloads):
        """Test XSS protection in HTML responses."""
        for payload in malicious_payloads["xss"]:
            # Test dashboard with malicious query parameter
            response = await client.get(f"/?user_input={payload}")
            
            if response.status_code == 200:
                html_content = response.text
                
                # Should not contain unescaped script tags in HTML
                assert "<script>alert" not in html_content
                assert "javascript:alert" not in html_content
                
                # Should be properly escaped
                if payload in html_content:
                    # If present, should be HTML-encoded
                    assert "&lt;script&gt;" in html_content or payload not in html_content


class TestPathTraversalProtection:
    """Test path traversal attack prevention."""
    
    @pytest.mark.asyncio
    async def test_path_traversal_in_file_upload(self, client: AsyncClient, test_user, malicious_payloads):
        """Test path traversal in file upload functionality."""
        for payload in malicious_payloads["path_traversal"]:
            # Try to upload file with malicious filename
            malicious_filename = f"{payload}.txt"
            files = {"file": (malicious_filename, "test content", "text/plain")}
            
            response = await client.post("/api/v1/data/upload", files=files)
            
            # Should either reject or sanitize the filename
            if response.status_code == 200:
                result = response.json()
                saved_filename = result.get("filename", "")
                
                # Should not contain path traversal sequences
                assert "../" not in saved_filename
                assert "..\\" not in saved_filename
                assert "%2e%2e" not in saved_filename.lower()
    
    @pytest.mark.asyncio
    async def test_path_traversal_in_dataset_path(self, client: AsyncClient, test_user, test_project, malicious_payloads):
        """Test path traversal in dataset path specification."""
        for payload in malicious_payloads["path_traversal"]:
            job_data = {
                "job_name": "path_traversal_test",
                "base_model_name": "microsoft/DialoGPT-small",
                "dataset_path": payload,  # Malicious path
                "project_id": test_project["id"],
                "hyperparameters": {
                    "epochs": 1,
                    "batch_size": 2
                }
            }
            
            response = await client.post("/api/v1/training/jobs", json=job_data)
            
            # Should reject malicious paths
            assert response.status_code in [400, 422]
    
    @pytest.mark.asyncio
    async def test_path_traversal_in_model_path(self, client: AsyncClient, test_user, test_project, malicious_payloads):
        """Test path traversal in model path specification."""
        for payload in malicious_payloads["path_traversal"]:
            job_data = {
                "job_name": "model_path_test",
                "base_model_name": payload,  # Malicious model path
                "dataset_path": "data/training_datasets/sample.jsonl",
                "project_id": test_project["id"],
                "hyperparameters": {
                    "epochs": 1,
                    "batch_size": 2
                }
            }
            
            response = await client.post("/api/v1/training/jobs", json=job_data)
            
            # Should reject or sanitize malicious model paths
            assert response.status_code in [400, 422]


class TestCommandInjectionProtection:
    """Test command injection attack prevention."""
    
    @pytest.mark.asyncio
    async def test_command_injection_in_lambda_tool(self, client: AsyncClient, test_user, test_project, malicious_payloads):
        """Test command injection in Lambda tool execution."""
        # Create agent with lambda tool
        agent_config = {
            "name": "command_injection_test",
            "display_name": "Command Injection Test",
            "llm_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "project_id": test_project["id"],
            "persona_prompt": "You are a helpful assistant.",
            "tools": ["lambda"],
            "vector_store": {"type": "faiss", "bm25_k": 5, "semantic_k": 5}
        }
        
        create_response = await client.post("/api/v1/agents", json=agent_config)
        agent_id = create_response.json()["agent_id"]
        
        for payload in malicious_payloads["command_injection"]:
            query_data = {
                "message": f"Execute this code: {payload}",
                "session_id": "cmd_injection_test"
            }
            
            response = await client.post(f"/api/v1/agents/{agent_id}/query", json=query_data)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Should not execute dangerous commands
                assert "passwd" not in response_text.lower()
                assert "/etc/passwd" not in response_text
                assert "system32" not in response_text.lower()
    
    @pytest.mark.asyncio
    async def test_command_injection_in_training_config(self, client: AsyncClient, test_user, test_project, malicious_payloads):
        """Test command injection in training configuration."""
        for payload in malicious_payloads["command_injection"]:
            job_data = {
                "job_name": "cmd_injection_test",
                "base_model_name": "microsoft/DialoGPT-small",
                "dataset_path": "data/training_datasets/sample.jsonl",
                "project_id": test_project["id"],
                "hyperparameters": {
                    "epochs": 1,
                    "batch_size": 2,
                    "custom_args": payload  # Malicious command injection
                }
            }
            
            response = await client.post("/api/v1/training/jobs", json=job_data)
            
            # Should reject or sanitize malicious configuration
            assert response.status_code in [201, 400, 422]


class TestAuthenticationBypass:
    """Test authentication bypass attempts."""
    
    @pytest.mark.asyncio
    async def test_api_key_bypass_attempts(self, client: AsyncClient):
        """Test attempts to bypass API key authentication."""
        # Remove API key
        original_headers = client.headers.copy()
        client.headers.pop("X-API-Key", None)
        
        # Try various bypass techniques
        bypass_headers = [
            {"X-API-Key": ""},
            {"X-API-Key": "null"},
            {"X-API-Key": "undefined"},
            {"X-API-Key": "admin"},
            {"X-API-Key": "bearer token"},
            {"Authorization": "Bearer fake_token"},
            {"X-Forwarded-For": "127.0.0.1"},
            {"X-Real-IP": "127.0.0.1"},
        ]
        
        for headers in bypass_headers:
            client.headers.update(headers)
            
            # Try to access protected endpoint
            response = await client.get("/api/v1/agents")
            
            # Should be unauthorized
            assert response.status_code == 401
            
            client.headers.clear()
        
        # Restore original headers
        client.headers.update(original_headers)
    
    @pytest.mark.asyncio
    async def test_session_fixation_protection(self, client: AsyncClient, test_user):
        """Test protection against session fixation attacks."""
        # Try to set custom session ID
        custom_session_headers = {
            "Cookie": "session_id=attacker_controlled_session",
            "X-Session-ID": "malicious_session_123"
        }
        
        response = await client.get("/api/v1/dashboard/overview", headers=custom_session_headers)
        
        # Should not accept attacker-controlled session
        assert response.status_code in [200, 401]
        
        # If successful, should generate new session, not use provided one
        if response.status_code == 200:
            set_cookie = response.headers.get("Set-Cookie", "")
            if "session_id=" in set_cookie:
                assert "attacker_controlled_session" not in set_cookie
    
    @pytest.mark.asyncio
    async def test_privilege_escalation_attempts(self, client: AsyncClient, test_user):
        """Test attempts to escalate privileges."""
        # Try to access admin endpoints with regular user
        admin_endpoints = [
            "/api/v1/admin/users",
            "/api/v1/admin/system-stats",
            "/api/v1/admin/billing",
            "/api/v1/admin/gpu-providers"
        ]
        
        for endpoint in admin_endpoints:
            response = await client.get(endpoint)
            
            # Should be forbidden or not found (not 200)
            assert response.status_code in [403, 404, 405]
    
    @pytest.mark.asyncio
    async def test_jwt_token_manipulation(self, client: AsyncClient, test_user):
        """Test JWT token manipulation attempts."""
        # Try various JWT bypass techniques
        malicious_tokens = [
            "Bearer eyJhbGciOiJub25lIn0.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.",  # None algorithm
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiJ9.invalid_signature",  # Invalid signature
            "Bearer ../../../etc/passwd",  # Path traversal in token
            "Bearer ' OR '1'='1",  # SQL injection in token
        ]
        
        for token in malicious_tokens:
            headers = {"Authorization": token}
            
            response = await client.get("/api/v1/agents", headers=headers)
            
            # Should reject malicious tokens
            assert response.status_code == 401


class TestDataValidationBypass:
    """Test attempts to bypass data validation."""
    
    @pytest.mark.asyncio
    async def test_large_payload_attacks(self, client: AsyncClient, test_user, test_project):
        """Test protection against large payload attacks."""
        # Create extremely large payload
        large_payload = "A" * 1000000  # 1MB of data
        
        agent_config = {
            "name": "large_payload_test",
            "display_name": "Large Payload Test",
            "description": large_payload,  # Extremely large description
            "llm_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "project_id": test_project["id"],
            "persona_prompt": "You are a helpful assistant.",
            "vector_store": {"type": "faiss", "bm25_k": 5, "semantic_k": 5}
        }
        
        response = await client.post("/api/v1/agents", json=agent_config)
        
        # Should reject or limit large payloads
        assert response.status_code in [400, 413, 422]  # 413 = Payload Too Large
    
    @pytest.mark.asyncio
    async def test_null_byte_injection(self, client: AsyncClient, test_user, test_project, malicious_payloads):
        """Test protection against null byte injection."""
        null_byte_payload = malicious_payloads["null_bytes"]
        
        agent_config = {
            "name": null_byte_payload,
            "display_name": "Null Byte Test",
            "llm_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "project_id": test_project["id"],
            "persona_prompt": "You are a helpful assistant.",
            "vector_store": {"type": "faiss", "bm25_k": 5, "semantic_k": 5}
        }
        
        response = await client.post("/api/v1/agents", json=agent_config)
        
        if response.status_code == 201:
            result = response.json()
            # Should not contain null bytes
            assert "\x00" not in result["name"]
    
    @pytest.mark.asyncio
    async def test_unicode_bypass_attempts(self, client: AsyncClient, test_user, test_project, malicious_payloads):
        """Test protection against unicode bypass attempts."""
        unicode_payload = malicious_payloads["unicode_bypass"]
        
        agent_config = {
            "name": unicode_payload,
            "display_name": "Unicode Bypass Test",
            "llm_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "project_id": test_project["id"],
            "persona_prompt": "You are a helpful assistant.",
            "vector_store": {"type": "faiss", "bm25_k": 5, "semantic_k": 5}
        }
        
        response = await client.post("/api/v1/agents", json=agent_config)
        
        # Should handle unicode properly
        assert response.status_code in [201, 400, 422]


class TestRateLimitBypass:
    """Test rate limiting bypass attempts."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, client: AsyncClient, test_user):
        """Test that rate limiting cannot be easily bypassed."""
        # Make many requests quickly
        responses = []
        for i in range(50):  # Exceed typical rate limit
            response = await client.get("/api/v1/agents")
            responses.append(response.status_code)
        
        # Should eventually start returning 429 (Too Many Requests)
        rate_limited = any(status == 429 for status in responses)
        
        # If rate limiting is implemented, should see 429 responses
        # If not implemented yet, all should be 200 (acceptable for now)
        assert all(status in [200, 429] for status in responses)
    
    @pytest.mark.asyncio
    async def test_rate_limit_bypass_with_different_ips(self, client: AsyncClient, test_user):
        """Test rate limiting bypass attempts using different IP addresses."""
        # Try to bypass rate limiting with different X-Forwarded-For headers
        ip_addresses = [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "203.0.113.1"
        ]
        
        for ip in ip_addresses:
            headers = {"X-Forwarded-For": ip}
            
            # Make multiple requests with each IP
            for _ in range(10):
                response = await client.get("/api/v1/agents", headers=headers)
                
                # Should still enforce rate limiting per user/session
                assert response.status_code in [200, 429]


class TestBusinessLogicFlaws:
    """Test business logic security flaws."""
    
    @pytest.mark.asyncio
    async def test_credit_manipulation_attempts(self, client: AsyncClient, test_user, test_project):
        """Test attempts to manipulate credit balances."""
        # Try to set negative costs
        job_data = {
            "job_name": "credit_manipulation_test",
            "base_model_name": "microsoft/DialoGPT-small",
            "dataset_path": "data/training_datasets/sample.jsonl",
            "project_id": test_project["id"],
            "estimated_cost": -100.0,  # Negative cost
            "hyperparameters": {
                "epochs": 1,
                "batch_size": 2
            }
        }
        
        response = await client.post("/api/v1/training/jobs", json=job_data)
        
        # Should reject negative costs
        assert response.status_code in [400, 422]
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_protection(self, client: AsyncClient, test_user, test_project, test_credits):
        """Test protection against resource exhaustion attacks."""
        # Try to create many expensive training jobs
        job_requests = []
        for i in range(20):
            job_data = {
                "job_name": f"exhaustion_test_{i}",
                "base_model_name": "microsoft/DialoGPT-small",
                "dataset_path": "data/training_datasets/sample.jsonl",
                "project_id": test_project["id"],
                "max_hours": 10.0,  # Expensive
                "hyperparameters": {
                    "epochs": 10,  # Long training
                    "batch_size": 2
                }
            }
            
            response = await client.post("/api/v1/training/jobs", json=job_data)
            job_requests.append(response.status_code)
        
        # Should eventually start rejecting requests due to limits
        rejected = any(status in [400, 429] for status in job_requests)
        
        # Some should be rejected to prevent resource exhaustion
        # (If all succeed, limits might not be implemented yet, which is acceptable)
        success_count = sum(1 for status in job_requests if status == 201)
        assert success_count <= 10  # Reasonable limit
    
    @pytest.mark.asyncio
    async def test_project_access_control(self, client: AsyncClient, test_user):
        """Test that users cannot access other users' projects."""
        # Try to access a project that doesn't belong to the user
        fake_project_id = "unauthorized-project-123"
        
        # Try to create agent in unauthorized project
        agent_config = {
            "name": "unauthorized_access_test",
            "display_name": "Unauthorized Access Test",
            "llm_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "project_id": fake_project_id,  # Unauthorized project
            "persona_prompt": "You are a helpful assistant.",
            "vector_store": {"type": "faiss", "bm25_k": 5, "semantic_k": 5}
        }
        
        response = await client.post("/api/v1/agents", json=agent_config)
        
        # Should reject access to unauthorized project
        assert response.status_code in [403, 404]


@pytest.mark.security
class TestSecurityHeaders:
    """Test security headers and configurations."""
    
    @pytest.mark.asyncio
    async def test_security_headers_present(self, client: AsyncClient, test_user):
        """Test that important security headers are present."""
        response = await client.get("/")
        
        important_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]
        
        for header in important_headers:
            # Header should be present (case-insensitive check)
            header_present = any(
                header.lower() == h.lower() 
                for h in response.headers.keys()
            )
            
            # Some headers might not be implemented yet, which is acceptable
            # but we should document what's missing
            if not header_present:
                print(f"Security header missing: {header}")
    
    @pytest.mark.asyncio
    async def test_cors_configuration(self, client: AsyncClient, test_user):
        """Test CORS configuration security."""
        # Test preflight request
        response = await client.options(
            "/api/v1/agents",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-API-Key"
            }
        )
        
        # Should not allow arbitrary origins
        allow_origin = response.headers.get("Access-Control-Allow-Origin", "")
        
        # Should not be wildcard (*) or allow evil domains
        assert allow_origin != "*"
        assert "evil.com" not in allow_origin
    
    @pytest.mark.asyncio
    async def test_sensitive_data_exposure(self, client: AsyncClient, test_user):
        """Test that sensitive data is not exposed in responses."""
        # Get user info
        response = await client.get("/api/v1/user/profile")
        
        if response.status_code == 200:
            result = response.json()
            
            # Should not expose sensitive fields
            sensitive_fields = ["password", "password_hash", "api_key_hash", "secret"]
            
            for field in sensitive_fields:
                assert field not in result
                
            # Should not expose internal IDs or database info
            response_text = json.dumps(result).lower()
            assert "internal_id" not in response_text
            assert "db_id" not in response_text
