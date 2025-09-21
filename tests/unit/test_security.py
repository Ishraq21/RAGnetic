# Unit tests for security, authentication, and authorization
import pytest
from unittest.mock import Mock, patch, MagicMock
import hashlib
import secrets
from datetime import datetime, timedelta
from app.core.security import (
    verify_api_key,
    hash_password,
    verify_password,
    create_api_key,
    PermissionChecker,
    get_current_user_from_api_key
)
from app.core.rate_limit import allow, _peer_ip
from app.schemas.security import User


class TestPasswordSecurity:
    """Test password hashing and verification."""
    
    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password_123"
        hashed = hash_password(password)
        
        assert hashed != password  # Should be different from plaintext
        assert len(hashed) > 50  # Reasonable hash length
        assert "$" in hashed  # Should contain bcrypt markers
    
    def test_hash_password_different_salts(self):
        """Test that same password produces different hashes."""
        password = "same_password"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        assert hash1 != hash2  # Different salts should produce different hashes
    
    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "correct_password"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) == True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        correct_password = "correct_password"
        wrong_password = "wrong_password"
        hashed = hash_password(correct_password)
        
        assert verify_password(wrong_password, hashed) == False
    
    def test_verify_password_empty(self):
        """Test password verification with empty password."""
        password = "test_password"
        hashed = hash_password(password)
        
        assert verify_password("", hashed) == False
        assert verify_password(None, hashed) == False
    
    def test_verify_password_malformed_hash(self):
        """Test password verification with malformed hash."""
        password = "test_password"
        malformed_hash = "not_a_valid_hash"
        
        assert verify_password(password, malformed_hash) == False
    
    def test_password_unicode_support(self):
        """Test password hashing with unicode characters."""
        unicode_password = "pässwörd_123_[SECURITY]"
        hashed = hash_password(unicode_password)
        
        assert verify_password(unicode_password, hashed) == True
        assert verify_password("password_123", hashed) == False
    
    def test_long_password(self):
        """Test hashing of very long passwords."""
        long_password = "a" * 1000  # 1000 character password
        hashed = hash_password(long_password)
        
        assert verify_password(long_password, hashed) == True
        assert verify_password("a" * 999, hashed) == False


class TestAPIKeyManagement:
    """Test API key creation and verification."""
    
    def test_create_api_key(self):
        """Test API key creation."""
        user_id = 1
        scope = "admin"
        
        api_key_data = create_api_key(user_id, "test_key", scope)
        
        assert "key" in api_key_data
        assert "key_hash" in api_key_data
        assert api_key_data["key"].startswith("sk_")  # Standard prefix
        assert len(api_key_data["key"]) >= 32  # Reasonable length
        assert api_key_data["key"] != api_key_data["key_hash"]  # Hash is different
    
    @pytest.mark.asyncio
    async def test_verify_api_key_valid(self, db_session):
        """Test API key verification with valid key."""
        user_id = 1
        scope = "editor"
        
        # Create key
        api_key_data = create_api_key(user_id, "test_key", scope)
        raw_key = api_key_data["key"]
        key_hash = api_key_data["key_hash"]
        
        # Mock database lookup
        with patch('app.core.security.get_api_key_from_hash') as mock_get:
            mock_get.return_value = {
                "user_id": user_id,
                "scope": scope,
                "is_active": True,
                "id": "key_123"
            }
            
            result = await verify_api_key(raw_key, db_session)
            assert result[1] == user_id  # user_id is second element in tuple
            assert result[2] == scope    # scope is third element in tuple
    
    @pytest.mark.asyncio
    async def test_verify_api_key_invalid(self, db_session):
        """Test API key verification with invalid key."""
        invalid_key = "sk_invalid_key_12345"
        
        with patch('app.core.security.get_api_key_from_hash') as mock_get:
            mock_get.return_value = None  # Key not found
            
            result = await verify_api_key(invalid_key, db_session)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_verify_api_key_inactive(self, db_session):
        """Test API key verification with inactive key."""
        user_id = 1
        api_key_data = create_api_key(user_id, "test_key", "viewer")
        raw_key = api_key_data["key"]
        
        with patch('app.core.security.get_api_key_from_hash') as mock_get:
            mock_get.return_value = {
                "user_id": user_id,
                "scope": "viewer",
                "is_active": False,  # Inactive key
                "id": "key_123"
            }
            
            result = await verify_api_key(raw_key, db_session)
            assert result is None  # Should reject inactive keys
    
    @pytest.mark.asyncio
    async def test_api_key_format_validation(self, db_session):
        """Test API key format validation."""
        # Valid formats
        valid_keys = [
            "sk_1234567890abcdef",
            "sk_test_1234567890abcdef1234567890abcdef",
            "sk_live_abcdef1234567890"
        ]
        
        for key in valid_keys:
            # Should not raise exception
            try:
                await verify_api_key(key, db_session)
            except ValueError:
                pytest.fail(f"Valid key {key} was rejected")
        
        # Invalid formats
        invalid_keys = [
            "invalid_key",  # Wrong prefix
            "sk_",  # Too short
            "sk_" + "a" * 1000,  # Too long
            "",  # Empty
            None  # None
        ]
        
        for key in invalid_keys:
            with pytest.raises((ValueError, TypeError)):
                await verify_api_key(key, db_session)


class TestPermissionChecker:
    """Test permission checking and authorization."""
    
    def test_permission_checker_valid_permission(self):
        """Test permission checker with valid permissions."""
        from datetime import datetime
        from app.schemas.security import Role
        
        required_perms = ["read:agents", "write:agents"]
        checker = PermissionChecker(required_perms)
        
        # Mock user with required permissions
        user = User(
            id=1,
            username="test_user",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            is_active=True,
            is_superuser=False,
            hashed_password="hashed_password",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            roles=[Role(id=1, name="developer", description="Developer role", permissions=["read:agents", "write:agents"])]
        )
        
        with patch('app.core.security.get_user_permissions') as mock_perms:
            mock_perms.return_value = ["read:agents", "write:agents", "read:models"]
            
            # Should not raise exception
            result = checker(user)
            assert result == user
    
    def test_permission_checker_missing_permission(self):
        """Test permission checker with missing permissions."""
        from datetime import datetime
        from app.schemas.security import Role
        
        required_perms = ["admin:users"]
        checker = PermissionChecker(required_perms)
        
        user = User(
            id=1,
            username="test_user",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            is_active=True,
            is_superuser=False,
            hashed_password="hashed_password",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            roles=[Role(id=1, name="viewer", description="Viewer role", permissions=["read:agents"])]
        )
        
        with patch('app.core.security.get_user_permissions') as mock_perms:
            mock_perms.return_value = ["read:agents"]  # Missing admin:users
            
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                checker(user)
            
            assert exc_info.value.status_code == 403
    
    def test_permission_checker_superuser_bypass(self):
        """Test that superusers bypass permission checks."""
        from datetime import datetime
        from app.schemas.security import Role
        
        required_perms = ["admin:everything"]
        checker = PermissionChecker(required_perms)
        
        superuser = User(
            id=1,
            username="admin",
            email="admin@example.com",
            first_name="Admin",
            last_name="User",
            is_active=True,
            is_superuser=True,  # Superuser
            hashed_password="hashed_password",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            roles=[Role(id=1, name="admin", description="Admin role", permissions=["admin:everything"])]
        )
        
        # Should pass without checking specific permissions
        result = checker(superuser)
        assert result == superuser
    
    def test_permission_checker_inactive_user(self):
        """Test permission checker with inactive user."""
        from datetime import datetime
        from app.schemas.security import Role
        
        required_perms = ["read:agents"]
        checker = PermissionChecker(required_perms)
        
        inactive_user = User(
            id=1,
            username="inactive",
            email="inactive@example.com",
            first_name="Inactive",
            last_name="User",
            is_active=False,  # Inactive
            is_superuser=False,
            hashed_password="hashed_password",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            roles=[Role(id=1, name="viewer", description="Viewer role", permissions=["read:agents"])]
        )
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            checker(inactive_user)
        
        assert exc_info.value.status_code == 403
    
    def test_permission_checker_multiple_permissions(self):
        """Test permission checker with multiple required permissions."""
        from datetime import datetime
        from app.schemas.security import Role
        
        required_perms = ["read:agents", "write:agents", "delete:agents"]
        checker = PermissionChecker(required_perms)
        
        user = User(
            id=1,
            username="developer",
            email="dev@example.com",
            first_name="Dev",
            last_name="User",
            is_active=True,
            is_superuser=False,
            hashed_password="hashed_password",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            roles=[Role(id=1, name="developer", description="Developer role", permissions=["read:agents", "write:agents", "delete:agents"])]
        )
        
        with patch('app.core.security.get_user_permissions') as mock_perms:
            # User has all required permissions
            mock_perms.return_value = ["read:agents", "write:agents", "delete:agents", "read:models"]
            
            result = checker(user)
            assert result == user
            
            # User missing one permission
            mock_perms.return_value = ["read:agents", "write:agents"]  # Missing delete:agents
            
            from fastapi import HTTPException
            with pytest.raises(HTTPException):
                checker(user)


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_allow_within_limit(self):
        """Test rate limiting allows requests within limit."""
        bucket = "test_bucket"
        key = "user_123"
        limit = 10
        window = 60  # 60 seconds
        
        # First request should be allowed
        assert allow(bucket, key, limit, window) == True
        
        # Multiple requests within limit should be allowed
        for i in range(9):  # 9 more requests (10 total)
            assert allow(bucket, key, limit, window) == True
    
    def test_rate_limit_deny_over_limit(self):
        """Test rate limiting denies requests over limit."""
        bucket = "test_bucket_2"
        key = "user_456"
        limit = 5
        window = 60
        
        # Use up the limit
        for i in range(5):
            assert allow(bucket, key, limit, window) == True
        
        # Next request should be denied
        assert allow(bucket, key, limit, window) == False
    
    def test_rate_limit_different_users(self):
        """Test rate limiting is per-user."""
        bucket = "test_bucket_3"
        limit = 3
        window = 60
        
        # User 1 uses up their limit
        for i in range(3):
            assert allow(bucket, "user1", limit, window) == True
        assert allow(bucket, "user1", limit, window) == False
        
        # User 2 should still have their full limit
        for i in range(3):
            assert allow(bucket, "user2", limit, window) == True
        assert allow(bucket, "user2", limit, window) == False
    
    def test_rate_limit_window_reset(self):
        """Test rate limiting window reset."""
        import time
        
        bucket = "test_bucket_4"
        key = "user_789"
        limit = 2
        window = 1  # 1 second window
        
        # Use up limit
        assert allow(bucket, key, limit, window) == True
        assert allow(bucket, key, limit, window) == True
        assert allow(bucket, key, limit, window) == False
        
        # Wait for window to reset
        time.sleep(1.1)
        
        # Should be allowed again
        assert allow(bucket, key, limit, window) == True
    
    def test_peer_ip_extraction(self):
        """Test peer IP extraction from request."""
        from fastapi import Request
        
        # Mock request with X-Forwarded-For
        request = Mock(spec=Request)
        request.headers = {"X-Forwarded-For": "192.168.1.1, 10.0.0.1"}
        request.client.host = "127.0.0.1"
        
        ip = _peer_ip(request)
        assert ip == "192.168.1.1"  # First IP in X-Forwarded-For
        
        # Mock request without X-Forwarded-For
        request.headers = {}
        ip = _peer_ip(request)
        assert ip == "127.0.0.1"  # Falls back to client.host
    
    def test_rate_limit_edge_cases(self):
        """Test rate limiting edge cases."""
        bucket = "edge_cases"
        
        # Zero limit should always deny
        assert allow(bucket, "user1", 0, 60) == False
        
        # Very large limit should allow
        assert allow(bucket, "user2", 1000000, 60) == True
        
        # Very short window
        assert allow(bucket, "user3", 1, 1) == True
        assert allow(bucket, "user3", 1, 1) == False


class TestSecurityHeaders:
    """Test security header handling."""
    
    def test_security_headers_present(self):
        """Test that security headers are properly set."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        response = client.get("/")
        
        # Check for important security headers
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security"
        ]
        
        for header in expected_headers:
            assert header in response.headers or header.lower() in response.headers
    
    def test_cors_headers(self):
        """Test CORS headers configuration."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Preflight request
        response = client.options(
            "/api/v1/agents",
            headers={
                "Origin": "https://app.ragnetic.ai",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-API-Key"
            }
        )
        
        # Should allow the origin (check for either exact match or wildcard)
        assert ("Access-Control-Allow-Origin" in response.headers or 
                "access-control-allow-origin" in response.headers or
                response.headers.get("Access-Control-Allow-Origin") == "https://app.ragnetic.ai" or
                response.headers.get("access-control-allow-origin") == "https://app.ragnetic.ai" or
                response.headers.get("Access-Control-Allow-Origin") == "*" or
                response.headers.get("access-control-allow-origin") == "*")


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--"
        ]
        
        from app.core.validation import sanitize_sql_input
        
        for malicious_input in malicious_inputs:
            sanitized = sanitize_sql_input(malicious_input)
            # Should escape or remove dangerous characters
            assert "DROP" not in sanitized.upper()
            assert "UNION" not in sanitized.upper()
            assert "--" not in sanitized
    
    def test_xss_prevention(self):
        """Test XSS prevention."""
        xss_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>"
        ]
        
        from app.core.validation import sanitize_html_input
        
        for xss_input in xss_inputs:
            sanitized = sanitize_html_input(xss_input)
            # Should remove or escape dangerous tags
            assert "<script>" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "onerror" not in sanitized.lower()
            assert "onload" not in sanitized.lower()
    
    def test_path_traversal_prevention(self):
        """Test path traversal prevention."""
        traversal_inputs = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        from app.core.validation import sanitize_path_input
        
        for traversal_input in traversal_inputs:
            sanitized = sanitize_path_input(traversal_input)
            # Should remove or escape path traversal sequences
            assert "../" not in sanitized
            assert "..\\" not in sanitized
            assert "%2e%2e" not in sanitized.lower()
    
    def test_command_injection_prevention(self):
        """Test command injection prevention."""
        injection_inputs = [
            "; ls -la",
            "| whoami",
            "&& cat /etc/passwd",
            "`whoami`",
            "$(whoami)"
        ]
        
        from app.core.validation import sanitize_command_input
        
        for injection_input in injection_inputs:
            sanitized = sanitize_command_input(injection_input)
            # Should remove or escape dangerous characters
            assert ";" not in sanitized
            assert "|" not in sanitized
            assert "&&" not in sanitized
            assert "`" not in sanitized
            assert "$(" not in sanitized


class TestSessionSecurity:
    """Test session security features."""
    
    def test_session_timeout(self):
        """Test session timeout functionality."""
        from app.core.security import is_session_valid
        from datetime import datetime, timedelta
        
        # Recent session should be valid
        recent_time = datetime.utcnow() - timedelta(minutes=5)
        assert is_session_valid(recent_time, timeout_minutes=30) == True
        
        # Old session should be invalid
        old_time = datetime.utcnow() - timedelta(hours=2)
        assert is_session_valid(old_time, timeout_minutes=30) == False
    
    def test_session_regeneration(self):
        """Test session ID regeneration."""
        from app.core.security import generate_session_id
        
        # Generate multiple session IDs
        session_ids = [generate_session_id() for _ in range(10)]
        
        # All should be unique
        assert len(set(session_ids)) == 10
        
        # All should be reasonable length
        for session_id in session_ids:
            assert len(session_id) >= 32
            assert session_id.isalnum() or "-" in session_id


class TestCryptographicSecurity:
    """Test cryptographic functions."""
    
    def test_secure_random_generation(self):
        """Test secure random number generation."""
        from app.core.security import generate_secure_token
        
        # Generate multiple tokens
        tokens = [generate_secure_token() for _ in range(100)]
        
        # All should be unique
        assert len(set(tokens)) == 100
        
        # All should be reasonable length
        for token in tokens:
            assert len(token) >= 32
    
    def test_constant_time_comparison(self):
        """Test constant-time string comparison."""
        from app.core.security import constant_time_compare
        
        # Same strings should match
        assert constant_time_compare("secret123", "secret123") == True
        
        # Different strings should not match
        assert constant_time_compare("secret123", "secret124") == False
        
        # Different lengths should not match
        assert constant_time_compare("short", "longer_string") == False
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        import time
        from app.core.security import constant_time_compare
        
        # Measure time for correct comparison
        start = time.perf_counter()
        for _ in range(1000):
            constant_time_compare("correct_password", "correct_password")
        correct_time = time.perf_counter() - start
        
        # Measure time for incorrect comparison
        start = time.perf_counter()
        for _ in range(1000):
            constant_time_compare("correct_password", "wrong_password")
        incorrect_time = time.perf_counter() - start
        
        # Times should be similar (within 50% difference)
        time_diff = abs(correct_time - incorrect_time)
        avg_time = (correct_time + incorrect_time) / 2
        assert time_diff / avg_time < 0.5


@pytest.mark.performance
class TestSecurityPerformance:
    """Performance tests for security functions."""
    
    def test_password_hashing_performance(self, benchmark):
        """Benchmark password hashing performance."""
        password = "test_password_123"
        
        def hash_password_bench():
            return hash_password(password)
        
        result = benchmark(hash_password_bench)
        assert len(result) > 50  # Reasonable hash length
    
    def test_password_verification_performance(self, benchmark):
        """Benchmark password verification performance."""
        password = "test_password_123"
        hashed = hash_password(password)
        
        def verify_password_bench():
            return verify_password(password, hashed)
        
        result = benchmark(verify_password_bench)
        assert result == True
    
    @pytest.mark.asyncio
    async def test_api_key_verification_performance(self, benchmark, db_session):
        """Benchmark API key verification performance."""
        user_id = 1
        api_key_data = create_api_key(user_id, "test_key", "admin")
        raw_key = api_key_data["key"]
        
        with patch('app.core.security.get_api_key_from_hash') as mock_get:
            mock_get.return_value = {
                "user_id": user_id,
                "scope": "admin",
                "is_active": True,
                "id": "key_123"
            }
            
            async def verify_key_bench():
                return await verify_api_key(raw_key, db_session)
            
            result = await benchmark(verify_key_bench)
            assert result[1] == user_id  # user_id is second element in tuple
