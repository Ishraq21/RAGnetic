# app/core/rate_limit.py

import time
import logging
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Max requests in a short burst


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_time: int
    limit: int
    retry_after: Optional[int] = None


class RateLimiter:
    """In-memory rate limiter with sliding window."""
    
    def __init__(self):
        # Store request timestamps for each key
        self._requests: Dict[str, deque] = defaultdict(deque)
        # Store last cleanup time
        self._last_cleanup = time.time()
        # Cleanup interval (5 minutes)
        self._cleanup_interval = 300
    
    def _cleanup_old_entries(self):
        """Remove old entries to prevent memory leaks."""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        cutoff_time = current_time - 86400  # 24 hours ago
        
        keys_to_remove = []
        for key, timestamps in self._requests.items():
            # Remove timestamps older than 24 hours
            while timestamps and timestamps[0] < cutoff_time:
                timestamps.popleft()
            
            # Remove empty entries
            if not timestamps:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._requests[key]
        
        self._last_cleanup = current_time
        logger.debug(f"Rate limiter cleanup: removed {len(keys_to_remove)} empty entries")
    
    def check_rate_limit(
        self, 
        key: str, 
        config: RateLimitConfig,
        current_time: Optional[float] = None
    ) -> RateLimitResult:
        """
        Check if request is within rate limits.
        
        Args:
            key: Unique identifier for the rate limit (e.g., API key)
            config: Rate limit configuration
            current_time: Current timestamp (for testing)
            
        Returns:
            RateLimitResult with rate limit status
        """
        if current_time is None:
            current_time = time.time()
        
        # Cleanup old entries periodically
        self._cleanup_old_entries()
        
        # Get request history for this key
        timestamps = self._requests[key]
        
        # Remove timestamps older than 1 minute for minute-based checks
        minute_cutoff = current_time - 60
        while timestamps and timestamps[0] < minute_cutoff:
            timestamps.popleft()
        
        # Check minute limit
        if len(timestamps) >= config.requests_per_minute:
            # Calculate retry after time
            oldest_request = timestamps[0]
            retry_after = int(60 - (current_time - oldest_request))
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=int(current_time + retry_after),
                limit=config.requests_per_minute,
                retry_after=retry_after
            )
        
        # Check burst limit (last 10 seconds)
        burst_cutoff = current_time - 10
        recent_requests = sum(1 for ts in timestamps if ts > burst_cutoff)
        
        if recent_requests >= config.burst_limit:
            retry_after = int(10 - (current_time - burst_cutoff))
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=int(current_time + retry_after),
                limit=config.burst_limit,
                retry_after=retry_after
            )
        
        # Check hourly limit (approximate)
        hour_cutoff = current_time - 3600
        hourly_requests = sum(1 for ts in timestamps if ts > hour_cutoff)
        
        if hourly_requests >= config.requests_per_hour:
            # Find oldest request in the last hour
            hour_timestamps = [ts for ts in timestamps if ts > hour_cutoff]
            if hour_timestamps:
                oldest_hourly = min(hour_timestamps)
                retry_after = int(3600 - (current_time - oldest_hourly))
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=int(current_time + retry_after),
                    limit=config.requests_per_hour,
                    retry_after=retry_after
                )
        
        # Check daily limit (approximate)
        day_cutoff = current_time - 86400
        daily_requests = sum(1 for ts in timestamps if ts > day_cutoff)
        
        if daily_requests >= config.requests_per_day:
            # Find oldest request in the last day
            day_timestamps = [ts for ts in timestamps if ts > day_cutoff]
            if day_timestamps:
                oldest_daily = min(day_timestamps)
                retry_after = int(86400 - (current_time - oldest_daily))
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=int(current_time + retry_after),
                    limit=config.requests_per_day,
                    retry_after=retry_after
                )
        
        # Request is allowed
        timestamps.append(current_time)
        
        # Calculate remaining requests
        remaining = config.requests_per_minute - len(timestamps)
        
        return RateLimitResult(
            allowed=True,
            remaining=max(0, remaining),
            reset_time=int(current_time + 60),
            limit=config.requests_per_minute
        )
    
    def get_stats(self, key: str) -> Dict[str, int]:
        """Get rate limit statistics for a key."""
        timestamps = self._requests[key]
        current_time = time.time()
        
        # Count requests in different time windows
        minute_requests = sum(1 for ts in timestamps if ts > current_time - 60)
        hour_requests = sum(1 for ts in timestamps if ts > current_time - 3600)
        day_requests = sum(1 for ts in timestamps if ts > current_time - 86400)
        
        return {
            "requests_last_minute": minute_requests,
            "requests_last_hour": hour_requests,
            "requests_last_day": day_requests,
            "total_requests": len(timestamps)
        }


def allow(bucket: str, key: str, limit: int, window_seconds: int) -> bool:
    """Simple sliding window rate limit: allow if under limit within window.

    This helper is DB-agnostic and used by tests; it shares the same global limiter store
    but applies its own window logic for simplicity.
    """
    if limit <= 0:
        return False
    now = time.time()
    # Use global limiter's storage
    timestamps = _rate_limiter._requests[f"{bucket}:{key}"]
    cutoff = now - max(1, window_seconds)
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()
    if len(timestamps) >= limit:
        return False
    timestamps.append(now)
    return True


def _peer_ip(request: Request) -> str:
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Global rate limiter instance
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    return _rate_limiter


def rate_limiter(scope: str = "generic", requests: int = 5, window_seconds: int = 60):
    """
    FastAPI dependency factory that enforces a simple rate limit per scope.

    Usage:
        @router.post("/login", dependencies=[Depends(rate_limiter("login", 5, 60))])

    Args:
        scope: logical scope name (e.g., "login") used for the limiter key prefix
        requests: allowed requests within the window
        window_seconds: time window length in seconds
    """
    limiter = get_rate_limiter()

    def _dep(request: Request):
        # Derive key from scope + client IP to keep behavior deterministic for public endpoints
        client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown").split(",")[0].strip()
        key = f"dep_scope:{scope}:ip:{client_ip}"

        # Map the simple requests/window to our RateLimitConfig minute window by scaling
        # For simplicity, use minute bucket proportionally if window <= 60, otherwise cap to minute behavior
        cfg = RateLimitConfig(
            requests_per_minute=requests if window_seconds <= 60 else max(1, int(requests * 60 / window_seconds)),
            requests_per_hour=1000000,
            requests_per_day=10000000,
            burst_limit=max(1, min(requests, 10)),
        )

        result = limiter.check_rate_limit(key, cfg)
        if not result.allowed:
            raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")

    return _dep

def check_api_rate_limit(
    api_key_id: int,
    config: Optional[RateLimitConfig] = None
) -> RateLimitResult:
    """
    Check rate limit for an API key.
    
    Args:
        api_key_id: The API key ID
        config: Rate limit configuration (uses default if None)
        
    Returns:
        RateLimitResult with rate limit status
    """
    if config is None:
        config = RateLimitConfig()
    
    # Use API key ID as the rate limit key
    key = f"api_key_{api_key_id}"
    
    return _rate_limiter.check_rate_limit(key, config)


def check_user_rate_limit(
    user_id: int,
    config: Optional[RateLimitConfig] = None
) -> RateLimitResult:
    """
    Check rate limit for a user.
    
    Args:
        user_id: The user ID
        config: Rate limit configuration (uses default if None)
        
    Returns:
        RateLimitResult with rate limit status
    """
    if config is None:
        config = RateLimitConfig()
    
    # Use user ID as the rate limit key
    key = f"user_{user_id}"
    
    return _rate_limiter.check_rate_limit(key, config)


def get_rate_limit_stats(api_key_id: int) -> Dict[str, int]:
    """Get rate limit statistics for an API key."""
    key = f"api_key_{api_key_id}"
    return _rate_limiter.get_stats(key)


def clear_rate_limit(api_key_id: int):
    """Clear rate limit data for an API key (useful for testing)."""
    key = f"api_key_{api_key_id}"
    if key in _rate_limiter._requests:
        del _rate_limiter._requests[key]
        logger.info(f"Cleared rate limit data for API key {api_key_id}")


# Default rate limit configurations for different scopes
DEFAULT_RATE_LIMITS = {
    "viewer": RateLimitConfig(
        requests_per_minute=30,
        requests_per_hour=500,
        requests_per_day=5000,
        burst_limit=5
    ),
    "editor": RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        requests_per_day=10000,
        burst_limit=10
    ),
    "admin": RateLimitConfig(
        requests_per_minute=120,
        requests_per_hour=2000,
        requests_per_day=20000,
        burst_limit=20
    )
}


def get_rate_limit_config(scope: str) -> RateLimitConfig:
    """Get rate limit configuration for a scope."""
    return DEFAULT_RATE_LIMITS.get(scope, DEFAULT_RATE_LIMITS["viewer"])