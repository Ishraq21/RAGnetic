# app/core/rate_limit.py
import time
from collections import defaultdict, deque
from typing import Callable, Optional

from fastapi import Request, Header, HTTPException, status

# bucket+key -> sliding window of timestamps
_hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

def allow(bucket: str, key: str, limit: int, window_sec: int) -> bool:
    """
    Simple in-memory sliding-window limiter.
    True if within limit, False if blocked.
    """
    now = time.time()
    cutoff = now - window_sec
    buf = _hits[(bucket, key)]

    # drop old timestamps
    while buf and buf[0] < cutoff:
        buf.popleft()

    if len(buf) >= limit:
        return False

    buf.append(now)
    return True

def _peer_ip(request: Request) -> str:
    # If behind a proxy/load balancer, prefer the first X-Forwarded-For hop
    xff = request.headers.get("x-forwarded-for")
    if xff:
        ip = xff.split(",")[0].strip()
        if ip:
            return ip
    return request.client.host if request.client else "unknown"

def _rate_key_from_request(request: Request, x_api_key: Optional[str]) -> str:
    # Prefer API key; otherwise client IP (proxy-aware). Prefix so key spaces don't collide.
    return f"key:{x_api_key[:12]}" if x_api_key else f"ip:{_peer_ip(request)}"

def rate_limiter(bucket: str, limit: int, window_sec: int) -> Callable:
    """
    FastAPI dependency factory:
    usage → dependencies=[Depends(rate_limiter("login", 5, 60))]
    """
    async def _dep(request: Request, x_api_key: Optional[str] = Header(None)):
        key = _rate_key_from_request(request, x_api_key)
        if not allow(bucket, key, limit, window_sec):
            buf = _hits[(bucket, key)]
            # earliest timestamp still in window -> compute Retry-After
            earliest = buf[0] if buf else time.time()
            retry_after = max(1, int(window_sec - (time.time() - earliest)))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
    return _dep
