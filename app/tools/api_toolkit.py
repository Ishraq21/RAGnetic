# app/tools/api_toolkit.py

import os
import logging
import asyncio
import random
import socket
import ipaddress
from typing import Any, Dict, Optional, Type, List
from urllib.parse import urlparse, urljoin

import httpx
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel

from app.schemas.api_toolkit import APIRequestToolInput, PaginationConfig

logger = logging.getLogger(__name__)


ALLOWED_SCHEMES = {"http", "https"}
BLOCK_PRIVATE_IPS = True
MAX_RESPONSE_BYTES = int(os.getenv("RAGNETIC_HTTP_MAX_BYTES", 5 * 1024 * 1024))  # 5 MB default
MAX_RETRIES = int(os.getenv("RAGNETIC_HTTP_MAX_RETRIES", 3))
TIMEOUTS = httpx.Timeout(
    connect=float(os.getenv("RAGNETIC_HTTP_CONNECT_TIMEOUT", "5")),
    read=float(os.getenv("RAGNETIC_HTTP_READ_TIMEOUT", "20")),
    write=float(os.getenv("RAGNETIC_HTTP_WRITE_TIMEOUT", "5")),
    pool=float(os.getenv("RAGNETIC_HTTP_POOL_TIMEOUT", "5")),
)

# Wildcard means allow everything (NOT recommended in prod).

# Example .env for Allowed Domains: RAGNETIC_HTTP_ALLOWED_DOMAINS=api.github.com,api.openai.com,jsonplaceholder.typicode.com

ALLOWED_DOMAINS = {
    d.strip().lower() for d in os.getenv("RAGNETIC_HTTP_ALLOWED_DOMAINS", "*").split(",")
} or {"*"}

RETRY_STATUS = {429, 500, 502, 503, 504}

SENSITIVE_H = {"authorization", "x-api-key", "api-key", "proxy-authorization"}
SENSITIVE_P = {"token", "apikey", "access_token", "signature"}

PRIVATE_NETS = [ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
    "127.0.0.0/8", "::1/128", "fc00::/7", "fe80::/10"
)]


def _is_private_ip(ip: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip)
        return any(ip_obj in net for net in PRIVATE_NETS)
    except ValueError:
        return True  # treat unknown as unsafe

def _domain_allowed(host: str) -> bool:
    if "*" in ALLOWED_DOMAINS:
        return True
    host = (host or "").lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS if d)

def _resolve_ips(host: str) -> List[str]:
    ips: List[str] = []
    for info in socket.getaddrinfo(host, None):
        try:
            ip = info[4][0]
            if ip:
                ips.append(ip)
        except Exception:
            continue
    return list(set(ips))

def _validate_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ToolException(f"Disallowed URL scheme: {parsed.scheme}")
    if not parsed.hostname:
        raise ToolException("URL missing hostname")
    if not _domain_allowed(parsed.hostname):
        raise ToolException(f"Domain not allowlisted: {parsed.hostname}")
    if BLOCK_PRIVATE_IPS:
        ips = _resolve_ips(parsed.hostname)
        if any(_is_private_ip(ip) for ip in ips):
            raise ToolException(f"Blocked private/loopback address: {parsed.hostname} -> {', '.join(ips)}")

def _redact(d: Optional[Dict[str, Any]], sens: set) -> Dict[str, Any]:
    d = d or {}
    return {k: ("***" if k.lower() in sens else v) for k, v in d.items()}

async def _read_capped(resp: httpx.Response, cap: int) -> bytes:
    # Early reject if Content-Length is huge
    cl = resp.headers.get("content-length")
    if cl and int(cl) > cap:
        raise ToolException(f"Response too large (> {cap} bytes)")
    total = 0
    parts: List[bytes] = []
    async for chunk in resp.aiter_bytes():
        total += len(chunk)
        if total > cap:
            raise ToolException(f"Response too large (> {cap} bytes)")
        parts.append(chunk)
    return b"".join(parts)

async def _process_response(resp: httpx.Response, mode: str) -> Any:
    resp.raise_for_status()
    raw = await _read_capped(resp, MAX_RESPONSE_BYTES)

    mode = (mode or "auto").lower()
    if mode == "json":
        try:
            return resp.json()
        except Exception as e:
            raise ToolException(f"Failed to parse JSON: {e}") from e
    if mode == "text":
        return raw.decode(resp.encoding or "utf-8", errors="replace")
    if mode == "bytes":
        return raw

    # auto
    ct = (resp.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        try:
            return resp.json()
        except Exception:
            return raw.decode(resp.encoding or "utf-8", errors="replace")
    if "text/" in ct:
        return raw.decode(resp.encoding or "utf-8", errors="replace")
    return raw

def _sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    clean: Dict[str, str] = {}
    for k, v in (headers or {}).items():
        if isinstance(v, str) and ("\r" in v or "\n" in v):
            raise ToolException(f"Invalid header value for {k}")
        clean[k] = v
    if "User-Agent" not in clean:
        clean["User-Agent"] = "ragnetic-api-toolkit/1.0"
    return clean

def _extract_next_link_from_headers(resp: httpx.Response) -> Optional[str]:
    # Minimal Link header parser for rel="next"
    link = resp.headers.get("Link") or resp.headers.get("link")
    if not link:
        return None
    # Example: <https://api.example.com/items?page=3>; rel="next", <...>; rel="prev"
    for part in link.split(","):
        seg = part.strip()
        if 'rel="next"' in seg or "rel=next" in seg:
            start = seg.find("<") + 1
            end = seg.find(">", start)
            if start > 0 and end > start:
                return seg[start:end]
    return None

def _jsonpath_get(data: Any, path: str) -> Optional[str]:
    # Tiny, safe JSONPath-ish getter for simple "$.a.b" or "$['a']['b']" usage.
    # For full power, pull in 'jsonpath-ng' — keeping it minimal here.
    try:
        if not path or not path.startswith("$"):
            return None
        cur = data
        # very basic parser: split on '.' and strip ["..."]
        parts = [p for p in path.replace("['", ".").replace("']", "").split(".") if p and p != "$"]
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None
        return cur
    except Exception:
        return None


class APIToolkit(BaseTool):
    name: str = "api_toolkit"
    description: str = (
        "A robust HTTP API tool supporting GET, POST, PUT, PATCH, DELETE with headers, "
        "parameters, JSON payloads, retries, SSRF protection, size limits, and multiple "
        "response formats. Supports optional pagination (page/cursor/link)."
    )
    args_schema: Type[BaseModel] = APIRequestToolInput

    async def _arun(self, **kwargs: Any) -> Any:
        tool_input = APIRequestToolInput(**kwargs)
        _validate_url(tool_input.url)

        method = tool_input.method.upper()
        headers = _sanitize_headers(tool_input.headers or {})
        params = dict(tool_input.params or {})
        json_data = tool_input.payload or {}
        response_mode = tool_input.response_mode
        pagination: Optional[PaginationConfig] = tool_input.pagination

        # Redacted audit log
        parsed = urlparse(tool_input.url)
        logger.info({
            "tool": "api_toolkit",
            "evt": "request",
            "method": method,
            "host": parsed.hostname,
            "path": parsed.path,
            "headers": _redact(headers, SENSITIVE_H),
            "params": _redact(params, SENSITIVE_P),
        })

        async def _one_request(client: httpx.AsyncClient, url: str) -> httpx.Response:
            # Validate on every hop (initial + redirects) via response hook
            return await client.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data if method in ("POST", "PUT", "PATCH", "DELETE") else None,
            )

        async def _on_response(resp: httpx.Response):
            # Defense-in-depth: validate the final URL (httpx validates redirects internally)
            try:
                _validate_url(str(resp.request.url))
            except ToolException as e:
                raise

        async with httpx.AsyncClient(
            timeout=TIMEOUTS,
            follow_redirects=True,
            event_hooks={"response": [_on_response]},
        ) as client:

            # Simple retry loop with jitter
            for attempt in range(MAX_RETRIES):
                try:
                    resp = await _one_request(client, tool_input.url)
                    if resp.status_code in RETRY_STATUS:
                        raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)

                    # Handle optional pagination
                    if pagination and pagination.max_pages > 1:
                        aggregate: List[Any] = []
                        pages_fetched = 0
                        next_url: Optional[str] = None
                        next_cursor: Optional[str] = None
                        page = pagination.page_start or 1

                        # seed first response
                        while True:
                            pages_fetched += 1
                            data = await _process_response(resp, response_mode)
                            aggregate.append(data)

                            # Stop if we hit page limit
                            if pages_fetched >= pagination.max_pages:
                                break

                            # Determine next request
                            if pagination.type == "page":
                                page += 1
                                if pagination.page_param:
                                    params[pagination.page_param] = str(page)
                                if pagination.per_page_param and pagination.per_page_value:
                                    params[pagination.per_page_param] = str(pagination.per_page_value)
                                # Fire next
                                resp = await _one_request(client, tool_input.url)

                            elif pagination.type == "cursor":
                                # Extract next cursor from body
                                cursor_path = pagination.next_cursor_jsonpath or "$.next_cursor"
                                next_cursor = _jsonpath_get(data, cursor_path)
                                if not next_cursor:
                                    break
                                if not pagination.cursor_param:
                                    break
                                params[pagination.cursor_param] = str(next_cursor)
                                resp = await _one_request(client, tool_input.url)

                            elif pagination.type == "link":
                                # Prefer Link header rel="next"
                                link_next = _extract_next_link_from_headers(resp)
                                if not link_next and pagination.next_link_jsonpath:
                                    link_next = _jsonpath_get(data, pagination.next_link_jsonpath)
                                if not link_next:
                                    break
                                # Absolute or relative
                                if not urlparse(link_next).netloc:
                                    link_next = urljoin(str(resp.request.url), link_next)
                                _validate_url(link_next)
                                resp = await _one_request(client, link_next)

                            else:
                                break  # unknown type

                        return aggregate

                    # No pagination
                    data = await _process_response(resp, response_mode)
                    return data

                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if attempt >= MAX_RETRIES - 1 or (status not in RETRY_STATUS and not isinstance(e, httpx.RequestError)):
                        raise ToolException(f"HTTP request failed: {e}") from e
                    backoff = (2 ** attempt) + random.random()
                    await asyncio.sleep(backoff)

                except ToolException:
                    raise  # bubble up our deliberate security/size errors

                except Exception as e:
                    logger.error(f"[api_toolkit] Unexpected error: {e}", exc_info=True)
                    raise ToolException(str(e))

    def _run(self, **kwargs: Any) -> Any:
        return asyncio.run(self._arun(**kwargs))
