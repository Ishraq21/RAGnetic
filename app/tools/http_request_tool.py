import requests
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class HTTPRequestTool:
    """
    A tool to make HTTP requests from a RAGnetic workflow.
    Supports GET, POST, PUT, PATCH, and DELETE methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def run(self, method: str, url: str, headers: Optional[Dict[str, str]] = None,
            body: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Makes an HTTP request to the specified URL.

        Args:
            method (str): The HTTP method (e.g., 'GET', 'POST', 'PUT').
            url (str): The URL to make the request to.
            headers (Optional[Dict]): A dictionary of HTTP headers.
            body (Optional[Dict]): A dictionary of the request body (for POST, PUT, etc.).
            params (Optional[Dict]): A dictionary of URL parameters.

        Returns:
            Dict: A dictionary containing the response status code, headers, and body.
        """
        if not url:
            return {"error": "URL is required for HTTP request."}

        try:
            logger.info(f"Making {method} request to {url} with params: {params} and body: {body}")
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=headers,
                json=body,
                params=params,
                timeout=15  # Set a reasonable timeout
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            logger.info(f"Request to {url} successful.")

            # Try to parse JSON response, fall back to text
            try:
                response_body = response.json()
            except json.JSONDecodeError:
                response_body = response.text

            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response_body
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request to {url} failed: {e}")
            return {"error": str(e), "status_code": None, "body": None}