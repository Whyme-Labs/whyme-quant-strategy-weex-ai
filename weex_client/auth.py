"""WEEX API Authentication and Signature Generation.

Based on WEEX AI Wars API documentation:
https://www.weex.com/api-doc/ai/QuickStart/RequestInteraction
"""

import time
import hmac
import hashlib
import base64
from typing import Optional


class WeexAuth:
    """Handles WEEX API authentication and request signing."""

    def __init__(self, api_key: str, secret_key: str, passphrase: str):
        """Initialize WEEX auth with credentials.

        Args:
            api_key: WEEX API key
            secret_key: WEEX secret key
            passphrase: WEEX access passphrase
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase

    def generate_signature(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        query_string: str = "",
        body: str = "",
    ) -> str:
        """Generate signature for POST requests.

        Args:
            timestamp: Unix timestamp in milliseconds
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            query_string: Query string for GET requests
            body: Request body for POST requests

        Returns:
            Base64-encoded HMAC-SHA256 signature
        """
        message = timestamp + method.upper() + request_path + query_string + body
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode()

    def generate_signature_get(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        query_string: str = "",
    ) -> str:
        """Generate signature for GET requests.

        Args:
            timestamp: Unix timestamp in milliseconds
            method: HTTP method
            request_path: API endpoint path
            query_string: Query parameters

        Returns:
            Base64-encoded HMAC-SHA256 signature
        """
        message = timestamp + method.upper() + request_path + query_string
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(signature).decode()

    def get_timestamp(self) -> str:
        """Get current timestamp in milliseconds."""
        return str(int(time.time() * 1000))

    def get_headers(
        self,
        method: str,
        request_path: str,
        query_string: str = "",
        body: str = "",
    ) -> dict:
        """Generate complete authentication headers for a request.

        Args:
            method: HTTP method
            request_path: API endpoint path
            query_string: Query parameters (for GET)
            body: Request body (for POST)

        Returns:
            Dictionary of authentication headers
        """
        timestamp = self.get_timestamp()

        if method.upper() == "GET":
            signature = self.generate_signature_get(
                timestamp, method, request_path, query_string
            )
        else:
            signature = self.generate_signature(
                timestamp, method, request_path, query_string, body
            )

        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-PASSPHRASE": self.passphrase,
            "ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
            "locale": "en-US",
        }
