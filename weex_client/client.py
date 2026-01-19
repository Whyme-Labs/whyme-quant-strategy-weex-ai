"""WEEX API Client for AI Wars Hackathon.

Implements the WEEX AI Wars API for:
- Market data
- Account management
- Trading operations
- AI log uploads
"""

import json
import httpx
from typing import Any, Dict, List, Optional
from loguru import logger

from .auth import WeexAuth


class WeexClient:
    """WEEX API client for the AI Wars hackathon."""

    # Symbol format mapping (user-friendly -> WEEX API format)
    SYMBOL_MAP = {
        "BTCUSDT": "cmt_btcusdt",
        "ETHUSDT": "cmt_ethusdt",
        "SOLUSDT": "cmt_solusdt",
        "BNBUSDT": "cmt_bnbusdt",
        "XRPUSDT": "cmt_xrpusdt",
        "DOGEUSDT": "cmt_dogeusdt",
    }

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        base_url: str = "https://api-contract.weex.com",
    ):
        """Initialize WEEX client.

        Args:
            api_key: WEEX API key
            secret_key: WEEX secret key
            passphrase: WEEX access passphrase
            base_url: API base URL
        """
        self.base_url = base_url.rstrip("/")
        self.auth = WeexAuth(api_key, secret_key, passphrase)
        self._client: Optional[httpx.AsyncClient] = None

    def _convert_symbol(self, symbol: str) -> str:
        """Convert user-friendly symbol to WEEX API format.

        Args:
            symbol: User symbol (e.g., 'BTCUSDT')

        Returns:
            WEEX API symbol format (e.g., 'cmt_btcusdt')
        """
        # If already in WEEX format, return as-is
        if symbol.startswith("cmt_"):
            return symbol
        # Convert from map or default to lowercase with cmt_ prefix
        return self.SYMBOL_MAP.get(symbol, f"cmt_{symbol.lower()}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make authenticated request to WEEX API.

        Args:
            method: HTTP method
            path: API endpoint path
            params: Query parameters
            data: Request body

        Returns:
            API response data
        """
        client = await self._get_client()
        url = f"{self.base_url}{path}"

        query_string = ""
        if params:
            query_string = "?" + "&".join(f"{k}={v}" for k, v in params.items())

        body = ""
        if data:
            body = json.dumps(data)

        headers = self.auth.get_headers(method, path, query_string, body)

        logger.debug(f"WEEX API Request: {method} {url}")

        if method.upper() == "GET":
            response = await client.get(url, params=params, headers=headers)
        else:
            response = await client.post(url, content=body, headers=headers)

        response.raise_for_status()
        result = response.json()

        # Handle different response formats
        # /capi/v2/ endpoints return data directly or with code field
        if isinstance(result, dict):
            # Check for error response
            if result.get("code") and result.get("code") not in ("00000", "0"):
                logger.error(f"WEEX API Error: {result}")
                raise Exception(f"WEEX API Error: {result.get('msg', 'Unknown error')}")
            # Return data field if present, otherwise return result
            return result.get("data", result)

        return result

    # ==================== Market APIs ====================

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker information for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT' or 'cmt_btcusdt')

        Returns:
            Ticker data with keys: symbol, last, best_ask, best_bid, high_24h, low_24h, volume_24h, etc.
        """
        weex_symbol = self._convert_symbol(symbol)
        return await self._request("GET", "/capi/v2/market/ticker", {"symbol": weex_symbol})

    async def get_orderbook(self, symbol: str, depth_type: str = "step0") -> Dict[str, Any]:
        """Get order book for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            depth_type: Depth aggregation level (step0, step1, step2, etc.)

        Returns:
            Order book data with asks, bids, and timestamp
        """
        weex_symbol = self._convert_symbol(symbol)
        return await self._request(
            "GET", "/capi/v2/market/depth", {"symbol": weex_symbol, "type": depth_type}
        )

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
    ) -> List[List[str]]:
        """Get candlestick/kline data.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1min, 5min, 15min, 30min, 1h, 4h, 12h, 1day, 1week)
            limit: Number of klines (max 200)

        Returns:
            List of klines: [timestamp, open, high, low, close, volume, quote_volume]
        """
        weex_symbol = self._convert_symbol(symbol)
        result = await self._request(
            "GET",
            "/capi/v2/market/candles",
            {"symbol": weex_symbol, "granularity": interval},
        )
        # Return only the requested number of candles
        if isinstance(result, list) and limit:
            return result[:limit]
        return result

    # ==================== Account APIs ====================

    async def get_account(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Get account information.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            Account data with equity, available balance, etc.
        """
        # Use /accounts endpoint which works without symbol param
        return await self._request("GET", "/capi/v2/account/accounts")

    async def get_account_info(self) -> Dict[str, Any]:
        """Get general account information.

        Returns:
            Account info with equity, available balance, etc.
        """
        return await self.get_account()

    async def get_assets(self) -> List[Dict[str, Any]]:
        """Get account assets/balances.

        Returns:
            List of assets with available, equity, frozen amounts
        """
        return await self._request("GET", "/capi/v2/account/assets")

    async def get_positions(self, symbol: str = "BTCUSDT") -> List[Dict[str, Any]]:
        """Get open positions.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            List of positions
        """
        weex_symbol = self._convert_symbol(symbol)
        return await self._request(
            "GET", "/capi/v2/account/position/singlePosition", {"symbol": weex_symbol}
        )

    # ==================== Trade APIs ====================

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: str,
        price: Optional[str] = None,
        client_oid: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Place an order.

        WEEX API uses snake_case parameters:
        - symbol: cmt_btcusdt format
        - client_oid: unique client order ID (required!)
        - size: contract size as string
        - side: "1" (buy/long) or "2" (sell/short) for open
        - type: "1" (limit) or "2" (market)
        - order_type: "0" (normal)
        - match_price: "1" for market orders
        - price: required for limit orders

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: 'buy' or 'sell' (will be converted to '1' or '2')
            order_type: 'limit' or 'market'
            size: Order size (contracts)
            price: Limit price (required for limit orders)
            client_oid: Client order ID (auto-generated if not provided)
            **kwargs: Additional order parameters

        Returns:
            Order response with orderId
        """
        import time

        weex_symbol = self._convert_symbol(symbol)

        # Generate client_oid if not provided (required by WEEX)
        if not client_oid:
            client_oid = str(int(time.time() * 1000))

        # Convert side to WEEX format
        # 1 = buy (open long), 2 = sell (open short)
        side_code = "1" if side.lower() in ("buy", "long", "buy_single") else "2"

        # Convert order_type to WEEX format
        # 1 = limit, 2 = market
        type_code = "1" if order_type.lower() == "limit" else "2"

        data = {
            "symbol": weex_symbol,
            "client_oid": client_oid,
            "size": str(size),
            "side": side_code,
            "type": type_code,
            "order_type": "0",  # Normal order
        }

        # For market orders, set match_price
        if order_type.lower() == "market":
            data["match_price"] = "1"

        # For limit orders, add price
        if order_type.lower() == "limit" and price:
            data["price"] = str(price)

        # Add any additional kwargs
        data.update(kwargs)

        logger.info(f"Placing order: {data}")
        return await self._request("POST", "/capi/v2/order/placeOrder", data=data)

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an order.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            order_id: Order ID to cancel

        Returns:
            Cancellation response
        """
        weex_symbol = self._convert_symbol(symbol)
        return await self._request(
            "POST",
            "/capi/v2/order/cancelOrder",
            data={"symbol": weex_symbol, "orderId": order_id},
        )

    async def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get order details.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            order_id: Order ID

        Returns:
            Order details
        """
        weex_symbol = self._convert_symbol(symbol)
        return await self._request(
            "GET",
            "/capi/v2/order/detail",
            {"symbol": weex_symbol, "orderId": order_id},
        )

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Get open orders.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            List of open orders
        """
        weex_symbol = self._convert_symbol(symbol)
        return await self._request(
            "GET", "/capi/v2/order/currentPlan", {"symbol": weex_symbol}
        )

    async def get_order_history(
        self, symbol: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get order history.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            limit: Number of orders

        Returns:
            List of historical orders
        """
        weex_symbol = self._convert_symbol(symbol)
        return await self._request(
            "GET",
            "/capi/v2/order/history",
            {"symbol": weex_symbol, "pageSize": str(limit)},
        )

    async def close_all_positions(self, symbol: str) -> Dict[str, Any]:
        """Close all positions for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            Close response
        """
        weex_symbol = self._convert_symbol(symbol)
        return await self._request(
            "POST", "/capi/v2/order/closePositions", data={"symbol": weex_symbol}
        )

    # ==================== AI Log API (Hackathon Requirement) ====================

    async def upload_ai_log(
        self,
        stage: str,
        model: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        explanation: str,
        order_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Upload AI log for hackathon verification.

        This is MANDATORY for the hackathon. Every trade must have an AI log.

        Args:
            stage: Trading stage (e.g., "Strategy Generation", "Risk Assessment")
            model: AI model name (e.g., "gpt-4-turbo", "claude-3-opus")
            input_data: Input to the AI model
            output_data: Output from the AI model
            explanation: Natural language explanation (max 1000 chars)
            order_id: Optional WEEX order ID

        Returns:
            Upload response
        """
        data = {
            "stage": stage,
            "model": model,
            "input": input_data,
            "output": output_data,
            "explanation": explanation[:1000],  # Max 1000 characters
        }

        if order_id:
            data["orderId"] = order_id

        logger.info(f"Uploading AI log: stage={stage}, model={model}")
        return await self._request("POST", "/capi/v2/order/uploadAiLog", data=data)

    # ==================== Utility Methods ====================

    async def test_connection(self) -> bool:
        """Test API connection.

        Returns:
            True if connection successful
        """
        try:
            await self.get_ticker("BTCUSDT")
            logger.info("WEEX API connection successful")
            return True
        except Exception as e:
            logger.error(f"WEEX API connection failed: {e}")
            return False
