#!/usr/bin/env python3
"""Simple API test with correct WEEX endpoint parameters."""
import time
import hmac
import hashlib
import base64
import json
import uuid

try:
    import httpx
except ImportError:
    print("Installing httpx...")
    import os
    os.system("pip install httpx")
    import httpx

# Hackathon credentials
API_KEY = "weex_f13a2ece6fa1d9cc316e06cbd9361fa4"
SECRET_KEY = "c0b76dd3d18b15715f047e1e6435165855892b4384471ca6eae08c2c8023974e"
PASSPHRASE = "weex8931679"
BASE_URL = "https://api-contract.weex.com"


def generate_signature(timestamp: str, method: str, path: str, query: str = "", body: str = "") -> str:
    """Generate HMAC-SHA256 signature."""
    # Format: timestamp + METHOD + path + ?query + body
    if query:
        message = timestamp + method.upper() + path + "?" + query + body
    else:
        message = timestamp + method.upper() + path + body
    print(f"  Signing: {message[:100]}...")
    signature = hmac.new(
        SECRET_KEY.encode(),
        message.encode(),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(signature).decode()


def get_headers(method: str, path: str, query: str = "", body: str = "") -> dict:
    """Get authentication headers."""
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, method, path, query, body)
    return {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
        "locale": "en-US",
    }


def main():
    print("="*60)
    print("WEEX API Test - Hackathon Qualification")
    print("="*60)
    print(f"API Key: {API_KEY[:30]}...")
    print(f"Base URL: {BASE_URL}")

    with httpx.Client(timeout=30.0) as client:
        # Test 1: Public endpoint
        print("\n[1] Testing public endpoint (no auth)...")
        try:
            r = client.get(f"{BASE_URL}/capi/v2/market/ticker?symbol=cmt_btcusdt")
            print(f"    Status: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                btc_price = float(data.get('last', 0))
                print(f"    BTC Price: ${btc_price:,.2f}")
        except Exception as e:
            print(f"    Error: {e}")
            btc_price = 95000

        # Test 2: Get account (CORRECT endpoint: /capi/v2/account/getAccounts)
        print("\n[2] Get account info (authenticated)...")
        path = "/capi/v2/account/getAccounts"
        headers = get_headers("GET", path)
        print(f"    Endpoint: {path}")
        try:
            r = client.get(f"{BASE_URL}{path}", headers=headers)
            print(f"    Status: {r.status_code}")
            print(f"    Response: {r.text[:500]}")
            if r.status_code == 200:
                print("    ✓ Account API works!")
        except Exception as e:
            print(f"    Error: {e}")

        # Test 3: Place order with CORRECT parameters
        print("\n[3] Place test order (10 USDT notional)...")
        path = "/capi/v2/order/placeOrder"

        # Generate unique client order ID
        client_oid = str(uuid.uuid4()).replace("-", "")[:32]

        # CORRECT order parameters per documentation:
        # - type: 1=Open long, 2=Open short, 3=Close long, 4=Close short
        # - order_type: 0=Normal, 1=Post-Only, 2=FOK, 3=IOC
        # - match_price: 0=Limit, 1=Market
        order_data = {
            "symbol": "cmt_btcusdt",  # From /contracts endpoint (lowercase)
            "client_oid": client_oid,
            "size": "1",           # 1 contract = 0.0001 BTC = ~$9.50
            "type": "1",           # 1 = Open long
            "order_type": "0",     # 0 = Normal
            "match_price": "1",    # 1 = Market order
        }

        body = json.dumps(order_data)
        headers = get_headers("POST", path, "", body)

        print(f"    Endpoint: {path}")
        print(f"    Order: {order_data}")
        try:
            r = client.post(f"{BASE_URL}{path}", content=body, headers=headers)
            print(f"    Status: {r.status_code}")
            print(f"    Response: {r.text[:500]}")

            if r.status_code == 200:
                result = r.json()
                if result.get("code") in ("00000", "0", 0):
                    print("\n    ✓ ORDER SUCCESS!")
                    order_id = result.get("data", {}).get("order_id", result.get("data", {}).get("orderId"))
                    print(f"    Order ID: {order_id}")

                    # Wait and close position
                    print("\n[4] Closing position...")
                    time.sleep(2)

                    close_oid = str(uuid.uuid4()).replace("-", "")[:32]
                    close_data = {
                        "symbol": "cmt_btcusdt",
                        "client_oid": close_oid,
                        "size": "1",
                        "type": "3",           # 3 = Close long
                        "order_type": "0",
                        "match_price": "1",    # Market
                    }
                    close_body = json.dumps(close_data)
                    close_headers = get_headers("POST", path, "", close_body)

                    r2 = client.post(f"{BASE_URL}{path}", content=close_body, headers=close_headers)
                    print(f"    Status: {r2.status_code}")
                    print(f"    Response: {r2.text[:300]}")
                else:
                    print(f"    API Error: {result.get('msg', result)}")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


if __name__ == "__main__":
    main()
