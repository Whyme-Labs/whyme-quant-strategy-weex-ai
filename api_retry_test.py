#!/usr/bin/env python3
"""
API Retry Test - Keeps trying until WEEX authenticated API works.
Run: python api_retry_test.py
"""
import time
import hmac
import hashlib
import base64
import json
import uuid
import os
from datetime import datetime

try:
    import httpx
except ImportError:
    os.system("pip install httpx")
    import httpx

# Hackathon credentials
API_KEY = "weex_f13a2ece6fa1d9cc316e06cbd9361fa4"
SECRET_KEY = "c0b76dd3d18b15715f047e1e6435165855892b4384471ca6eae08c2c8023974e"
PASSPHRASE = "weex8931679"
BASE_URL = "https://api-contract.weex.com"

# Discord webhook for notifications (optional)
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")


def send_discord(msg):
    """Send notification to Discord."""
    if DISCORD_WEBHOOK:
        try:
            httpx.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=10)
        except:
            pass


def generate_signature(timestamp: str, method: str, path: str, body: str = "") -> str:
    message = timestamp + method.upper() + path + body
    signature = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256).digest()
    return base64.b64encode(signature).decode()


def test_api():
    """Test the API and return True if authenticated endpoints work."""
    timestamp = str(int(time.time() * 1000))
    path = "/capi/v2/account/getAccounts"
    signature = generate_signature(timestamp, "GET", path)

    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
        "locale": "en-US",
    }

    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{BASE_URL}{path}", headers=headers)
        return r.status_code, r.text


def place_test_order():
    """Place a test order with 10 USDT notional."""
    timestamp = str(int(time.time() * 1000))
    path = "/capi/v2/order/placeOrder"
    client_oid = str(uuid.uuid4()).replace("-", "")[:32]

    order_data = {
        "symbol": "cmt_btcusdt",
        "client_oid": client_oid,
        "size": "1",           # Minimum size
        "type": "1",           # Open long
        "order_type": "0",     # Normal
        "match_price": "1",    # Market
    }

    body = json.dumps(order_data)
    signature = generate_signature(timestamp, "POST", path, body)

    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json",
        "locale": "en-US",
    }

    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{BASE_URL}{path}", content=body, headers=headers)
        return r.status_code, r.text


def main():
    print("="*60)
    print("WEEX API Retry Test")
    print("Will keep trying until authenticated API works")
    print("Press Ctrl+C to stop")
    print("="*60)

    retry_count = 0
    retry_interval = 60  # seconds

    while True:
        retry_count += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n[{now}] Attempt #{retry_count}...")

        try:
            status, response = test_api()
            print(f"  Account API: {status}")

            if status == 200:
                print(f"  Response: {response[:200]}")
                result = json.loads(response)

                if result.get("code") in ("00000", "0", 0, None):
                    print("\n" + "="*60)
                    print("✓ AUTHENTICATED API IS WORKING!")
                    print("="*60)

                    # Try to place order
                    print("\nPlacing test order...")
                    order_status, order_response = place_test_order()
                    print(f"  Order API: {order_status}")
                    print(f"  Response: {order_response[:300]}")

                    if order_status == 200:
                        order_result = json.loads(order_response)
                        if order_result.get("code") in ("00000", "0", 0, None):
                            print("\n" + "="*60)
                            print("✓✓✓ ORDER PLACED SUCCESSFULLY! ✓✓✓")
                            print("API TEST COMPLETED!")
                            print("="*60)
                            send_discord("🎉 WEEX API Test PASSED! Order placed successfully!")
                            return True
                        else:
                            print(f"  Order error: {order_result.get('msg', order_result)}")

                    send_discord(f"✓ WEEX API working! Account status: {status}")
                else:
                    print(f"  API Error: {result.get('msg', result)}")

            elif status == 521:
                print("  Still getting 521 (server down)")
            else:
                print(f"  Response: {response[:200]}")

        except Exception as e:
            print(f"  Error: {e}")

        print(f"  Retrying in {retry_interval} seconds...")
        time.sleep(retry_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
