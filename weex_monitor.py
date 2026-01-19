#!/usr/bin/env python3
"""
WEEX API Monitor - Continuously tests API and notifies when working.
Run: python weex_monitor.py
"""
import time
import hmac
import hashlib
import base64
import json
import urllib.request
import urllib.error
import ssl
from datetime import datetime

# Hackathon credentials
API_KEY = "weex_f13a2ece6fa1d9cc316e06cbd9361fa4"
SECRET_KEY = "c0b76dd3d18b15715f047e1e6435165855892b4384471ca6eae08c2c8023974e"
PASSPHRASE = "weex8931679"
BASE_URL = "https://api-contract.weex.com"

# Discord webhook for notifications
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1461966152199704617/Qq0IUdPCdms0X-ban6vuAydk6dhStNU1Ue7cv0mYej5z3M4pE5yHPSOFC4eU92reK8pp"

ctx = ssl.create_default_context()


def send_discord(msg):
    """Send notification to Discord."""
    try:
        data = json.dumps({"content": msg}).encode()
        req = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "WEEX-Monitor/1.0"  # Required to avoid Cloudflare 403
            }
        )
        urllib.request.urlopen(req, timeout=10, context=ctx)
        print(f"  Discord: Sent notification")
    except Exception as e:
        print(f"  Discord: Failed - {e}")


def generate_signature(timestamp: str, method: str, path: str, body: str = "") -> str:
    message = timestamp + method.upper() + path + body
    signature = hmac.new(SECRET_KEY.encode(), message.encode(), hashlib.sha256).digest()
    return base64.b64encode(signature).decode()


def test_api():
    """Test the API. Returns (status_code, response_text, is_working)."""
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
    
    try:
        req = urllib.request.Request(f"{BASE_URL}{path}", headers=headers)
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            body = resp.read().decode()
            return resp.status, body, True
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return e.code, body, False
    except Exception as e:
        return 0, str(e), False


def main():
    print("=" * 60)
    print("WEEX API Monitor")
    print("Will keep testing until authenticated API works")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    retry_count = 0
    retry_interval = 60  # seconds
    last_notified = None
    
    while True:
        retry_count += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n[{now}] Attempt #{retry_count}...")
        
        status, response, is_working = test_api()
        print(f"  Status: {status}")
        
        if status == 200 and is_working:
            try:
                result = json.loads(response)
                if result.get("code") in ("00000", "0", 0, None):
                    print(f"  Response: {response[:200]}")
                    print("\n" + "=" * 60)
                    print("✓ AUTHENTICATED API IS WORKING!")
                    print("=" * 60)
                    
                    send_discord(f"🎉 **WEEX API IS WORKING!**\nRun `python simple_api_test.py` to complete hackathon task!")
                    
                    # Keep running to confirm it stays up
                    continue
                else:
                    print(f"  API Error: {result.get('msg', result)}")
            except json.JSONDecodeError:
                print(f"  Response: {response[:200]}")
        elif status == 521:
            print("  Still getting 521 (server unreachable)")
            
            # Notify every hour if still down
            hour_now = datetime.now().hour
            if last_notified != hour_now and retry_count > 1:
                send_discord(f"⏳ WEEX API still returning 521 after {retry_count} attempts ({now})")
                last_notified = hour_now
        else:
            print(f"  Response: {response[:200] if response else '(empty)'}")
        
        print(f"  Next check in {retry_interval} seconds...")
        time.sleep(retry_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
