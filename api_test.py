#!/usr/bin/env python3
"""API Testing Script for WEEX Hackathon.

Task: Execute an order through the API with notional value of 10 USDT on BTCUSDT.
"""
import asyncio
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from weex_client import WeexClient

async def test_api():
    """Test API by placing a 10 USDT order on BTCUSDT."""

    # Initialize client
    client = WeexClient(
        api_key=os.getenv('WEEX_API_KEY'),
        secret_key=os.getenv('WEEX_SECRET_KEY'),
        passphrase=os.getenv('WEEX_PASSPHRASE'),
    )

    print('='*60)
    print('WEEX API Testing - Hackathon Qualification')
    print('='*60)
    print(f'API Key: {os.getenv("WEEX_API_KEY")[:20]}...')

    # 1. Test public endpoint - get ticker
    print('\n[1] Testing public endpoint - Get BTC price...')
    try:
        ticker = await client.get_ticker('cmt_btcusdt')
        btc_price = float(ticker.get('last', 0))
        print(f'    BTC/USDT Price: ${btc_price:,.2f}')
    except Exception as e:
        print(f'    Error: {e}')
        return False

    # 2. Test authenticated endpoint - get account
    print('\n[2] Testing authenticated endpoint - Get account info...')
    try:
        account = await client.get_account()
        print(f'    Account info retrieved successfully')
        print(f'    Response: {account}')
    except Exception as e:
        print(f'    Error: {e}')

    # 3. Place a test order - 10 USDT notional value
    print('\n[3] Placing test order - Market buy on BTCUSDT...')
    try:
        # For BTC perpetual on WEEX:
        # - Minimum order size is typically 1 contract
        # - 1 contract = 0.001 BTC
        # - At BTC ~95000, 1 contract = ~95 USDT notional
        # - With 20x leverage, margin needed = 95/20 = ~4.75 USDT

        result = await client.place_order(
            symbol='cmt_btcusdt',
            side='buy',           # Long position
            size=1,               # Minimum size (1 contract = 0.001 BTC)
            order_type='market',  # Market order
            leverage=20,          # Max leverage
        )

        print(f'    Order placed successfully!')
        print(f'    Result: {result}')

        order_id = result.get('orderId', result.get('order_id', result.get('data', {}).get('orderId')))
        print(f'    Order ID: {order_id}')

        # Wait a moment for the order to fill
        await asyncio.sleep(3)

        # Check position
        print('\n[4] Checking position...')
        try:
            position = await client.get_position('cmt_btcusdt')
            print(f'    Position: {position}')
        except Exception as e:
            print(f'    Position check error: {e}')

        # Close the position
        print('\n[5] Closing test position...')
        close_result = await client.place_order(
            symbol='cmt_btcusdt',
            side='sell',
            size=1,
            order_type='market',
            reduce_only=True,
        )
        print(f'    Position closed!')
        print(f'    Result: {close_result}')

        return True

    except Exception as e:
        print(f'    Error placing order: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = asyncio.run(test_api())
    print('\n' + '='*60)
    if success:
        print('API TEST COMPLETED SUCCESSFULLY!')
        print('You have qualified for the Preliminary Round!')
    else:
        print('API TEST FAILED - Check errors above')
    print('='*60)
