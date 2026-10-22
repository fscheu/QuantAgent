#!/usr/bin/env python
"""Submit and cancel one deliberately distant SPY limit order on Alpaca paper."""
import argparse
import os
from decimal import Decimal

import yfinance as yf
from alpaca.trading.client import TradingClient
from quantagent.models import Order, OrderSide, OrderType
from quantagent.trading.alpaca_broker import AlpacaBroker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="send a real paper-api order")
    args = parser.parse_args()
    if not args.real:
        parser.error("pass --real to submit an order to Alpaca paper")

    close = float(yf.Ticker("SPY").history(period="5d")["Close"].dropna().iloc[-1])
    order = Order(symbol="SPY", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                  quantity=Decimal("1"), price=Decimal(str(round(close / 2, 2))))
    client = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    broker = AlpacaBroker(client)
    broker.submit_order(order)
    before = broker.client.get_order_by_id(order.alpaca_order_id)
    before_status = str(getattr(before.status, "value", before.status)).lower()
    if before_status not in {"new", "accepted"}:
        raise RuntimeError(f"order was not left open: {before.status}")
    if not broker.cancel_order(order.alpaca_order_id):
        raise RuntimeError("Alpaca did not cancel the order")
    after = broker.client.get_order_by_id(order.alpaca_order_id)
    after_status = str(getattr(after.status, "value", after.status)).lower()
    print(f"paper order={order.alpaca_order_id} before={before_status} after={after_status}")
    if after_status != "canceled":
        raise RuntimeError(f"order cancellation not confirmed: {after.status}")


if __name__ == "__main__":
    main()
