"""Test script to verify migrations work correctly."""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from quantagent.database import engine, init_db, SessionLocal, Base
from quantagent.models import Order, Fill, Position, Signal, Trade, MarketData


def test_create_tables():
    """Test creating all tables."""
    print("Creating all tables...")
    init_db()
    print("✓ Tables created successfully")


def test_insert_market_data():
    """Test inserting market data."""
    from datetime import datetime, timezone
    from decimal import Decimal

    print("\nInserting market data...")
    db = SessionLocal()
    try:
        market_data = MarketData(
            symbol="BTC",
            timeframe="1h",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("45000.00"),
            high=Decimal("45500.00"),
            low=Decimal("44800.00"),
            close=Decimal("45200.00"),
            volume=Decimal("100.5")
        )
        db.add(market_data)
        db.commit()
        print(f"✓ Market data inserted: BTC 1h candle")
        db.refresh(market_data)
        print(f"  ID: {market_data.id}, Close: {market_data.close}")
    finally:
        db.close()


def test_insert_signal():
    """Test inserting trading signal."""
    from datetime import datetime
    from quantagent.models import TradeSignal

    print("\nInserting trading signal...")
    db = SessionLocal()
    try:
        signal = Signal(
            symbol="BTC",
            signal=TradeSignal.LONG,
            confidence=0.85,
            timeframe="1h",
            rsi=65.5,
            macd=250.5,
            stochastic=78.3,
            roc=2.5,
            williams_r=-25.0,
            pattern="Bullish Engulfing",
            trend="Uptrend",
            analysis_summary="Strong bullish signal based on technical indicators"
        )
        db.add(signal)
        db.commit()
        print(f"✓ Signal inserted: {signal.signal.value} for {signal.symbol}")
        db.refresh(signal)
        print(f"  ID: {signal.id}, Confidence: {signal.confidence}, Pattern: {signal.pattern}")
    finally:
        db.close()


def test_insert_order():
    """Test inserting trading order."""
    from datetime import datetime
    from decimal import Decimal
    from quantagent.models import OrderSide, OrderType, OrderStatus

    print("\nInserting order...")
    db = SessionLocal()
    try:
        order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.5"),
            price=Decimal("45000.00"),
            status=OrderStatus.PENDING,
            comment="Test buy order"
        )
        db.add(order)
        db.commit()
        print(f"✓ Order inserted: BUY {order.quantity} {order.symbol}")
        db.refresh(order)
        print(f"  ID: {order.id}, Price: {order.price}, Status: {order.status.value}")
    finally:
        db.close()


def test_insert_fill():
    """Test inserting order fill."""
    from datetime import datetime
    from decimal import Decimal
    from quantagent.models import OrderSide, OrderType, OrderStatus

    print("\nInserting order fill...")
    db = SessionLocal()
    try:
        # First create an order
        order = Order(
            symbol="ETH",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),
            status=OrderStatus.FILLED
        )
        db.add(order)
        db.flush()

        # Then create a fill for it
        fill = Fill(
            order_id=order.id,
            quantity=Decimal("10.0"),
            price=Decimal("2500.00"),
            commission=Decimal("2.50")
        )
        db.add(fill)
        db.commit()
        print(f"✓ Fill inserted: {fill.quantity} {order.symbol} @ {fill.price}")
        db.refresh(fill)
        print(f"  ID: {fill.id}, Order ID: {fill.order_id}, Commission: {fill.commission}")
    finally:
        db.close()


def test_insert_trade():
    """Test inserting completed trade."""
    from datetime import datetime
    from decimal import Decimal
    from quantagent.models import OrderSide

    print("\nInserting trade...")
    db = SessionLocal()
    try:
        trade = Trade(
            symbol="BTC",
            entry_price=Decimal("45000.00"),
            exit_price=Decimal("45500.00"),
            quantity=Decimal("0.5"),
            side=OrderSide.BUY,
            pnl=Decimal("250.00"),
            pnl_pct=0.556,
            commission=Decimal("22.50"),
            entry_signal="LONG",
            exit_signal="TAKE_PROFIT",
            timeframe="1h",
            notes="Test trade"
        )
        db.add(trade)
        db.commit()
        print(f"✓ Trade inserted: {trade.side.value} {trade.quantity} {trade.symbol}")
        db.refresh(trade)
        print(f"  ID: {trade.id}, P&L: {trade.pnl} ({trade.pnl_pct}%)")
    finally:
        db.close()


def test_insert_position():
    """Test inserting open position."""
    from decimal import Decimal
    from quantagent.models import OrderSide

    print("\nInserting position...")
    db = SessionLocal()
    try:
        position = Position(
            symbol="BTC",
            quantity=Decimal("0.5"),
            average_entry_price=Decimal("45000.00"),
            current_price=Decimal("45500.00"),
            unrealized_pnl=Decimal("250.00"),
            unrealized_pnl_pct=0.556,
            side=OrderSide.BUY
        )
        db.add(position)
        db.commit()
        print(f"✓ Position inserted: {position.side.value} {position.quantity} {position.symbol}")
        db.refresh(position)
        print(f"  ID: {position.id}, Current Price: {position.current_price}, Unrealized P&L: {position.unrealized_pnl}")
    finally:
        db.close()


def test_query_data():
    """Test querying all created data."""
    print("\n" + "="*50)
    print("QUERYING DATA")
    print("="*50)

    db = SessionLocal()
    try:
        print(f"\nMarketData entries: {db.query(MarketData).count()}")
        print(f"Signal entries: {db.query(Signal).count()}")
        print(f"Order entries: {db.query(Order).count()}")
        print(f"Fill entries: {db.query(Fill).count()}")
        print(f"Trade entries: {db.query(Trade).count()}")
        print(f"Position entries: {db.query(Position).count()}")
    finally:
        db.close()


def cleanup():
    """Clean up test data."""
    print("\nCleaning up...")
    from quantagent.database import drop_all_tables
    drop_all_tables()
    print("✓ All tables dropped")


if __name__ == "__main__":
    print("=" * 50)
    print("ALEMBIC MIGRATION TEST")
    print("=" * 50)

    try:
        # Test database operations
        test_create_tables()
        test_insert_market_data()
        test_insert_signal()
        test_insert_order()
        test_insert_fill()
        test_insert_trade()
        test_insert_position()
        test_query_data()

        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        cleanup()
