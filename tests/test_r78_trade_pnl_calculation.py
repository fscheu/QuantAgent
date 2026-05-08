"""Unit tests for QuantAgent-r78: Trade P&L Calculation.

Tests validate that closing LONG/SHORT positions correctly calculate
Trade.pnl and Trade.pnl_pct fields per acceptance criteria in:
docs/05_acceptance_tests/QuantAgent-r78-AC-trade-pnl-calculation.md

Following TESTING_PATTERNS.md:
- Structure & type validation (Decimal vs float)
- Constraint validation (formulas, ranges)
- Error handling (invalid prices)
- Edge cases (opening positions, zero prices)

Note: These tests require PostgreSQL due to JSON column usage in models.
They are marked as integration tests to exclude from default pytest run.
"""

import os
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from quantagent.models import (
    Base,
    Environment,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Trade,
)
from quantagent.portfolio.manager import PortfolioManager


@pytest.fixture
def test_db():
    """Create database session for testing using DATABASE_URL if available."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        engine = create_engine(database_url)
    else:
        # Fallback to SQLite for local development
        engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    if database_url:
        tables = ", ".join(table.name for table in reversed(Base.metadata.sorted_tables))
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    TestSession = sessionmaker(bind=engine)
    db = TestSession()
    yield db
    db.close()
    if database_url:
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
def portfolio(test_db):
    """Create portfolio manager with initial capital."""
    return PortfolioManager(
        initial_cash=100000.0, environment=Environment.BACKTEST, db=test_db
    )


class TestTradePnLStructure:
    """Tests for Trade P&L structure and types (AC invariants)."""

    def test_trade_pnl_is_decimal_or_none(self, portfolio, test_db):
        """Verify Trade.pnl is Decimal type (not float) or None."""
        # Open position
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=60000.0)

        # Close position
        close_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("65000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(close_order)
        test_db.commit()
        portfolio.execute_trade(close_order, fill_price=65000.0)

        # Query closed trade
        closed_trade = (
            test_db.query(Trade).filter(Trade.closed_at.isnot(None)).first()
        )

        assert closed_trade is not None
        assert isinstance(closed_trade.pnl, (Decimal, type(None)))
        assert closed_trade.pnl is not None  # Closing trade must have pnl
        assert isinstance(closed_trade.pnl, Decimal)  # Must be Decimal, not float

    def test_trade_pnl_pct_is_float_or_none(self, portfolio, test_db):
        """Verify Trade.pnl_pct is float type or None."""
        # Open and close position
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=60000.0)

        close_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("65000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(close_order)
        test_db.commit()
        portfolio.execute_trade(close_order, fill_price=65000.0)

        closed_trade = (
            test_db.query(Trade).filter(Trade.closed_at.isnot(None)).first()
        )

        assert closed_trade is not None
        assert isinstance(closed_trade.pnl_pct, (float, type(None)))
        assert closed_trade.pnl_pct is not None
        assert isinstance(closed_trade.pnl_pct, float)


class TestLongPositionPnL:
    """Tests for LONG position P&L calculation (AC-1, AC-2)."""

    def test_long_position_profit(self, portfolio, test_db):
        """AC-1: LONG position closed at profit calculates positive P&L.

        Given: LONG position opened at $60,000
        When: Closed at $65,000 with quantity 0.1
        Then: pnl = $500.00, pnl_pct ≈ 8.33%
        """
        # Open LONG at 60,000
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=60000.0)

        # Close LONG at 65,000
        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("65000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        portfolio.execute_trade(sell_order, fill_price=65000.0)

        # Verify closed trade
        closed_trade = (
            test_db.query(Trade).filter(Trade.closed_at.isnot(None)).first()
        )

        assert closed_trade is not None
        assert closed_trade.pnl == Decimal("500.00")
        assert abs(closed_trade.pnl_pct - 8.33) < 0.01

    def test_long_position_loss(self, portfolio, test_db):
        """AC-2: LONG position closed at loss calculates negative P&L.

        Given: LONG position opened at $60,000
        When: Closed at $55,000 with quantity 0.1
        Then: pnl = -$500.00, pnl_pct ≈ -8.33%
        """
        # Open LONG at 60,000
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=60000.0)

        # Close LONG at 55,000
        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("55000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        portfolio.execute_trade(sell_order, fill_price=55000.0)

        # Verify closed trade
        closed_trade = (
            test_db.query(Trade).filter(Trade.closed_at.isnot(None)).first()
        )

        assert closed_trade is not None
        assert closed_trade.pnl == Decimal("-500.00")
        assert abs(closed_trade.pnl_pct - (-8.33)) < 0.01


class TestShortPositionPnL:
    """Tests for SHORT position P&L calculation (AC-3, AC-4)."""

    def test_short_position_profit(self, portfolio, test_db):
        """AC-3: SHORT position closed at profit calculates positive P&L.

        Given: SHORT position opened at $65,000
        When: Closed at $60,000 with quantity 0.1
        Then: pnl = $500.00, pnl_pct ≈ 7.69%
        """
        # Open SHORT at 65,000
        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("65000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        portfolio.execute_trade(sell_order, fill_price=65000.0)

        # Close SHORT at 60,000
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=60000.0)

        # Verify closed trade
        closed_trade = (
            test_db.query(Trade).filter(Trade.closed_at.isnot(None)).first()
        )

        assert closed_trade is not None
        assert closed_trade.pnl == Decimal("500.00")
        assert abs(closed_trade.pnl_pct - 7.69) < 0.01

    def test_short_position_loss(self, portfolio, test_db):
        """AC-4: SHORT position closed at loss calculates negative P&L.

        Given: SHORT position opened at $60,000
        When: Closed at $65,000 with quantity 0.1
        Then: pnl = -$500.00, pnl_pct ≈ -8.33%
        """
        # Open SHORT at 60,000
        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        portfolio.execute_trade(sell_order, fill_price=60000.0)

        # Close SHORT at 65,000
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("65000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=65000.0)

        # Verify closed trade
        closed_trade = (
            test_db.query(Trade).filter(Trade.closed_at.isnot(None)).first()
        )

        assert closed_trade is not None
        assert closed_trade.pnl == Decimal("-500.00")
        assert abs(closed_trade.pnl_pct - (-8.33)) < 0.01


class TestOpeningPositionNoPnL:
    """Tests for opening positions (AC-5)."""

    def test_opening_position_has_no_pnl(self, portfolio, test_db):
        """AC-5: Opening position has pnl=None and pnl_pct=None.

        Given: No existing position
        When: New position is opened
        Then: trade.pnl = None, trade.pnl_pct = None
        """
        # Open new position
        buy_order = Order(
            symbol="ETH-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            filled_quantity=Decimal("1.0"),
            price=Decimal("3000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=3000.0)

        # Verify opening trade
        opening_trade = test_db.query(Trade).filter(Trade.closed_at.is_(None)).first()

        assert opening_trade is not None
        assert opening_trade.pnl is None
        assert opening_trade.pnl_pct is None

    def test_increasing_position_has_no_pnl(self, portfolio, test_db):
        """AC-5: Increasing existing position has pnl=None.

        Given: Existing LONG position
        When: Position is increased (additional BUY)
        Then: new trade has pnl=None, pnl_pct=None
        """
        # Open initial position
        buy_order_1 = Order(
            symbol="ETH-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            filled_quantity=Decimal("1.0"),
            price=Decimal("3000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order_1)
        test_db.commit()
        portfolio.execute_trade(buy_order_1, fill_price=3000.0)

        # Increase position
        buy_order_2 = Order(
            symbol="ETH-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            price=Decimal("3100"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order_2)
        test_db.commit()
        portfolio.execute_trade(buy_order_2, fill_price=3100.0)

        # Verify both trades have no pnl
        all_trades = test_db.query(Trade).filter(Trade.closed_at.is_(None)).all()

        assert len(all_trades) == 2
        for trade in all_trades:
            assert trade.pnl is None
            assert trade.pnl_pct is None


class TestPnLConstraints:
    """Tests for P&L calculation constraints and formulas."""

    def test_long_pnl_formula(self, portfolio, test_db):
        """Verify LONG P&L formula: (exit - entry) * quantity."""
        entry = 50000.0
        exit_price = 52000.0
        qty = 0.25

        # Open and close LONG
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal(str(qty)),
            filled_quantity=Decimal(str(qty)),
            price=Decimal(str(entry)),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=entry)

        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal(str(qty)),
            filled_quantity=Decimal(str(qty)),
            price=Decimal(str(exit_price)),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        portfolio.execute_trade(sell_order, fill_price=exit_price)

        closed_trade = (
            test_db.query(Trade).filter(Trade.closed_at.isnot(None)).first()
        )

        expected_pnl = Decimal(str((exit_price - entry) * qty))
        assert closed_trade.pnl == expected_pnl

    def test_short_pnl_formula(self, portfolio, test_db):
        """Verify SHORT P&L formula: (entry - exit) * quantity."""
        entry = 52000.0
        exit_price = 50000.0
        qty = 0.25

        # Open and close SHORT
        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal(str(qty)),
            filled_quantity=Decimal(str(qty)),
            price=Decimal(str(entry)),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        portfolio.execute_trade(sell_order, fill_price=entry)

        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal(str(qty)),
            filled_quantity=Decimal(str(qty)),
            price=Decimal(str(exit_price)),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=exit_price)

        closed_trade = (
            test_db.query(Trade).filter(Trade.closed_at.isnot(None)).first()
        )

        expected_pnl = Decimal(str((entry - exit_price) * qty))
        assert closed_trade.pnl == expected_pnl

    def test_pnl_pct_formula(self, portfolio, test_db):
        """Verify pnl_pct formula: ((exit - entry) / entry) * 100."""
        entry = 60000.0
        exit_price = 63000.0

        # Open and close LONG
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal(str(entry)),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=entry)

        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal(str(exit_price)),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        portfolio.execute_trade(sell_order, fill_price=exit_price)

        closed_trade = (
            test_db.query(Trade).filter(Trade.closed_at.isnot(None)).first()
        )

        expected_pnl_pct = ((exit_price - entry) / entry) * 100
        assert abs(closed_trade.pnl_pct - expected_pnl_pct) < 0.01


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_closed_trade_has_exit_price(self, portfolio, test_db):
        """Invariant: Closed trades always have exit_price set."""
        # Open and close
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=60000.0)

        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("65000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        portfolio.execute_trade(sell_order, fill_price=65000.0)

        closed_trade = (
            test_db.query(Trade).filter(Trade.closed_at.isnot(None)).first()
        )

        assert closed_trade.exit_price is not None
        assert closed_trade.exit_price > 0

    def test_opening_trade_has_no_exit_price(self, portfolio, test_db):
        """Invariant: Opening trades have exit_price = None."""
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        portfolio.execute_trade(buy_order, fill_price=60000.0)

        opening_trade = test_db.query(Trade).filter(Trade.closed_at.is_(None)).first()

        assert opening_trade.exit_price is None
