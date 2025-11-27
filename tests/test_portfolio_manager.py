"""Unit tests for PortfolioManager class.

Following TESTING_PATTERNS.md:
- Validate structure and type
- Validate constraints (calculations, ranges)
- Validate error handling
- Validate state preservation
- Test edge cases (empty data, extreme values)
"""

import pytest
from decimal import Decimal
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.models import (
    Order, OrderSide, OrderType, OrderStatus, Trade, Position,
    Signal, TradeSignal, Environment, Base
)
from quantagent.portfolio.manager import PortfolioManager
from quantagent.database import SessionLocal


@pytest.fixture
def test_db():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    db = TestSession()
    yield db
    db.close()


@pytest.fixture
def portfolio(test_db):
    """Create portfolio manager with initial capital."""
    return PortfolioManager(
        initial_cash=100000.0,
        environment=Environment.PAPER,
        db=test_db
    )


@pytest.fixture
def sample_buy_order(test_db):
    """Create a sample BUY order."""
    order = Order(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.5"),
        price=Decimal("42000.0"),
        status=OrderStatus.PENDING,
        environment=Environment.PAPER,
    )
    test_db.add(order)
    test_db.commit()
    test_db.refresh(order)
    return order


class TestPortfolioManagerStructure:
    """Test PortfolioManager type and structure."""

    def test_portfolio_is_instance(self, portfolio):
        """Verify PortfolioManager instance is created correctly."""
        assert isinstance(portfolio, PortfolioManager)
        assert hasattr(portfolio, "cash")
        assert hasattr(portfolio, "positions")
        assert hasattr(portfolio, "environment")

    def test_initial_state(self, portfolio):
        """Verify initial portfolio state is correct."""
        assert portfolio.cash == 100000.0
        assert portfolio.initial_cash == 100000.0
        assert len(portfolio.positions) == 0
        assert portfolio.environment == Environment.PAPER

    def test_required_methods_exist(self, portfolio):
        """Verify all required methods exist."""
        required_methods = [
            "execute_trade",
            "get_total_value",
            "get_unrealized_pnl",
            "get_realized_pnl",
            "update_prices",
            "get_position",
            "get_positions",
            "get_cash",
        ]
        for method in required_methods:
            assert hasattr(portfolio, method)
            assert callable(getattr(portfolio, method))


class TestPortfolioManagerBuySell:
    """Test buying and selling operations."""

    def test_execute_buy_creates_position(self, portfolio, test_db, sample_buy_order):
        """Verify BUY order creates new position."""
        sample_buy_order.filled_quantity = Decimal("0.5")
        test_db.commit()

        trade = portfolio.execute_trade(sample_buy_order, fill_price=42000.0)

        # Validate trade was created
        assert isinstance(trade, Trade)
        assert trade.symbol == "BTC"
        assert trade.quantity == Decimal("0.5")

        # Validate position was created
        assert "BTC" in portfolio.positions
        pos = portfolio.positions["BTC"]
        assert pos["qty"] == 0.5
        assert pos["avg_cost"] == 42000.0

    def test_execute_buy_reduces_cash(self, portfolio, test_db, sample_buy_order):
        """Verify BUY order reduces available cash."""
        initial_cash = portfolio.cash
        sample_buy_order.filled_quantity = Decimal("0.5")
        test_db.commit()

        portfolio.execute_trade(sample_buy_order, fill_price=42000.0)

        expected_spend = 0.5 * 42000.0
        assert portfolio.cash == pytest.approx(initial_cash - expected_spend)

    def test_execute_multiple_buys_average_cost(self, portfolio, test_db):
        """Verify average cost calculation with multiple buys."""
        # First buy: 0.5 BTC @ 42000
        order1 = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            price=Decimal("42000.0"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(order1)
        test_db.commit()
        test_db.refresh(order1)

        portfolio.execute_trade(order1, fill_price=42000.0)

        # Second buy: 0.5 BTC @ 44000
        order2 = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            price=Decimal("44000.0"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(order2)
        test_db.commit()
        test_db.refresh(order2)

        portfolio.execute_trade(order2, fill_price=44000.0)

        # Average cost should be (42000 + 44000) / 2 = 43000
        pos = portfolio.positions["BTC"]
        assert pos["avg_cost"] == pytest.approx(43000.0)
        assert pos["qty"] == pytest.approx(1.0)

    def test_execute_sell_requires_position(self, portfolio, test_db):
        """Verify SELL order fails without existing position."""
        sell_order = Order(
            symbol="BTC",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(sell_order)
        test_db.commit()
        test_db.refresh(sell_order)

        with pytest.raises(ValueError, match="No position"):
            portfolio.execute_trade(sell_order, fill_price=44000.0)

    def test_execute_sell_closes_position(self, portfolio, test_db):
        """Verify SELL order reduces position to zero."""
        # Setup: Buy 0.5 BTC
        buy_order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(buy_order)
        test_db.commit()
        test_db.refresh(buy_order)
        portfolio.execute_trade(buy_order, fill_price=42000.0)

        # Sell: 0.5 BTC
        sell_order = Order(
            symbol="BTC",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(sell_order)
        test_db.commit()
        test_db.refresh(sell_order)

        portfolio.execute_trade(sell_order, fill_price=44000.0)

        pos = portfolio.positions["BTC"]
        assert pos["qty"] == pytest.approx(0.0)
        assert pos["avg_cost"] == pytest.approx(0.0)


class TestPortfolioManagerCalculations:
    """Test P&L and value calculations."""

    def test_get_total_value_cash_only(self, portfolio):
        """Verify total value calculation with no positions."""
        total = portfolio.get_total_value()
        assert total == pytest.approx(100000.0)

    def test_get_total_value_with_positions(self, portfolio, test_db):
        """Verify total value = cash + position values."""
        # Buy 0.5 BTC @ 42000
        buy_order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(buy_order)
        test_db.commit()
        test_db.refresh(buy_order)
        portfolio.execute_trade(buy_order, fill_price=42000.0)

        # Update price to 44000
        portfolio.update_prices({"BTC": 44000.0})

        # Expected: 100000 - 21000 (spent) + 22000 (position value) = 101000
        expected = 100000.0 - 21000.0 + 22000.0
        assert portfolio.get_total_value() == pytest.approx(expected)

    def test_get_unrealized_pnl_profit(self, portfolio, test_db):
        """Verify unrealized P&L calculation with profit."""
        # Buy 0.5 BTC @ 42000
        buy_order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(buy_order)
        test_db.commit()
        test_db.refresh(buy_order)
        portfolio.execute_trade(buy_order, fill_price=42000.0)

        # Price goes to 44000
        portfolio.update_prices({"BTC": 44000.0})

        # P&L = 0.5 * (44000 - 42000) = 1000
        unrealized = portfolio.get_unrealized_pnl()
        assert unrealized == pytest.approx(1000.0)

    def test_get_unrealized_pnl_loss(self, portfolio, test_db):
        """Verify unrealized P&L calculation with loss."""
        # Buy 0.5 BTC @ 42000
        buy_order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(buy_order)
        test_db.commit()
        test_db.refresh(buy_order)
        portfolio.execute_trade(buy_order, fill_price=42000.0)

        # Price drops to 40000
        portfolio.update_prices({"BTC": 40000.0})

        # P&L = 0.5 * (40000 - 42000) = -1000
        unrealized = portfolio.get_unrealized_pnl()
        assert unrealized == pytest.approx(-1000.0)

    def test_get_position_exists(self, portfolio, test_db):
        """Verify get_position returns position data."""
        # Setup position
        buy_order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(buy_order)
        test_db.commit()
        test_db.refresh(buy_order)
        portfolio.execute_trade(buy_order, fill_price=42000.0)

        pos = portfolio.get_position("BTC")
        assert pos is not None
        assert pos["qty"] == pytest.approx(0.5)
        assert pos["avg_cost"] == pytest.approx(42000.0)

    def test_get_position_not_found(self, portfolio):
        """Verify get_position returns None for non-existent position."""
        pos = portfolio.get_position("ETH")
        assert pos is None


class TestPortfolioManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_insufficient_cash_buy(self, portfolio, test_db):
        """Verify error when buying with insufficient cash."""
        # Try to buy 10 BTC @ 50000 (500k > 100k cash)
        buy_order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),
            filled_quantity=Decimal("10.0"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(buy_order)
        test_db.commit()
        test_db.refresh(buy_order)

        with pytest.raises(ValueError, match="insufficient"):
            portfolio.execute_trade(buy_order, fill_price=50000.0)

    def test_multiple_positions(self, portfolio, test_db):
        """Verify portfolio handles multiple symbol positions."""
        # Buy 0.5 BTC @ 42000
        btc_order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(btc_order)
        test_db.commit()
        test_db.refresh(btc_order)
        portfolio.execute_trade(btc_order, fill_price=42000.0)

        # Buy 100 SPX @ 5000
        spx_order = Order(
            symbol="SPX",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            filled_quantity=Decimal("100"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(spx_order)
        test_db.commit()
        test_db.refresh(spx_order)
        portfolio.execute_trade(spx_order, fill_price=500.0)

        positions = portfolio.get_positions()
        assert len(positions) == 2
        assert "BTC" in positions
        assert "SPX" in positions

    def test_update_prices_multiple_symbols(self, portfolio, test_db):
        """Verify update_prices works for multiple symbols."""
        # Setup positions
        btc_order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(btc_order)
        test_db.commit()
        test_db.refresh(btc_order)
        portfolio.execute_trade(btc_order, fill_price=42000.0)

        spx_order = Order(
            symbol="SPX",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            filled_quantity=Decimal("100"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(spx_order)
        test_db.commit()
        test_db.refresh(spx_order)
        portfolio.execute_trade(spx_order, fill_price=500.0)

        # Update both prices
        portfolio.update_prices({"BTC": 44000.0, "SPX": 5500.0})

        assert portfolio.positions["BTC"]["current_price"] == pytest.approx(44000.0)
        assert portfolio.positions["SPX"]["current_price"] == pytest.approx(5500.0)

    def test_empty_positions_get_positions(self, portfolio):
        """Verify get_positions returns empty dict when no positions."""
        positions = portfolio.get_positions()
        assert isinstance(positions, dict)
        assert len(positions) == 0

    def test_cash_decreases_with_buys(self, portfolio, test_db):
        """Verify cash balance decreases with multiple buys."""
        initial_cash = portfolio.cash

        # Buy 1: 0.5 BTC @ 42000
        order1 = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(order1)
        test_db.commit()
        test_db.refresh(order1)
        portfolio.execute_trade(order1, fill_price=42000.0)
        cash_after_1 = portfolio.cash

        # Buy 2: 100 SPX @ 5000
        order2 = Order(
            symbol="SPX",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            filled_quantity=Decimal("100"),
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(order2)
        test_db.commit()
        test_db.refresh(order2)
        portfolio.execute_trade(order2, fill_price=500.0)
        cash_after_2 = portfolio.cash

        assert cash_after_1 < initial_cash
        assert cash_after_2 < cash_after_1
