"""Unit tests for QuantAgent-les: Commission Support in P&L Calculation.

Tests validate that commissions are correctly:
- Calculated based on commission model (none/fixed/pct)
- Persisted in Fill and Trade records
- Deducted from gross P&L to produce net P&L
- Reflected in pnl_pct calculations

Following TESTING_PATTERNS.md and acceptance criteria in:
docs/05_acceptance_tests/QuantAgent-les-AC-commissions-pnl.md
"""

from decimal import Decimal

import pytest
from sqlalchemy import create_engine
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
from quantagent.trading.paper_broker import PaperBroker


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
        initial_cash=100000.0, environment=Environment.BACKTEST, db=test_db
    )


class TestCommissionPersistence:
    """Tests for AC-1: Commission is persisted on trades."""

    def test_commission_persisted_on_trade(self, portfolio, test_db):
        """Given filled order with commission, Trade.commission equals that value."""
        broker = PaperBroker(commission_model="fixed", commission_fixed=10.0)

        order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.PENDING,
            environment=Environment.BACKTEST,
        )
        test_db.add(order)
        test_db.commit()

        # Broker fills order and creates Fill with commission
        filled_order = broker.place_order(order)
        test_db.commit()

        # Execute trade
        portfolio.execute_trade(filled_order, fill_price=float(filled_order.average_fill_price))

        # Verify commission persisted
        trade = test_db.query(Trade).filter(Trade.order_id == order.id).first()
        assert trade is not None
        assert trade.commission == Decimal("10.0")


class TestNetPnLLongClose:
    """Tests for AC-2: Net P&L is reduced by commission (LONG close)."""

    def test_long_close_with_commission(self, portfolio, test_db):
        """Given LONG close with $10 commission, net P&L = gross - commission."""
        broker = PaperBroker(slippage_pct=0, commission_model="fixed", commission_fixed=10.0)

        # Open LONG
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.PENDING,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        filled_buy = broker.place_order(buy_order)
        test_db.commit()
        portfolio.execute_trade(filled_buy, fill_price=60000.0)

        # Close LONG at $65,000
        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("65000"),
            status=OrderStatus.PENDING,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        filled_sell = broker.place_order(sell_order)
        test_db.commit()
        portfolio.execute_trade(filled_sell, fill_price=65000.0)

        # Verify net P&L
        closing_trade = (
            test_db.query(Trade)
            .filter(Trade.closed_at.isnot(None))
            .first()
        )
        assert closing_trade is not None
        assert closing_trade.commission == Decimal("10.0")
        
        # Gross P&L = (65000 - 60000) * 0.1 = $500
        # Net P&L = $500 - $10 = $490
        expected_pnl = Decimal("490.0")
        assert closing_trade.pnl == expected_pnl


class TestNetPnLShortClose:
    """Tests for AC-3: Net P&L is reduced by commission (SHORT close)."""

    def test_short_close_with_commission(self, portfolio, test_db):
        """Given SHORT close with $10 commission, net P&L = gross - commission."""
        broker = PaperBroker(slippage_pct=0, commission_model="fixed", commission_fixed=10.0)

        # Open SHORT
        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("65000"),
            status=OrderStatus.PENDING,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        filled_sell = broker.place_order(sell_order)
        test_db.commit()
        portfolio.execute_trade(filled_sell, fill_price=65000.0)

        # Close SHORT at $60,000
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.PENDING,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        filled_buy = broker.place_order(buy_order)
        test_db.commit()
        portfolio.execute_trade(filled_buy, fill_price=60000.0)

        # Verify net P&L
        closing_trade = (
            test_db.query(Trade)
            .filter(Trade.closed_at.isnot(None))
            .first()
        )
        assert closing_trade is not None
        assert closing_trade.commission == Decimal("10.0")
        
        # Gross P&L = (65000 - 60000) * 0.1 = $500
        # Net P&L = $500 - $10 = $490
        expected_pnl = Decimal("490.0")
        assert closing_trade.pnl == expected_pnl


class TestNetPnLPercent:
    """Tests for AC-4: Net pnl_pct reflects costs."""

    def test_pnl_pct_includes_commission(self, portfolio, test_db):
        """Given closing trade with commission, pnl_pct = (net_pnl / entry_notional) * 100."""
        broker = PaperBroker(slippage_pct=0, commission_model="fixed", commission_fixed=10.0)

        # Open LONG at $60,000
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.PENDING,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        filled_buy = broker.place_order(buy_order)
        test_db.commit()
        portfolio.execute_trade(filled_buy, fill_price=60000.0)

        # Close LONG at $65,000
        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("65000"),
            status=OrderStatus.PENDING,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        filled_sell = broker.place_order(sell_order)
        test_db.commit()
        portfolio.execute_trade(filled_sell, fill_price=65000.0)

        # Verify pnl_pct
        closing_trade = (
            test_db.query(Trade)
            .filter(Trade.closed_at.isnot(None))
            .first()
        )
        assert closing_trade is not None
        
        # Net P&L = $490, Entry notional = $6,000
        # pnl_pct = (490 / 6000) * 100 = 8.1667%
        expected_pnl_pct = (Decimal("490") / Decimal("6000")) * 100
        assert closing_trade.pnl_pct is not None
        assert abs(closing_trade.pnl_pct - float(expected_pnl_pct)) < 0.01


class TestCommissionModelFixed:
    """Tests for AC-5: Commission model = fixed fee."""

    def test_fixed_commission_model(self, test_db):
        """Given fixed commission model, commission equals fixed fee."""
        broker = PaperBroker(commission_model="fixed", commission_fixed=2.50)

        order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.PENDING,
            environment=Environment.BACKTEST,
        )
        test_db.add(order)
        test_db.commit()

        filled_order = broker.place_order(order)
        test_db.commit()

        assert len(filled_order.fills) == 1
        assert filled_order.fills[0].commission == Decimal("2.50")


class TestCommissionModelPercentage:
    """Tests for AC-6: Commission model = percentage of notional."""

    def test_percentage_commission_model(self, test_db):
        """Given 0.10% commission, $10,000 notional results in $10 commission."""
        broker = PaperBroker(
            slippage_pct=0,
            commission_model="pct",
            commission_pct=0.001  # 0.1%
        )

        order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),  # 0.1 BTC
            price=Decimal("100000"),  # $100,000 per BTC
            status=OrderStatus.PENDING,
            environment=Environment.BACKTEST,
        )
        test_db.add(order)
        test_db.commit()

        filled_order = broker.place_order(order)
        test_db.commit()

        # Notional = 0.1 * 100,000 = $10,000
        # Commission = $10,000 * 0.001 = $10
        assert len(filled_order.fills) == 1
        assert filled_order.fills[0].commission == Decimal("10.0")


class TestCommissionDefaults:
    """Tests for AC-7: Defaults preserve prior behavior."""

    def test_default_no_commission(self, portfolio, test_db):
        """Given no commission config, commission = 0 and P&L equals prior behavior."""
        broker = PaperBroker()  # Default: commission_model="none"

        # Open LONG
        buy_order = Order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("60000"),
            status=OrderStatus.PENDING,
            environment=Environment.BACKTEST,
        )
        test_db.add(buy_order)
        test_db.commit()
        filled_buy = broker.place_order(buy_order)
        test_db.commit()
        portfolio.execute_trade(filled_buy, fill_price=float(filled_buy.average_fill_price))

        # Close LONG
        sell_order = Order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("65000"),
            status=OrderStatus.PENDING,
            environment=Environment.BACKTEST,
        )
        test_db.add(sell_order)
        test_db.commit()
        filled_sell = broker.place_order(sell_order)
        test_db.commit()
        portfolio.execute_trade(filled_sell, fill_price=float(filled_sell.average_fill_price))

        # Verify zero commission
        closing_trade = (
            test_db.query(Trade)
            .filter(Trade.closed_at.isnot(None))
            .first()
        )
        assert closing_trade is not None
        assert closing_trade.commission == Decimal("0")
        
        # P&L should be gross (no commission deduction)
        # With 1% slippage: buy at ~60600, sell at ~64350
        # Approximate P&L around $375 (exact value depends on slippage)
        assert closing_trade.pnl is not None
        assert closing_trade.pnl > 0
