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
        # Net P&L = $500 - $10 de entrada - $10 de salida = $480 (QuantAgent-j62)
        expected_pnl = Decimal("480.0")
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
        # Net P&L = $500 - $10 de entrada - $10 de salida = $480 (QuantAgent-j62)
        expected_pnl = Decimal("480.0")
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

        # Net P&L = $480 (resta las dos comisiones, QuantAgent-j62), Entry notional = $6,000
        # pnl_pct = (480 / 6000) * 100 = 8.0%
        expected_pnl_pct = (Decimal("480") / Decimal("6000")) * 100
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


class TestCommissionChargedOnceInCashAndPnl:
    """QuantAgent-j62: cada comisión baja el efectivo y el PnL del trade, una sola vez."""

    def test_round_trip_nets_both_commissions(self, portfolio, test_db):
        """Valida efectivo, PnL de la fila del run e identidad equity - capital = pnl."""
        from types import SimpleNamespace

        from quantagent.backtesting.backtest import Backtest

        c, qty, entry, exit_ = Decimal("0.001"), Decimal("10"), Decimal("100"), Decimal("110")
        broker = PaperBroker(slippage_pct=0, commission_model="pct", commission_pct=float(c))

        def fill(side, price):
            order = Order(
                symbol="SPY",
                side=side,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=price,
                status=OrderStatus.PENDING,
                environment=Environment.BACKTEST,
            )
            test_db.add(order)
            test_db.commit()
            broker.place_order(order)
            return order, portfolio.execute_trade(order, fill_price=float(price))

        _, opening_trade = fill(OrderSide.BUY, entry)
        entry_commission = c * qty * entry  # 1.00
        assert portfolio.cash == pytest.approx(float(100000 - qty * entry - entry_commission))

        close_order, _ = fill(OrderSide.SELL, exit_)
        exit_commission = c * qty * exit_  # 1.10
        expected_pnl = (exit_ - entry) * qty - entry_commission - exit_commission  # 97.90
        assert portfolio.cash == pytest.approx(float(100000 + expected_pnl))

        # El motor deja una sola fila por trade: la de apertura con los datos del cierre.
        position = SimpleNamespace(trade_id=opening_trade.id, closed_at=None)
        Backtest._sync_linked_trade_exit(
            SimpleNamespace(db=test_db), position, "take_profit", float(exit_), close_order
        )

        rows = test_db.query(Trade).all()
        assert len(rows) == 1
        assert float(rows[0].pnl) == pytest.approx(float(expected_pnl))
        assert rows[0].pnl_pct == pytest.approx(float(expected_pnl / (entry * qty) * 100))
        assert portfolio.cash - portfolio.initial_cash == pytest.approx(float(rows[0].pnl))


class TestEntryCommissionNettedByPortfolio:
    """QuantAgent-j62: el camino de paper (PortfolioManager + PaperBroker, sin motor de backtest)."""

    C, ENTRY, EXIT = Decimal("0.001"), Decimal("100"), Decimal("110")

    def _fill(self, test_db, paper, side, qty, price):
        order = Order(
            symbol="SPY",
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty,
            price=price,
            status=OrderStatus.PENDING,
            environment=Environment.PAPER,
        )
        test_db.add(order)
        test_db.commit()
        PaperBroker(slippage_pct=0, commission_model="pct", commission_pct=float(self.C)).place_order(order)
        return paper.execute_trade(order, fill_price=float(price))

    def test_closing_trade_pnl_nets_entry_and_exit_commission(self, test_db):
        """Valida que el pnl que devuelve execute_trade al cerrar ya resta las dos comisiones."""
        paper = PortfolioManager(initial_cash=100000.0, environment=Environment.PAPER, db=test_db)
        qty = Decimal("10")
        self._fill(test_db, paper, OrderSide.BUY, qty, self.ENTRY)
        closing = self._fill(test_db, paper, OrderSide.SELL, qty, self.EXIT)

        expected = (self.EXIT - self.ENTRY) * qty - self.C * qty * self.ENTRY - self.C * qty * self.EXIT  # 97.90
        assert float(closing.pnl) == pytest.approx(float(expected))
        assert closing.pnl_pct == pytest.approx(float(expected / (self.ENTRY * qty) * 100))
        assert paper.cash - paper.initial_cash == pytest.approx(float(expected))

    def test_partial_closes_split_entry_commission_by_quantity(self, test_db):
        """Valida el cierre en dos mitades: cada una carga media comisión de entrada y no queda resto."""
        paper = PortfolioManager(initial_cash=100000.0, environment=Environment.PAPER, db=test_db)
        qty, half, exit_2 = Decimal("10"), Decimal("5"), Decimal("120")
        self._fill(test_db, paper, OrderSide.BUY, qty, self.ENTRY)
        entry_commission = self.C * qty * self.ENTRY  # 1.00
        first = self._fill(test_db, paper, OrderSide.SELL, half, self.EXIT)
        second = self._fill(test_db, paper, OrderSide.SELL, half, exit_2)

        expected_first = (self.EXIT - self.ENTRY) * half - entry_commission / 2 - self.C * half * self.EXIT  # 48.95
        expected_second = (exit_2 - self.ENTRY) * half - entry_commission / 2 - self.C * half * exit_2  # 98.90
        assert float(first.pnl) == pytest.approx(float(expected_first))
        assert float(second.pnl) == pytest.approx(float(expected_second))
        assert paper.cash - paper.initial_cash == pytest.approx(float(expected_first + expected_second))
        assert paper.positions["SPY"]["entry_commission"] == 0
