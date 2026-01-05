"""
Unit tests for trading components: PositionSizer, RiskManager, PaperBroker, OrderManager.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from quantagent.models import (Order, OrderSide, OrderStatus, OrderType,
                               TradeSignal)
from quantagent.trading.order_manager import OrderManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager


class TestPositionSizer:
    """Test PositionSizer: calculates order size based on confidence."""

    def test_init_valid(self):
        """Test PositionSizer initialization with valid parameters."""
        sizer = PositionSizer(base_position_pct=0.05)
        assert sizer.base_position_pct == 0.05

    def test_init_invalid_pct(self):
        """Test PositionSizer rejects invalid position percentages."""
        with pytest.raises(ValueError, match="must be between 0% and 10%"):
            PositionSizer(base_position_pct=0.15)

        with pytest.raises(ValueError, match="must be between 0% and 10%"):
            PositionSizer(base_position_pct=0.0)

    def test_calculate_size_low_confidence(self):
        """Test position size calculation with low confidence (50%)."""
        sizer = PositionSizer(base_position_pct=0.05)

        # Portfolio: $100,000
        # Base: 5%, Confidence: 50%
        # Position value: $100,000 * 0.05 * 0.5 = $2,500
        # Price: $42,000
        # Qty: $2,500 / $42,000 ≈ 0.0595 BTC

        qty = sizer.calculate_size(
            symbol="BTC",
            signal_confidence=0.5,
            current_price=42000.0,
            portfolio_value=100000.0,
        )

        assert qty == pytest.approx(2500.0 / 42000.0, rel=0.001)
        assert qty == pytest.approx(0.0595, rel=0.01)

    def test_calculate_size_high_confidence(self):
        """Test position size calculation with high confidence (100%)."""
        sizer = PositionSizer(base_position_pct=0.05)

        # Portfolio: $100,000
        # Base: 5%, Confidence: 100%
        # Position value: $100,000 * 0.05 * 1.0 = $5,000
        # Price: $42,000
        # Qty: $5,000 / $42,000 ≈ 0.119 BTC

        qty = sizer.calculate_size(
            symbol="BTC",
            signal_confidence=1.0,
            current_price=42000.0,
            portfolio_value=100000.0,
        )

        assert qty == pytest.approx(5000.0 / 42000.0, rel=0.001)
        assert qty == pytest.approx(0.119, rel=0.01)

    def test_calculate_size_invalid_confidence(self):
        """Test position size calculation rejects invalid confidence."""
        sizer = PositionSizer(base_position_pct=0.05)

        with pytest.raises(ValueError, match="must be between 0 and 1.0"):
            sizer.calculate_size(
                symbol="BTC",
                signal_confidence=1.5,
                current_price=42000.0,
                portfolio_value=100000.0,
            )

    def test_calculate_size_invalid_price(self):
        """Test position size calculation rejects invalid price."""
        sizer = PositionSizer(base_position_pct=0.05)

        with pytest.raises(ValueError, match="must be positive"):
            sizer.calculate_size(
                symbol="BTC",
                signal_confidence=0.5,
                current_price=-100.0,
                portfolio_value=100000.0,
            )

    def test_calculate_size_invalid_portfolio_value(self):
        """Test position size calculation rejects invalid portfolio value."""
        sizer = PositionSizer(base_position_pct=0.05)

        with pytest.raises(ValueError, match="must be positive"):
            sizer.calculate_size(
                symbol="BTC",
                signal_confidence=0.5,
                current_price=42000.0,
                portfolio_value=0.0,
            )


class TestPaperBroker:
    """Test PaperBroker: simulates order execution with slippage."""

    def test_init(self):
        """Test PaperBroker initialization."""
        broker = PaperBroker(slippage_pct=0.01)
        assert broker.slippage_pct == 0.01

    def test_init_invalid_slippage(self):
        """Test PaperBroker rejects invalid slippage."""
        with pytest.raises(ValueError, match="should be between 0% and 5%"):
            PaperBroker(slippage_pct=0.10)

    def test_place_buy_order(self):
        """Test BUY order execution with slippage."""
        broker = PaperBroker(slippage_pct=0.01)

        order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            quantity=0.1,
            price=42000.0,
            order_type=OrderType.MARKET,
        )

        filled = broker.place_order(order)

        # BUY: fill_price = 42000 * (1 + 0.01) = 42420 (1% slippage)
        assert filled.average_fill_price == pytest.approx(42420.0, rel=0.001)
        assert filled.filled_quantity == pytest.approx(0.1, rel=0.001)
        assert filled.status == OrderStatus.FILLED
        assert filled.filled_at is not None

    def test_place_sell_order(self):
        """Test SELL order execution with slippage."""
        broker = PaperBroker(slippage_pct=0.01)

        order = Order(
            symbol="BTC",
            side=OrderSide.SELL,
            quantity=0.1,
            price=42000.0,
            order_type=OrderType.MARKET,
        )

        filled = broker.place_order(order)

        # SELL: fill_price = 42000 * (1 - 0.01) = 41580 (1% slippage)
        assert filled.average_fill_price == pytest.approx(41580.0, rel=0.001)
        assert filled.filled_quantity == pytest.approx(0.1, rel=0.001)
        assert filled.status == OrderStatus.FILLED
        assert filled.filled_at is not None

    def test_place_order_no_price(self):
        """Test order execution fails if price is not set."""
        broker = PaperBroker(slippage_pct=0.01)

        order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            quantity=0.1,
            price=None,
            order_type=OrderType.MARKET,
        )

        with pytest.raises(ValueError, match="Order price must be set"):
            broker.place_order(order)


class TestRiskManager:
    """Test RiskManager: validates trades before execution."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock PortfolioManager
        self.portfolio = Mock()
        self.portfolio.cash = 100000.0
        self.portfolio.positions = {}
        self.portfolio.get_total_value.return_value = 100000.0
        self.portfolio.get_unrealized_pnl.return_value = 0.0

        self.risk_manager = RiskManager(
            portfolio_manager=self.portfolio,
            max_daily_loss_pct=0.05,
            max_position_pct=0.10,
            db=None,
        )

    def test_init(self):
        """Test RiskManager initialization."""
        assert self.risk_manager.max_daily_loss_pct == 0.05
        assert self.risk_manager.max_position_pct == 0.10
        assert not self.risk_manager.circuit_breaker_triggered

    def test_validate_trade_valid(self):
        """Test validation passes for valid trade."""
        is_valid, reason = self.risk_manager.validate_trade(
            symbol="BTC",
            qty=0.1,
            price=42000.0,
        )

        assert is_valid is True
        assert reason is None

    def test_validate_trade_insufficient_capital(self):
        """Test validation fails if insufficient capital."""
        self.portfolio.cash = 1000.0  # Only $1,000

        is_valid, reason = self.risk_manager.validate_trade(
            symbol="BTC",
            qty=0.1,
            price=42000.0,  # Trade value: $4,200
        )

        assert is_valid is False
        assert "Insufficient capital" in reason

    def test_validate_trade_position_too_large(self):
        """Test validation fails if position exceeds 10% limit."""
        # Trade value: 0.3 * 42000 = $12,600 (12.6% of $100k)
        is_valid, reason = self.risk_manager.validate_trade(
            symbol="BTC",
            qty=0.3,
            price=42000.0,
        )

        assert is_valid is False
        assert "Position too large" in reason

    def test_validate_trade_daily_loss_exceeded(self):
        """Test validation fails if daily loss exceeded."""
        # Daily loss: -$6,000 (exceeds -5% of $100k = -$5,000)
        self.risk_manager.daily_pnl_tracker[__import__("datetime").date.today()] = (
            -6000.0
        )

        is_valid, reason = self.risk_manager.validate_trade(
            symbol="SPX",
            qty=1.0,
            price=5000.0,
        )

        assert is_valid is False
        assert "Daily loss limit exceeded" in reason

    def test_validate_trade_circuit_breaker_active(self):
        """Test validation fails if circuit breaker is active."""
        self.risk_manager.circuit_breaker_triggered = True

        is_valid, reason = self.risk_manager.validate_trade(
            symbol="BTC",
            qty=0.1,
            price=42000.0,
        )

        assert is_valid is False
        assert "Circuit breaker" in reason

    def test_get_daily_pnl_no_db(self):
        """Test daily P&L calculation without database."""
        today = __import__("datetime").date.today()
        self.risk_manager.daily_pnl_tracker[today] = -1000.0

        pnl = self.risk_manager.get_daily_pnl()
        assert pnl == -1000.0

    def test_on_trade_executed(self):
        """Test updating P&L after trade execution."""
        trade = Mock()
        trade.pnl = Decimal("500.00")

        self.risk_manager.on_trade_executed(trade)

        today = __import__("datetime").date.today()
        assert self.risk_manager.daily_pnl_tracker[today] == 500.0

    def test_on_trade_executed_triggers_circuit_breaker(self):
        """Test circuit breaker triggers on excessive loss."""
        self.portfolio.get_total_value.return_value = 100000.0

        # Trade that causes -6% loss
        trade = Mock()
        trade.pnl = Decimal("-6000.00")

        self.risk_manager.on_trade_executed(trade)

        assert self.risk_manager.circuit_breaker_triggered is True

    def test_reset_daily_tracker(self):
        """Test resetting daily tracker."""
        today = __import__("datetime").date.today()
        self.risk_manager.daily_pnl_tracker[today] = -1000.0
        self.risk_manager.circuit_breaker_triggered = True

        self.risk_manager.reset_daily_tracker()

        assert self.risk_manager.circuit_breaker_triggered is False
        assert self.risk_manager.daily_pnl_tracker[today] == 0.0


class TestOrderManager:
    """Test OrderManager: orchestrates order execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.position_sizer = PositionSizer(base_position_pct=0.05)

        # Mock dependencies
        self.portfolio = Mock()
        self.portfolio.cash = 100000.0
        self.portfolio.positions = {}
        self.portfolio.get_total_value.return_value = 100000.0

        self.risk_manager = RiskManager(self.portfolio, db=None)
        self.broker = PaperBroker(slippage_pct=0.01)
        self.db = Mock()

        self.order_manager = OrderManager(
            position_sizer=self.position_sizer,
            risk_manager=self.risk_manager,
            broker=self.broker,
            portfolio_manager=self.portfolio,
            db=self.db,
        )

    def test_execute_decision_hold(self):
        """Test HOLD decision returns None."""
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision=TradeSignal.NEUTRAL,
            confidence=0.8,
            current_price=42000.0,
        )

        assert result is None

    def test_execute_decision_long_valid(self):
        """Test LONG decision executes successfully."""
        # Mock portfolio.execute_trade to return a trade
        trade = Mock()
        trade.pnl = Decimal("0.00")
        self.portfolio.execute_trade.return_value = trade

        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        assert result is not None
        assert result.symbol == "BTC"
        assert result.side == OrderSide.BUY
        assert result.status == OrderStatus.FILLED

    def test_execute_decision_short_valid(self):
        """Test SHORT decision executes successfully."""
        # Mock portfolio.execute_trade to return a trade
        trade = Mock()
        trade.pnl = Decimal("0.00")
        self.portfolio.execute_trade.return_value = trade

        # No existing position (clean open SHORT)
        self.portfolio.positions = {}

        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="SHORT",
            confidence=0.8,
            current_price=42000.0,
        )

        assert result is not None
        assert result.side == OrderSide.SELL

    def test_execute_decision_insufficient_capital(self):
        """Test decision is rejected if insufficient capital."""
        self.portfolio.cash = 1000.0  # Not enough capital

        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        assert result is None
        # Verify broker was never called for invalid trade
        self.portfolio.execute_trade.assert_not_called()

    def test_execute_decision_position_too_large(self):
        """Test decision is rejected if position exceeds limit."""
        # Use custom position sizer that returns larger qty
        large_sizer = Mock()
        large_sizer.calculate_size.return_value = (
            1.0  # 1 BTC = $42,000 (420% of $10k portfolio!)
        )
        self.order_manager.position_sizer = large_sizer

        self.portfolio.get_total_value.return_value = 10000.0  # Smaller portfolio

        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=1.0,
            current_price=42000.0,
        )

        assert result is None
        # Verify portfolio.execute_trade was never called
        self.portfolio.execute_trade.assert_not_called()


class TestFullEndToEndIntegration:
    """
    Test full end-to-end integration: Decision → Size → Validate → Execute → Update → Log.

    This tests the complete workflow as specified in the Phase 1 roadmap (Week 5-6, Task 3.3).
    """

    def setup_method(self):
        """Set up test fixtures with real components."""
        self.position_sizer = PositionSizer(base_position_pct=0.05)

        # Mock portfolio with realistic state
        self.portfolio = Mock()
        self.portfolio.cash = 100000.0
        self.portfolio.positions = {}
        self.portfolio.get_total_value.return_value = 100000.0
        self.portfolio.get_unrealized_pnl.return_value = 0.0

        self.risk_manager = RiskManager(self.portfolio, db=None)
        self.broker = PaperBroker(slippage_pct=0.01)
        self.db = Mock()

        self.order_manager = OrderManager(
            position_sizer=self.position_sizer,
            risk_manager=self.risk_manager,
            broker=self.broker,
            portfolio_manager=self.portfolio,
            db=self.db,
        )

    def test_full_flow_long_valid_trade_executes_all_steps(self):
        """Test LONG decision executes complete chain: Size → Validate → Broker → Portfolio → DB."""
        # Mock portfolio.execute_trade to return a trade
        trade = Mock()
        trade.pnl = Decimal("500.00")
        self.portfolio.execute_trade.return_value = trade

        # Execute decision
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Critical validations: Order must be filled (reached broker)
        assert result is not None, "Valid LONG decision must return filled order"
        assert (
            result.status == OrderStatus.FILLED
        ), f"Order status should be FILLED, got {result.status}"
        assert result.filled_at is not None, "Order must have fill timestamp"

        # Validate slippage was applied (proves broker executed)
        # BUY slippage: price * 1.01
        expected_fill_price = 42000.0 * 1.01
        assert result.average_fill_price == pytest.approx(
            expected_fill_price, rel=0.001
        ), f"Expected fill price {expected_fill_price}, got {result.average_fill_price}"

        # Validate quantity was sized correctly (proves position_sizer was called)
        # Expected qty = (portfolio_value * base_pct * confidence) / price
        expected_qty = (100000.0 * 0.05 * 0.8) / 42000.0
        assert result.filled_quantity == pytest.approx(
            expected_qty, rel=0.001
        ), f"Expected qty {expected_qty}, got {result.filled_quantity}"

        # Critical: verify portfolio AND database were updated (full chain executed)
        assert (
            self.portfolio.execute_trade.called
        ), "Portfolio.execute_trade should have been called"
        self.portfolio.execute_trade.assert_called_once()
        assert (
            self.db.add.called
        ), "Database.add should have been called to persist trade"
        self.db.add.assert_called()
        assert self.db.commit.called, "Database.commit should have been called"
        self.db.commit.assert_called()

    def test_full_flow_short_valid_trade_executes_all_steps(self):
        """Test SHORT decision executes complete chain: Size → Validate → Broker → Portfolio → DB."""
        # No existing position (clean SHORT open)
        self.portfolio.positions = {}

        # Mock portfolio.execute_trade to return a trade
        trade = Mock()
        trade.pnl = Decimal("-200.00")  # Loss on this trade
        self.portfolio.execute_trade.return_value = trade

        # Execute SHORT decision
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="SHORT",
            confidence=0.6,
            current_price=42000.0,
        )

        # Critical validations: Order must be filled (reached broker)
        assert result is not None, "Valid SHORT decision must return filled order"
        assert result.side == OrderSide.SELL, "SHORT decision must create SELL order"
        assert (
            result.status == OrderStatus.FILLED
        ), f"Order status should be FILLED, got {result.status}"

        # Validate slippage was applied (proves broker executed)
        # SELL slippage: price * 0.99
        expected_fill_price = 42000.0 * 0.99
        assert result.average_fill_price == pytest.approx(
            expected_fill_price, rel=0.001
        ), f"Expected fill price {expected_fill_price}, got {result.average_fill_price}"

        # Validate quantity was sized correctly
        expected_qty = (100000.0 * 0.05 * 0.6) / 42000.0
        assert result.filled_quantity == pytest.approx(
            expected_qty, rel=0.001
        ), f"Expected qty {expected_qty}, got {result.filled_quantity}"

        # Critical: verify chain of execution
        # 1. Portfolio must be updated
        assert (
            self.portfolio.execute_trade.called
        ), "Portfolio.execute_trade should have been called"
        self.portfolio.execute_trade.assert_called_once()

        # 2. Database must be updated
        assert (
            self.db.add.called
        ), "Database.add should have been called to persist trade"
        assert (
            self.db.commit.called
        ), "Database.commit should have been called to finalize trade"

    def test_full_flow_invalid_trade_rejected_before_broker(self):
        """Test invalid trade is REJECTED before reaching broker (validation gate)."""
        # Setup insufficient capital
        self.portfolio.cash = 500.0  # Only $500, not enough for BTC order

        # Execute decision
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Verify execution was rejected
        assert result is None

        # CRITICAL: Verify portfolio.execute_trade was NEVER called
        # (Order never reached the broker or portfolio)
        self.portfolio.execute_trade.assert_not_called()

        # Verify database was NOT called (no trade to log)
        self.db.add.assert_not_called()
        self.db.commit.assert_not_called()

    def test_full_flow_position_too_large_rejected(self):
        """Test position size exceeding 10% limit is rejected before broker."""
        # Use custom position sizer returning large qty
        large_sizer = Mock()
        large_sizer.calculate_size.return_value = (
            3.0  # 3 BTC = $126,000 (126% of portfolio!)
        )
        self.order_manager.position_sizer = large_sizer

        # Execute decision
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=1.0,
            current_price=42000.0,
        )

        # Verify rejection
        assert result is None

        # Verify broker was never called for this invalid trade
        self.portfolio.execute_trade.assert_not_called()

    def test_full_flow_circuit_breaker_active(self):
        """Test circuit breaker prevents all trades if triggered."""
        # Trigger circuit breaker via large loss
        trade = Mock()
        trade.pnl = Decimal("-6000.00")  # 6% loss
        self.portfolio.execute_trade.return_value = trade

        # First trade should execute and trigger circuit breaker
        result1 = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )
        assert result1 is not None  # First trade succeeds

        # Now circuit breaker should be active
        assert self.risk_manager.circuit_breaker_triggered is True

        # Second trade should be rejected
        result2 = self.order_manager.execute_decision(
            symbol="SPX",
            decision="LONG",
            confidence=0.8,
            current_price=5000.0,
        )
        assert result2 is None  # Second trade rejected

    def test_broker_slippage_consistency(self):
        """Test that broker consistently applies 2% slippage (±1%)."""
        broker = PaperBroker(slippage_pct=0.01)

        # Test multiple BUY orders
        for price in [42000.0, 50000.0, 30000.0]:
            buy_order = Order(
                symbol="BTC",
                side=OrderSide.BUY,
                quantity=0.1,
                price=price,
                order_type=OrderType.MARKET,
            )
            filled = broker.place_order(buy_order)
            expected_price = price * 1.01
            assert filled.average_fill_price == pytest.approx(expected_price, rel=0.001)

        # Test multiple SELL orders
        for price in [42000.0, 50000.0, 30000.0]:
            sell_order = Order(
                symbol="BTC",
                side=OrderSide.SELL,
                quantity=0.1,
                price=price,
                order_type=OrderType.MARKET,
            )
            filled = broker.place_order(sell_order)
            expected_price = price * 0.99
            assert filled.average_fill_price == pytest.approx(expected_price, rel=0.001)

    def test_order_status_transitions(self):
        """Test proper order status transitions (PENDING → FILLED)."""
        order = Order(
            symbol="BTC",
            side=OrderSide.BUY,
            quantity=0.1,
            price=42000.0,
            order_type=OrderType.MARKET,
            status=OrderStatus.PENDING,  # Initial state
        )

        broker = PaperBroker(slippage_pct=0.01)
        filled = broker.place_order(order)

        # Verify status transition
        assert filled.status == OrderStatus.FILLED
        assert filled.filled_at is not None

    def test_daily_pnl_tracking_across_trades(self):
        """Test daily P&L tracking accumulates correctly across multiple trades."""
        # Mock multiple trades with different P&L
        self.portfolio.execute_trade.side_effect = [
            Mock(pnl=Decimal("500.00")),  # +$500
            Mock(pnl=Decimal("-200.00")),  # -$200
            Mock(pnl=Decimal("300.00")),  # +$300
        ]

        # Execute three trades
        for i in range(3):
            self.order_manager.execute_decision(
                symbol=["BTC", "SPX", "CL"][i],
                decision="LONG",
                confidence=0.8,
                current_price=42000.0,
            )

        # Verify daily P&L is accumulated
        daily_pnl = self.risk_manager.get_daily_pnl()
        # Expected: 500 - 200 + 300 = 600
        assert daily_pnl == pytest.approx(600.0, rel=0.001)

    def test_short_to_long_reversal(self):
        """Test reversal from SHORT to LONG position."""
        # Setup: existing SHORT position
        self.portfolio.positions = {
            "BTC": {
                "qty": -0.033094,
                "avg_cost": 105000.0,
                "current_price": 106000.0,
                "pnl": -33.09,
                "pnl_pct": -0.95,
            }
        }
        self.portfolio.cash = 103500.0

        # Mock portfolio.execute_trade to return trades
        close_trade = Mock()
        close_trade.pnl = Decimal("0.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")

        call_count = [0]

        def mock_execute_trade(order, fill_price):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: close SHORT position
                self.portfolio.positions["BTC"]["qty"] = 0.0
                return close_trade
            else:
                # Second call: open LONG position
                self.portfolio.positions["BTC"]["qty"] = 0.034277
                return open_trade

        self.portfolio.execute_trade.side_effect = mock_execute_trade

        # Execute: LONG decision should trigger reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision=TradeSignal.LONG,
            confidence=0.68,
            current_price=106045.33,
        )

        # Verify: reversal executed successfully
        assert result is not None
        assert self.portfolio.positions["BTC"]["qty"] > 0
        assert self.portfolio.execute_trade.call_count == 2
        assert self.db.add.call_count >= 2  # At least 2 orders + 2 trades

    def test_long_to_short_reversal(self):
        """Test reversal from LONG to SHORT position."""
        # Setup: existing LONG position
        self.portfolio.positions = {
            "ETH": {
                "qty": 2.5,
                "avg_cost": 3000.0,
                "current_price": 3100.0,
                "pnl": 250.0,
                "pnl_pct": 3.33,
            }
        }
        self.portfolio.cash = 92500.0

        # Mock portfolio.execute_trade
        close_trade = Mock()
        close_trade.pnl = Decimal("250.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")

        call_count = [0]

        def mock_execute_trade(order, fill_price):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: close LONG position
                self.portfolio.positions["ETH"]["qty"] = 0.0
                return close_trade
            else:
                # Second call: open SHORT position
                self.portfolio.positions["ETH"]["qty"] = -2.0
                return open_trade

        self.portfolio.execute_trade.side_effect = mock_execute_trade

        # Execute: SHORT decision should trigger reversal
        result = self.order_manager.execute_decision(
            symbol="ETH",
            decision=TradeSignal.SHORT,
            confidence=0.75,
            current_price=3100.0,
        )

        # Verify: reversal executed successfully
        assert result is not None
        assert self.portfolio.positions["ETH"]["qty"] < 0
        assert self.portfolio.execute_trade.call_count == 2

    def test_reversal_with_different_sizes(self):
        """Test reversal where close qty != open qty."""
        # Setup: small SHORT position
        self.portfolio.positions = {
            "BTC": {
                "qty": -0.01,
                "avg_cost": 100000.0,
                "current_price": 105000.0,
                "pnl": -50.0,
                "pnl_pct": -5.0,
            }
        }
        self.portfolio.cash = 99000.0

        # Mock portfolio.execute_trade
        close_trade = Mock()
        close_trade.pnl = Decimal("-50.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")

        call_count = [0]

        def mock_execute_trade(order, fill_price):
            call_count[0] += 1
            if call_count[0] == 1:
                # Close: BUY 0.01
                assert abs(order.quantity - 0.01) < 0.0001
                self.portfolio.positions["BTC"]["qty"] = 0.0
                return close_trade
            else:
                # Open: BUY calculated size (larger)
                assert order.quantity > 0.01
                self.portfolio.positions["BTC"]["qty"] = order.quantity
                return open_trade

        self.portfolio.execute_trade.side_effect = mock_execute_trade

        # Execute: LONG decision
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision=TradeSignal.LONG,
            confidence=0.80,
            current_price=105000.0,
        )

        # Verify: reversal completed with two different sizes
        assert result is not None
        assert self.portfolio.execute_trade.call_count == 2
        assert self.portfolio.positions["BTC"]["qty"] > 0.01

    def test_reversal_close_order_fails(self):
        """Test reversal stops if close order fails."""
        # Setup: SHORT position exists
        self.portfolio.positions = {
            "BTC": {
                "qty": -0.05,
                "avg_cost": 100000.0,
                "current_price": 105000.0,
                "pnl": -250.0,
                "pnl_pct": -5.0,
            }
        }

        # Mock broker to fail on first order (close)
        self.broker.place_order = Mock(side_effect=Exception("Broker error"))

        # Execute: LONG decision should attempt reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision=TradeSignal.LONG,
            confidence=0.70,
            current_price=105000.0,
        )

        # Verify: reversal failed, no portfolio update
        assert result is None
        assert self.portfolio.execute_trade.call_count == 0
        assert self.portfolio.positions["BTC"]["qty"] == -0.05  # unchanged

    def test_non_reversal_unchanged(self):
        """Test non-reversal trades still work as before."""
        # Setup: no existing position
        self.portfolio.positions = {}
        self.portfolio.cash = 100000.0

        # Mock portfolio.execute_trade
        trade = Mock()
        trade.pnl = Decimal("0.00")
        self.portfolio.execute_trade.return_value = trade

        # Execute: LONG decision (not a reversal)
        result = self.order_manager.execute_decision(
            symbol="SOL",
            decision=TradeSignal.LONG,
            confidence=0.70,
            current_price=150.0,
        )

        # Verify: single order executed
        assert result is not None
        assert self.portfolio.execute_trade.call_count == 1

    def test_reversal_order_objects_created(self):
        """Test correct Order objects are created during reversal (AC-1 enhanced)."""
        # Setup: SHORT position
        self.portfolio.positions = {
            "BTC": {
                "qty": -0.05,
                "avg_cost": 100000.0,
                "current_price": 105000.0,
                "pnl": -250.0,
                "pnl_pct": -5.0,
            }
        }
        self.portfolio.cash = 95000.0

        # Mock portfolio.execute_trade
        close_trade = Mock()
        close_trade.pnl = Decimal("-250.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")

        call_count = [0]

        def mock_execute_trade(order, fill_price):
            call_count[0] += 1
            if call_count[0] == 1:
                self.portfolio.positions["BTC"]["qty"] = 0.0
                return close_trade
            else:
                self.portfolio.positions["BTC"]["qty"] = order.quantity
                return open_trade

        self.portfolio.execute_trade.side_effect = mock_execute_trade

        # Execute reversal using TradeSignal enum
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision=TradeSignal.LONG,
            confidence=0.75,
            current_price=105000.0,
        )

        # Verify: result is not None
        assert result is not None

        # Verify: 2 orders + 2 trades were added to DB
        assert self.db.add.call_count >= 4

        # Extract created orders from mock calls
        created_orders = [
            call[0][0]
            for call in self.db.add.call_args_list
            if len(call[0]) > 0 and isinstance(call[0][0], Order)
        ]

        # Verify: at least 2 orders were created
        assert len(created_orders) >= 2

        # Verify first order (close SHORT)
        close_order = created_orders[0]
        assert close_order.side == OrderSide.BUY
        assert (
            abs(close_order.quantity - 0.05) < 0.0001
        ), f"Close order quantity should be 0.05, got {close_order.quantity}"
        assert close_order.symbol == "BTC"
        assert close_order.order_type == OrderType.MARKET

        # Verify second order (open LONG)
        open_order = created_orders[1]
        assert open_order.side == OrderSide.BUY
        assert open_order.quantity > 0, "Open order must have positive quantity"
        assert open_order.symbol == "BTC"
        assert open_order.order_type == OrderType.MARKET

    def test_reversal_broker_receives_correct_sequence(self):
        """Test broker receives orders in correct sequence (close then open)."""
        # Setup: LONG position
        self.portfolio.positions = {
            "ETH": {
                "qty": 2.5,
                "avg_cost": 3000.0,
                "current_price": 3100.0,
                "pnl": 250.0,
                "pnl_pct": 3.33,
            }
        }
        self.portfolio.cash = 92500.0

        # Spy on broker calls
        broker_calls = []

        def spy_place_order(order):
            broker_calls.append(order)
            # Return a filled order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = order.price * 0.99  # Simulate slippage
            order.filled_timestamp = datetime.now()
            return order

        self.broker.place_order = Mock(side_effect=spy_place_order)

        # Mock portfolio.execute_trade
        call_count = [0]

        def mock_execute_trade(order, fill_price):
            call_count[0] += 1
            if call_count[0] == 1:
                self.portfolio.positions["ETH"]["qty"] = 0.0
                trade = Mock()
                trade.pnl = Decimal("250.00")
                return trade
            else:
                self.portfolio.positions["ETH"]["qty"] = -order.quantity
                trade = Mock()
                trade.pnl = Decimal("0.00")
                return trade

        self.portfolio.execute_trade.side_effect = mock_execute_trade

        # Execute reversal using TradeSignal enum
        result = self.order_manager.execute_decision(
            symbol="ETH",
            decision=TradeSignal.SHORT,
            confidence=0.80,
            current_price=3100.0,
        )

        # Verify: reversal completed
        assert result is not None

        # Verify: broker received exactly 2 orders
        assert len(broker_calls) == 2, f"Expected 2 broker calls, got {len(broker_calls)}"

        # Verify first call: close LONG (SELL)
        close_order = broker_calls[0]
        assert close_order.side == OrderSide.SELL
        assert (
            abs(close_order.quantity - 2.5) < 0.0001
        ), f"Close order should sell 2.5 ETH, got {close_order.quantity}"
        assert close_order.symbol == "ETH"

        # Verify second call: open SHORT (SELL)
        open_order = broker_calls[1]
        assert open_order.side == OrderSide.SELL
        assert open_order.quantity > 0, "Open order must have positive quantity"
        assert open_order.symbol == "ETH"

    def test_reversal_using_tradesiganl_enum(self):
        """Test reversal works correctly with TradeSignal enum (not strings)."""
        # Setup: SHORT position
        self.portfolio.positions = {
            "BTC": {
                "qty": -0.02,
                "avg_cost": 100000.0,
                "current_price": 102000.0,
                "pnl": -40.0,
                "pnl_pct": -2.0,
            }
        }

        # Mock portfolio.execute_trade
        call_count = [0]

        def mock_execute_trade(order, fill_price):
            call_count[0] += 1
            if call_count[0] == 1:
                self.portfolio.positions["BTC"]["qty"] = 0.0
                trade = Mock()
                trade.pnl = Decimal("-40.00")
                return trade
            else:
                self.portfolio.positions["BTC"]["qty"] = order.quantity
                trade = Mock()
                trade.pnl = Decimal("0.00")
                return trade

        self.portfolio.execute_trade.side_effect = mock_execute_trade

        # Execute using TradeSignal.LONG enum (not string)
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision=TradeSignal.LONG,
            confidence=0.65,
            current_price=102000.0,
        )

        # Verify: execution succeeded
        assert result is not None
        assert self.portfolio.positions["BTC"]["qty"] > 0
        assert self.portfolio.execute_trade.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
