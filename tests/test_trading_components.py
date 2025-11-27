"""
Unit tests for trading components: PositionSizer, RiskManager, PaperBroker, OrderManager.
"""

import pytest
from unittest.mock import Mock, MagicMock
from decimal import Decimal

from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.trading.order_manager import OrderManager
from quantagent.models import Order, OrderSide, OrderStatus, OrderType


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
        self.risk_manager.daily_pnl_tracker[__import__('datetime').date.today()] = -6000.0

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
        today = __import__('datetime').date.today()
        self.risk_manager.daily_pnl_tracker[today] = -1000.0

        pnl = self.risk_manager.get_daily_pnl()
        assert pnl == -1000.0

    def test_on_trade_executed(self):
        """Test updating P&L after trade execution."""
        trade = Mock()
        trade.pnl = Decimal("500.00")

        self.risk_manager.on_trade_executed(trade)

        today = __import__('datetime').date.today()
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
        today = __import__('datetime').date.today()
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
            decision="HOLD",
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

        # First need a position to short
        self.portfolio.positions["BTC"] = {"qty": 1.0, "avg_cost": 42000.0}

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
        large_sizer.calculate_size.return_value = 1.0  # 1 BTC = $42,000 (420% of $10k portfolio!)
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
        assert result.status == OrderStatus.FILLED, f"Order status should be FILLED, got {result.status}"
        assert result.filled_at is not None, "Order must have fill timestamp"

        # Validate slippage was applied (proves broker executed)
        # BUY slippage: price * 1.01
        expected_fill_price = 42000.0 * 1.01
        assert result.average_fill_price == pytest.approx(expected_fill_price, rel=0.001), \
            f"Expected fill price {expected_fill_price}, got {result.average_fill_price}"

        # Validate quantity was sized correctly (proves position_sizer was called)
        # Expected qty = (portfolio_value * base_pct * confidence) / price
        expected_qty = (100000.0 * 0.05 * 0.8) / 42000.0
        assert result.filled_quantity == pytest.approx(expected_qty, rel=0.001), \
            f"Expected qty {expected_qty}, got {result.filled_quantity}"

        # Critical: verify portfolio AND database were updated (full chain executed)
        assert self.portfolio.execute_trade.called, "Portfolio.execute_trade should have been called"
        self.portfolio.execute_trade.assert_called_once()
        assert self.db.add.called, "Database.add should have been called to persist trade"
        self.db.add.assert_called()
        assert self.db.commit.called, "Database.commit should have been called"
        self.db.commit.assert_called()

    def test_full_flow_short_valid_trade_executes_all_steps(self):
        """Test SHORT decision executes complete chain: Size → Validate → Broker → Portfolio → DB."""
        # Setup existing position to short
        self.portfolio.positions["BTC"] = {"qty": 1.0, "avg_cost": 42000.0}

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
        assert result.status == OrderStatus.FILLED, f"Order status should be FILLED, got {result.status}"

        # Validate slippage was applied (proves broker executed)
        # SELL slippage: price * 0.99
        expected_fill_price = 42000.0 * 0.99
        assert result.average_fill_price == pytest.approx(expected_fill_price, rel=0.001), \
            f"Expected fill price {expected_fill_price}, got {result.average_fill_price}"

        # Validate quantity was sized correctly
        expected_qty = (100000.0 * 0.05 * 0.6) / 42000.0
        assert result.filled_quantity == pytest.approx(expected_qty, rel=0.001), \
            f"Expected qty {expected_qty}, got {result.filled_quantity}"

        # Critical: verify chain of execution
        # 1. Portfolio must be updated
        assert self.portfolio.execute_trade.called, "Portfolio.execute_trade should have been called"
        self.portfolio.execute_trade.assert_called_once()

        # 2. Database must be updated
        assert self.db.add.called, "Database.add should have been called to persist trade"
        assert self.db.commit.called, "Database.commit should have been called to finalize trade"

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
        large_sizer.calculate_size.return_value = 3.0  # 3 BTC = $126,000 (126% of portfolio!)
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
            Mock(pnl=Decimal("500.00")),   # +$500
            Mock(pnl=Decimal("-200.00")),  # -$200
            Mock(pnl=Decimal("300.00")),   # +$300
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
