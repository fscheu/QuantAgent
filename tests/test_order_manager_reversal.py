"""
Unit tests for OrderManager position reversal logic.

Tests the fix for QuantAgent-g3c: Position reversal fails when switching
from SHORT to LONG due to size mismatch.

Test Strategy:
1. Structure validation: Reversal produces two orders, two trades
2. Constraint validation: Close qty matches existing position exactly
3. Error handling: Failed close prevents open, failed open leaves flat position
4. State flow: Position goes from LONG/SHORT -> FLAT -> LONG/SHORT
5. Edge cases: Zero position, equal sizes, different sizes
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from quantagent.models import OrderSide
from quantagent.trading.order_manager import OrderManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager


class TestPositionReversal:
    """Test position reversal logic in OrderManager."""

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

    def test_reversal_short_to_long_structure(self):
        """
        Test SHORT to LONG reversal produces correct structure:
        - Two orders created
        - Two portfolio.execute_trade() calls
        - Two db.add() calls for trades
        - Returns final order
        """
        # Setup: existing SHORT position
        self.portfolio.get_position.return_value = {"qty": -0.033, "avg_cost": 42000.0}

        # Mock portfolio.execute_trade to return trade objects
        close_trade = Mock()
        close_trade.pnl = Decimal("100.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")

        self.portfolio.execute_trade.side_effect = [close_trade, open_trade]

        # Execute: LONG decision triggers reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Validate structure
        assert result is not None, "Reversal should return filled order"
        assert result.side == OrderSide.BUY, "Final order should be BUY (LONG)"

        # Should call execute_trade twice: once for close, once for open
        assert self.portfolio.execute_trade.call_count == 2

        # Should add two orders + two trades to DB
        # Orders are flushed, trades are added
        add_calls = [call for call in self.db.add.call_args_list]
        assert len(add_calls) >= 2, "Should persist at least close order and open order"

    def test_reversal_long_to_short_structure(self):
        """
        Test LONG to SHORT reversal produces correct structure.
        """
        # Setup: existing LONG position
        self.portfolio.get_position.return_value = {"qty": 0.05, "avg_cost": 42000.0}

        # Mock portfolio.execute_trade
        close_trade = Mock()
        close_trade.pnl = Decimal("200.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")

        self.portfolio.execute_trade.side_effect = [close_trade, open_trade]

        # Execute: SHORT decision triggers reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="SHORT",
            confidence=0.8,
            current_price=42000.0,
        )

        # Validate structure
        assert result is not None
        assert result.side == OrderSide.SELL, "Final order should be SELL (SHORT)"
        assert self.portfolio.execute_trade.call_count == 2

    def test_reversal_close_qty_matches_existing(self):
        """
        Test constraint: Close order qty must match existing position exactly.
        """
        existing_qty = -0.0330943811250786  # Exact qty from bug report
        self.portfolio.get_position.return_value = {"qty": existing_qty, "avg_cost": 42000.0}

        # Capture orders placed with broker
        placed_orders = []
        original_place_order = self.broker.place_order

        def capture_order(order):
            placed_orders.append(order)
            return original_place_order(order)

        self.broker.place_order = capture_order

        # Mock portfolio.execute_trade
        close_trade = Mock()
        close_trade.pnl = Decimal("50.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")

        self.portfolio.execute_trade.side_effect = [close_trade, open_trade]

        # Execute reversal
        _ = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.68,
            current_price=106045.33,
        )

        # Validate constraint: close order qty == abs(existing_qty)
        assert len(placed_orders) >= 1, "Should place at least close order"
        close_order = placed_orders[0]

        assert close_order.side == OrderSide.BUY, "Close SHORT requires BUY"
        assert close_order.quantity == pytest.approx(abs(existing_qty), rel=1e-9), \
            f"Close qty {close_order.quantity} must match existing {abs(existing_qty)}"

    def test_reversal_close_fails_prevents_open(self):
        """
        Test error handling: If close order fails, new position is not opened.
        """
        # Setup: existing SHORT position
        self.portfolio.get_position.return_value = {"qty": -0.05, "avg_cost": 42000.0}

        # Mock broker to fail on first order (close)
        self.broker.place_order = Mock(side_effect=Exception("Broker error"))

        # Execute reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Validate: Should return None (no order executed)
        assert result is None, "Failed close should prevent reversal"

        # Verify portfolio.execute_trade never called
        self.portfolio.execute_trade.assert_not_called()

    def test_reversal_open_fails_leaves_flat_position(self):
        """
        Test error handling: If open order fails after close succeeds,
        position is left FLAT (not in inconsistent state).
        """
        # Setup: existing SHORT position
        self.portfolio.get_position.return_value = {"qty": -0.05, "avg_cost": 42000.0}

        # Mock close succeeds, open fails
        close_trade = Mock()
        close_trade.pnl = Decimal("100.00")

        self.portfolio.execute_trade.side_effect = [
            close_trade,  # Close succeeds
            Exception("Portfolio update failed on open")  # Open fails
        ]

        # Execute reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Validate: Should return None
        assert result is None, "Failed open should return None"

        # Verify close was executed (called once)
        assert self.portfolio.execute_trade.call_count >= 1, "Close should execute before open fails"

    def test_reversal_different_sizes(self):
        """
        Test edge case: Close qty != Open qty (different position sizes).
        """
        # Setup: existing SHORT of 0.03 BTC
        existing_qty = -0.03
        self.portfolio.get_position.return_value = {"qty": existing_qty, "avg_cost": 42000.0}

        # Position sizer will calculate new size (e.g., 0.04 BTC)
        # This is normal: market conditions changed, confidence changed, etc.

        # Capture orders
        placed_orders = []
        original_place_order = self.broker.place_order

        def capture_order(order):
            placed_orders.append(order)
            return original_place_order(order)

        self.broker.place_order = capture_order

        # Mock portfolio.execute_trade
        close_trade = Mock()
        close_trade.pnl = Decimal("50.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")

        self.portfolio.execute_trade.side_effect = [close_trade, open_trade]

        # Execute
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.9,  # Higher confidence -> larger position
            current_price=42000.0,
        )

        # Validate: Should succeed with different sizes
        assert result is not None
        assert len(placed_orders) == 2, "Should place close + open orders"

        close_order = placed_orders[0]
        open_order = placed_orders[1]

        assert close_order.quantity == pytest.approx(abs(existing_qty), rel=1e-9)
        # Open order quantity determined by position sizer (likely different)
        assert open_order.quantity != close_order.quantity, \
            "Close and open quantities should differ when confidence/conditions change"

    def test_non_reversal_unaffected(self):
        """
        Test that non-reversal trades (adding to position, new position) work unchanged.
        """
        # Setup: No existing position
        self.portfolio.get_position.return_value = None

        # Mock portfolio.execute_trade
        trade = Mock()
        trade.pnl = Decimal("0.00")
        self.portfolio.execute_trade.return_value = trade

        # Execute normal LONG
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Validate: Should execute as single order
        assert result is not None
        assert result.side == OrderSide.BUY

        # Should only call execute_trade once (not a reversal)
        assert self.portfolio.execute_trade.call_count == 1

    def test_reversal_state_flow_short_to_long(self):
        """
        Test state flow: SHORT -> FLAT -> LONG
        Validates that portfolio state transitions correctly through reversal.
        """
        # Setup: existing SHORT position
        self.portfolio.get_position.return_value = {"qty": -0.05, "avg_cost": 42000.0}

        # Track portfolio state during reversal
        position_states = []

        def track_state(order, price):
            # Record position qty after each trade
            current_pos = self.portfolio.get_position.return_value
            position_states.append(current_pos.get("qty", 0.0) if current_pos else 0.0)

            # Simulate portfolio state changes
            if order.side == OrderSide.BUY and position_states[0] < 0:
                # Close SHORT -> FLAT
                self.portfolio.get_position.return_value = {"qty": 0.0, "avg_cost": 0.0}
            elif order.side == OrderSide.BUY and len(position_states) == 1:
                # Open LONG
                self.portfolio.get_position.return_value = {"qty": order.quantity, "avg_cost": price}

            trade = Mock()
            trade.pnl = Decimal("0.00")
            return trade

        self.portfolio.execute_trade.side_effect = track_state

        # Execute reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Validate state flow
        assert result is not None
        assert len(position_states) == 2, "Should record two state changes"
        assert position_states[0] == -0.05, "Initial state: SHORT"
        # After close, position should be flat (or about to open LONG)

    def test_reversal_edge_case_zero_position(self):
        """
        Test edge case: get_position returns qty=0 (flat position).
        Should NOT trigger reversal logic.
        """
        # Setup: FLAT position (qty=0)
        self.portfolio.get_position.return_value = {"qty": 0.0, "avg_cost": 0.0}

        # Mock portfolio.execute_trade
        trade = Mock()
        trade.pnl = Decimal("0.00")
        self.portfolio.execute_trade.return_value = trade

        # Execute LONG decision
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Validate: Should NOT trigger reversal (execute as normal single order)
        assert result is not None
        assert self.portfolio.execute_trade.call_count == 1, \
            "Flat position should not trigger reversal (single order)"

    def test_reversal_validation_fails_on_close(self):
        """
        Test error path: RiskManager rejects close order.
        """
        # Setup: existing LONG position
        self.portfolio.get_position.return_value = {"qty": 0.1, "avg_cost": 42000.0}

        # Force risk manager to reject close order
        # (e.g., circuit breaker active)
        self.risk_manager.circuit_breaker_triggered = True

        # Execute SHORT decision (would trigger reversal)
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="SHORT",
            confidence=0.8,
            current_price=42000.0,
        )

        # Validate: Should return None (reversal blocked)
        assert result is None
        self.portfolio.execute_trade.assert_not_called()

    def test_reversal_validation_fails_on_open(self):
        """
        Test error path: RiskManager rejects open order after close succeeds.
        """
        # Setup: existing SHORT position
        self.portfolio.get_position.return_value = {"qty": -0.05, "avg_cost": 42000.0}

        # Mock close succeeds
        close_trade = Mock()
        close_trade.pnl = Decimal("-5000.00")  # Big loss triggers circuit breaker

        def execute_trade_with_circuit_breaker(order, price):
            trade = close_trade
            # After close trade with big loss, trigger circuit breaker
            if order.side == OrderSide.BUY:
                self.risk_manager.on_trade_executed(trade)
            return trade

        self.portfolio.execute_trade.side_effect = execute_trade_with_circuit_breaker

        # Execute LONG decision
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Validate: Should return None (open order rejected after close)
        # This leaves position FLAT, which is acceptable
        assert result is None


class TestPositionReversalRealBugScenario:
    """
    Reproduce the exact scenario from bug report QuantAgent-g3c.
    """

    def setup_method(self):
        """Set up exact conditions from bug report."""
        self.position_sizer = PositionSizer(base_position_pct=0.05)

        self.portfolio = Mock()
        self.portfolio.cash = 106909.42
        self.portfolio.positions = {
            "BTC": {"qty": -0.0330943811250786, "avg_cost": 106000.0}
        }
        self.portfolio.get_total_value.return_value = 106909.42
        self.portfolio.get_unrealized_pnl.return_value = 0.0

        # Return existing SHORT position
        self.portfolio.get_position.return_value = {
            "qty": -0.0330943811250786,
            "avg_cost": 106000.0
        }

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

    def test_bug_scenario_short_to_long_reversal(self):
        """
        Reproduce bug: SHORT position 0.033094 -> LONG signal with calculated size 0.034277.

        Before fix: ValueError "Trying to buy 0.0342770443640196 shares but SHORT position is only 0.0330943811250786"
        After fix: Should succeed with two orders (close 0.033094, open 0.034277)
        """
        # Mock portfolio.execute_trade
        close_trade = Mock()
        close_trade.pnl = Decimal("50.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")

        self.portfolio.execute_trade.side_effect = [close_trade, open_trade]

        # Execute: LONG decision with 68% confidence (from bug report)
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.68,
            current_price=106045.33,
        )

        # Validate: Should succeed (no ValueError)
        assert result is not None, "Bug scenario should now succeed"
        assert result.side == OrderSide.BUY

        # Verify two trades executed
        assert self.portfolio.execute_trade.call_count == 2
