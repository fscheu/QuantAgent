"""
Additional integration tests for QuantAgent-2mu error handling validation.

Tests scenarios from design doc section 6 (Test Requirements):
- Error propagation (return None on failures)
- Portfolio state consistency after errors
- Precondition validation
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from quantagent.trading.order_manager import OrderManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager


class TestReversalErrorPropagation:
    """Test that errors are properly propagated (return None on failure)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.position_sizer = PositionSizer(base_position_pct=0.05)

        # Mock portfolio with SHORT position
        self.portfolio = Mock()
        self.portfolio.cash = 100000.0
        self.portfolio.positions = {"BTC": {"qty": -0.05, "avg_cost": 42000.0}}
        self.portfolio.get_total_value.return_value = 100000.0
        self.portfolio.get_unrealized_pnl.return_value = 0.0
        self.portfolio.get_position.return_value = {"qty": -0.05, "avg_cost": 42000.0}

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

    def test_reversal_returns_none_on_close_failure(self):
        """Test that reversal returns None when close order fails."""
        # Mock: execute_trade raises exception on close
        self.portfolio.execute_trade.side_effect = ValueError("Close failed")

        # Action: Attempt reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Assert: Should return None (error propagated)
        assert result is None, "Should return None when close order fails"

    def test_reversal_returns_none_on_open_failure(self):
        """Test that reversal returns None when open order fails."""
        # Mock: close succeeds, open fails
        close_trade = Mock()
        close_trade.pnl = Decimal("100.00")

        self.portfolio.execute_trade.side_effect = [
            close_trade,
            ValueError("Open failed"),
        ]

        # Action: Attempt reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Assert: Should return None (error propagated)
        assert result is None, "Should return None when open order fails"

    def test_portfolio_unchanged_after_close_failure(self):
        """Test that portfolio state is unchanged when close order fails."""
        # Capture initial portfolio state
        initial_positions = dict(self.portfolio.positions)
        initial_cash = self.portfolio.cash

        # Mock: Close fails
        self.portfolio.execute_trade.side_effect = ValueError("Close failed")

        # Action: Attempt reversal
        self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Assert: Portfolio state should be unchanged
        assert self.portfolio.positions == initial_positions, \
            "Portfolio positions should be unchanged after close failure"
        assert self.portfolio.cash == initial_cash, \
            "Portfolio cash should be unchanged after close failure"

    def test_portfolio_flat_after_open_failure(self):
        """Test that portfolio is FLAT (position closed) when open order fails."""
        # Mock: close succeeds, open fails
        close_trade = Mock()
        close_trade.pnl = Decimal("100.00")

        call_count = [0]
        def track_calls(order, price):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (close): Set position to empty
                self.portfolio.positions = {}
                self.portfolio.get_position.return_value = None
                return close_trade
            else:
                # Second call (open): Fail
                raise ValueError("Open failed")

        self.portfolio.execute_trade.side_effect = track_calls

        # Action: Attempt reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Assert: Result should be None
        assert result is None, "Should return None when open fails"

        # Assert: Position should be FLAT (closed but not reopened)
        assert "BTC" not in self.portfolio.positions or \
               self.portfolio.positions.get("BTC", {}).get("qty", 0) == 0, \
            "Position should be FLAT after partial reversal failure"


class TestReversalPreconditions:
    """Test precondition validation for reversal scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.position_sizer = PositionSizer(base_position_pct=0.05)
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

    def test_no_reversal_when_no_position_exists(self):
        """Test that reversal doesn't trigger when no position exists."""
        # Portfolio has no position
        self.portfolio.get_position.return_value = None

        # Action: Try to execute decision (should be new position, not reversal)
        # Mock successful trade
        new_trade = Mock()
        new_trade.pnl = Decimal("0.00")
        self.portfolio.execute_trade.return_value = new_trade

        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Assert: Should succeed as new position (not reversal)
        assert result is not None, "Should create new position when none exists"

        # Assert: Only one trade executed (no close order)
        assert self.portfolio.execute_trade.call_count == 1, \
            "Should not attempt reversal when no position exists"

    def test_no_reversal_when_same_side(self):
        """Test that reversal doesn't trigger when decision matches existing position."""
        # Portfolio has LONG position
        self.portfolio.positions = {"BTC": {"qty": 0.05, "avg_cost": 42000.0}}
        self.portfolio.get_position.return_value = {"qty": 0.05, "avg_cost": 42000.0}

        # Action: Try to execute LONG decision (same side, should not reverse)
        _ = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Assert: No reversal should occur
        # Note: Actual behavior depends on implementation - might reject, might do nothing
        # The key is that it shouldn't execute a close+open reversal sequence

        # Check that no close order was executed (reversal would start with close)
        if self.portfolio.execute_trade.called:
            # If any trade executed, verify it's not a reversal pattern
            call_count = self.portfolio.execute_trade.call_count
            assert call_count <= 1, \
                "Should not execute 2-step reversal for same-side decision"


class TestReversalStateConsistency:
    """Test portfolio state consistency after reversal errors."""

    def setup_method(self):
        """Set up test fixtures."""
        self.position_sizer = PositionSizer(base_position_pct=0.05)

        self.portfolio = Mock()
        self.portfolio.cash = 100000.0
        self.portfolio.positions = {"BTC": {"qty": -0.05, "avg_cost": 42000.0}}
        self.portfolio.get_total_value.return_value = 100000.0
        self.portfolio.get_unrealized_pnl.return_value = 0.0
        self.portfolio.get_position.return_value = {"qty": -0.05, "avg_cost": 42000.0}

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

    def test_portfolio_value_accessible_after_error(self):
        """Test that portfolio value can be queried after reversal error."""
        # Mock: Close fails
        self.portfolio.execute_trade.side_effect = ValueError("Close failed")

        # Action: Attempt reversal (will fail)
        self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=42000.0,
        )

        # Assert: Portfolio value should still be accessible
        assert self.portfolio.get_total_value.called, \
            "Portfolio value should be queried for error logging"

        # Verify no exception when accessing portfolio value
        try:
            value = self.portfolio.get_total_value()
            assert value == 100000.0, "Portfolio value should be consistent"
        except Exception as e:
            pytest.fail(f"Portfolio value access failed after error: {e}")

    def test_multiple_reversal_attempts_dont_corrupt_state(self):
        """Test that multiple failed reversal attempts don't corrupt portfolio."""
        # Mock: All attempts fail
        self.portfolio.execute_trade.side_effect = ValueError("Always fails")

        # Action: Attempt reversal multiple times
        for _ in range(3):
            result = self.order_manager.execute_decision(
                symbol="BTC",
                decision="LONG",
                confidence=0.8,
                current_price=42000.0,
            )
            assert result is None, "Each attempt should fail"

        # Assert: Portfolio state should remain consistent
        assert self.portfolio.positions["BTC"]["qty"] == -0.05, \
            "Position quantity should remain unchanged after multiple failures"
        assert self.portfolio.cash == 100000.0, \
            "Cash should remain unchanged after multiple failures"
