"""
Tests for enhanced error logging in position reversal (QuantAgent-2mu).

Validates that error messages include:
- Exception type
- Portfolio state (value)
- Clear status indicators (ABORTED, PARTIAL FAILURE, COMPLETE)
"""

import logging
from decimal import Decimal
from unittest.mock import Mock

import pytest

from quantagent.models import OrderSide
from quantagent.trading.order_manager import OrderManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager


class TestReversalErrorLogging:
    """Test enhanced error logging for position reversal (QuantAgent-2mu)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.position_sizer = PositionSizer(base_position_pct=0.05)

        # Mock portfolio - start with SHORT position
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
        
        # Helper to update positions during test for state transitions
        self._update_portfolio_positions = lambda pos: setattr(self.portfolio, 'positions', pos)

    def test_reversal_aborted_error_includes_exception_type(self, caplog):
        """Test that ABORTED errors include exception type and portfolio value."""
        # Mock portfolio update to raise specific exception
        self.portfolio.execute_trade.side_effect = ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            result = self.order_manager.execute_decision(
                symbol="BTC",
                decision="LONG",
                confidence=0.8,
                current_price=42000.0,
            )

        # Validate: Should return None
        assert result is None

        # Find the ABORTED log message
        error_logs = [record.message for record in caplog.records if record.levelname == "ERROR"]
        aborted_log = next((log for log in error_logs if "ABORTED" in log), None)

        assert aborted_log is not None, "Should log ABORTED message"
        assert "ValueError" in aborted_log, "Should include exception type"
        assert "Portfolio value:" in aborted_log, "Should include portfolio value"
        assert "$" in aborted_log, "Should format portfolio value as currency"

    def test_reversal_partial_failure_logged_when_open_fails(self, caplog):
        """Test that PARTIAL FAILURE is logged when close succeeds but open fails."""
        # Mock: close succeeds, open fails
        close_trade = Mock()
        close_trade.pnl = Decimal("100.00")

        self.portfolio.execute_trade.side_effect = [
            close_trade,
            ValueError("Open failed"),
        ]

        with caplog.at_level(logging.ERROR):
            result = self.order_manager.execute_decision(
                symbol="BTC",
                decision="LONG",
                confidence=0.8,
                current_price=42000.0,
            )

        # Validate: Should return None
        assert result is None

        # Find PARTIAL FAILURE log
        error_logs = [record.message for record in caplog.records if record.levelname == "ERROR"]
        partial_log = next((log for log in error_logs if "PARTIAL FAILURE" in log), None)

        assert partial_log is not None, "Should log PARTIAL FAILURE"
        assert "FLAT" in partial_log, "Should indicate position is FLAT"
        assert "Portfolio value:" in partial_log, "Should include portfolio value"

    def test_reversal_complete_includes_final_state(self, caplog):
        """Test that successful reversal logs COMPLETE with final position state."""
        # Mock successful reversal
        close_trade = Mock()
        close_trade.pnl = Decimal("100.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")

        # Simulate state transitions during execute_trade
        def execute_trade_with_state_update(order, price):
            if self.portfolio.execute_trade.call_count == 0:
                # First call: close SHORT position -> FLAT
                self.portfolio.positions = {}
                return close_trade
            else:
                # Second call: open LONG position
                self.portfolio.positions = {"BTC": {"qty": 0.05, "avg_cost": 42000.0}}
                return open_trade
        
        self.portfolio.execute_trade = Mock(side_effect=lambda o, p: (
            close_trade if self.portfolio.positions.get("BTC", {}).get("qty", 0) < 0
            else (setattr(self.portfolio, 'positions', {}), close_trade)[1]
        ))
        
        # Actually, let's just mock it simply
        call_count = [0]
        def track_calls(order, price):
            call_count[0] += 1
            if call_count[0] == 1:
                self.portfolio.positions = {}  # FLAT
                return close_trade
            else:
                self.portfolio.positions = {"BTC": {"qty": 0.05, "avg_cost": 42000.0}}
                return open_trade
        
        self.portfolio.execute_trade.side_effect = track_calls

        with caplog.at_level(logging.INFO):
            result = self.order_manager.execute_decision(
                symbol="BTC",
                decision="LONG",
                confidence=0.8,
                current_price=42000.0,
            )

        # Validate: Should succeed
        assert result is not None

        # Find COMPLETE log
        info_logs = [record.message for record in caplog.records if record.levelname == "INFO"]
        complete_log = next((log for log in info_logs if "COMPLETE" in log), None)

        assert complete_log is not None, "Should log COMPLETE message"
        assert "New position:" in complete_log, "Should include new position details"
        assert "Portfolio value:" in complete_log, "Should include portfolio value"

    def test_reversal_starting_log_includes_initial_state(self, caplog):
        """Test that reversal start log includes initial state."""
        # Mock successful reversal
        close_trade = Mock()
        close_trade.pnl = Decimal("100.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")
        self.portfolio.execute_trade.side_effect = [close_trade, open_trade]

        with caplog.at_level(logging.INFO):
            self.order_manager.execute_decision(
                symbol="BTC",
                decision="LONG",
                confidence=0.8,
                current_price=42000.0,
            )

        # Find starting log
        info_logs = [record.message for record in caplog.records if record.levelname == "INFO"]
        start_log = next((log for log in info_logs if "Starting position reversal" in log), None)

        assert start_log is not None, "Should log reversal start"
        assert "Current:" in start_log, "Should include current position"
        assert "Target:" in start_log, "Should include target position"
        assert "Portfolio value:" in start_log, "Should include portfolio value"
