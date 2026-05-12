"""
Test for QuantAgent-8vb: Fix ConversionSyntax error when closing SHORT positions.

This test validates that the duplicate method removal and correct side/quantity
calculation prevent the ConversionSyntax error that was occurring when reversing
from SHORT to LONG positions.

Bug Details:
- Error: "Conversion 'ConversionSyntax' received SELL -0.04807692307692308 for attribute 'side'"
- Root Cause: Duplicate _execute_reversal() method with incorrect signature
- Fix: Removed duplicate, kept correct implementation with proper side calculation

Test Strategy:
- Validate that close side is BUY for SHORT positions
- Validate that close quantity is always positive
- Validate that no SQLAlchemy conversion errors occur
- Validate the full reversal flow completes successfully
"""

import pytest
from unittest.mock import Mock
from decimal import Decimal

from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.trading.order_manager import OrderManager
from quantagent.models import OrderSide


class TestQuantAgent8vbConversionFix:
    """Test the fix for ConversionSyntax error when closing SHORT positions."""

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

    def test_close_short_uses_buy_side_not_sell(self):
        """
        CRITICAL: Verify that closing a SHORT position uses OrderSide.BUY, not SELL.
        
        This is the core validation for QuantAgent-8vb.
        Before fix: Would incorrectly use SELL with negative quantity -> ConversionSyntax
        After fix: Must use BUY with positive quantity
        """
        # Setup: SHORT position (negative quantity)
        existing_qty = -0.04807692307692308  # Exact value from bug report
        self.portfolio.get_position.return_value = {"qty": existing_qty, "avg_cost": 96382.0}

        # Capture the close order using real broker (like test_order_manager_reversal.py)
        placed_orders = []
        original_place_order = self.broker.place_order
        
        def capture_order(order):
            placed_orders.append(order)
            return original_place_order(order)  # Use real broker to fill
        
        self.broker.place_order = capture_order

        # Mock portfolio.execute_trade
        close_trade = Mock()
        close_trade.pnl = Decimal("10.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")
        self.portfolio.execute_trade.side_effect = [close_trade, open_trade]

        # Execute: LONG decision triggers SHORT->LONG reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=96382.0,
        )

        # CRITICAL VALIDATIONS for QuantAgent-8vb fix:
        
        # 1. Close order must exist
        assert len(placed_orders) >= 1, "Should create close order"
        close_order = placed_orders[0]
        
        # 2. Close side MUST be BUY (not SELL)
        assert close_order.side == OrderSide.BUY, \
            f"Close SHORT requires BUY side, got {close_order.side}"
        
        # 3. Close quantity MUST be positive
        assert close_order.quantity > 0, \
            f"Close quantity must be positive, got {close_order.quantity}"
        
        # 4. Close quantity must match absolute value of position
        assert close_order.quantity == pytest.approx(abs(existing_qty), rel=1e-9), \
            f"Close qty {close_order.quantity} must equal abs(existing_qty) {abs(existing_qty)}"
        
        # 5. Overall reversal must succeed
        assert result is not None, "Reversal should succeed"
        assert result.side == OrderSide.BUY, "Final position should be LONG (BUY)"

    def test_no_conversion_syntax_error_on_short_reversal(self):
        """
        Validate that the ConversionSyntax error no longer occurs.
        
        This test ensures that the combination of side + quantity that is passed
        to SQLAlchemy is always valid (no SELL with negative quantity).
        """
        # Setup: SHORT position from bug scenario
        self.portfolio.get_position.return_value = {"qty": -0.048, "avg_cost": 96382.0}

        # Track all orders created using real broker
        created_orders = []
        original_place_order = self.broker.place_order
        
        def track_order_creation(order):
            created_orders.append({
                "side": order.side,
                "quantity": order.quantity,
                "symbol": order.symbol
            })
            return original_place_order(order)  # Use real broker
        
        self.broker.place_order = track_order_creation

        # Mock portfolio trades
        close_trade = Mock()
        close_trade.pnl = Decimal("5.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")
        self.portfolio.execute_trade.side_effect = [close_trade, open_trade]

        # Execute reversal
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=96500.0,
        )

        # Validate no invalid combinations
        for order in created_orders:
            # Rule 1: SELL side must have positive quantity
            if order["side"] == OrderSide.SELL:
                assert order["quantity"] > 0, \
                    f"SELL order has negative/zero quantity: {order['quantity']}"
            
            # Rule 2: BUY side must have positive quantity
            if order["side"] == OrderSide.BUY:
                assert order["quantity"] > 0, \
                    f"BUY order has negative/zero quantity: {order['quantity']}"
            
            # Rule 3: Quantity must NEVER be negative
            assert order["quantity"] > 0, \
                f"Order quantity must be positive, got {order['quantity']}"

        # Should have two orders: close (BUY) + open (BUY)
        assert len(created_orders) == 2, "Should create close + open orders"
        assert created_orders[0]["side"] == OrderSide.BUY, "First order (close) must be BUY"
        assert created_orders[0]["quantity"] > 0, "Close quantity must be positive"

    def test_exact_bug_scenario_from_report(self):
        """
        Reproduce the EXACT scenario from the bug report that triggered ConversionSyntax.
        
        From docs/02_planning/QuantAgent-8vb-backtest-analysis.md:
        - Existing qty: -0.04807692307692308
        - New side: OrderSide.BUY
        - Price: ~$96382
        
        Before fix: ERROR - "Conversion 'ConversionSyntax' received SELL -0.04807..."
        After fix: Should execute successfully with BUY order
        """
        # Exact values from bug report
        existing_qty = -0.04807692307692308
        price = 96382.0
        
        self.portfolio.get_position.return_value = {"qty": existing_qty, "avg_cost": price}

        # Capture the exact order details using real broker
        placed_orders = []
        original_place_order = self.broker.place_order
        
        def capture_order(order):
            placed_orders.append(order)
            return original_place_order(order)
        
        self.broker.place_order = capture_order

        # Mock trades
        close_trade = Mock()
        close_trade.pnl = Decimal("0.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")
        self.portfolio.execute_trade.side_effect = [close_trade, open_trade]

        # Execute the exact scenario from bug
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="LONG",
            confidence=0.8,
            current_price=price,
        )

        # The bug manifested as: "Conversion 'ConversionSyntax' received SELL -0.04807..."
        # This means the close order had SELL side with negative quantity
        
        # Validate the fix:
        assert len(placed_orders) >= 1, "Close order should be created"
        captured_close_order = placed_orders[0]
        
        # MUST NOT be: side=SELL, quantity=negative
        # MUST be: side=BUY, quantity=positive
        assert captured_close_order.side == OrderSide.BUY, \
            f"Bug reproduced: Expected BUY to close SHORT, got {captured_close_order.side}"
        
        assert captured_close_order.quantity == pytest.approx(abs(existing_qty), rel=1e-9), \
            f"Bug reproduced: Expected positive quantity {abs(existing_qty)}, got {captured_close_order.quantity}"
        
        # Reversal must complete successfully (no exception)
        assert result is not None, "Bug reproduced: Reversal failed"

    def test_long_to_short_reversal_still_works(self):
        """
        Regression test: Ensure LONG->SHORT reversals still work correctly.
        
        The fix for SHORT->LONG should not break existing LONG->SHORT functionality.
        """
        # Setup: LONG position
        self.portfolio.get_position.return_value = {"qty": 0.05, "avg_cost": 42000.0}

        placed_orders = []
        original_place_order = self.broker.place_order
        
        def capture_order(order):
            placed_orders.append(order)
            return original_place_order(order)  # Use real broker
        
        self.broker.place_order = capture_order

        # Mock trades
        close_trade = Mock()
        close_trade.pnl = Decimal("100.00")
        open_trade = Mock()
        open_trade.pnl = Decimal("0.00")
        self.portfolio.execute_trade.side_effect = [close_trade, open_trade]

        # Execute SHORT decision
        result = self.order_manager.execute_decision(
            symbol="BTC",
            decision="SHORT",
            confidence=0.8,
            current_price=43000.0,
        )

        # Validate LONG->SHORT still works
        assert result is not None
        assert len(placed_orders) == 2
        
        # Close LONG requires SELL
        close_order = placed_orders[0]
        assert close_order.side == OrderSide.SELL, "Close LONG requires SELL"
        assert close_order.quantity > 0, "Close quantity must be positive"
        
        # Open SHORT is also SELL (new short position)
        open_order = placed_orders[1]
        assert open_order.side == OrderSide.SELL, "Open SHORT uses SELL"
