"""
Trading module: Position sizing, risk management, order execution.

Components:
- PositionSizer: Calculates order size based on confidence
- RiskManager: Validates trades before execution
- OrderManager: Orchestrates the complete order flow
- PaperBroker: Simulates order execution with slippage
"""

from .order_manager import OrderManager
from .paper_broker import PaperBroker
from .position_sizer import PositionSizer
from .risk_manager import RiskManager
from .scheduler import TradingScheduler

__all__ = [
    "PositionSizer",
    "RiskManager",
    "OrderManager",
    "PaperBroker",
    "TradingScheduler",
]
