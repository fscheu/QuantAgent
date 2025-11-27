"""Risk Manager for trade validation and monitoring."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Tuple, Optional, Dict
from sqlalchemy.orm import Session

from quantagent.models import Trade, Order, Environment
from quantagent.database import SessionLocal
from quantagent.portfolio.manager import PortfolioManager


class RiskManager:
    """Manages pre-trade and post-trade risk validation.

    Attributes:
        max_position_size_pct: Max % of capital per trade (default 10%)
        max_daily_loss_pct: Max % daily loss before circuit breaker (default 5%)
        circuit_breaker_active: Whether circuit breaker is active
    """

    def __init__(
        self,
        initial_capital: float,
        portfolio: PortfolioManager,
        max_position_size_pct: float = 2.0,
        max_daily_loss_pct: float = 5.0,
        environment: Environment = Environment.PAPER,
        db: Optional[Session] = None,
    ):
        """Initialize risk manager.

        Args:
            initial_capital: Initial portfolio capital
            portfolio: PortfolioManager instance for position tracking
            max_position_size_pct: Max position as % of capital (default 10%)
            max_daily_loss_pct: Max daily loss as % of capital (default 5%)
            environment: Execution environment tag
            db: SQLAlchemy session (uses SessionLocal if not provided)
        """
        self.initial_capital = float(initial_capital)
        self.portfolio = portfolio
        self.max_position_size_pct = max_position_size_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.environment = environment
        self.db = db or SessionLocal()
        self.circuit_breaker_active = False

    def validate_trade(
        self, symbol: str, qty: float, price: float
    ) -> Tuple[bool, str]:
        """Validate trade before execution.

        Checks:
        1. Sufficient capital available
        2. Position size within limits (max 10% of capital)
        3. Daily loss limit not exceeded

        Args:
            symbol: Trading symbol
            qty: Order quantity
            price: Expected fill price

        Returns:
            (is_valid, reason) tuple
        """
        # Check if circuit breaker is active
        if self.circuit_breaker_active:
            return False, "Circuit breaker is active - no trading allowed"

        # Check 1: Sufficient capital
        trade_value = qty * price
        if self.portfolio.cash < trade_value:
            return False, f"Insufficient capital: need {trade_value}, have {self.portfolio.cash}"

        # Check 2: Position size limit
        max_position_value = (self.max_position_size_pct / 100.0) * self.portfolio.get_total_value()
        if trade_value > max_position_value:
            return False, (
                f"Position too large: {trade_value} exceeds max {max_position_value} "
                f"({self.max_position_size_pct}% of capital)"
            )

        # Check 3: Daily loss limit
        daily_loss = self._get_daily_loss()
        max_daily_loss = (self.max_daily_loss_pct / 100.0) * self.initial_capital
        if abs(daily_loss) >= max_daily_loss:
            return False, (
                f"Daily loss limit exceeded: {abs(daily_loss)} >= {max_daily_loss} "
                f"({self.max_daily_loss_pct}% of capital)"
            )

        return True, "Trade approved"

    def check_circuit_breaker(self) -> Tuple[bool, str]:
        """Check if circuit breaker should be activated.

        Returns:
            (breaker_active, reason) tuple
        """
        daily_loss = self._get_daily_loss()
        max_daily_loss = (self.max_daily_loss_pct / 100.0) * self.initial_capital

        if abs(daily_loss) >= max_daily_loss:
            self.circuit_breaker_active = True
            return True, f"Circuit breaker activated: daily loss {abs(daily_loss)} >= limit {max_daily_loss}"

        return False, "Circuit breaker not active"

    def on_trade_executed(self, trade: Trade) -> None:
        """Post-trade monitoring and updates.

        Args:
            trade: Executed Trade object
        """
        # Check circuit breaker after trade
        self.check_circuit_breaker()

        # Log trade execution
        if self.circuit_breaker_active:
            # Log warning but don't prevent already-executed trade
            pass

    def load_profile(self, profile_name: str, config: Optional[Dict] = None) -> None:
        """Load risk management profile configuration.

        Args:
            profile_name: Name of the profile to load
            config: Optional dict to override defaults (e.g., {'max_position_size_pct': 15})
        """
        if config:
            if "max_position_size_pct" in config:
                self.max_position_size_pct = config["max_position_size_pct"]
            if "max_daily_loss_pct" in config:
                self.max_daily_loss_pct = config["max_daily_loss_pct"]

    def _get_daily_loss(self) -> float:
        """Calculate daily loss from trades closed today.

        Returns:
            Daily P&L (negative if loss)
        """
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        daily_trades = self.db.query(Trade).filter(
            Trade.environment == self.environment,
            Trade.closed_at >= today_start,
            Trade.pnl.isnot(None),
        ).all()

        total_daily_pnl = sum(float(t.pnl) for t in daily_trades)
        return total_daily_pnl

    def get_max_position_size(self) -> float:
        """Calculate max position size in capital value.

        Returns:
            Max position value in base currency
        """
        return (self.max_position_size_pct / 100.0) * self.portfolio.get_total_value()

    def get_max_daily_loss(self) -> float:
        """Calculate max daily loss in capital value.

        Returns:
            Max daily loss in base currency
        """
        return (self.max_daily_loss_pct / 100.0) * self.initial_capital

    def get_daily_loss(self) -> float:
        """Get current daily loss.

        Returns:
            Daily P&L (negative if loss)
        """
        return self._get_daily_loss()

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (typically at start of trading day)."""
        self.circuit_breaker_active = False

    def close(self) -> None:
        """Close database session."""
        if self.db:
            self.db.close()
