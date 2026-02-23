"""PositionMonitor for tracking and managing active positions."""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy.orm import Session

from ..models import ActivePosition, OrderSide


class PositionMonitor:
    """
    Manages state of active positions in DB.
    Does NOT decide when to exit - that is TradingStrategy's responsibility.
    """

    def __init__(self, db_session: Session, backtest_run_id: Optional[int] = None):
        self.db = db_session
        self.backtest_run_id = backtest_run_id

    def set_backtest_run_id(self, backtest_run_id: Optional[int]) -> None:
        """Set or update backtest run context."""
        self.backtest_run_id = backtest_run_id

    def get_active_position(self, symbol: str) -> Optional[ActivePosition]:
        """Get active position for a symbol."""
        query = self.db.query(ActivePosition).filter(
            ActivePosition.symbol == symbol,
            ActivePosition.is_active.is_(True),
        )

        if self.backtest_run_id is not None:
            query = query.filter(ActivePosition.backtest_run_id == self.backtest_run_id)

        return query.first()

    def open_position(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        quantity: Decimal,
        exit_policy: str,
        trade_id: Optional[int] = None,
        signal_id: Optional[int] = None,
        trailing_stop_pct: Optional[float] = None,
        max_hold_candles: Optional[int] = None,
        prediction_horizon: int = 3,
        backtest_run_id: Optional[int] = None,
    ) -> ActivePosition:
        """Create new active position."""
        run_context = (
            backtest_run_id if backtest_run_id is not None else self.backtest_run_id
        )

        position = ActivePosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            decision_timestamp=datetime.utcnow(),
            exit_policy=exit_policy,
            max_hold_candles=max_hold_candles,
            trailing_stop_pct=trailing_stop_pct,
            prediction_horizon=prediction_horizon,
            trade_id=trade_id,
            signal_id=signal_id,
            backtest_run_id=run_context,
            candles_direction=[],
        )

        self.db.add(position)
        self.db.commit()
        self.db.refresh(position)
        return position

    def update_candle_tracking(
        self,
        position: ActivePosition,
        current_price: float,
        prev_close: float,
    ) -> None:
        """
        Update tracking for paper metrics (3-candle accuracy).
        Only state management, no decision logic.
        """
        from sqlalchemy.orm.attributes import flag_modified

        position.candles_since_entry += 1

        if len(position.candles_direction) < position.prediction_horizon:
            direction = "up" if current_price > prev_close else "down"
            position.candles_direction.append(direction)
            flag_modified(position, "candles_direction")

        self.db.commit()

    def close_position(
        self,
        position: ActivePosition,
        reason: str,
        exit_price: float,
    ) -> None:
        """Close position and calculate accuracy."""
        position.is_active = False
        position.closed_at = datetime.utcnow()
        position.close_reason = reason

        if position.candles_direction:
            expected_direction = "up" if position.side == OrderSide.BUY else "down"
            correct = sum(
                1 for d in position.candles_direction if d == expected_direction
            )
            position.accuracy = correct / len(position.candles_direction)

        self.db.commit()
