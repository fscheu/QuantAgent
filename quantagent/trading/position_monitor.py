"""PositionMonitor for tracking and managing active positions."""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy.orm import Session

from ..models import ActivePosition, Environment, Order, OrderSide, OrderStatus, Trade
from .stop_management import StopDecision, nuevo_stop

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Manages state of active positions in DB.
    Does NOT decide when to exit - that is TradingStrategy's responsibility.
    """

    def __init__(
        self,
        db_session: Session,
        backtest_run_id: Optional[int] = None,
        environment: Optional[Environment] = None,
    ):
        self.db = db_session
        self.backtest_run_id = backtest_run_id
        self.environment = environment

    def actualizar_stop(self, position: ActivePosition, propuesto: float) -> StopDecision:
        """Actualiza el stop solo si la propuesta lo ajusta a favor."""
        decision = nuevo_stop(position.stop_loss, propuesto, position.side)
        if decision.stop != position.stop_loss:
            position.stop_loss = decision.stop
            self.db.commit()
        return decision

    def get_active_position(self, symbol: str) -> Optional[ActivePosition]:
        """Get active position for a symbol (ignores zero-quantity positions)."""
        from decimal import Decimal
        query = self.db.query(ActivePosition).filter(
            ActivePosition.symbol == symbol,
            ActivePosition.is_active.is_(True),
            ActivePosition.quantity > Decimal("0"),  # Ignore zero-quantity positions
        )

        if self.environment is not None:
            query = query.filter(ActivePosition.environment == self.environment)

        if self.backtest_run_id is not None:
            query = query.filter(ActivePosition.backtest_run_id == self.backtest_run_id)

        return query.order_by(ActivePosition.decision_timestamp.desc()).first()

    def reconcile_alpaca_positions(self, broker) -> list[str]:
        """Close broker-stopped positions, then report remaining PAPER divergences."""
        local = {
            position.symbol: position
            for position in self.db.query(ActivePosition).filter(
                ActivePosition.is_active.is_(True),
                ActivePosition.quantity > Decimal("0"),
                ActivePosition.environment == Environment.PAPER,
            ).all()
        }
        remote = broker.get_positions()
        differences = []
        for position in list(local.values()):
            if not position.broker_stop_order_id:
                continue
            stop_order = self.db.query(Order).filter(
                Order.alpaca_order_id == position.broker_stop_order_id
            ).first()
            if stop_order is None:
                continue
            broker.refresh_order(stop_order)
            if stop_order.status != OrderStatus.FILLED:
                self.db.commit()
                continue
            exit_price = float(stop_order.average_fill_price)
            self.close_position(position, reason="stop_broker", exit_price=exit_price)
            if position.trade_id:
                trade = self.db.query(Trade).filter(Trade.id == position.trade_id).first()
                if trade:
                    trade.exit_signal = "stop_broker"
                    trade.closed_at = position.closed_at
                    self.db.commit()
            local.pop(position.symbol)
            remote.pop(position.symbol, None)
        for symbol in sorted(local.keys() | remote.keys()):
            position = local.get(symbol)
            alpaca_position = remote.get(symbol)
            if position is None:
                detail = f"remote-only qty={alpaca_position.qty} side={alpaca_position.side}"
            elif alpaca_position is None:
                detail = f"local-only qty={position.quantity} side={position.side.value}"
            else:
                alpaca_side = getattr(alpaca_position.side, "value", alpaca_position.side)
                remote_side = "buy" if alpaca_side == "long" else "sell"
                if position.quantity == Decimal(str(alpaca_position.qty)) and position.side.value == remote_side:
                    continue
                detail = (f"local qty={position.quantity} side={position.side.value}; "
                          f"Alpaca qty={alpaca_position.qty} side={alpaca_side}")
            differences.append(f"{symbol}: {detail}")
            logger.warning(
                "Alpaca position reconciliation divergence: %s",
                differences[-1],
                extra={"environment": Environment.PAPER.value, "symbol": symbol,
                       "event_type": "position_reconciliation", "extra_data": {"detail": detail}},
            )
        return differences

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
        environment: Optional[Environment] = None,
        timestamp: Optional[datetime] = None,
    ) -> ActivePosition:
        """Create new active position.

        Args:
            timestamp: Simulated candle time for backtests; None (wall-clock) for
                live/paper trading
        """
        run_context = (
            backtest_run_id if backtest_run_id is not None else self.backtest_run_id
        )
        position_environment = (
            environment if environment is not None else self.environment
        )

        position = ActivePosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            decision_timestamp=timestamp or datetime.utcnow(),
            exit_policy=exit_policy,
            max_hold_candles=max_hold_candles,
            trailing_stop_pct=trailing_stop_pct,
            prediction_horizon=prediction_horizon,
            trade_id=trade_id,
            signal_id=signal_id,
            backtest_run_id=run_context,
            candles_direction=[],
            environment=position_environment or Environment.BACKTEST,
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
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Close position and calculate accuracy.

        Args:
            timestamp: Simulated candle time for backtests; None (wall-clock) for
                live/paper trading
        """
        position.is_active = False
        position.closed_at = timestamp or datetime.utcnow()
        position.close_reason = reason

        if position.candles_direction:
            expected_direction = "up" if position.side == OrderSide.BUY else "down"
            correct = sum(
                1 for d in position.candles_direction if d == expected_direction
            )
            position.accuracy = correct / len(position.candles_direction)

        self.db.commit()
