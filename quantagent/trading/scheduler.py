"""TradingScheduler orchestrates periodic analysis + paper trading."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session

from quantagent import settings
from quantagent.data.provider import DataProvider
from quantagent.models import Environment, SchedulerHeartbeat, Signal, Trade, TradeSignal
from quantagent.static_util import format_ohlcv_for_agents
from quantagent.strategy.base import TradingSignal as StrategyTradingSignal
from quantagent.strategy.llm_agent_strategy import LLMAgentStrategy
from quantagent.trading.order_manager import OrderManager
from quantagent.trading.position_monitor import PositionMonitor
from quantagent.trading_graph import TradingGraph

logger = logging.getLogger(__name__)


class SchedulerError(Exception):
    """Base class for scheduler errors."""


class DataFetchError(SchedulerError):
    """Raised when market data cannot be retrieved."""


class AnalysisError(SchedulerError):
    """Raised when the analysis pipeline fails."""


class ExecutionError(SchedulerError):
    """Raised when trade execution fails."""


class TradingScheduler:
    """Run TradingGraph analysis and order execution on an interval."""

    JOB_ID = "trading_scheduler_job"

    def __init__(
        self,
        *,
        trading_graph: TradingGraph,
        order_manager: OrderManager,
        data_provider: DataProvider,
        db_session: Session,
        scheduler_settings: Optional[settings.SchedulerSettings] = None,
        scheduler_factory: Optional[Callable[[], BackgroundScheduler]] = None,
        strategy: Optional[LLMAgentStrategy] = None,
    ) -> None:
        self.trading_graph = trading_graph
        self.order_manager = order_manager
        self.data_provider = data_provider
        self.db = db_session
        self.config = scheduler_settings or settings.scheduler
        self.environment = Environment(self.config.environment)
        self.strategy = strategy or LLMAgentStrategy(self.trading_graph)

        # Initialize PositionMonitor for tracking active positions
        self.position_monitor = PositionMonitor(
            db_session=db_session,
            backtest_run_id=None,  # Paper trading doesn't use backtest_run_id
            environment=self.environment,
        )

        self.scheduler = (
            scheduler_factory()
            if scheduler_factory is not None
            else BackgroundScheduler(
                job_defaults={"coalesce": True, "max_instances": 1}, timezone="UTC"
            )
        )
        self.is_running = False
        self.last_run_stats: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def start(self, *, immediate: bool = True) -> bool:
        """Start APScheduler background loop."""
        if not self.config.enabled:
            logger.warning(
                "TradingScheduler is disabled; set TRADING_SCHEDULER_ENABLED=1 to run"
            )
            return False
        if self.is_running:
            logger.warning("TradingScheduler already running; ignoring start() request")
            return False

        trigger = IntervalTrigger(hours=self.config.interval_hours)
        next_run = datetime.utcnow() if immediate else None
        self.scheduler.add_job(
            self.analyze_and_trade,
            trigger=trigger,
            id=self.JOB_ID,
            replace_existing=True,
            next_run_time=next_run,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()
        self.is_running = True
        logger.info(
            "Scheduler started, interval=%.3fh, assets=%s",
            self.config.interval_hours,
            self.config.assets,
            extra={
                "event_type": "scheduler.start",
                "environment": self.environment.value,
                "extra_data": {
                    "interval_hours": self.config.interval_hours,
                    "assets": self.config.assets,
                    "timeframe": self.config.timeframe,
                },
            },
        )
        return True

    def stop(self) -> None:
        """Stop APScheduler loop."""
        if not self.is_running:
            logger.warning("TradingScheduler not running; ignoring stop() request")
            return
        self.scheduler.shutdown(wait=True)
        self.is_running = False
        logger.info(
            "TradingScheduler stopped",
            extra={
                "event_type": "scheduler.stop",
                "environment": self.environment.value,
            },
        )

    def run_once(self) -> Dict[str, float]:
        """Run a single analyze+trade cycle synchronously."""
        return self.analyze_and_trade()

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def analyze_and_trade(self) -> Dict[str, float]:
        cycle_start = datetime.utcnow()
        processed = 0
        errors = 0

        heartbeat = self._upsert_heartbeat_start(cycle_start)
        try:
            for symbol in self.config.assets:
                try:
                    self._process_asset(symbol)
                    processed += 1
                except DataFetchError as exc:
                    errors += 1
                    logger.warning(
                        "Failed to fetch data for %s: %s",
                        symbol,
                        exc,
                        extra={
                            "event_type": "scheduler.data_error",
                            "symbol": symbol,
                            "environment": self.environment.value,
                        },
                    )
                except AnalysisError as exc:
                    errors += 1
                    logger.error(
                        "Analysis failed for %s: %s",
                        symbol,
                        exc,
                        extra={
                            "event_type": "scheduler.analysis_error",
                            "symbol": symbol,
                            "environment": self.environment.value,
                        },
                    )
                except ExecutionError as exc:
                    errors += 1
                    logger.error(
                        "Execution failed for %s: %s",
                        symbol,
                        exc,
                        extra={
                            "event_type": "scheduler.execution_error",
                            "symbol": symbol,
                            "environment": self.environment.value,
                        },
                    )
                except Exception:  # pragma: no cover - defensive logging
                    errors += 1
                    logger.exception(
                        "Unexpected scheduler error for %s",
                        symbol,
                        extra={
                            "event_type": "scheduler.unexpected_error",
                            "symbol": symbol,
                            "environment": self.environment.value,
                        },
                    )
        except Exception as exc:  # pragma: no cover - cycle-level safeguard
            stats = self._build_cycle_stats(cycle_start, processed, errors)
            self.last_run_stats = stats
            self._upsert_heartbeat_error(heartbeat, stats, str(exc))
            raise

        stats = self._build_cycle_stats(cycle_start, processed, errors)
        self.last_run_stats = stats

        self._upsert_heartbeat_complete(heartbeat, stats)

        logger.info(
            "Analysis cycle completed: %s/%s processed, %s errors (%.2fs)",
            processed,
            len(self.config.assets),
            errors,
            stats["duration_seconds"],
            extra={
                "event_type": "scheduler.cycle_complete",
                "environment": self.environment.value,
                "extra_data": stats,
            },
        )
        return stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _process_asset(self, symbol: str) -> None:
        df = self._fetch_market_data(symbol)
        kline_data = format_ohlcv_for_agents(df)
        current_price = float(df["close"].iloc[-1])

        # NEW: Position monitoring and exit check
        position = self.position_monitor.get_active_position(symbol)
        if position:
            # Calculate prev_close (fallback to current if not enough data)
            prev_close = (
                float(df["close"].iloc[-2]) if len(df) >= 2
                else current_price
            )

            # Update tracking
            self.position_monitor.update_candle_tracking(
                position, current_price, prev_close
            )

            logger.debug(
                "Position tracked: %s (candles=%d)",
                symbol, position.candles_since_entry,
                extra={
                    "event_type": "scheduler.position_tracked",
                    "symbol": symbol,
                    "environment": self.environment.value,
                }
            )

            # Check exit conditions
            should_exit, exit_reason = self._check_exit_conditions(
                position, current_price
            )

            if should_exit:
                self._execute_position_exit(
                    position, exit_reason, current_price, symbol
                )
                logger.info(
                    "Position exit: %s reason=%s price=%.2f",
                    symbol, exit_reason, current_price,
                    extra={
                        "event_type": "scheduler.position_exit",
                        "symbol": symbol,
                        "environment": self.environment.value,
                        "extra_data": {
                            "reason": exit_reason,
                            "exit_price": current_price
                        }
                    }
                )
                return  # Skip LLM analysis after exit

        thread_id = self._make_thread_id(symbol)

        try:
            signal = self.strategy.generate_signal(
                kline_data,
                symbol,
                self.config.timeframe,
                current_price,
                thread_id=thread_id,
            )
        except Exception as exc:  # pragma: no cover - strategy errors
            raise AnalysisError(str(exc)) from exc

        if signal is None:
            logger.info(
                "No action for %s: signal=HOLD",
                symbol,
                extra={
                    "event_type": "scheduler.hold",
                    "symbol": symbol,
                    "environment": self.environment.value,
                },
            )
            return

        trade_signal = self._map_trade_signal(signal)
        if trade_signal == TradeSignal.NEUTRAL:
            logger.info(
                "No action for %s: signal=HOLD",
                symbol,
                extra={
                    "event_type": "scheduler.hold",
                    "symbol": symbol,
                    "environment": self.environment.value,
                },
            )
            return

        db_signal = self._persist_signal(symbol, trade_signal, signal, thread_id)

        try:
            order = self.order_manager.execute_decision(
                symbol=symbol,
                decision=trade_signal,
                confidence=signal.confidence,
                current_price=current_price,
                environment=self.environment,
                trigger_signal_id=db_signal.id if db_signal else None,
            )
        except Exception as exc:
            self.db.rollback()
            raise ExecutionError(str(exc)) from exc

        if order:
            trade_id = self._get_trade_id_for_order(getattr(order, "id", None))

            # NEW: Open ActivePosition for tracking
            # Extract stop loss and take profit from signal (with fallbacks)
            stop_loss = getattr(signal, "stop_loss", None) or (
                current_price * 0.98 if trade_signal == TradeSignal.LONG
                else current_price * 1.02
            )
            take_profit = getattr(signal, "take_profit", None) or (
                current_price * 1.04 if trade_signal == TradeSignal.LONG
                else current_price * 0.96
            )

            self.position_monitor.open_position(
                symbol=symbol,
                side=order.side,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=order.quantity,
                exit_policy="sl_tp_only",
                trade_id=trade_id,
                signal_id=db_signal.id if db_signal else None,
                backtest_run_id=None,
                environment=self.environment,
            )

            logger.info(
                "Order executed: %s %s [%s] [entry=%.2f, sl=%.2f, tp=%.2f]",
                symbol,
                trade_signal.value.upper(),
                self.environment.value,
                current_price,
                stop_loss,
                take_profit,
                extra={
                    "event_type": "scheduler.order_executed",
                    "symbol": symbol,
                    "environment": self.environment.value,
                    "extra_data": {
                        "decision": trade_signal.value,
                        "confidence": signal.confidence,
                        "entry_price": current_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    },
                },
            )
        else:
            self.db.commit()
            logger.info(
                "Decision rejected for %s by execution layer",
                symbol,
                extra={
                    "event_type": "scheduler.order_skipped",
                    "symbol": symbol,
                    "environment": self.environment.value,
                },
            )

    def _fetch_market_data(self, symbol: str) -> pd.DataFrame:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=self.config.lookback_hours)
        try:
            df = self.data_provider.get_ohlc(
                symbol=symbol,
                timeframe=self.config.timeframe,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:  # pragma: no cover - provider errors
            raise DataFetchError(str(exc)) from exc

        if df.empty:
            raise DataFetchError("no data returned")
        if "close" not in df.columns:
            raise DataFetchError("missing close column in OHLC data")
        return df

    def _persist_signal(
        self,
        symbol: str,
        decision: TradeSignal,
        signal: StrategyTradingSignal,
        thread_id: Optional[str],
    ) -> Signal:
        record = Signal(
            symbol=symbol,
            signal=decision,
            confidence=signal.confidence,
            timeframe=self.config.timeframe,
            analysis_summary=signal.reasoning,
            generated_at=datetime.utcnow(),
            environment=self.environment,
            model_provider=settings.AGENT_LLM_PROVIDER,
            model_name=settings.AGENT_LLM_MODEL,
            temperature=settings.AGENT_LLM_TEMPERATURE,
            thread_id=thread_id,
            state_snapshot={
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "trailing_stop_pct": signal.trailing_stop_pct,
            },
        )

        self.db.add(record)
        self.db.flush()
        if getattr(record, "id", None) is None:
            self.db.refresh(record)
        return record

    @staticmethod
    def _map_trade_signal(signal: StrategyTradingSignal) -> TradeSignal:
        decision = signal.decision.upper()
        if decision == "LONG":
            return TradeSignal.LONG
        if decision == "SHORT":
            return TradeSignal.SHORT
        return TradeSignal.NEUTRAL

    def _make_thread_id(self, symbol: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        return f"scheduler_{symbol}_{timestamp}"

    def _build_cycle_stats(
        self, cycle_start: datetime, processed: int, errors: int
    ) -> Dict[str, float]:
        duration = (datetime.utcnow() - cycle_start).total_seconds()
        return {
            "processed": processed,
            "errors": errors,
            "duration_seconds": duration,
            "total": len(self.config.assets),
        }

    def _upsert_heartbeat_start(
        self, started_at: datetime
    ) -> Optional[SchedulerHeartbeat]:
        try:
            existing = (
                self.db.query(SchedulerHeartbeat)
                .filter(SchedulerHeartbeat.environment == self.environment)
                .order_by(SchedulerHeartbeat.id)
                .first()
            )
            if existing is not None:
                if existing.status == "running" and existing.completed_at is None:
                    previous_started = (
                        existing.timestamp.isoformat()
                        if existing.timestamp is not None
                        else "unknown"
                    )
                    existing.error_message = (
                        "Recovered stale running heartbeat from "
                        f"{previous_started} before starting a new cycle."
                    )
                else:
                    existing.error_message = None
                existing.timestamp = started_at
                existing.status = "running"
                existing.completed_at = None
                existing.assets = list(self.config.assets)
                existing.stats = None
                self.db.commit()
                self.db.refresh(existing)
                return existing

            hb = SchedulerHeartbeat(
                timestamp=started_at,
                status="running",
                environment=self.environment,
                assets=list(self.config.assets),
                error_message=None,
            )
            self.db.add(hb)
            self.db.commit()
            self.db.refresh(hb)
            return hb
        except Exception:
            logger.exception("Heartbeat start failed; continuing cycle")
            return None

    def _upsert_heartbeat_complete(
        self, heartbeat: Optional[SchedulerHeartbeat], stats: Dict[str, float]
    ) -> None:
        if heartbeat is None:
            return
        try:
            last_trade = (
                self.db.query(Trade)
                .filter(Trade.environment == self.environment)
                .order_by(Trade.id.desc())
                .first()
            )
            heartbeat.status = "completed"
            heartbeat.completed_at = datetime.utcnow()
            heartbeat.stats = stats
            heartbeat.last_trade_id = last_trade.id if last_trade else None
            self.db.commit()
        except Exception:
            logger.exception("Heartbeat complete failed")

    def _upsert_heartbeat_error(
        self,
        heartbeat: Optional[SchedulerHeartbeat],
        stats: Dict[str, float],
        error_message: str,
    ) -> None:
        if heartbeat is None:
            return
        try:
            heartbeat.status = "error"
            heartbeat.completed_at = datetime.utcnow()
            heartbeat.stats = stats
            heartbeat.error_message = error_message
            self.db.commit()
        except Exception:
            logger.exception("Heartbeat error update failed")

    def _get_trade_id_for_order(self, order_id: Optional[int]) -> Optional[int]:
        if order_id is None or not isinstance(order_id, int):
            return None
        trade = self.db.query(Trade).filter(Trade.order_id == order_id).first()
        return trade.id if trade is not None else None

    def _check_exit_conditions(
        self, position, current_price: float
    ) -> tuple[bool, Optional[str]]:
        """
        Check if position should exit based on stop loss, take profit, or max hold.

        Args:
            position: ActivePosition instance
            current_price: Current market price

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        from quantagent.models import OrderSide

        # Stop loss check
        if position.side == OrderSide.BUY:  # LONG position
            if current_price <= float(position.stop_loss):
                return (True, "stop_loss")
        else:  # SHORT position
            if current_price >= float(position.stop_loss):
                return (True, "stop_loss")

        # Take profit check
        if position.side == OrderSide.BUY:
            if current_price >= float(position.take_profit):
                return (True, "take_profit")
        else:
            if current_price <= float(position.take_profit):
                return (True, "take_profit")

        # Max hold check
        if position.max_hold_candles:
            if position.candles_since_entry >= position.max_hold_candles:
                return (True, "max_hold")

        return (False, None)

    def _execute_position_exit(
        self,
        position,
        exit_reason: str,
        exit_price: float,
        symbol: str,
    ) -> None:
        """
        Execute position exit by closing position and placing exit order.

        Args:
            position: ActivePosition to close
            exit_reason: Reason for exit (stop_loss, take_profit, max_hold)
            exit_price: Price at exit
            symbol: Trading symbol

        Raises:
            ExecutionError: If exit order fails
        """
        from quantagent.models import OrderSide, Trade

        # Determine exit signal (opposite of position)
        exit_signal = (
            TradeSignal.SHORT
            if position.side == OrderSide.BUY
            else TradeSignal.LONG
        )

        # Close position in tracking system
        self.position_monitor.close_position(
            position, reason=exit_reason, exit_price=exit_price
        )

        # Execute exit order
        try:
            _order = self.order_manager.execute_decision(
                symbol=symbol,
                decision=exit_signal,
                confidence=1.0,  # Exit is mandatory, not confidence-based
                current_price=exit_price,
                environment=self.environment,
                trigger_signal_id=position.signal_id,
            )
            # Note: Exit order execution is fire-and-forget in current architecture
            # OrderManager logs the result internally
        except Exception as exc:
            self.db.rollback()
            logger.error(
                "Exit order failed for %s: %s",
                symbol,
                exc,
                extra={
                    "event_type": "scheduler.exit_order_error",
                    "symbol": symbol,
                    "environment": self.environment.value,
                },
            )
            raise ExecutionError(f"Exit failed: {exc}") from exc

        # Update Trade record with exit reason
        if position.trade_id:
            trade = self.db.query(Trade).filter(Trade.id == position.trade_id).first()
            if trade:
                trade.exit_signal = exit_reason
                trade.closed_at = position.closed_at
                self.db.commit()
