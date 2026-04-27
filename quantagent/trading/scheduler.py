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
from quantagent.models import Environment, SchedulerHeartbeat, Signal, TradeSignal
from quantagent.static_util import format_ohlcv_for_agents
from quantagent.strategy.base import TradingSignal as StrategyTradingSignal
from quantagent.strategy.llm_agent_strategy import LLMAgentStrategy
from quantagent.trading.order_manager import OrderManager
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

        # Write heartbeat at cycle start
        heartbeat = self._upsert_heartbeat_start(cycle_start)

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

        duration = (datetime.utcnow() - cycle_start).total_seconds()
        stats = {
            "processed": processed,
            "errors": errors,
            "duration_seconds": duration,
            "total": len(self.config.assets),
        }
        self.last_run_stats = stats

        # Update heartbeat at cycle end
        self._upsert_heartbeat_complete(heartbeat, stats)

        logger.info(
            "Analysis cycle completed: %s/%s processed, %s errors (%.2fs)",
            processed,
            len(self.config.assets),
            errors,
            duration,
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
            logger.info(
                "Order executed: %s %s [%s]",
                symbol,
                trade_signal.value.upper(),
                self.environment.value,
                extra={
                    "event_type": "scheduler.order_executed",
                    "symbol": symbol,
                    "environment": self.environment.value,
                    "extra_data": {
                        "decision": trade_signal.value,
                        "confidence": signal.confidence,
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

    def _upsert_heartbeat_start(self, cycle_start: datetime) -> Optional[SchedulerHeartbeat]:
        """
        Write or update heartbeat at cycle start.

        Args:
            cycle_start: Timestamp of cycle start

        Returns:
            SchedulerHeartbeat instance or None if write fails
        """
        try:
            # Upsert pattern: single row per environment
            heartbeat = (
                self.db.query(SchedulerHeartbeat)
                .filter_by(environment=self.environment)
                .first()
            )

            if heartbeat:
                # Update existing
                heartbeat.timestamp = cycle_start
                heartbeat.completed_at = None
                heartbeat.status = "running"
                heartbeat.assets = self.config.assets
                heartbeat.stats = None
                heartbeat.error_message = None
            else:
                # Create new
                heartbeat = SchedulerHeartbeat(
                    timestamp=cycle_start,
                    status="running",
                    environment=self.environment,
                    assets=self.config.assets,
                )
                self.db.add(heartbeat)

            self.db.commit()
            logger.debug(
                "Heartbeat started for environment=%s",
                self.environment.value,
                extra={"event_type": "scheduler.heartbeat_start"},
            )
            return heartbeat

        except Exception as exc:  # pragma: no cover - defensive
            # Heartbeat failure should not break scheduler
            logger.warning(
                "Failed to write heartbeat: %s",
                exc,
                extra={"event_type": "scheduler.heartbeat_error"},
            )
            try:
                self.db.rollback()
            except Exception:  # pragma: no cover
                pass
            return None

    def _upsert_heartbeat_complete(
        self, heartbeat: Optional[SchedulerHeartbeat], stats: Dict[str, float]
    ) -> None:
        """
        Update heartbeat at cycle end.

        Args:
            heartbeat: Heartbeat instance from cycle start (may be None if write failed)
            stats: Cycle statistics
        """
        if heartbeat is None:
            return

        try:
            heartbeat.completed_at = datetime.utcnow()
            heartbeat.status = "completed"
            heartbeat.stats = stats

            # Get last trade ID if any trades exist
            from quantagent.models import Trade

            last_trade = (
                self.db.query(Trade)
                .filter_by(environment=self.environment)
                .order_by(Trade.id.desc())
                .first()
            )
            if last_trade:
                heartbeat.last_trade_id = last_trade.id

            self.db.commit()
            logger.debug(
                "Heartbeat completed for environment=%s",
                self.environment.value,
                extra={"event_type": "scheduler.heartbeat_complete"},
            )

        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to update heartbeat: %s",
                exc,
                extra={"event_type": "scheduler.heartbeat_error"},
            )
            try:
                self.db.rollback()
            except Exception:  # pragma: no cover
                pass
