"""Backtesting engine for strategy validation."""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from quantagent.agent_models import TradingDecision
from quantagent.data.asset_types import AssetType, get_asset_type
from quantagent.data.market_calendar import get_market_calendar
from quantagent.data.provider import DataProvider
from quantagent.database import SessionLocal
from quantagent.models import (
    ActivePosition,
    BacktestRun,
    Environment,
    OrderSide,
    Signal,
    Trade,
    TradeSignal,
)
from quantagent.static_util import format_ohlcv_for_agents
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.strategy.base import TradingStrategy
from quantagent.strategy.llm_agent_strategy import LLMAgentStrategy
from quantagent.trading.position_monitor import PositionMonitor

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_return_pct: float

    # Phase 4: Active Position Monitoring metrics
    agent_invocations: int = 0
    invocations_saved: int = 0
    invocation_reduction_pct: float = 0.0
    mean_directional_accuracy: float = 0.0
    accuracy_by_candle: Dict[int, float] = None
    close_reasons: Dict[str, int] = None

    def __post_init__(self):
        if self.accuracy_by_candle is None:
            self.accuracy_by_candle = {}
        if self.close_reasons is None:
            self.close_reasons = {}


class Backtest:
    """
    Backtesting engine for validating trading strategies.

    Workflow:
    1. Loop through historical dates
    2. Fetch OHLC data for each date using DataProvider (cached)
    3. Execute analysis using TradingGraph
    4. Simulate trade execution using OrderManager
    5. Track portfolio performance
    6. Calculate and persist metrics

    Features:
    - Uses DataProvider for 10x faster data access (caching)
    - Executes same agents as live trading
    - Stores full provenance (config snapshot, model metadata)
    - Supports replay with different risk/portfolio profiles
    """

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        assets: List[str],
        timeframe: str = "1h",
        initial_capital: float = 100000.0,
        config: Optional[Dict] = None,
        db_session: Optional[Session] = None,
        use_checkpointing: bool = False,
        strategy: Optional[TradingStrategy] = None,
    ):
        """
        Initialize Backtest.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            assets: List of asset symbols (e.g., ["BTC", "SPX", "CL"])
            timeframe: Timeframe for analysis (e.g., "1h", "4h", "1d")
            initial_capital: Starting portfolio value
            config: Configuration dict (portfolio/risk params, model settings)
            db_session: Database session (creates new if None)
            use_checkpointing: Enable LangGraph checkpointing for state persistence
            strategy: Optional TradingStrategy. If None, uses LLMAgentStrategy with TradingGraph
        """
        self.start_date = start_date
        self.end_date = end_date
        self.assets = assets
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.config = config or {}
        self.use_checkpointing = use_checkpointing

        # Database
        self.db = db_session or SessionLocal()
        self._own_session = db_session is None

        # Data provider (caching layer)
        self.data_provider = DataProvider(self.db)

        # Resolve config via StrategyAssembler and build components (unify DB session)
        from quantagent import settings

        resolved = StrategyAssembler.from_snapshot(
            {
                "initial_cash": initial_capital,
                "base_position_pct": self.config.get("base_position_pct", settings.TRADING_BASE_POSITION_PCT),
                "max_daily_loss_pct": self.config.get("max_daily_loss_pct", settings.TRADING_MAX_DAILY_LOSS_PCT),
                "max_position_pct": self.config.get("max_position_pct", settings.TRADING_MAX_POSITION_PCT),
                "slippage_pct": self.config.get("slippage_pct", settings.TRADING_SLIPPAGE_PCT),
                # Normalize model fields into generic ones; accept both
                "model_provider": self.config.get(
                    "agent_llm_provider", self.config.get("model_provider", settings.AGENT_LLM_PROVIDER)
                ),
                "model_name": self.config.get(
                    "agent_llm_model", self.config.get("model_name", settings.AGENT_LLM_MODEL)
                ),
                "temperature": self.config.get(
                    "agent_llm_temperature", self.config.get("temperature", settings.AGENT_LLM_TEMPERATURE)
                ),
                "use_checkpointing": use_checkpointing,
                "universe": self.config.get("universe", settings.get_trading_universe()),
            },
            environment=Environment.BACKTEST,
        )
        components = StrategyAssembler.build_components(resolved, db_session=self.db)

        # Trading graph (analysis engine)
        self.trading_graph = components.graph

        # Trading strategy (use provided or default to LLMAgentStrategy)
        self.strategy = strategy or LLMAgentStrategy(self.trading_graph)

        # Position monitor for active position tracking
        self.position_monitor = PositionMonitor(self.db, environment=Environment.BACKTEST)

        # Trading components
        self.portfolio = components.portfolio_manager
        self.order_manager = components.order_manager

        # Backtest state
        self.current_date = start_date
        self.backtest_run_id: Optional[int] = None
        self._replay_trade_order_ids: Optional[set[int]] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

        # Phase 4: Invocation tracking
        self.agent_invocations = 0
        self.total_candles_processed = 0

        # Market hours filtering
        self.market_hours_filter = self.config.get("market_hours_filter", True)
        self._market_calendar = (
            get_market_calendar() if self.market_hours_filter else None
        )

        # Cache asset types for each symbol
        self._asset_types: Dict[str, AssetType] = {
            asset: get_asset_type(asset) for asset in assets
        }

    def run(self, name: Optional[str] = None) -> BacktestMetrics:
        """
        Run backtest and return metrics.

        Args:
            name: Optional name for this backtest run

        Returns:
            BacktestMetrics with performance statistics
        """
        logger.info(
            f"Starting backtest: {self.start_date} to {self.end_date}",
            extra={"event_type": "backtest_start", "environment": "backtest"},
        )
        logger.info(
            f"Assets: {self.assets}, Timeframe: {self.timeframe}",
            extra={"event_type": "backtest_start"},
        )
        logger.info(
            f"Initial capital: ${self.initial_capital:,.2f}",
            extra={"event_type": "backtest_start"},
        )
        logger.info(
            f"Market hours filter: {self.market_hours_filter}",
            extra={"event_type": "backtest_start"},
        )

        self._replay_trade_order_ids = None

        # NEW: Clean up any stale active positions BEFORE creating backtest run
        # This prevents contamination from previous incomplete runs
        self._cleanup_stale_positions()

        # Create backtest run record
        self._create_backtest_run(name)

        # Process each asset with its filtered date range
        total_periods = 0
        for asset in self.assets:
            asset_dates = self._get_date_range_for_asset(asset)
            total_periods += len(asset_dates)

            asset_type = self._asset_types.get(asset, AssetType.UNKNOWN)
            logger.info(
                f"Asset {asset} ({asset_type.value}): {len(asset_dates)} analysis periods",
                extra={"event_type": "backtest_start"},
            )

        logger.info(
            f"Backtesting {total_periods} total analysis periods",
            extra={"event_type": "backtest_start"},
        )

        # Track progress across all periods
        periods_completed = 0

        # Loop through assets (outer) and their dates (inner)
        for asset in self.assets:
            asset_dates = self._get_date_range_for_asset(asset)

            for i, current_date in enumerate(asset_dates):
                self.current_date = current_date

                # Reset daily P&L tracking at start of each day
                if i == 0 or current_date.date() != asset_dates[i - 1].date():
                    self.order_manager.reset_daily_tracker()

                try:
                    self._analyze_and_trade(asset, current_date)
                except Exception as e:
                    logger.error(
                        f"Error analyzing {asset} at {current_date}: {e}",
                        exc_info=True,
                        extra={"event_type": "backtest_error", "symbol": asset},
                    )
                    continue

                # Record equity at end of period
                self._record_equity(current_date)

                periods_completed += 1

                # Log progress
                if periods_completed % 100 == 0 or periods_completed == total_periods:
                    progress = (periods_completed / total_periods) * 100
                    logger.info(
                        f"Progress: {progress:.1f}% ({periods_completed}/{total_periods})",
                        extra={"event_type": "backtest_progress"},
                    )

        # Close any remaining active positions to prevent stale position contamination
        self._close_remaining_positions()

        # Calculate metrics after forcing final exits so linked trades carry
        # realized P&L from backtest-end closures as well.
        metrics = self._calculate_metrics()

        # Update backtest run with results
        self._update_backtest_run(metrics)

        logger.info(
            f"Backtest complete: {metrics.total_trades} trades, Win rate: {metrics.win_rate:.2%}",
            extra={"event_type": "backtest_end", "environment": "backtest"},
        )
        logger.info(
            f"Total P&L: ${metrics.total_pnl:,.2f} ({metrics.total_return_pct:.2%})",
            extra={"event_type": "backtest_end"},
        )

        return metrics

    def run_replay(
        self,
        source_run_id: int,
        name: Optional[str] = None,
    ) -> "BacktestMetrics":
        """
        Execute replay using signals stored from source_run_id.

        No LLM calls are made — signals are read from the DB and re-applied with
        the current portfolio/risk config (set at Backtest init time).

        Args:
            source_run_id: ID of the completed BacktestRun to replay.
            name: Optional name for the new replay BacktestRun record.

        Returns:
            BacktestMetrics from replay execution.

        Raises:
            ValueError: If source run does not exist or has no scoped signals.
        """
        source_run = (
            self.db.query(BacktestRun).filter(BacktestRun.id == source_run_id).first()
        )
        if source_run is None:
            raise ValueError(f"Source BacktestRun {source_run_id} not found")

        # Load only signals scoped to this source run — prevents cross-run contamination.
        signals = (
            self.db.query(Signal)
            .filter(Signal.backtest_run_id == source_run_id)
            .all()
        )

        logger.info(
            f"[REPLAY] Starting replay: source={source_run_id}, signals={len(signals)}",
            extra={"event_type": "replay_start"},
        )

        if not signals:
            raise ValueError(
                f"No stored signals found for source run {source_run_id}. "
                "Run the source backtest first (signals must have backtest_run_id set)."
            )

        # Build lookup: (symbol, generated_at) -> signal
        signal_map: Dict = {}
        for sig in signals:
            signal_map[(sig.symbol, sig.generated_at)] = sig

        # Reset in-memory state
        self.equity_curve = []
        self.trades = []
        self.agent_invocations = 0
        self.total_candles_processed = 0
        self._replay_trade_order_ids = set()

        self._cleanup_stale_positions()

        # Override date/asset/timeframe from source run so replay matches it exactly
        self.start_date = source_run.start_date
        self.end_date = source_run.end_date
        self.assets = source_run.assets
        self.timeframe = source_run.timeframe

        self._create_backtest_run(
            name or f"Replay_{source_run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            replay_source_run_id=source_run_id,
        )

        for asset in self.assets:
            asset_dates = self._get_date_range_for_asset(asset)

            for i, current_date in enumerate(asset_dates):
                self.current_date = current_date

                if i == 0 or current_date.date() != asset_dates[i - 1].date():
                    self.order_manager.reset_daily_tracker()

                try:
                    self._replay_and_trade(asset, current_date, signal_map)
                except Exception as e:
                    logger.error(
                        f"[REPLAY] Error replaying {asset} at {current_date}: {e}",
                        exc_info=True,
                        extra={"event_type": "replay_error", "symbol": asset},
                    )
                    continue

                self._record_equity(current_date)

        self._close_remaining_positions()
        metrics = self._calculate_metrics()
        self._update_backtest_run(metrics)

        logger.info(
            f"[REPLAY] Completed: {metrics.total_trades} trades, P&L ${metrics.total_pnl:,.2f}",
            extra={"event_type": "replay_end"},
        )
        return metrics

    def _replay_and_trade(
        self, asset: str, current_date: datetime, signal_map: Dict
    ) -> None:
        """Execute a single candle in replay mode — uses stored signal, no LLM call."""
        lookback_days = 30
        data_start = current_date - timedelta(days=lookback_days)
        df = self.data_provider.get_ohlc(
            symbol=asset,
            timeframe=self.timeframe,
            start_date=data_start,
            end_date=current_date,
        )

        if df.empty or len(df) < 2:
            return

        current_price = float(df.iloc[-1]["close"])
        active_pos = self.position_monitor.get_active_position(asset)
        self.total_candles_processed += 1

        if active_pos:
            should_exit, reason = self.strategy.should_exit(
                active_pos, current_price, df
            )
            if should_exit:
                self._close_position_with_trade_sync(
                    active_pos, reason, current_price
                )
                logger.debug(
                    f"[REPLAY] {asset}: Closed position ({reason}) @ ${current_price:.2f}"
                )
            else:
                prev_close = float(df.iloc[-2]["close"])
                self.position_monitor.update_candle_tracking(
                    active_pos, current_price, prev_close
                )
                return

        stored_signal = signal_map.get((asset, current_date))
        if stored_signal is None or stored_signal.signal == TradeSignal.NEUTRAL:
            return

        logger.debug(
            f"[REPLAY] {asset} @ {current_date}: using stored signal {stored_signal.id}"
        )

        trading_signal = stored_signal.signal
        order = self.order_manager.execute_decision(
            symbol=asset,
            decision=trading_signal,
            confidence=stored_signal.confidence,
            current_price=current_price,
            environment=Environment.BACKTEST,
            trigger_signal_id=stored_signal.id,
        )

        if order and order.filled_quantity and order.filled_quantity > 0:
            if order.id is not None and self._replay_trade_order_ids is not None:
                self._replay_trade_order_ids.add(order.id)

            side = (
                OrderSide.BUY if trading_signal == TradeSignal.LONG else OrderSide.SELL
            )
            trade_id = None
            if order.id:
                trade = self.db.query(Trade).filter(Trade.order_id == order.id).first()
                if trade:
                    trade_id = trade.id

            self.position_monitor.open_position(
                symbol=asset,
                side=side,
                entry_price=current_price,
                stop_loss=(
                    current_price * 0.98
                    if side == OrderSide.BUY
                    else current_price * 1.02
                ),
                take_profit=(
                    current_price * 1.03
                    if side == OrderSide.BUY
                    else current_price * 0.97
                ),
                quantity=order.filled_quantity,
                exit_policy="sl_tp_only",
                trade_id=trade_id,
                signal_id=stored_signal.id,
            )

            logger.info(
                f"[REPLAY] Executed {trading_signal.value} for {asset} "
                f"@ ${current_price:.2f}, qty: {order.filled_quantity}",
                extra={"event_type": "replay_trade", "symbol": asset},
            )

    def _create_backtest_run(
        self, name: Optional[str], replay_source_run_id: Optional[int] = None
    ) -> None:
        """Create BacktestRun record in database."""
        run = BacktestRun(
            name=name or f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timeframe=self.timeframe,
            assets=self.assets,
            start_date=self.start_date,
            end_date=self.end_date,
            config_snapshot=self._build_config_snapshot(),
            replay_source_run_id=replay_source_run_id,
        )

        self.db.add(run)
        self.db.commit()
        self.backtest_run_id = run.id

        if self.position_monitor is not None:
            self.position_monitor.set_backtest_run_id(self.backtest_run_id)

        logger.info(
            f"Created backtest run #{self.backtest_run_id}: {run.name}",
            extra={"event_type": "backtest_init"},
        )

    def _build_config_snapshot(self) -> Dict:
        """Build immutable config snapshot for reproducibility."""
        from quantagent import settings

        # Re-generate snapshot via assembler to keep alignment
        resolved = StrategyAssembler.from_snapshot(
            {
                "initial_cash": self.initial_capital,
                "base_position_pct": self.config.get("base_position_pct", settings.TRADING_BASE_POSITION_PCT),
                "max_daily_loss_pct": self.config.get("max_daily_loss_pct", settings.TRADING_MAX_DAILY_LOSS_PCT),
                "max_position_pct": self.config.get("max_position_pct", settings.TRADING_MAX_POSITION_PCT),
                "slippage_pct": self.config.get("slippage_pct", settings.TRADING_SLIPPAGE_PCT),
                "model_provider": self.config.get(
                    "agent_llm_provider", self.config.get("model_provider", settings.AGENT_LLM_PROVIDER)
                ),
                "model_name": self.config.get(
                    "agent_llm_model", self.config.get("model_name", settings.AGENT_LLM_MODEL)
                ),
                "temperature": self.config.get(
                    "agent_llm_temperature", self.config.get("temperature", settings.AGENT_LLM_TEMPERATURE)
                ),
                "use_checkpointing": self.use_checkpointing,
                "universe": self.config.get("universe", settings.get_trading_universe()),
            },
            environment=Environment.BACKTEST,
        )
        return StrategyAssembler.config_snapshot(resolved)

    def _get_date_range(self) -> List[datetime]:
        """
        Get list of dates to backtest.

        For hourly/intraday: every N hours
        For daily: every day
        """
        dates = []
        current = self.start_date

        # Determine step size based on timeframe
        if self.timeframe in ["1h", "4h"]:
            step_hours = int(self.timeframe.replace("h", ""))
            step = timedelta(hours=step_hours)
        elif self.timeframe == "1d":
            step = timedelta(days=1)
        elif self.timeframe == "1w":
            step = timedelta(weeks=1)
        else:
            # Default to hourly
            step = timedelta(hours=1)

        while current <= self.end_date:
            dates.append(current)
            current += step

        return dates

    def _get_date_range_for_asset(self, asset: str) -> List[datetime]:
        """
        Get date range filtered by market hours for specific asset.

        Args:
            asset: Asset symbol

        Returns:
            List of valid trading timestamps for this asset
        """
        all_dates = self._get_date_range()

        if not self.market_hours_filter or self._market_calendar is None:
            return all_dates

        asset_type = self._asset_types.get(asset, AssetType.UNKNOWN)

        return self._market_calendar.filter_to_trading_hours(all_dates, asset_type)

    def _analyze_and_trade(self, asset: str, current_date: datetime) -> None:
        """
        Execute analysis and trade for a single asset at a given time.

        NEW FLOW (Hybrid Model with PositionMonitor):
        1. Check for active position
        2. If active: strategy.should_exit() → close or update tracking (NO invoke)
        3. If not active: strategy.generate_signal() → open position if signal != HOLD

        Args:
            asset: Asset symbol
            current_date: Current backtest date
        """
        # Get historical data for analysis (need lookback period)
        lookback_bars = self.strategy.required_history_bars
        if lookback_bars <= 0:
            lookback_bars = 30

        data_start = current_date - timedelta(
            days=self._bars_to_calendar_days(lookback_bars)
        )

        df = self.data_provider.get_ohlc(
            symbol=asset,
            timeframe=self.timeframe,
            start_date=data_start,
            end_date=current_date,
        )

        if df.empty or len(df) < lookback_bars:
            logger.warning(
                f"Insufficient data for {asset} at {current_date} "
                f"(got {len(df)}, need {lookback_bars})",
                extra={"event_type": "backtest_data_warning", "symbol": asset},
            )
            return

        current_price = float(df.iloc[-1]["close"])

        # Check for active position
        active_pos = self.position_monitor.get_active_position(asset)

        # Phase 4: Count total candles processed
        self.total_candles_processed += 1

        if active_pos:
            # Position active: check exit conditions via strategy
            should_exit, reason = self.strategy.should_exit(
                active_pos, current_price, df
            )

            if should_exit:
                self._close_position_with_trade_sync(
                    active_pos, reason, current_price
                )
                logger.info(
                    f"{asset}: Closed position - {reason} @ ${current_price:.2f}"
                )
                # Continue to potentially open new position below
            else:
                # Position still active: update tracking only (NO INVOKE)
                prev_close = (
                    float(df.iloc[-2]["close"]) if len(df) >= 2 else current_price
                )
                self.position_monitor.update_candle_tracking(
                    active_pos, current_price, prev_close
                )
                return  # Early return - invocation saved

        # No active position (or just closed): generate signal
        if isinstance(self.strategy, LLMAgentStrategy):
            kline_data = format_ohlcv_for_agents(df)
        else:
            kline_data = df.to_dict(orient="records")

        # Phase 4: Count agent invocations
        self.agent_invocations += 1

        # Generate thread_id for checkpointing (if enabled)
        thread_id = None
        if self.use_checkpointing:
            from quantagent.strategy.assembler import StrategyAssembler

            thread_id = StrategyAssembler.make_thread_id(
                self.backtest_run_id, asset, current_date
            )

        signal_kwargs = {}
        if thread_id is not None:
            signal_kwargs["thread_id"] = thread_id

        signal = self.strategy.generate_signal(
            kline_data, asset, self.timeframe, current_price, **signal_kwargs
        )

        if signal is None or signal.decision == "HOLD":
            return

        # Execute trade
        trading_signal = (
            TradeSignal.LONG if signal.decision == "LONG" else TradeSignal.SHORT
        )

        # Store signal in database
        db_signal = self._create_signal_from_strategy(
            asset=asset,
            decision=trading_signal,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            current_date=current_date,
        )

        # Execute order
        order = self.order_manager.execute_decision(
            symbol=asset,
            decision=trading_signal,
            confidence=signal.confidence,
            current_price=current_price,
            environment=Environment.BACKTEST,
            trigger_signal_id=db_signal.id if db_signal else None,
        )

        if order and order.filled_quantity and order.filled_quantity > 0:
            # Create ActivePosition
            side = (
                OrderSide.BUY if trading_signal == TradeSignal.LONG else OrderSide.SELL
            )

            # Get trade_id from order
            trade_id = None
            if order.id:
                trade = self.db.query(Trade).filter(Trade.order_id == order.id).first()
                if trade:
                    trade_id = trade.id

            self.position_monitor.open_position(
                symbol=asset,
                side=side,
                entry_price=signal.entry_price or current_price,
                stop_loss=signal.stop_loss
                or (
                    current_price * 0.98
                    if side == OrderSide.BUY
                    else current_price * 1.02
                ),
                take_profit=signal.take_profit
                or (
                    current_price * 1.03
                    if side == OrderSide.BUY
                    else current_price * 0.97
                ),
                quantity=order.filled_quantity,
                exit_policy=signal.exit_policy.value,
                trade_id=trade_id,
                signal_id=db_signal.id if db_signal else None,
                trailing_stop_pct=signal.trailing_stop_pct,
                max_hold_candles=signal.max_hold_candles,
            )

            logger.info(
                f"Executed {trading_signal.value} for {asset} "
                f"@ "
                f"${current_price:.2f}, qty: {order.filled_quantity}"
            )

    def _bars_to_calendar_days(self, bars: int) -> int:
        """Convert required trading bars to a calendar-day lookback window."""
        if self.timeframe == "1d":
            return math.ceil(bars * 365 / 252)
        if self.timeframe == "1h":
            return math.ceil(bars / 6.5 * 7 / 5)
        if self.timeframe == "4h":
            return math.ceil(bars * 4 / 6.5 * 7 / 5)
        return bars * 2

    def _parse_decision(self, decision: TradingDecision) -> (TradeSignal, float):
        """Parse decision text to extract LONG/SHORT/HOLD and confidence."""
        decision_upper = decision.decision.upper()
        signal = TradeSignal.NEUTRAL

        if "LONG" in decision_upper or "BUY" in decision_upper:
            signal = TradeSignal.LONG
        elif "SHORT" in decision_upper or "SELL" in decision_upper:
            signal = TradeSignal.SHORT
        else:
            signal = TradeSignal.NEUTRAL

        ind = float(decision.confidence)

        return signal, ind

    def _create_signal_from_strategy(
        self,
        asset: str,
        decision: TradeSignal,
        confidence: float,
        reasoning: str,
        current_date: datetime,
    ) -> Optional[Signal]:
        """Create Signal record from strategy output (simplified)."""
        from quantagent import settings

        try:
            signal = Signal(
                symbol=asset,
                signal=decision,
                confidence=confidence,
                timeframe=self.timeframe,
                analysis_summary=reasoning,
                generated_at=current_date,
                environment=Environment.BACKTEST,
                backtest_run_id=self.backtest_run_id,
                model_provider=self.config.get("agent_llm_provider", settings.AGENT_LLM_PROVIDER),
                model_name=self.config.get("agent_llm_model", settings.AGENT_LLM_MODEL),
                temperature=self.config.get("agent_llm_temperature", settings.AGENT_LLM_TEMPERATURE),
            )

            self.db.add(signal)
            self.db.commit()
            return signal

        except Exception as e:
            logger.error(f"Error creating signal: {e}", exc_info=True)
            return None

    def _create_signal(
        self,
        asset: str,
        decision: TradeSignal,
        confidence: float,
        result: Dict,
        current_date: datetime,
        thread_id: Optional[str] = None,
    ) -> Optional[Signal]:
        """Create and persist Signal record."""
        try:
            # Extract technical indicators
            rsi = None
            macd = None
            stochastic = None
            roc = None
            williams_r = None
            pattern = None
            trend = None

            if "rsi" in result and result["rsi"]:
                rsi = (
                    float(result["rsi"][-1])
                    if isinstance(result["rsi"], list)
                    else float(result["rsi"])
                )

            if "macd" in result and result["macd"]:
                macd = (
                    float(result["macd"][-1])
                    if isinstance(result["macd"], list)
                    else float(result["macd"])
                )

            indicator_report = result.get("indicator_report")
            if indicator_report and hasattr(indicator_report, "rsi"):
                rsi = float(indicator_report.rsi)
            if indicator_report and hasattr(indicator_report, "macd"):
                macd = float(indicator_report.macd)
            if indicator_report and hasattr(indicator_report, "stochastic"):
                stochastic = float(indicator_report.stochastic)
            if indicator_report and hasattr(indicator_report, "roc"):
                roc = float(indicator_report.roc)
            if indicator_report and hasattr(indicator_report, "willr"):
                williams_r = float(indicator_report.willr)

            # Get pattern and trend from reports
            pattern_report = result.get("pattern_report")
            if pattern_report and hasattr(pattern_report, "primary_pattern"):
                pattern = pattern_report.primary_pattern

            trend_report = result.get("trend_report")
            if trend_report and hasattr(trend_report, "trend_direction"):
                trend = trend_report.trend_direction

            # Create signal
            from quantagent import settings

            signal = Signal(
                symbol=asset,
                signal=decision,
                confidence=confidence,
                timeframe=self.timeframe,
                rsi=rsi,
                macd=macd,
                stochastic=stochastic,
                roc=roc,
                williams_r=williams_r,
                pattern=pattern,
                trend=trend,
                analysis_summary=result.get("reasoning", ""),
                generated_at=current_date,
                environment=Environment.BACKTEST,
                backtest_run_id=self.backtest_run_id,
                thread_id=thread_id,
                model_provider=self.config.get("agent_llm_provider", settings.AGENT_LLM_PROVIDER),
                model_name=self.config.get("agent_llm_model", settings.AGENT_LLM_MODEL),
                temperature=self.config.get("agent_llm_temperature", settings.AGENT_LLM_TEMPERATURE),
            )

            self.db.add(signal)
            self.db.commit()

            return signal

        except Exception as e:
            logger.error(
                f"Error creating signal: {e}",
                exc_info=True,
                extra={"event_type": "backtest_error"},
            )
            return None

    def _record_equity(self, current_date: datetime) -> None:
        """Record equity curve data point."""
        total_value = self.portfolio.get_total_value()

        self.equity_curve.append(
            {
                "date": current_date,
                "equity": total_value,
                "cash": self.portfolio.cash,
                "positions_value": total_value - self.portfolio.cash,
            }
        )

    def _close_remaining_positions(self) -> None:
        """Close any remaining active positions at the end of backtest to prevent stale position contamination."""
        for asset in self.assets:
            active_pos = self.position_monitor.get_active_position(asset)
            if active_pos:
                # Get final price for closing
                df = self.data_provider.get_ohlc(
                    symbol=asset,
                    timeframe=self.timeframe,
                    start_date=self.end_date - timedelta(days=1),
                    end_date=self.end_date,
                )
                if not df.empty:
                    final_price = float(df.iloc[-1]["close"])
                    self._close_position_with_trade_sync(
                        active_pos, "backtest_end", final_price
                    )
                    logger.info(
                        f"Closed remaining position for {asset} at backtest end @ ${final_price:.2f}",
                        extra={"event_type": "backtest_position_cleanup", "symbol": asset},
                    )

    def _cleanup_stale_positions(self) -> None:
        """
        Close ALL stale active positions for assets in this backtest.

        Called at backtest START to prevent contamination from previous incomplete runs.
        Unlike _close_remaining_positions() which only closes positions for CURRENT
        backtest_run_id, this closes ALL active positions regardless of run_id.
        """
        if not self.assets:
            return

        logger.info(
            "Checking for stale active positions before backtest start",
            extra={"event_type": "stale_position_check"},
        )

        total_cleaned = 0

        for asset in self.assets:
            # Query ALL active positions for this asset + environment
            stale_positions = (
                self.db.query(ActivePosition)
                .filter(
                    ActivePosition.symbol == asset,
                    ActivePosition.is_active.is_(True),
                    ActivePosition.environment == Environment.BACKTEST,
                )
                .all()
            )

            if not stale_positions:
                continue

            logger.warning(
                f"Found {len(stale_positions)} stale active positions for {asset}",
                extra={
                    "event_type": "stale_positions_found",
                    "symbol": asset,
                    "count": len(stale_positions),
                    "run_ids": [p.backtest_run_id for p in stale_positions],
                },
            )

            for pos in stale_positions:
                # Get price data for closing
                df = self.data_provider.get_ohlc(
                    symbol=asset,
                    timeframe=self.timeframe,
                    start_date=self.start_date - timedelta(days=1),
                    end_date=self.start_date,
                )

                # Fallback: try position's timestamp if no data at start_date
                if df.empty:
                    df = self.data_provider.get_ohlc(
                        symbol=asset,
                        timeframe=self.timeframe,
                        start_date=pos.decision_timestamp - timedelta(days=1),
                        end_date=pos.decision_timestamp + timedelta(days=1),
                    )

                if not df.empty:
                    final_price = float(df.iloc[-1]["close"])

                    self._close_position_with_trade_sync(
                        pos,
                        "stale_cleanup",
                        final_price,
                    )

                    logger.info(
                        f"Closed stale position ID {pos.id} from run {pos.backtest_run_id} @ ${final_price:.2f}",
                        extra={
                            "event_type": "stale_position_closed",
                            "position_id": pos.id,
                            "backtest_run_id": pos.backtest_run_id,
                        },
                    )
                    total_cleaned += 1
                else:
                    # No price data, force close
                    logger.warning(
                        f"No price data for stale position {pos.id}, force closing",
                        extra={"event_type": "stale_position_force_close", "position_id": pos.id},
                    )
                    pos.is_active = False
                    pos.closed_at = datetime.utcnow()
                    pos.close_reason = "stale_cleanup_no_price"
                    self.db.commit()
                    total_cleaned += 1

        if total_cleaned > 0:
            logger.warning(
                f"Cleaned up {total_cleaned} stale positions before backtest start",
                extra={"event_type": "stale_cleanup_complete", "count": total_cleaned},
            )

    def _close_position_with_trade_sync(
        self,
        position: ActivePosition,
        reason: str,
        exit_price: float,
    ) -> None:
        """Close the tracked position and sync realized exit data onto its linked opening trade."""
        self.position_monitor.close_position(position, reason, exit_price)

        close_order = None
        if position.trade_id:
            close_order = self.order_manager.close_trade(
                position.trade_id,
                exit_price,
                environment=Environment.BACKTEST,
            )

        self._sync_linked_trade_exit(position, reason, exit_price, close_order)

    def _sync_linked_trade_exit(
        self,
        position: ActivePosition,
        reason: str,
        exit_price: float,
        close_order,
    ) -> None:
        """Copy exit metadata onto the opening trade used to scope this backtest run."""
        if not position.trade_id:
            return

        opening_trade = self.db.query(Trade).filter(Trade.id == position.trade_id).first()
        if opening_trade is None:
            return

        opening_trade.exit_price = Decimal(str(exit_price))
        opening_trade.closed_at = position.closed_at
        opening_trade.exit_signal = reason

        if opening_trade.entry_price is not None:
            qty = Decimal(str(opening_trade.quantity))
            entry_price = Decimal(str(opening_trade.entry_price))
            exit_price_decimal = Decimal(str(exit_price))

            if opening_trade.side == OrderSide.BUY:
                opening_trade.pnl = (exit_price_decimal - entry_price) * qty
            else:
                opening_trade.pnl = (entry_price - exit_price_decimal) * qty

            entry_notional = entry_price * qty
            if entry_notional > 0:
                opening_trade.pnl_pct = float((opening_trade.pnl / entry_notional) * 100)

        close_order_id = getattr(close_order, "id", None)
        if close_order_id is not None:
            close_trade = self.db.query(Trade).filter(Trade.order_id == close_order_id).first()
            if close_trade is not None:
                opening_trade.pnl = close_trade.pnl
                opening_trade.pnl_pct = close_trade.pnl_pct

        self.db.commit()

    def _calculate_metrics(self) -> BacktestMetrics:
        """
        Calculate backtest performance metrics.

        Metrics:
        - Win rate: % of winning trades
        - Profit factor: sum(wins) / abs(sum(losses))
        - Sharpe ratio: (return - risk_free) / volatility
        - Max drawdown: worst peak-to-trough decline
        - Total P&L: sum of all trade P&L
        """
        # Get all trades scoped to this backtest run.
        # Time-window-only filtering bleeds trades from previous runs that used the
        # same date range, which is exactly how you get the clown show of 4H reusing
        # 1H metrics.
        trades_query = self.db.query(Trade).filter(
            Trade.environment == Environment.BACKTEST,
        )

        if self.backtest_run_id is not None:
            trade_ids_subquery = (
                self.db.query(ActivePosition.trade_id)
                .filter(
                    ActivePosition.backtest_run_id == self.backtest_run_id,
                    ActivePosition.trade_id.is_not(None),
                )
                .subquery()
            )
            trades_query = trades_query.filter(Trade.id.in_(trade_ids_subquery))
            trades_query = trades_query.filter(
                Trade.closed_at.is_not(None),
                Trade.exit_price.is_not(None),
            )

        trades = trades_query.all()

        if self._replay_trade_order_ids is not None:
            trades = [
                trade
                for trade in trades
                if trade.order_id in self._replay_trade_order_ids
            ]

        if not trades:
            logger.warning(
                "No trades executed during backtest",
                extra={"event_type": "backtest_warning"},
            )

            # Phase 4: Even with no trades, calculate MDA if positions were opened
            mda, accuracy_by_candle = self._calculate_directional_accuracy()
            close_reasons = self._calculate_close_reasons()

            invocations_saved = self.total_candles_processed - self.agent_invocations
            invocation_reduction_pct = (
                (invocations_saved / self.total_candles_processed * 100)
                if self.total_candles_processed > 0
                else 0.0
            )

            return BacktestMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_pnl=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                total_return_pct=0.0,
                agent_invocations=self.agent_invocations,
                invocations_saved=invocations_saved,
                invocation_reduction_pct=invocation_reduction_pct,
                mean_directional_accuracy=mda,
                accuracy_by_candle=accuracy_by_candle,
                close_reasons=close_reasons,
            )

        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl and float(t.pnl) > 0]
        losing_trades = [t for t in trades if t.pnl and float(t.pnl) < 0]

        total_wins = sum(float(t.pnl) for t in winning_trades)
        total_losses = abs(sum(float(t.pnl) for t in losing_trades))

        # Win rate
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        # Profit factor
        profit_factor = (
            total_wins / total_losses
            if total_losses > 0
            else float("inf") if total_wins > 0 else 0.0
        )

        # Total P&L
        total_pnl = sum(float(t.pnl) for t in trades if t.pnl)

        # Average win/loss
        avg_win = total_wins / len(winning_trades) if winning_trades else 0.0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0.0

        # Largest win/loss
        largest_win = max((float(t.pnl) for t in winning_trades), default=0.0)
        largest_loss = min((float(t.pnl) for t in losing_trades), default=0.0)

        # Total return %
        total_return_pct = (
            (total_pnl / self.initial_capital) * 100
            if self.initial_capital > 0
            else 0.0
        )

        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()

        # Phase 4: Calculate MDA and accuracy metrics
        mda, accuracy_by_candle = self._calculate_directional_accuracy()

        # Phase 4: Calculate close reasons distribution
        close_reasons = self._calculate_close_reasons()

        # Phase 4: Calculate invocation reduction
        invocations_saved = self.total_candles_processed - self.agent_invocations
        invocation_reduction_pct = (
            (invocations_saved / self.total_candles_processed * 100)
            if self.total_candles_processed > 0
            else 0.0
        )

        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            total_return_pct=total_return_pct,
            agent_invocations=self.agent_invocations,
            invocations_saved=invocations_saved,
            invocation_reduction_pct=invocation_reduction_pct,
            mean_directional_accuracy=mda,
            accuracy_by_candle=accuracy_by_candle,
            close_reasons=close_reasons,
        )

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.

        Formula: (return - risk_free) / volatility

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)

        Returns:
            Sharpe ratio
        """
        if len(self.equity_curve) < 2:
            return 0.0

        # Calculate returns
        equity_series = pd.Series([e["equity"] for e in self.equity_curve])
        returns = equity_series.pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Annualize based on timeframe
        periods_per_year = self._get_periods_per_year()

        # Calculate Sharpe
        excess_return = returns.mean() - (risk_free_rate / periods_per_year)
        sharpe = (excess_return / returns.std()) * np.sqrt(periods_per_year)

        return float(sharpe)

    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.

        Formula: max((peak - trough) / peak)

        Returns:
            Max drawdown as decimal (e.g., 0.15 = 15%)
        """
        if len(self.equity_curve) < 2:
            return 0.0

        equity_series = pd.Series([e["equity"] for e in self.equity_curve])

        # Calculate running maximum
        running_max = equity_series.expanding().max()

        # Calculate drawdown
        drawdown = (equity_series - running_max) / running_max

        # Maximum drawdown (most negative)
        max_dd = abs(drawdown.min())

        return float(max_dd)

    def _calculate_directional_accuracy(self) -> tuple[float, Dict[int, float]]:
        """
        Calculate Mean Directional Accuracy (MDA) and per-candle accuracy.

        MDA = (correct_candles / total_candles_evaluated)

        Returns:
            Tuple of (mean_directional_accuracy, accuracy_by_candle)
        """
        query = self.db.query(ActivePosition).filter(
            ActivePosition.is_active.is_(False),
            ActivePosition.decision_timestamp >= self.start_date,
            ActivePosition.decision_timestamp <= self.end_date,
        )

        if self.backtest_run_id is not None:
            query = query.filter(ActivePosition.backtest_run_id == self.backtest_run_id)

        positions = query.all()

        if not positions:
            return 0.0, {}

        # Track correct predictions per candle index
        correct_by_candle = {}
        total_by_candle = {}

        for pos in positions:
            expected_direction = "up" if pos.side == OrderSide.BUY else "down"

            # Evaluate up to prediction_horizon candles
            for i, direction in enumerate(
                pos.candles_direction[: pos.prediction_horizon]
            ):
                candle_idx = i + 1  # 1-indexed (candle 1, 2, 3...)

                if candle_idx not in correct_by_candle:
                    correct_by_candle[candle_idx] = 0
                    total_by_candle[candle_idx] = 0

                total_by_candle[candle_idx] += 1
                if direction == expected_direction:
                    correct_by_candle[candle_idx] += 1

        # Calculate accuracy per candle
        accuracy_by_candle = {
            candle: correct_by_candle[candle] / total_by_candle[candle]
            for candle in sorted(total_by_candle.keys())
        }

        # Calculate overall MDA
        total_correct = sum(correct_by_candle.values())
        total_candles = sum(total_by_candle.values())
        mda = total_correct / total_candles if total_candles > 0 else 0.0

        return mda, accuracy_by_candle

    def _calculate_close_reasons(self) -> Dict[str, int]:
        """
        Calculate distribution of position close reasons.

        Returns:
            Dict mapping close_reason to count
        """
        query = self.db.query(ActivePosition).filter(
            ActivePosition.is_active.is_(False),
            ActivePosition.decision_timestamp >= self.start_date,
            ActivePosition.decision_timestamp <= self.end_date,
        )

        if self.backtest_run_id is not None:
            query = query.filter(ActivePosition.backtest_run_id == self.backtest_run_id)

        positions = query.all()

        close_reasons = {}
        for pos in positions:
            reason = pos.close_reason or "unknown"
            close_reasons[reason] = close_reasons.get(reason, 0) + 1

        return close_reasons

    def _get_periods_per_year(self) -> int:
        """Get number of periods per year based on timeframe."""
        if self.timeframe == "1h":
            return 252 * 6.5  # Trading days * hours per day
        elif self.timeframe == "4h":
            return 252 * 1.625  # Approx 1.6 periods per day
        elif self.timeframe == "1d":
            return 252
        elif self.timeframe == "1w":
            return 52
        else:
            return 252

    def _update_backtest_run(self, metrics: BacktestMetrics) -> None:
        """Update BacktestRun record with final metrics."""
        if not self.backtest_run_id:
            return

        run = (
            self.db.query(BacktestRun)
            .filter(BacktestRun.id == self.backtest_run_id)
            .first()
        )

        if run:
            run.total_trades = metrics.total_trades
            run.win_rate = metrics.win_rate
            run.profit_factor = metrics.profit_factor
            run.sharpe_ratio = metrics.sharpe_ratio
            run.max_drawdown = metrics.max_drawdown
            run.total_pnl = Decimal(str(metrics.total_pnl))

            self.db.commit()
            logger.info(
                f"Updated backtest run #{self.backtest_run_id} with final metrics",
                extra={"event_type": "backtest_complete"},
            )

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve as DataFrame.

        Returns:
            DataFrame with columns: date, equity, cash, positions_value
        """
        return pd.DataFrame(self.equity_curve)

    def __del__(self):
        """Cleanup database session if we own it."""
        if self._own_session and self.db:
            self.db.close()
