"""Backtesting engine for strategy validation."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from quantagent.agent_models import TradingDecision
from quantagent.backtesting.batch import (
    BacktestLLMRequest,
    BatchConfig,
    BatchExecutorFactory,
    BatchSignalCollector,
    TraceMetadata,
    _build_single_prompt_messages,
)
from quantagent.data.asset_types import AssetType, get_asset_type
from quantagent.data.market_calendar import get_market_calendar
from quantagent.data.provider import DataProvider
from quantagent.database import SessionLocal
from quantagent.models import (ActivePosition, BacktestRun, Environment, OrderSide, Signal, Trade, TradeSignal)
from quantagent.static_util import format_ohlcv_for_agents
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.strategy.base import TradingStrategy
from quantagent.strategy.llm_agent_strategy import LLMAgentStrategy
from quantagent.trading.order_manager import OrderManager  # noqa: F401 (used via components)
from quantagent.trading.paper_broker import PaperBroker  # noqa: F401 (used via components)
from quantagent.trading.position_monitor import PositionMonitor
from quantagent.trading.position_sizer import PositionSizer  # noqa: F401 (used via components)
from quantagent.trading.risk_manager import RiskManager  # noqa: F401 (used via components)
from quantagent.trading_graph import TradingGraph  # noqa: F401 (patched in tests)

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

    # QuantAgent-um8: Batch processing metrics
    batch_enabled: bool = False
    invocations_total: int = 0
    batched_total: int = 0
    failed_total: int = 0

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
        batch_config: Optional[BatchConfig] = None,
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
            batch_config: Optional BatchConfig for batch LLM processing
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
        resolved = StrategyAssembler.from_snapshot(
            {
                "initial_cash": initial_capital,
                "base_position_pct": self.config.get("base_position_pct", 0.05),
                "max_daily_loss_pct": self.config.get("max_daily_loss_pct", 0.05),
                "max_position_pct": self.config.get("max_position_pct", 0.10),
                "slippage_pct": self.config.get("slippage_pct", 0.01),
                # Normalize model fields into generic ones; accept both
                "model_provider": self.config.get(
                    "agent_llm_provider", self.config.get("model_provider", "openai")
                ),
                "model_name": self.config.get(
                    "agent_llm_model", self.config.get("model_name", "gpt-4o-mini")
                ),
                "temperature": self.config.get(
                    "agent_llm_temperature", self.config.get("temperature", 0.1)
                ),
                "use_checkpointing": use_checkpointing,
                "universe": self.config.get("universe", []),
            },
            environment=Environment.BACKTEST,
        )
        components = StrategyAssembler.build_components(resolved, db_session=self.db)

        # Trading graph (analysis engine)
        self.trading_graph = components.graph

        # Trading strategy (use provided or default to LLMAgentStrategy)
        self.strategy = strategy or LLMAgentStrategy(self.trading_graph)

        # Position monitor for active position tracking
        self.position_monitor = PositionMonitor(self.db)

        # Trading components
        self.portfolio = components.portfolio_manager
        self.position_sizer = components.position_sizer
        self.risk_manager = components.risk_manager
        self.broker = components.broker
        self.order_manager = components.order_manager

        # Backtest state
        self.current_date = start_date
        self.backtest_run_id: Optional[int] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

        # Phase 4: Invocation tracking
        self.agent_invocations = 0
        self.total_candles_processed = 0

        # QuantAgent-um8: Batch processing
        self.batch_config = batch_config or BatchConfig()
        self._batch_collector: Optional[BatchSignalCollector] = None
        if self.batch_config.batch_enabled:
            self._setup_batch_collector()

        # Market hours filtering
        self.market_hours_filter = self.config.get("market_hours_filter", True)
        self._market_calendar = (
            get_market_calendar() if self.market_hours_filter else None
        )

        # Cache asset types for each symbol
        self._asset_types: Dict[str, AssetType] = {
            asset: get_asset_type(asset) for asset in assets
        }

    def _setup_batch_collector(self) -> None:
        """Initialize the BatchSignalCollector based on batch_config and LLM provider."""
        provider_name = self.config.get(
            "agent_llm_provider", self.config.get("model_provider", "openai")
        )
        model_name = self.config.get(
            "agent_llm_model", self.config.get("model_name", "gpt-4o-mini")
        )
        batch_mode = self.config.get("batch_mode", "concurrent")

        from quantagent import settings as _settings

        api_key = ""
        if provider_name == "openai":
            api_key = _settings.OPENAI_API_KEY
        elif provider_name == "anthropic":
            api_key = _settings.ANTHROPIC_API_KEY

        executor = BatchExecutorFactory.create(
            provider=batch_mode,
            model=model_name,
            trading_graph=self.trading_graph,
            api_key=api_key,
            batch_config=self.batch_config,
            concurrent_workers=self.config.get("batch_concurrent_workers", 4),
        )
        self._batch_collector = BatchSignalCollector(executor, self.batch_config)
        logger.info(
            f"Batch collector initialized (mode={batch_mode}, size={self.batch_config.batch_size})",
            extra={"event_type": "batch_init", "batch_mode": batch_mode},
        )

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
                    self.risk_manager.reset_daily_tracker()

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

        # Flush any pending batch requests before calculating metrics
        if self._batch_collector:
            self._batch_collector.flush()
            batch_summary = self._batch_collector.summary()
            logger.info(
                "Batch processing summary",
                extra={"event_type": "batch_summary", **batch_summary},
            )

        # Calculate metrics
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

    def _create_backtest_run(self, name: Optional[str]) -> None:
        """Create BacktestRun record in database."""
        run = BacktestRun(
            name=name or f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timeframe=self.timeframe,
            assets=self.assets,
            start_date=self.start_date,
            end_date=self.end_date,
            config_snapshot=self._build_config_snapshot(),
        )

        self.db.add(run)
        self.db.commit()
        self.backtest_run_id = run.id

        logger.info(
            f"Created backtest run #{self.backtest_run_id}: {run.name}",
            extra={"event_type": "backtest_init"},
        )

    def _build_config_snapshot(self) -> Dict:
        """Build immutable config snapshot for reproducibility."""
        # Re-generate snapshot via assembler to keep alignment
        resolved = StrategyAssembler.from_snapshot(
            {
                "initial_cash": self.initial_capital,
                "base_position_pct": self.config.get("base_position_pct", 0.05),
                "max_daily_loss_pct": self.config.get("max_daily_loss_pct", 0.05),
                "max_position_pct": self.config.get("max_position_pct", 0.10),
                "slippage_pct": self.config.get("slippage_pct", 0.01),
                "model_provider": self.config.get(
                    "agent_llm_provider", self.config.get("model_provider", "openai")
                ),
                "model_name": self.config.get(
                    "agent_llm_model", self.config.get("model_name", "gpt-4o-mini")
                ),
                "temperature": self.config.get(
                    "agent_llm_temperature", self.config.get("temperature", 0.1)
                ),
                "use_checkpointing": self.use_checkpointing,
                "universe": self.config.get("universe", []),
            },
            environment=Environment.BACKTEST,
        )
        snapshot = StrategyAssembler.config_snapshot(resolved)
        # QuantAgent-um8: persist batch config in snapshot for reproducibility
        if self.batch_config.batch_enabled:
            from dataclasses import asdict
            snapshot["batch_config"] = asdict(self.batch_config)
        return snapshot

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
        lookback_days = 30
        data_start = current_date - timedelta(days=lookback_days)

        df = self.data_provider.get_ohlc(
            symbol=asset,
            timeframe=self.timeframe,
            start_date=data_start,
            end_date=current_date,
        )

        if df.empty or len(df) < 30:
            logger.warning(
                f"Insufficient data for {asset} at {current_date} (got {len(df)} records)",
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
                # Close position
                self.position_monitor.close_position(active_pos, reason, current_price)

                # Execute close via OrderManager
                if active_pos.trade_id:
                    self.order_manager.close_trade(
                        active_pos.trade_id,
                        current_price,
                        environment=Environment.BACKTEST,
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
        kline_data = format_ohlcv_for_agents(df)

        # Phase 4: Count agent invocations
        self.agent_invocations += 1

        # Generate thread_id for checkpointing (if enabled)
        thread_id = None
        if self.use_checkpointing:
            from quantagent.strategy.assembler import StrategyAssembler

            thread_id = StrategyAssembler.make_thread_id(
                self.backtest_run_id, asset, current_date
            )

        # QuantAgent-um8: Use batch executor if enabled, else direct strategy call
        if self._batch_collector is not None:
            signal = self._generate_signal_via_batch(
                kline_data, asset, current_price, current_date
            )
        else:
            signal = self.strategy.generate_signal(
                kline_data, asset, self.timeframe, current_price, thread_id=thread_id
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

    def _generate_signal_via_batch(
        self,
        kline_data,
        asset: str,
        current_price: float,
        current_date: datetime,
    ):
        """
        Generate signal using the batch executor.

        Builds a BacktestLLMRequest, submits to BatchSignalCollector,
        flushes immediately (batch_size=1 for inline use), and returns
        a TradingSignal parsed from the batch result.
        """
        from quantagent.strategy.base import TradingSignal

        candle_index = int(current_date.timestamp())
        trace = TraceMetadata(
            backtest_run_id=self.backtest_run_id,
            symbol=asset,
            timeframe=self.timeframe,
            candle_index=candle_index,
        )
        messages = _build_single_prompt_messages(kline_data, asset, self.timeframe)
        req = BacktestLLMRequest(
            custom_id=trace.to_custom_id(),
            provider=self.config.get("agent_llm_provider", "openai"),
            model=self.config.get("agent_llm_model", "gpt-4o-mini"),
            payload=messages,
            trace=trace,
            kline_data=kline_data,
            current_price=current_price,
        )

        self._batch_collector.add(req)
        # For inline use (sync/concurrent): force immediate flush if batch_size not reached
        # so we have the result available now.
        self._batch_collector.flush()

        result = self._batch_collector.get_result(req.custom_id)
        if result is None or not result.is_success():
            if self.batch_config.fail_fast:
                raise RuntimeError(
                    f"Batch signal failed for {req.custom_id}: {result.error if result else 'no result'}"
                )
            if self.batch_config.batch_allow_fallback_to_sync:
                logger.warning(
                    f"Batch failed for {asset}@{current_date}, falling back to sync",
                    extra={"event_type": "batch_fallback", "custom_id": req.custom_id},
                )
                return self.strategy.generate_signal(
                    kline_data, asset, self.timeframe, current_price
                )
            return None

        decision_dict = result.parse_decision()
        if decision_dict is None:
            return None

        decision = str(decision_dict.get("decision", "HOLD")).upper()
        if decision not in ("LONG", "SHORT"):
            return None

        confidence = float(decision_dict.get("confidence", 0.5))
        reasoning = str(decision_dict.get("reasoning", ""))

        if decision == "LONG":
            stop_loss = current_price * 0.98
            take_profit = current_price * 1.03
        else:
            stop_loss = current_price * 1.02
            take_profit = current_price * 0.97

        return TradingSignal(
            decision=decision,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            trailing_stop_pct=0.05,
        )

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
        try:
            signal = Signal(
                symbol=asset,
                signal=decision,
                confidence=confidence,
                timeframe=self.timeframe,
                analysis_summary=reasoning,
                generated_at=current_date,
                environment=Environment.BACKTEST,
                model_provider=self.config.get("agent_llm_provider", "openai"),
                model_name=self.config.get("agent_llm_model", "gpt-4o-mini"),
                temperature=self.config.get("agent_llm_temperature", 0.1),
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
                thread_id=thread_id,
                model_provider=self.config.get("agent_llm_provider", "openai"),
                model_name=self.config.get("agent_llm_model", "gpt-4o-mini"),
                temperature=self.config.get("agent_llm_temperature", 0.1),
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
        # Get all trades from database for this backtest
        trades = (
            self.db.query(Trade)
            .filter(
                Trade.environment == Environment.BACKTEST,
                Trade.opened_at >= self.start_date,
                Trade.opened_at <= self.end_date,
            )
            .all()
        )

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

            batch_summary = (
                self._batch_collector.summary() if self._batch_collector else {}
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
                batch_enabled=self.batch_config.batch_enabled,
                invocations_total=batch_summary.get("invocations_total", 0),
                batched_total=batch_summary.get("batched_total", 0),
                failed_total=batch_summary.get("failed_total", 0),
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

        # QuantAgent-um8: Batch counters
        batch_summary = (
            self._batch_collector.summary() if self._batch_collector else {}
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
            batch_enabled=self.batch_config.batch_enabled,
            invocations_total=batch_summary.get("invocations_total", 0),
            batched_total=batch_summary.get("batched_total", 0),
            failed_total=batch_summary.get("failed_total", 0),
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
        positions = (
            self.db.query(ActivePosition)
            .filter(
                ActivePosition.is_active.is_(False),
                ActivePosition.decision_timestamp >= self.start_date,
                ActivePosition.decision_timestamp <= self.end_date,
            )
            .all()
        )

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
        positions = (
            self.db.query(ActivePosition)
            .filter(
                ActivePosition.is_active.is_(False),
                ActivePosition.decision_timestamp >= self.start_date,
                ActivePosition.decision_timestamp <= self.end_date,
            )
            .all()
        )

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
