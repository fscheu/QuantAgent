"""Backtesting engine for strategy validation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from decimal import Decimal

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from quantagent.data.provider import DataProvider
from quantagent.trading_graph import TradingGraph
from quantagent.trading.order_manager import OrderManager
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.portfolio.manager import PortfolioManager
from quantagent.models import BacktestRun, Signal, Order, Trade, Environment, TradeSignal
from quantagent.database import SessionLocal
from quantagent.strategy.assembler import StrategyAssembler

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
        use_checkpointing: bool = False
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
                'initial_cash': initial_capital,
                'base_position_pct': self.config.get('base_position_pct', 0.05),
                'max_daily_loss_pct': self.config.get('max_daily_loss_pct', 0.05),
                'max_position_pct': self.config.get('max_position_pct', 0.10),
                'slippage_pct': self.config.get('slippage_pct', 0.01),
                # Normalize model fields into generic ones; accept both
                'model_provider': self.config.get('agent_llm_provider', self.config.get('model_provider', 'openai')),
                'model_name': self.config.get('agent_llm_model', self.config.get('model_name', 'gpt-4o-mini')),
                'temperature': self.config.get('agent_llm_temperature', self.config.get('temperature', 0.1)),
                'use_checkpointing': use_checkpointing,
                'universe': self.config.get('universe', []),
            },
            environment=Environment.BACKTEST,
        )
        components = StrategyAssembler.build_components(resolved, db_session=self.db)

        # Trading graph (analysis engine)
        self.trading_graph = components.graph

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

    def run(self, name: Optional[str] = None) -> BacktestMetrics:
        """
        Run backtest and return metrics.

        Args:
            name: Optional name for this backtest run

        Returns:
            BacktestMetrics with performance statistics
        """
        logger.info(f"Starting backtest: {self.start_date} to {self.end_date}")
        logger.info(f"Assets: {self.assets}, Timeframe: {self.timeframe}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")

        # Create backtest run record
        self._create_backtest_run(name)

        # Get date range for iteration
        date_range = self._get_date_range()
        total_periods = len(date_range) * len(self.assets)

        logger.info(f"Backtesting {total_periods} analysis periods ({len(date_range)} dates x {len(self.assets)} assets)")

        # Loop through dates
        for i, current_date in enumerate(date_range):
            self.current_date = current_date

            # Reset daily P&L tracking at start of each day
            if i == 0 or current_date.date() != date_range[i-1].date():
                self.risk_manager.reset_daily_tracker()

            # Analyze each asset
            for asset in self.assets:
                try:
                    self._analyze_and_trade(asset, current_date)
                except Exception as e:
                    logger.error(f"Error analyzing {asset} at {current_date}: {e}", exc_info=True)
                    continue

            # Record equity at end of period
            self._record_equity(current_date)

            # Log progress
            if (i + 1) % 100 == 0 or i == len(date_range) - 1:
                progress = ((i + 1) / len(date_range)) * 100
                logger.info(f"Progress: {progress:.1f}% ({i+1}/{len(date_range)} dates)")

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Update backtest run with results
        self._update_backtest_run(metrics)

        logger.info(f"Backtest complete: {metrics.total_trades} trades, Win rate: {metrics.win_rate:.2%}")
        logger.info(f"Total P&L: ${metrics.total_pnl:,.2f} ({metrics.total_return_pct:.2%})")

        return metrics

    def _create_backtest_run(self, name: Optional[str]) -> None:
        """Create BacktestRun record in database."""
        run = BacktestRun(
            name=name or f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timeframe=self.timeframe,
            assets=self.assets,
            start_date=self.start_date,
            end_date=self.end_date,
            config_snapshot=self._build_config_snapshot()
        )

        self.db.add(run)
        self.db.commit()
        self.backtest_run_id = run.id

        logger.info(f"Created backtest run #{self.backtest_run_id}: {run.name}")

    def _build_config_snapshot(self) -> Dict:
        """Build immutable config snapshot for reproducibility."""
        # Re-generate snapshot via assembler to keep alignment
        resolved = StrategyAssembler.from_snapshot(
            {
                'initial_cash': self.initial_capital,
                'base_position_pct': self.config.get('base_position_pct', 0.05),
                'max_daily_loss_pct': self.config.get('max_daily_loss_pct', 0.05),
                'max_position_pct': self.config.get('max_position_pct', 0.10),
                'slippage_pct': self.config.get('slippage_pct', 0.01),
                'model_provider': self.config.get('agent_llm_provider', self.config.get('model_provider', 'openai')),
                'model_name': self.config.get('agent_llm_model', self.config.get('model_name', 'gpt-4o-mini')),
                'temperature': self.config.get('agent_llm_temperature', self.config.get('temperature', 0.1)),
                'use_checkpointing': self.use_checkpointing,
                'universe': self.config.get('universe', []),
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
        if self.timeframe in ['1h', '4h']:
            step_hours = int(self.timeframe.replace('h', ''))
            step = timedelta(hours=step_hours)
        elif self.timeframe == '1d':
            step = timedelta(days=1)
        elif self.timeframe == '1w':
            step = timedelta(weeks=1)
        else:
            # Default to hourly
            step = timedelta(hours=1)

        while current <= self.end_date:
            dates.append(current)
            current += step

        return dates

    def _analyze_and_trade(self, asset: str, current_date: datetime) -> None:
        """
        Execute analysis and trade for a single asset at a given time.

        Args:
            asset: Asset symbol
            current_date: Current backtest date
        """
        # Get historical data for analysis (need lookback period)
        lookback_days = 30  # Use last 30 days for analysis
        data_start = current_date - timedelta(days=lookback_days)

        df = self.data_provider.get_ohlc(
            symbol=asset,
            timeframe=self.timeframe,
            start_date=data_start,
            end_date=current_date
        )

        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient data for {asset} at {current_date} (got {len(df)} records)")
            return

        # Convert to kline_data format
        kline_data = self._df_to_kline_data(df)

        # Execute analysis using TradingGraph
        initial_state = {
            "kline_data": kline_data,
            "time_frame": self.timeframe,
            "stock_name": asset,
            "messages": []
        }

        # Run analysis with thread_id for checkpointing
        thread_id = f"backtest_{self.backtest_run_id}_{asset}_{current_date.isoformat()}"
        config = {"configurable": {"thread_id": thread_id}} if self.use_checkpointing else None

        result = self.trading_graph.graph.invoke(initial_state, config=config)

        # Extract decision
        decision_text = result.get("final_trade_decision", "HOLD")

        # Parse decision (extract LONG/SHORT/HOLD)
        decision = self._parse_decision(decision_text)

        # Get confidence from indicator report (default to 0.5 if not found)
        confidence = self._extract_confidence(result)

        # Get current price
        current_price = float(df.iloc[-1]['close'])

        # Store signal in database
        signal = self._create_signal(
            asset=asset,
            decision=decision,
            confidence=confidence,
            result=result,
            current_date=current_date,
            thread_id=thread_id if self.use_checkpointing else None
        )

        # Execute trade if not HOLD
        if decision != "NEUTRAL":
            order = self.order_manager.execute_decision(
                symbol=asset,
                decision=decision.value if decision != "NEUTRAL" else "HOLD",
                confidence=confidence,
                current_price=current_price,
                environment=Environment.BACKTEST,
                trigger_signal_id=signal.id if signal else None
            )

            if order:
                logger.info(f"Executed {decision.value} for {asset} @ ${current_price:.2f}, qty: {order.filled_quantity}")

    def _df_to_kline_data(self, df: pd.DataFrame) -> Dict:
        """Convert DataFrame to kline_data dict format."""
        return {
            'timestamps': df['timestamp'].astype(str).tolist(),
            'opens': df['open'].tolist(),
            'highs': df['high'].tolist(),
            'lows': df['low'].tolist(),
            'closes': df['close'].tolist(),
            'volumes': df['volume'].tolist()
        }

    def _parse_decision(self, decision_text: str) -> TradeSignal:
        """Parse decision text to extract LONG/SHORT/HOLD."""
        decision_upper = decision_text.upper()

        if "LONG" in decision_upper or "BUY" in decision_upper:
            return TradeSignal.LONG
        elif "SHORT" in decision_upper or "SELL" in decision_upper:
            return TradeSignal.SHORT
        else:
            return TradeSignal.NEUTRAL

    def _extract_confidence(self, result: Dict) -> float:
        """Extract confidence from analysis result."""
        # Try to get confidence from indicator_report
        indicator_report = result.get('indicator_report')

        if indicator_report and hasattr(indicator_report, 'confidence'):
            return float(indicator_report.confidence)

        # Default to medium confidence
        return 0.5

    def _create_signal(
        self,
        asset: str,
        decision: TradeSignal,
        confidence: float,
        result: Dict,
        current_date: datetime,
        thread_id: Optional[str] = None
    ) -> Optional[Signal]:
        """Create and persist Signal record."""
        try:
            # Extract technical indicators
            rsi = None
            macd = None
            pattern = None
            trend = None

            if 'rsi' in result and result['rsi']:
                rsi = float(result['rsi'][-1]) if isinstance(result['rsi'], list) else float(result['rsi'])

            if 'macd' in result and result['macd']:
                macd = float(result['macd'][-1]) if isinstance(result['macd'], list) else float(result['macd'])

            # Get pattern and trend from reports
            pattern_report = result.get('pattern_report')
            if pattern_report and hasattr(pattern_report, 'primary_pattern'):
                pattern = pattern_report.primary_pattern

            trend_report = result.get('trend_report')
            if trend_report and hasattr(trend_report, 'trend_direction'):
                trend = trend_report.trend_direction

            # Create signal
            signal = Signal(
                symbol=asset,
                signal=decision,
                confidence=confidence,
                timeframe=self.timeframe,
                rsi=rsi,
                macd=macd,
                pattern=pattern,
                trend=trend,
                analysis_summary=result.get('final_trade_decision', ''),
                generated_at=current_date,
                environment=Environment.BACKTEST,
                thread_id=thread_id,
                model_provider=self.config.get('agent_llm_provider', 'openai'),
                model_name=self.config.get('agent_llm_model', 'gpt-4o-mini'),
                temperature=self.config.get('agent_llm_temperature', 0.1)
            )

            self.db.add(signal)
            self.db.commit()

            return signal

        except Exception as e:
            logger.error(f"Error creating signal: {e}", exc_info=True)
            return None

    def _record_equity(self, current_date: datetime) -> None:
        """Record equity curve data point."""
        total_value = self.portfolio.get_total_value()

        self.equity_curve.append({
            'date': current_date,
            'equity': total_value,
            'cash': self.portfolio.cash,
            'positions_value': total_value - self.portfolio.cash
        })

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
        trades = self.db.query(Trade).filter(
            Trade.environment == Environment.BACKTEST,
            Trade.opened_at >= self.start_date,
            Trade.opened_at <= self.end_date
        ).all()

        if not trades:
            logger.warning("No trades executed during backtest")
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
                total_return_pct=0.0
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
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0

        # Total P&L
        total_pnl = sum(float(t.pnl) for t in trades if t.pnl)

        # Average win/loss
        avg_win = total_wins / len(winning_trades) if winning_trades else 0.0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0.0

        # Largest win/loss
        largest_win = max((float(t.pnl) for t in winning_trades), default=0.0)
        largest_loss = min((float(t.pnl) for t in losing_trades), default=0.0)

        # Total return %
        total_return_pct = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0

        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()

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
            total_return_pct=total_return_pct
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
        equity_series = pd.Series([e['equity'] for e in self.equity_curve])
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

        equity_series = pd.Series([e['equity'] for e in self.equity_curve])

        # Calculate running maximum
        running_max = equity_series.expanding().max()

        # Calculate drawdown
        drawdown = (equity_series - running_max) / running_max

        # Maximum drawdown (most negative)
        max_dd = abs(drawdown.min())

        return float(max_dd)

    def _get_periods_per_year(self) -> int:
        """Get number of periods per year based on timeframe."""
        if self.timeframe == '1h':
            return 252 * 6.5  # Trading days * hours per day
        elif self.timeframe == '4h':
            return 252 * 1.625  # Approx 1.6 periods per day
        elif self.timeframe == '1d':
            return 252
        elif self.timeframe == '1w':
            return 52
        else:
            return 252

    def _update_backtest_run(self, metrics: BacktestMetrics) -> None:
        """Update BacktestRun record with final metrics."""
        if not self.backtest_run_id:
            return

        run = self.db.query(BacktestRun).filter(BacktestRun.id == self.backtest_run_id).first()

        if run:
            run.total_trades = metrics.total_trades
            run.win_rate = metrics.win_rate
            run.profit_factor = metrics.profit_factor
            run.sharpe_ratio = metrics.sharpe_ratio
            run.max_drawdown = metrics.max_drawdown
            run.total_pnl = Decimal(str(metrics.total_pnl))

            self.db.commit()
            logger.info(f"Updated backtest run #{self.backtest_run_id} with final metrics")

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
