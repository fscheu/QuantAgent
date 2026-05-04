#!/usr/bin/env python3
"""
Seed DEV/QA databases with realistic test data for QuantAgent.

Usage:
    python scripts/seed_dev.py --reset
    python scripts/seed_dev.py --reset --db-url postgresql://user:pass@host:5432/db
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal

# Allow running from repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from quantagent.models import (
    ActivePosition,
    BacktestRun,
    Environment,
    ExitPolicy,
    Fill,
    MarketData,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Signal,
    StrategyConfig,
    Trade,
    TradeSignal,
)

# FK-safe truncation order (children before parents).
# signals ↔ orders is a nullable circular FK; TRUNCATE ... CASCADE handles it.
TRUNCATE_TABLES = (
    "active_positions",
    "trades",
    "fills",
    "orders",
    "signals",
    "backtest_runs",
    "strategy_configs",
    "market_data",
)

STRATEGY_CONFIGS = [
    {
        "name": "RSI_Mean_Reversion",
        "kind": "portfolio",
        "json_config": {
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "position_size_pct": 0.10,
            "max_positions": 3,
        },
    },
    {
        "name": "MACD_Momentum",
        "kind": "portfolio",
        "json_config": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "position_size_pct": 0.15,
            "max_positions": 5,
        },
    },
    {
        "name": "Triple_Screen",
        "kind": "combined",
        "json_config": {
            "weekly_trend_ema": 13,
            "daily_momentum_stoch": {"k_period": 5, "d_period": 3},
            "entry_intraday_tf": "1h",
            "stop_loss_atr_multiplier": 2.0,
            "take_profit_risk_ratio": 3.0,
        },
    },
    {
        "name": "Risk_Management_Default",
        "kind": "risk",
        "json_config": {
            "max_drawdown_pct": 0.15,
            "max_position_size_pct": 0.20,
            "max_portfolio_exposure_pct": 0.80,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
        },
    },
]

# BTC uses 1h interval (4h not universally supported by yfinance/Yahoo).
# Timeframe label stored in DB matches conceptual granularity.
MARKET_DATA_ASSETS = [
    {"ticker": "BTC-USD", "symbol": "BTC-USD", "timeframe": "1h", "interval": "1h", "period": "60d"},
    {"ticker": "AAPL",    "symbol": "AAPL",    "timeframe": "1d", "interval": "1d", "period": "6mo"},
    {"ticker": "SPY",     "symbol": "SPY",     "timeframe": "1d", "interval": "1d", "period": "6mo"},
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed DEV/QA database with test data.")
    parser.add_argument("--db-url", default=None, help="Database URL (overrides DATABASE_URL env var)")
    parser.add_argument("--reset", action="store_true", help="Truncate all seed tables before inserting")
    return parser.parse_args()


def resolve_db_url(db_url_arg: str | None) -> str:
    if db_url_arg:
        return db_url_arg
    # Load .env from project root before checking env.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(root, ".env"))
    load_dotenv(os.path.join(root, ".env.local"), override=True)
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("ERROR: DATABASE_URL not set. Pass --db-url or export DATABASE_URL.")
        sys.exit(1)
    return url


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def reset_tables(session) -> None:
    print("  Truncating tables (CASCADE)...")
    tables_sql = ", ".join(TRUNCATE_TABLES)
    session.execute(text(f"TRUNCATE {tables_sql} CASCADE"))
    session.commit()
    print("  Tables cleared.")


# ---------------------------------------------------------------------------
# Masters
# ---------------------------------------------------------------------------


def seed_strategy_configs(session) -> int:
    now = datetime.utcnow()
    for cfg in STRATEGY_CONFIGS:
        session.add(
            StrategyConfig(
                name=cfg["name"],
                kind=cfg["kind"],
                json_config=cfg["json_config"],
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
    session.commit()
    return session.query(StrategyConfig).count()


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


def _strip_tz(ts) -> datetime:
    """Convert pandas Timestamp to naive UTC datetime."""
    dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def seed_market_data(session) -> int:
    total = 0
    now = datetime.utcnow()

    for asset in MARKET_DATA_ASSETS:
        ticker = asset["ticker"]
        symbol = asset["symbol"]
        timeframe = asset["timeframe"]
        interval = asset["interval"]
        period = asset["period"]

        print(f"  Downloading {ticker} (interval={interval}, period={period})...")
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
        except TypeError:
            # Older yfinance without multi_level_index param.
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

        if df is None or df.empty:
            print(f"  WARNING: no data returned for {ticker} — skipping.")
            continue

        # Flatten multi-level columns if present (yfinance >= 0.2.x with single ticker).
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.droplevel(1)

        df.columns = [str(c).lower() for c in df.columns]

        records = [
            MarketData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=_strip_tz(ts),
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
                volume=float(row.get("volume", 0)),
                created_at=now,
            )
            for ts, row in df.iterrows()
        ]

        session.bulk_save_objects(records)
        session.commit()
        count = session.query(MarketData).filter_by(symbol=symbol).count()
        print(f"  {symbol}/{timeframe}: {count} rows")
        total += count

    return total


# ---------------------------------------------------------------------------
# Transactional scenarios
# ---------------------------------------------------------------------------


def _sig(session, *, symbol, signal, confidence, timeframe, generated_at, env=Environment.PAPER, **kw) -> Signal:
    s = Signal(
        symbol=symbol,
        signal=signal,
        confidence=confidence,
        timeframe=timeframe,
        generated_at=generated_at,
        environment=env,
        model_provider="anthropic",
        model_name="claude-haiku-4-5-20251001",
        temperature=0.1,
        agent_version="1.0",
        **kw,
    )
    session.add(s)
    session.flush()
    return s


def _ord(session, *, symbol, side, order_type, quantity, price=None, status, created_at, filled_at=None,
         filled_qty=Decimal("0"), avg_fill=None, comment="", env=Environment.PAPER, signal_id=None) -> Order:
    o = Order(
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        status=status,
        created_at=created_at,
        updated_at=filled_at or created_at,
        filled_at=filled_at,
        filled_quantity=filled_qty,
        average_fill_price=avg_fill,
        comment=comment,
        environment=env,
        trigger_signal_id=signal_id,
    )
    session.add(o)
    session.flush()
    return o


def _fill(session, *, order_id, quantity, price, commission, filled_at) -> Fill:
    f = Fill(order_id=order_id, quantity=quantity, price=price, commission=commission, filled_at=filled_at)
    session.add(f)
    session.flush()
    return f


def _trade(session, *, symbol, order_id, entry_price, exit_price=None, quantity, side, pnl=None,
           pnl_pct=None, commission=Decimal("0"), entry_signal=None, exit_signal=None,
           timeframe=None, opened_at, closed_at=None, notes="", env=Environment.PAPER) -> Trade:
    t = Trade(
        symbol=symbol,
        order_id=order_id,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        side=side,
        pnl=pnl,
        pnl_pct=pnl_pct,
        commission=commission,
        entry_signal=entry_signal,
        exit_signal=exit_signal,
        timeframe=timeframe,
        opened_at=opened_at,
        closed_at=closed_at,
        notes=notes,
        environment=env,
    )
    session.add(t)
    session.flush()
    return t


def _pos(session, *, symbol, side, entry_price, stop_loss, take_profit, quantity,
         decision_timestamp, exit_policy, candles_direction, signal_id=None, trade_id=None,
         backtest_run_id=None, is_active=True, closed_at=None, close_reason=None,
         accuracy=None, highest=None, lowest=None, max_hold=None,
         env=Environment.PAPER) -> ActivePosition:
    p = ActivePosition(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        quantity=quantity,
        decision_timestamp=decision_timestamp,
        candles_since_entry=0 if is_active else 3,
        exit_policy=exit_policy,
        max_hold_candles=max_hold,
        prediction_horizon=3,
        candles_direction=candles_direction,
        highest_price_seen=highest,
        lowest_price_seen=lowest,
        trade_id=trade_id,
        signal_id=signal_id,
        backtest_run_id=backtest_run_id,
        is_active=is_active,
        closed_at=closed_at,
        close_reason=close_reason,
        accuracy=accuracy,
        environment=env,
    )
    session.add(p)
    session.flush()
    return p


def seed_transactional_scenarios(session) -> None:
    now = datetime.utcnow()

    # ------------------------------------------------------------------
    # Scenario 1: Winning trade — LONG BTC-USD, closed take_profit
    # ------------------------------------------------------------------
    t1_sig_at = now - timedelta(days=10)
    t1_ord_at = t1_sig_at + timedelta(minutes=5)
    t1_fill_at = t1_ord_at + timedelta(minutes=1)
    t1_close_at = t1_sig_at + timedelta(hours=8)

    sig1 = _sig(
        session,
        symbol="BTC-USD", signal=TradeSignal.LONG, confidence=0.85,
        timeframe="1h", generated_at=t1_sig_at,
        rsi=35.0, macd=0.002, stochastic=28.0, roc=1.5, williams_r=-75.0,
        pattern="double_bottom", trend="upward",
        analysis_summary="RSI oversold + MACD bullish cross + double bottom",
    )
    ord1 = _ord(
        session,
        symbol="BTC-USD", side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=Decimal("0.01"), status=OrderStatus.FILLED,
        created_at=t1_ord_at, filled_at=t1_fill_at,
        filled_qty=Decimal("0.01"), avg_fill=Decimal("42000.00"),
        comment="S1: winning trade", signal_id=sig1.id,
    )
    sig1.order_id = ord1.id
    session.flush()
    _fill(session, order_id=ord1.id, quantity=Decimal("0.01"), price=Decimal("42000.00"),
          commission=Decimal("0.42"), filled_at=t1_fill_at)
    trade1 = _trade(
        session,
        symbol="BTC-USD", order_id=ord1.id,
        entry_price=Decimal("42000.00"), exit_price=Decimal("44520.00"),
        quantity=Decimal("0.01"), side=OrderSide.BUY,
        pnl=Decimal("25.20"), pnl_pct=0.06, commission=Decimal("0.84"),
        entry_signal="long", exit_signal="take_profit", timeframe="1h",
        opened_at=t1_fill_at, closed_at=t1_close_at, notes="S1: take_profit hit",
    )
    _pos(
        session,
        symbol="BTC-USD", side=OrderSide.BUY,
        entry_price=Decimal("42000.00"), stop_loss=Decimal("40320.00"),
        take_profit=Decimal("44520.00"), quantity=Decimal("0.01"),
        decision_timestamp=t1_sig_at, exit_policy=ExitPolicy.SL_TP_ONLY, max_hold=12,
        candles_direction=[1, 1, 0],
        highest=Decimal("44600.00"), lowest=Decimal("41800.00"),
        trade_id=trade1.id, signal_id=sig1.id,
        is_active=False, closed_at=t1_close_at, close_reason="take_profit", accuracy=1.0,
    )

    # ------------------------------------------------------------------
    # Scenario 2: Losing trade — LONG AAPL, closed stop_loss
    # ------------------------------------------------------------------
    t2_sig_at = now - timedelta(days=7)
    t2_ord_at = t2_sig_at + timedelta(minutes=5)
    t2_fill_at = t2_ord_at + timedelta(minutes=1)
    t2_close_at = t2_sig_at + timedelta(hours=26)

    sig2 = _sig(
        session,
        symbol="AAPL", signal=TradeSignal.LONG, confidence=0.60,
        timeframe="1d", generated_at=t2_sig_at,
        rsi=55.0, macd=0.15, stochastic=52.0, roc=0.8, williams_r=-45.0,
        pattern="ascending_triangle", trend="sideways",
        analysis_summary="Moderate LONG, ascending triangle with mixed momentum",
    )
    ord2 = _ord(
        session,
        symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=Decimal("5"), status=OrderStatus.FILLED,
        created_at=t2_ord_at, filled_at=t2_fill_at,
        filled_qty=Decimal("5"), avg_fill=Decimal("185.50"),
        comment="S2: losing trade", signal_id=sig2.id,
    )
    sig2.order_id = ord2.id
    session.flush()
    _fill(session, order_id=ord2.id, quantity=Decimal("5"), price=Decimal("185.50"),
          commission=Decimal("0.93"), filled_at=t2_fill_at)
    trade2 = _trade(
        session,
        symbol="AAPL", order_id=ord2.id,
        entry_price=Decimal("185.50"), exit_price=Decimal("179.94"),
        quantity=Decimal("5"), side=OrderSide.BUY,
        pnl=Decimal("-27.80"), pnl_pct=-0.03, commission=Decimal("1.86"),
        entry_signal="long", exit_signal="stop_loss", timeframe="1d",
        opened_at=t2_fill_at, closed_at=t2_close_at, notes="S2: stop_loss triggered",
    )
    _pos(
        session,
        symbol="AAPL", side=OrderSide.BUY,
        entry_price=Decimal("185.50"), stop_loss=Decimal("179.94"),
        take_profit=Decimal("196.63"), quantity=Decimal("5"),
        decision_timestamp=t2_sig_at, exit_policy=ExitPolicy.SL_TP_ONLY, max_hold=5,
        candles_direction=[-1, -1, 0],
        highest=Decimal("186.20"), lowest=Decimal("179.90"),
        trade_id=trade2.id, signal_id=sig2.id,
        is_active=False, closed_at=t2_close_at, close_reason="stop_loss", accuracy=0.0,
    )

    # ------------------------------------------------------------------
    # Scenario 3: Open trade — LONG SPY, currently active (is_active=True)
    # ------------------------------------------------------------------
    t3_sig_at = now - timedelta(hours=6)
    t3_ord_at = t3_sig_at + timedelta(minutes=5)
    t3_fill_at = t3_ord_at + timedelta(minutes=1)

    sig3 = _sig(
        session,
        symbol="SPY", signal=TradeSignal.LONG, confidence=0.72,
        timeframe="1d", generated_at=t3_sig_at,
        rsi=42.0, macd=0.05, stochastic=38.0, roc=0.3, williams_r=-62.0,
        pattern="bull_flag", trend="upward",
        analysis_summary="Bull flag with volume confirmation, RSI approaching oversold",
    )
    ord3 = _ord(
        session,
        symbol="SPY", side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=Decimal("10"), status=OrderStatus.FILLED,
        created_at=t3_ord_at, filled_at=t3_fill_at,
        filled_qty=Decimal("10"), avg_fill=Decimal("510.25"),
        comment="S3: open trade", signal_id=sig3.id,
    )
    sig3.order_id = ord3.id
    session.flush()
    _fill(session, order_id=ord3.id, quantity=Decimal("10"), price=Decimal("510.25"),
          commission=Decimal("1.53"), filled_at=t3_fill_at)
    # No trade yet — position still open.
    _pos(
        session,
        symbol="SPY", side=OrderSide.BUY,
        entry_price=Decimal("510.25"), stop_loss=Decimal("494.94"),
        take_profit=Decimal("541.47"), quantity=Decimal("10"),
        decision_timestamp=t3_sig_at, exit_policy=ExitPolicy.SL_TP_ONLY, max_hold=10,
        candles_direction=[],
        highest=Decimal("511.00"), lowest=Decimal("509.80"),
        signal_id=sig3.id, is_active=True,
        env=Environment.PAPER,
    )

    # ------------------------------------------------------------------
    # Scenario 4: Signal without order — NEUTRAL BTC-USD
    # ------------------------------------------------------------------
    _sig(
        session,
        symbol="BTC-USD", signal=TradeSignal.NEUTRAL, confidence=0.30,
        timeframe="1h", generated_at=now - timedelta(hours=4),
        rsi=52.0, macd=0.0, stochastic=50.0, roc=0.0, williams_r=-50.0,
        pattern="none", trend="sideways",
        analysis_summary="Mixed signals, no clear direction. Staying out.",
    )
    # No order created.

    # ------------------------------------------------------------------
    # Scenario 5: Cancelled limit order — AAPL
    # ------------------------------------------------------------------
    t5_at = now - timedelta(days=3)
    sig5 = _sig(
        session,
        symbol="AAPL", signal=TradeSignal.LONG, confidence=0.65,
        timeframe="1d", generated_at=t5_at,
        rsi=38.0, macd=0.08, stochastic=32.0,
        analysis_summary="Potential entry; limit order placed but cancelled before fill",
    )
    ord5 = _ord(
        session,
        symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        quantity=Decimal("3"), price=Decimal("182.00"),
        status=OrderStatus.CANCELLED,
        created_at=t5_at, filled_at=None, filled_qty=Decimal("0"),
        comment="S5: limit order cancelled — price never reached", signal_id=sig5.id,
    )
    sig5.order_id = ord5.id
    session.flush()
    # No fill, no trade, no active_position.

    # ------------------------------------------------------------------
    # Scenario 6: Complete backtest run — 12 closed positions, metrics set
    # ------------------------------------------------------------------
    bt_start = now - timedelta(days=90)
    bt_end = now - timedelta(days=30)

    bt6 = BacktestRun(
        name="Seed_Backtest_Complete_RSI_90d",
        timeframe="1h",
        assets=["BTC-USD", "AAPL", "SPY"],
        start_date=bt_start,
        end_date=bt_end,
        data_source="yfinance",
        config_snapshot={"strategy": "RSI_Mean_Reversion", "rsi_period": 14,
                         "stop_loss_pct": 0.03, "take_profit_pct": 0.06},
        created_at=bt_start,
        total_trades=12,
        win_rate=0.583,
        profit_factor=2.15,
        sharpe_ratio=1.82,
        max_drawdown=0.112,
        total_pnl=Decimal("3847.50"),
    )
    session.add(bt6)
    session.flush()

    bt6_rows = [
        ("BTC-USD", 40000, True),  ("BTC-USD", 41000, True),  ("BTC-USD", 42000, True),
        ("AAPL",    180,   True),  ("AAPL",    182,   False), ("AAPL",    185,   True),
        ("SPY",     500,   False), ("SPY",     505,   True),  ("SPY",     510,   True),
        ("BTC-USD", 39000, False), ("AAPL",    178,   False), ("SPY",     498,   True),
    ]
    for i, (sym, entry, is_win) in enumerate(bt6_rows):
        qty = Decimal("0.01") if sym == "BTC-USD" else Decimal("5") if sym == "AAPL" else Decimal("10")
        pos_ts = bt_start + timedelta(days=i * 5)
        close_ts = pos_ts + timedelta(hours=12)
        _pos(
            session,
            symbol=sym, side=OrderSide.BUY,
            entry_price=Decimal(str(entry)),
            stop_loss=Decimal(str(round(entry * 0.97, 2))),
            take_profit=Decimal(str(round(entry * 1.06, 2))),
            quantity=qty,
            decision_timestamp=pos_ts,
            exit_policy=ExitPolicy.SL_TP_ONLY, max_hold=20,
            candles_direction=[1, 0, -1],
            backtest_run_id=bt6.id,
            is_active=False, closed_at=close_ts,
            close_reason="take_profit" if is_win else "stop_loss",
            accuracy=1.0 if is_win else 0.0,
            env=Environment.BACKTEST,
        )

    # ------------------------------------------------------------------
    # Scenario 7: Backtest run in progress — metrics NULL, some active positions
    # ------------------------------------------------------------------
    bt7_start = now - timedelta(days=30)
    bt7_end = now + timedelta(days=60)

    bt7 = BacktestRun(
        name="Seed_Backtest_InProgress_MACD_Multi",
        timeframe="1d",
        assets=["AAPL", "SPY"],
        start_date=bt7_start,
        end_date=bt7_end,
        data_source="yfinance",
        config_snapshot={"strategy": "MACD_Momentum", "fast_period": 12, "slow_period": 26},
        created_at=bt7_start,
        total_trades=None,
        win_rate=None,
        profit_factor=None,
        sharpe_ratio=None,
        max_drawdown=None,
        total_pnl=None,
    )
    session.add(bt7)
    session.flush()

    bt7_rows = [
        ("AAPL", 183.0, True),   # still open
        ("SPY",  507.0, True),   # still open
        ("AAPL", 186.0, False),  # closed, one position done
    ]
    for i, (sym, entry, is_act) in enumerate(bt7_rows):
        qty = Decimal("5") if sym == "AAPL" else Decimal("10")
        pos_ts = bt7_start + timedelta(days=i * 3)
        _pos(
            session,
            symbol=sym, side=OrderSide.BUY,
            entry_price=Decimal(str(entry)),
            stop_loss=Decimal(str(round(entry * 0.97, 2))),
            take_profit=Decimal(str(round(entry * 1.06, 2))),
            quantity=qty,
            decision_timestamp=pos_ts,
            exit_policy=ExitPolicy.REEVALUATE,
            candles_direction=[1] if is_act else [1, -1, 0, 1],
            backtest_run_id=bt7.id,
            is_active=is_act,
            closed_at=None if is_act else pos_ts + timedelta(days=5),
            close_reason=None if is_act else "take_profit",
            env=Environment.BACKTEST,
        )

    session.commit()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(session) -> None:
    table_models = [
        ("strategy_configs", StrategyConfig),
        ("market_data",      MarketData),
        ("signals",          Signal),
        ("orders",           Order),
        ("fills",            Fill),
        ("trades",           Trade),
        ("active_positions", ActivePosition),
        ("backtest_runs",    BacktestRun),
    ]

    print("\n=== Seed Summary ===")
    grand_total = 0
    for table_name, model in table_models:
        count = session.query(model).count()
        print(f"  {table_name:<22} {count:>6}")
        grand_total += count
    print(f"  {'TOTAL':<22} {grand_total:>6}")

    active = session.query(ActivePosition).filter_by(is_active=True).count()
    winning = session.execute(text("SELECT COUNT(*) FROM trades WHERE pnl > 0")).scalar()
    losing  = session.execute(text("SELECT COUNT(*) FROM trades WHERE pnl < 0")).scalar()
    mkt     = session.query(MarketData).count()
    print(f"\n  market_data rows         : {mkt}")
    print(f"  active_positions (open)  : {active}")
    print(f"  trades with pnl > 0      : {winning}")
    print(f"  trades with pnl < 0      : {losing}")

    if mkt < 500:
        print(f"\n  WARNING: market_data has only {mkt} rows (< 500). "
              "Check yfinance connectivity.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    db_url = resolve_db_url(args.db_url)

    masked = db_url.split("@")[-1] if "@" in db_url else db_url
    print(f"Connecting to: ...@{masked}")

    engine = create_engine(db_url, pool_pre_ping=True, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        if args.reset:
            print("[1/4] Resetting tables...")
            reset_tables(session)

        print("[2/4] Seeding strategy_configs...")
        n = seed_strategy_configs(session)
        print(f"  strategy_configs: {n}")

        print("[3/4] Downloading and seeding market_data...")
        seed_market_data(session)

        print("[4/4] Seeding transactional scenarios...")
        seed_transactional_scenarios(session)

        print_summary(session)
        print("\nDone.")

    except Exception:
        session.rollback()
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
