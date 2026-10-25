"""CLI entry point for the TradingScheduler (paper trading)."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.orm import Session

from quantagent import settings
from quantagent.data.provider import DataProvider
from quantagent.data.sources import build_source
from quantagent.data.sources.fred import FredClient
from quantagent.database import Base, SessionLocal
from quantagent.logging_config import setup_logging
from quantagent.models import Environment
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.strategy.registry import build_strategy
from quantagent.trading.scheduler import TradingScheduler

logger = logging.getLogger("quantagent.paper_trading")


def _parse_assets(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    assets = [asset.strip().upper() for asset in value.split(",") if asset.strip()]
    return assets or None


def _build_scheduler(
    config: settings.SchedulerSettings,
    strategy_name: str = "LLMAgentStrategy",
    strategy_params: Optional[dict] = None,
) -> tuple[TradingScheduler, Session]:
    session = SessionLocal()
    try:
        resolved = StrategyAssembler.from_profiles(
            overrides={"universe": config.assets},
            environment=Environment(config.environment),
        )
        components = StrategyAssembler.build_components(resolved, db_session=session)
        data_provider = DataProvider(session, source=build_source())
        scheduler = TradingScheduler(
            trading_graph=components.graph,
            order_manager=components.order_manager,
            data_provider=data_provider,
            db_session=session,
            scheduler_settings=config,
            strategy=build_strategy(strategy_name, **(strategy_params or {})),
        )
        return scheduler, session
    except Exception:
        session.close()
        raise


def _build_portfolio_scheduler(config, path: str):
    hypothesis = Path(path)
    if not hypothesis.is_file():
        raise ValueError(f"cartera inexistente: {path}")
    from quantagent.fabrica.hipotesis import cargar_hipotesis

    cartera = cargar_hipotesis(hypothesis)
    macro = FredClient() if any(p.clase == "curva-pendiente" for p in cartera.piernas) else None
    db_path = Path("local-data") / f"{cartera.id}.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        resolved = StrategyAssembler.from_profiles(
            overrides={"universe": config.assets}, environment=Environment(config.environment)
        )
        components = StrategyAssembler.build_components(resolved, db_session=session, build_graph=False)
        provider = DataProvider(session, source=build_source("alpaca"))
        scheduler = TradingScheduler(
            trading_graph=None, order_manager=components.order_manager, data_provider=provider,
            db_session=session, scheduler_settings=config,
            paper_portfolio={"hipotesis": cartera, "reloj": datetime.utcnow, "macro": macro,
                             "sombra": str(db_path.with_suffix(".sombra.json")), "cartera": cartera.id},
        )
        return scheduler, session
    except Exception:
        session.close()
        engine.dispose()
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QuantAgent TradingScheduler for automatic paper trading",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        help="Override scheduler interval in hours (default: env setting)",
    )
    parser.add_argument(
        "--assets",
        type=str,
        help="Comma-separated asset list override (e.g., BTC,SPX,QQQ)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        help="Timeframe override (e.g., 1h,4h)",
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        help="Lookback window (hours) override for data pulls",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Trading environment override: paper, backtest, prod (default: from env/settings)",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Force-enable scheduler even if env flag is off",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy name from registry (default: LLMAgentStrategy)",
    )
    parser.add_argument("--portfolio", action="append", help="Path to a paper portfolio hypothesis YAML")
    parser.add_argument(
        "--strategy-params",
        type=str,
        default="{}",
        help="JSON-encoded dict of strategy constructor params",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single analysis cycle and exit",
    )
    parser.add_argument(
        "--no-immediate",
        action="store_true",
        help="Start background schedule without an immediate run",
    )
    return parser.parse_args()


def _apply_overrides(args: argparse.Namespace) -> settings.SchedulerSettings:
    overrides = {}
    if args.interval_hours is not None:
        overrides["interval_hours"] = args.interval_hours
    if args.assets:
        parsed_assets = _parse_assets(args.assets)
        if parsed_assets:
            overrides["assets"] = parsed_assets
    if args.timeframe:
        overrides["timeframe"] = args.timeframe
    if args.lookback_hours is not None:
        overrides["lookback_hours"] = args.lookback_hours
    if args.enable:
        overrides["enabled"] = True
    if args.environment is not None:
        overrides["environment"] = args.environment

    config = settings.scheduler
    if overrides:
        config = config.with_overrides(**overrides)
    return config


def main() -> int:
    args = _parse_args()
    config = _apply_overrides(args)
    strategy_params = json.loads(args.strategy_params or "{}")
    portfolio_paths = getattr(args, "portfolio", None) or []
    if len(portfolio_paths) > 1 or (portfolio_paths and (args.strategy or strategy_params)):
        raise ValueError("selección de cartera ambigua: use un YAML sin estrategia suelta")

    setup_logging(environment=config.environment)
    logger.info(
        "Scheduler CLI start: interval=%.3fh, assets=%s",
        config.interval_hours,
        config.assets,
        extra={
            "event_type": "scheduler.cli_start",
            "environment": config.environment,
            "extra_data": {
                "assets": config.assets,
                "interval_hours": config.interval_hours,
                "timeframe": config.timeframe,
            },
        },
    )

    scheduler, session = (
        _build_portfolio_scheduler(config, portfolio_paths[0]) if portfolio_paths else
        _build_scheduler(config, strategy_name=args.strategy or "LLMAgentStrategy", strategy_params=strategy_params)
    )

    def _shutdown(sig=None, _frame=None):
        logger.info(
            "Received signal %s, shutting down...",
            sig if sig is not None else "manual",
        )
        try:
            scheduler.stop()
        finally:
            session.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    if args.run_once:
        scheduler.run_once()
        _shutdown("run_once")

    started = scheduler.start(immediate=not args.no_immediate)
    if not started:
        logger.error("Scheduler start aborted; enable it via env or --enable flag")
        _shutdown("disabled")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        _shutdown("SIGINT")


if __name__ == "__main__":
    raise SystemExit(main())
