"""CLI entry point for the TradingScheduler (paper trading)."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import List, Optional

from sqlalchemy.orm import Session

from quantagent import settings
from quantagent.data.provider import DataProvider
from quantagent.database import SessionLocal
from quantagent.logging_config import setup_logging
from quantagent.models import Environment
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.trading.scheduler import TradingScheduler

logger = logging.getLogger("quantagent.paper_trading")


def _parse_assets(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    assets = [asset.strip().upper() for asset in value.split(",") if asset.strip()]
    return assets or None


def _build_scheduler(
    config: settings.SchedulerSettings,
) -> tuple[TradingScheduler, Session]:
    session = SessionLocal()
    try:
        resolved = StrategyAssembler.from_profiles(
            overrides={"universe": config.assets},
            environment=Environment(config.environment),
        )
        components = StrategyAssembler.build_components(resolved, db_session=session)
        data_provider = DataProvider(session)
        scheduler = TradingScheduler(
            trading_graph=components.graph,
            order_manager=components.order_manager,
            data_provider=data_provider,
            db_session=session,
            scheduler_settings=config,
        )
        return scheduler, session
    except Exception:
        session.close()
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

    scheduler, session = _build_scheduler(config)

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
