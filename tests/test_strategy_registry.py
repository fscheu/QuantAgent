from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from quantagent.settings import SchedulerSettings
from quantagent.strategy import (
    STRATEGY_REGISTRY,
    LLMAgentStrategy,
    RSIMeanReversionStrategy,
    build_strategy,
    get_strategy_registry,
)
from quantagent.trading.scheduler import TradingScheduler


class DummySession:
    def __init__(self):
        self._objects = []

    def add(self, obj):
        self._objects.append(obj)

    def flush(self):
        for idx, obj in enumerate(self._objects, start=1):
            if getattr(obj, "id", None) is None:
                setattr(obj, "id", idx)

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            setattr(obj, "id", len(self._objects))
        return obj

    def commit(self):
        return None

    def rollback(self):
        return None

    def query(self, model):
        mock_q = Mock()
        mock_q.filter.return_value = mock_q
        mock_q.order_by.return_value = mock_q
        mock_q.first.return_value = None
        mock_q.all.return_value = []
        return mock_q


class DummyScheduler:
    def add_job(self, func, **kwargs):
        return None

    def start(self):
        return None

    def shutdown(self, wait=True):
        return None


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000_000, 1_100_000, 1_200_000],
        }
    )


def test_strategy_package_import_smoke_without_talib_stub():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from quantagent.strategy import STRATEGY_REGISTRY, build_strategy; "
                "print(sorted(STRATEGY_REGISTRY.keys())); "
                "print(type(build_strategy('RSIMeanReversionStrategy')).__name__)"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "RSIMeanReversionStrategy" in result.stdout
    assert "LLMAgentStrategy" in result.stdout


def test_registry_has_expected_entries():
    registry = get_strategy_registry()
    assert set(registry) == {
        "RSIMeanReversionStrategy",
        "FiftyTwoWeekHighStrategy",
        "TripleScreenStrategy",
        "LLMAgentStrategy",
    }
    for entry in registry.values():
        assert {"cls", "type", "display_name", "description", "params", "min_bars"} <= set(entry)


def test_build_strategy_custom_and_default_params():
    custom = build_strategy(
        "RSIMeanReversionStrategy",
        rsi_period=20,
        oversold_threshold=25.0,
    )
    default = build_strategy("RSIMeanReversionStrategy")
    assert isinstance(custom, RSIMeanReversionStrategy)
    assert custom.rsi_period == 20
    assert custom.oversold_threshold == 25.0
    assert default.rsi_period == STRATEGY_REGISTRY["RSIMeanReversionStrategy"]["params"]["rsi_period"]["default"]


def test_describe_metadata_matches_registry_types():
    assert RSIMeanReversionStrategy.describe()["type"] == "deterministic"
    assert LLMAgentStrategy.describe()["type"] == "llm"
    assert STRATEGY_REGISTRY["RSIMeanReversionStrategy"]["display_name"] == RSIMeanReversionStrategy.describe()["display_name"]


def test_scheduler_uses_provided_strategy_and_keeps_llm_default():
    config = SchedulerSettings(enabled=True, assets=["BTC"], environment="paper")
    provided_strategy = RSIMeanReversionStrategy()

    scheduler_with_rsi = TradingScheduler(
        trading_graph=Mock(),
        order_manager=Mock(),
        data_provider=Mock(),
        db_session=DummySession(),
        scheduler_settings=config,
        scheduler_factory=DummyScheduler,
        strategy=provided_strategy,
    )
    default_scheduler = TradingScheduler(
        trading_graph=Mock(),
        order_manager=Mock(),
        data_provider=Mock(),
        db_session=DummySession(),
        scheduler_settings=config,
        scheduler_factory=DummyScheduler,
    )

    assert scheduler_with_rsi.strategy is provided_strategy
    assert isinstance(default_scheduler.strategy, LLMAgentStrategy)


def test_process_asset_with_deterministic_strategy_does_not_raise_type_error():
    config = SchedulerSettings(enabled=True, assets=["BTC"], environment="paper", timeframe="1h")
    scheduler = TradingScheduler(
        trading_graph=Mock(),
        order_manager=Mock(),
        data_provider=Mock(),
        db_session=DummySession(),
        scheduler_settings=config,
        scheduler_factory=DummyScheduler,
        strategy=RSIMeanReversionStrategy(),
    )
    scheduler.position_monitor.get_active_position = Mock(return_value=None)
    scheduler._fetch_market_data = Mock(return_value=_sample_df())

    scheduler._process_asset("BTC")
