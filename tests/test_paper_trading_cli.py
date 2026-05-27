from __future__ import annotations

import argparse
import json

from apps import paper_trading


class _FakeScheduler:
    def __init__(self):
        self.run_once_calls = 0
        self.stop_calls = 0
        self.start_calls = []

    def run_once(self):
        self.run_once_calls += 1
        return {}

    def stop(self):
        self.stop_calls += 1

    def start(self, *, immediate: bool):
        self.start_calls.append(immediate)
        return True


class _FakeSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeConfig:
    def __init__(self):
        self.environment = "paper"
        self.interval_hours = 1.0
        self.assets = ["BTC"]
        self.timeframe = "1h"
        self.overrides = None

    def with_overrides(self, **kwargs):
        clone = _FakeConfig()
        clone.overrides = kwargs
        clone.environment = kwargs.get("environment", self.environment)
        clone.interval_hours = kwargs.get("interval_hours", self.interval_hours)
        clone.assets = kwargs.get("assets", self.assets)
        clone.timeframe = kwargs.get("timeframe", self.timeframe)
        return clone


def test_parse_assets_normalizes_values():
    assert paper_trading._parse_assets(" btc , spx ,, qqq ") == ["BTC", "SPX", "QQQ"]


def test_paper_trading_cli_parses_strategy_args(monkeypatch):
    monkeypatch.setattr(
        paper_trading.sys,
        "argv",
        [
            "paper_trading.py",
            "--strategy",
            "RSIMeanReversionStrategy",
            "--strategy-params",
            '{"rsi_period": 10}',
            "--enable",
        ],
    )

    args = paper_trading._parse_args()

    assert args.strategy == "RSIMeanReversionStrategy"
    assert json.loads(args.strategy_params) == {"rsi_period": 10}
    assert args.enable is True


def test_apply_overrides_includes_environment(monkeypatch):
    fake_scheduler = _FakeConfig()
    monkeypatch.setattr(paper_trading.settings, "scheduler", fake_scheduler)
    args = argparse.Namespace(
        interval_hours=0.5,
        assets="btc,spx",
        timeframe="4h",
        lookback_hours=24.0,
        environment="paper",
        enable=True,
    )

    config = paper_trading._apply_overrides(args)

    assert config.overrides == {
        "interval_hours": 0.5,
        "assets": ["BTC", "SPX"],
        "timeframe": "4h",
        "lookback_hours": 24.0,
        "enabled": True,
        "environment": "paper",
    }


def test_main_run_once_executes_single_cycle_and_closes_session(monkeypatch):
    scheduler = _FakeScheduler()
    session = _FakeSession()
    config = _FakeConfig()

    monkeypatch.setattr(
        paper_trading,
        "_parse_args",
        lambda: argparse.Namespace(
            interval_hours=None,
            assets=None,
            timeframe=None,
            lookback_hours=None,
            environment=None,
            enable=False,
            strategy="LLMAgentStrategy",
            strategy_params="{}",
            run_once=True,
            no_immediate=False,
        ),
    )
    monkeypatch.setattr(paper_trading, "_apply_overrides", lambda _args: config)
    monkeypatch.setattr(paper_trading, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        paper_trading,
        "_build_scheduler",
        lambda _config, strategy_name=None, strategy_params=None: (scheduler, session),
    )
    monkeypatch.setattr(paper_trading.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(paper_trading.logger, "info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(paper_trading.sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    try:
        paper_trading.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("main() should exit after run_once")

    assert scheduler.run_once_calls == 1
    assert scheduler.stop_calls == 1
    assert session.closed is True
    assert scheduler.start_calls == []
