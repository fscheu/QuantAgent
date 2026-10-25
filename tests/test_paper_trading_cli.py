from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

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


def test_portfolio_path_must_exist(tmp_path):
    with pytest.raises(ValueError, match="cartera inexistente"):
        paper_trading._build_portfolio_scheduler(_FakeConfig(), str(tmp_path / "missing.yaml"))


def test_each_portfolio_builds_an_isolated_unstarted_scheduler(monkeypatch, tmp_path):
    source = Path("hipotesis/cartera-transversal.yaml").resolve()
    second = Path("hipotesis/cartera-nucleo-rsi-regimen.yaml").resolve()
    third = Path("hipotesis/cartera-nucleo-curva.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    Path("local-data").mkdir()
    monkeypatch.setattr(paper_trading.StrategyAssembler, "build_components", lambda *a, **k: SimpleNamespace(order_manager=object()))
    monkeypatch.setattr(paper_trading, "build_source", lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(paper_trading, "FredClient", lambda: object())
    built = []
    monkeypatch.setattr(paper_trading, "TradingScheduler", lambda **kwargs: built.append(kwargs) or SimpleNamespace())
    for yaml_path in (source, second, third):
        paper_trading._build_portfolio_scheduler(_FakeConfig(), str(yaml_path))
    configs = [item["paper_portfolio"] for item in built]
    assert [item["cartera"] for item in configs] == ["cartera-transversal", "cartera-nucleo-rsi-regimen", "cartera-nucleo-curva"]
    assert len({str(item["sombra"]) for item in configs}) == 3
    assert len({str(item["db_session"].bind.url) for item in built}) == 3
    assert all(item["trading_graph"] is None for item in built)
    assert all(item["data_provider"].source.name == "alpaca" for item in built)
    assert built[0]["paper_portfolio"]["macro"] is None
    assert built[1]["paper_portfolio"]["macro"] is None
    assert built[2]["paper_portfolio"]["macro"] is not None
    for item in built:
        item["db_session"].close()


def test_curva_portfolio_uses_injected_fred_client_without_nan(monkeypatch, tmp_path):
    source = Path("hipotesis/cartera-nucleo-curva.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    Path("local-data").mkdir()
    monkeypatch.setattr(paper_trading.StrategyAssembler, "build_components", lambda *a, **k: SimpleNamespace(order_manager=object()))
    monkeypatch.setattr(paper_trading, "build_source", lambda name: SimpleNamespace(name=name))
    client = SimpleNamespace(observaciones=lambda *a, **k: pd.DataFrame({
        "fecha": [pd.Timestamp("2026-10-24")], "valor": [0.25],
        "realtime_start": [pd.Timestamp("2026-10-24")], "realtime_end": [pd.Timestamp("2262-04-11")],
    }))
    monkeypatch.setattr(paper_trading, "FredClient", lambda: client)
    built = []
    monkeypatch.setattr(paper_trading, "TradingScheduler", lambda **kwargs: built.append(kwargs) or SimpleNamespace())
    scheduler, session = paper_trading._build_portfolio_scheduler(_FakeConfig(), str(source))
    config = built[0]["paper_portfolio"]
    assert config["macro"] is client
    from quantagent.trading.cartera_paper import exposicion_piernas
    exposure = exposicion_piernas(config["hipotesis"], pd.DataFrame({"TLT": [100]}, index=[pd.Timestamp("2026-10-25")]), macro=config["macro"])
    assert not exposure[(2, "TLT")].isna().any()
    session.close()


def test_curva_portfolio_fails_clearly_without_fred_key(monkeypatch, tmp_path):
    source = Path("hipotesis/cartera-nucleo-curva.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(paper_trading, "FredClient", lambda: (_ for _ in ()).throw(RuntimeError("falta FRED_API_KEY")))
    with pytest.raises(RuntimeError, match="falta FRED_API_KEY"):
        paper_trading._build_portfolio_scheduler(_FakeConfig(), str(source))


def test_main_rejects_two_portfolios(monkeypatch):
    monkeypatch.setattr(paper_trading, "_parse_args", lambda: argparse.Namespace(
        portfolio=["a.yaml", "b.yaml"], strategy=None, strategy_params="{}"))
    monkeypatch.setattr(paper_trading, "_apply_overrides", lambda _args: _FakeConfig())
    with pytest.raises(ValueError, match="cartera"):
        paper_trading.main()

    monkeypatch.setattr(paper_trading, "_parse_args", lambda: argparse.Namespace(
        portfolio=["a.yaml"], strategy="RSIMeanReversionStrategy", strategy_params="{}"))
    with pytest.raises(ValueError, match="cartera"):
        paper_trading.main()


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
            portfolio=None,
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
