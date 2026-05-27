from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def _build_app_test(tmp_path: Path, db_run_ids: list[int]) -> AppTest:
    script_path = tmp_path / "backtesting_app.py"
    script_path.write_text(
        f'''
from datetime import datetime
from types import SimpleNamespace

from apps.streamlit.views.backtesting import render


class FakeRun:
    def __init__(self, run_id):
        self.id = run_id
        self.created_at = datetime(2026, 1, 1)
        self.total_trades = None
        self.assets = ["BTC"]
        self.timeframe = "1h"
        self.start_date = datetime(2026, 1, 1)
        self.end_date = datetime(2026, 1, 2)
        self.config_snapshot = {{
            "model_preset": "default",
            "profile": None,
            "mode": "Generate + Execute",
            "artifacts": "path-only",
        }}
        self.win_rate = None
        self.profit_factor = None
        self.sharpe_ratio = None
        self.max_drawdown = None
        self.total_pnl = None


class FakeBacktestRunModel:
    class created_at:
        @staticmethod
        def desc():
            return FakeBacktestRunModel.created_at


class FakeStrategyConfigModel:
    name = "name"


class FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter_by(self, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def all(self):
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class FakeSession:
    def __init__(self, backtest_runs):
        self._backtest_runs = backtest_runs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def query(self, model):
        if model is FakeBacktestRunModel:
            return FakeQuery(self._backtest_runs)
        if model is FakeStrategyConfigModel:
            return FakeQuery([])
        raise AssertionError(f"Unexpected model queried: {{model!r}}")


class FakeDB:
    ok = True
    models = SimpleNamespace(
        BacktestRun=FakeBacktestRunModel,
        StrategyConfig=FakeStrategyConfigModel,
    )

    def __init__(self, backtest_runs):
        self._backtest_runs = backtest_runs

    def SessionLocal(self):
        return FakeSession(self._backtest_runs)


render(FakeDB([FakeRun(run_id) for run_id in {db_run_ids!r}]), "backtest")
'''
    )
    at = AppTest.from_file(str(script_path), default_timeout=30)
    at.session_state["backtest_runs"] = [
        {
            "id": 1,
            "created_at": "2026-01-01T00:00:00",
            "status": "pending",
            "progress": 0,
            "assets": ["BTC"],
            "timeframe": "1h",
            "range_start": "2026-01-01",
            "range_end": "2026-01-02",
            "model_preset": "default",
            "profile": None,
            "mode": "Generate + Execute",
            "artifacts": "path-only",
            "environment": "backtest",
        }
    ]
    at.session_state["ui_profiles"] = {"portfolio": {}, "risk": {}, "combined": {}}
    at.session_state["model_presets"] = {
        "default": {"provider": "openai", "model_name": "gpt-4o-mini", "temperature": 0.1}
    }
    at.session_state["default_strategy"] = {"paper": None, "backtest": None}
    return at


def test_render_deduplicates_session_and_db_runs(tmp_path: Path):
    at = _build_app_test(tmp_path, [1])

    at.run()

    df = at.dataframe[0].value
    assert df["id"].tolist() == [1]


def test_render_keeps_distinct_session_and_db_runs(tmp_path: Path):
    at = _build_app_test(tmp_path, [2])

    at.run()

    df = at.dataframe[0].value
    assert df["id"].tolist() == [1, 2]


def test_backtesting_strategy_selector_uses_default_from_session(tmp_path: Path):
    at = _build_app_test(tmp_path, [])
    at.session_state["default_strategy"] = {
        "paper": None,
        "backtest": "TripleScreenStrategy",
    }

    at.run()

    strategy_selector = next(widget for widget in at.selectbox if widget.label == "Estrategia")
    assert strategy_selector.value == "TripleScreenStrategy"
