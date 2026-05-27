from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def _build_app_test(tmp_path: Path) -> AppTest:
    script_path = tmp_path / "configuration_app.py"
    script_path.write_text(
        '''
from apps.streamlit.views.configuration import render


class FakeDB:
    ok = False


render(FakeDB(), "paper")
'''
    )
    at = AppTest.from_file(str(script_path), default_timeout=30)
    at.session_state["ui_profiles"] = {
        "portfolio": {"growth": {"universe": ["BTC", "SPX"]}},
        "risk": {},
        "combined": {},
    }
    at.session_state["model_presets"] = {
        "default": {"provider": "openai", "model_name": "gpt-4o-mini", "temperature": 0.1}
    }
    at.session_state["default_profiles"] = {"paper": None, "backtest": None}
    at.session_state["default_strategy"] = {"paper": None, "backtest": None}
    return at


def test_configuration_strategy_default_selectors_render(tmp_path: Path):
    at = _build_app_test(tmp_path)

    at.run()

    paper_selector = next(
        widget for widget in at.selectbox if widget.label == "Paper default strategy"
    )
    backtest_selector = next(
        widget for widget in at.selectbox if widget.label == "Backtest default strategy"
    )

    assert "TripleScreenStrategy" in paper_selector.options
    assert "LLMAgentStrategy" in backtest_selector.options


def test_configuration_strategy_default_set_on_button_click(tmp_path: Path):
    at = _build_app_test(tmp_path)

    at.run()

    paper_selector = next(
        widget for widget in at.selectbox if widget.label == "Paper default strategy"
    )
    paper_selector.select("TripleScreenStrategy")
    at.run()

    button = next(
        widget for widget in at.button if widget.key == "btn_default_strategy_paper"
    )
    button.click()
    at.run()

    assert at.session_state["default_strategy"]["paper"] == "TripleScreenStrategy"
