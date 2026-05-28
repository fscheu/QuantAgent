from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from quantagent.llm.registry import supported_providers


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
    at.session_state["provider_routing_presets"] = {
        "default": {
            "deep_reasoning": {
                "provider": "anthropic",
                "model_name": "claude-haiku-4-5-20251001",
                "temperature": 0.1,
            },
            "lite": {
                "provider": "openai",
                "model_name": "gpt-4o-mini",
                "temperature": 0.2,
            },
            "image": None,
        }
    }
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


def test_configuration_provider_selector_uses_registry_options(tmp_path: Path):
    at = _build_app_test(tmp_path)

    at.run()

    provider_selector = next(widget for widget in at.selectbox if widget.key == "model_provider")

    assert list(provider_selector.options) == supported_providers()


def test_configuration_saves_routing_preset_to_session(tmp_path: Path):
    at = _build_app_test(tmp_path)

    at.run()

    lite_provider = next(
        widget for widget in at.selectbox if widget.key == "routing_provider_lite"
    )
    lite_provider.select("qwen")
    at.run()

    lite_model = next(
        widget for widget in at.text_input if widget.key == "routing_model_lite"
    )
    lite_model.set_value("qwen3-max")
    at.run()

    save_name = next(
        widget for widget in at.text_input if widget.key == "routing_preset_save_name"
    )
    save_name.set_value("cost-efficient")
    at.run()

    save_button = next(
        widget for widget in at.button if widget.key == "save_provider_routing_preset"
    )
    save_button.click()
    at.run()

    saved = at.session_state["provider_routing_presets"]["cost-efficient"]
    assert saved["lite"]["provider"] == "qwen"
    assert saved["lite"]["model_name"] == "qwen3-max"
