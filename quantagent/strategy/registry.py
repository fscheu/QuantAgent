"""Central registry of available trading strategies."""

from typing import Any

from .base import TradingStrategy
from .fifty_two_week_high_strategy import FiftyTwoWeekHighStrategy
from .llm_agent_strategy import LLMAgentStrategy
from .rsi_strategy import RSIMeanReversionStrategy
from .triple_screen_strategy import TripleScreenStrategy

STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "RSIMeanReversionStrategy": {
        "cls": RSIMeanReversionStrategy,
        "type": "deterministic",
        "display_name": RSIMeanReversionStrategy.describe()["display_name"],
        "description": RSIMeanReversionStrategy.describe()["description"],
        "min_bars": 15,
        "params": {
            "rsi_period": {
                "type": int,
                "default": 14,
                "description": "RSI calculation period.",
            },
            "oversold_threshold": {
                "type": float,
                "default": 30.0,
                "description": "RSI level that triggers a long setup.",
            },
            "overbought_threshold": {
                "type": float,
                "default": 70.0,
                "description": "RSI level that triggers a short setup.",
            },
            "stop_loss_pct": {
                "type": float,
                "default": 0.02,
                "description": "Stop loss percentage.",
            },
            "take_profit_pct": {
                "type": float,
                "default": 0.03,
                "description": "Take profit percentage.",
            },
            "trailing_stop_pct": {
                "type": float,
                "default": 0.05,
                "description": "Trailing stop percentage.",
            },
        },
    },
    "FiftyTwoWeekHighStrategy": {
        "cls": FiftyTwoWeekHighStrategy,
        "type": "deterministic",
        "display_name": FiftyTwoWeekHighStrategy.describe()["display_name"],
        "description": FiftyTwoWeekHighStrategy.describe()["description"],
        "min_bars": 303,
        "params": {
            "lookback_days": {
                "type": int,
                "default": 252,
                "description": "Rolling lookback window for the 52-week high.",
            },
            "proximity_threshold": {
                "type": float,
                "default": 0.98,
                "description": "Minimum proximity ratio to the prior 52-week high.",
            },
            "trend_ma_period": {
                "type": int,
                "default": 50,
                "description": "Trend moving average period.",
            },
            "volume_ma_period": {
                "type": int,
                "default": 20,
                "description": "Volume moving average period.",
            },
            "volume_factor": {
                "type": float,
                "default": 1.5,
                "description": "Required multiple over average volume.",
            },
            "stop_loss_pct": {
                "type": float,
                "default": 0.05,
                "description": "Stop loss percentage.",
            },
            "take_profit_pct": {
                "type": float,
                "default": 0.15,
                "description": "Take profit percentage.",
            },
            "trailing_stop_pct": {
                "type": float,
                "default": 0.08,
                "description": "Trailing stop percentage.",
            },
        },
    },
    "TripleScreenStrategy": {
        "cls": TripleScreenStrategy,
        "type": "deterministic",
        "display_name": TripleScreenStrategy.describe()["display_name"],
        "description": TripleScreenStrategy.describe()["description"],
        "min_bars": 73,
        "params": {
            "weekly_bars": {
                "type": int,
                "default": 5,
                "description": "Bars aggregated into the higher timeframe trend view.",
            },
            "trend_ema_period": {
                "type": int,
                "default": 13,
                "description": "EMA period for the trend filter.",
            },
            "stoch_k_period": {
                "type": int,
                "default": 5,
                "description": "Stochastic percent-K period.",
            },
            "stoch_d_period": {
                "type": int,
                "default": 3,
                "description": "Stochastic percent-D period.",
            },
            "stoch_oversold": {
                "type": float,
                "default": 20.0,
                "description": "Oversold threshold for long pullbacks.",
            },
            "stoch_overbought": {
                "type": float,
                "default": 80.0,
                "description": "Overbought threshold for short pullbacks.",
            },
            "stop_loss_pct": {
                "type": float,
                "default": 0.02,
                "description": "Stop loss percentage.",
            },
            "take_profit_pct": {
                "type": float,
                "default": 0.04,
                "description": "Take profit percentage.",
            },
            "trailing_stop_pct": {
                "type": float,
                "default": 0.05,
                "description": "Trailing stop percentage.",
            },
        },
    },
    "LLMAgentStrategy": {
        "cls": LLMAgentStrategy,
        "type": "llm",
        "display_name": LLMAgentStrategy.describe()["display_name"],
        "description": LLMAgentStrategy.describe()["description"],
        "min_bars": 30,
        "params": {},
    },
}


def get_strategy_registry() -> dict[str, dict[str, Any]]:
    """Return registered strategy metadata."""
    return STRATEGY_REGISTRY


def get_strategy_names() -> list[str]:
    """Return registered strategy names."""
    return list(STRATEGY_REGISTRY.keys())


def build_strategy(name: str, **kwargs: Any) -> TradingStrategy:
    """Instantiate a registered strategy with optional constructor params."""
    return STRATEGY_REGISTRY[name]["cls"](**kwargs)
