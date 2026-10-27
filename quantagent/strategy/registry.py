"""Central registry of available trading strategies."""

from typing import Any

from .arquetipo_a_strategy import ArquetipoAStrategy
from .arquetipo_b_strategy import ArquetipoBStrategy
from .base import TradingStrategy
from .curva_pendiente_strategy import CurvaPendienteStrategy
from .fifty_two_week_high_strategy import FiftyTwoWeekHighStrategy
from .halloween_strategy import HalloweenStrategy
from .llm_agent_strategy import LLMAgentStrategy
from .momentum_12m_strategy import Momentum12mStrategy
from .rsi_regimen_strategy import RsiRegimenStrategy
from .rsi_strategy import RSIMeanReversionStrategy
from .senal_externa_strategy import SenalExterna
from .sma_cross_strategy import SmaCrossStrategy
from .triple_screen_strategy import TripleScreenStrategy
from .turn_of_the_month_strategy import TurnOfTheMonthStrategy

STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "RSIMeanReversionStrategy": {
        "cls": RSIMeanReversionStrategy,
        "type": "deterministic",
        "cli_alias": "rsi",
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
        "cli_alias": "fifty-two-week-high",
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
        "cli_alias": "triple-screen",
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
    "ArquetipoAStrategy": {
        "cls": ArquetipoAStrategy,
        "type": "deterministic",
        "cli_alias": "arquetipo-a",
        "display_name": ArquetipoAStrategy.describe()["display_name"],
        "description": ArquetipoAStrategy.describe()["description"],
        "min_bars": 185,
        "params": {
            "weekly_bars": {"type": int, "default": 5, "description": "Bars per higher timeframe bar."},
            "trend_ema_period": {"type": int, "default": 26, "description": "Weekly EMA trend period."},
            "macd_fast": {"type": int, "default": 12, "description": "Weekly MACD fast period."},
            "macd_slow": {"type": int, "default": 26, "description": "Weekly MACD slow period."},
            "macd_signal": {"type": int, "default": 9, "description": "Weekly MACD signal period."},
            "force_index_period": {"type": int, "default": 2, "description": "Force Index period."},
            "rsi_period": {"type": int, "default": 3, "description": "RSI period."},
            "rsi_oversold": {"type": float, "default": 30.0, "description": "RSI oversold threshold."},
            "ema_trend_period": {"type": int, "default": 22, "description": "Daily EMA trend period."},
            "entry_atr_mult": {"type": float, "default": 0.1, "description": "ATR multiple for buy stop."},
            "entry_expiry_bars": {"type": int, "default": 3, "description": "Expiry bars for buy stop."},
            "atr_period": {"type": int, "default": 14, "description": "ATR period."},
            "swing_window": {"type": int, "default": 5, "description": "Lookback for swing low."},
            "swing_atr_mult": {"type": float, "default": 2.0, "description": "ATR multiple below swing low."},
            "safezone_period": {"type": int, "default": 10, "description": "SafeZone period."},
            "safezone_coef": {"type": float, "default": 2.0, "description": "SafeZone coefficient."},
            "chandelier_mult": {"type": float, "default": 3.0, "description": "ATR multiple for Chandelier exit."},
        },
    },
    "ArquetipoBStrategy": {
        "cls": ArquetipoBStrategy,
        "type": "deterministic",
        "cli_alias": "arquetipo-b",
        "display_name": ArquetipoBStrategy.describe()["display_name"],
        "description": ArquetipoBStrategy.describe()["description"],
        "min_bars": 201,
        "params": {
            "bollinger_period": {"type": int, "default": 20, "description": "Período del ancho de Bollinger."},
            "bollinger_k": {"type": float, "default": 2.0, "description": "Multiplicador k de Bollinger."},
            "bollinger_lookback": {"type": int, "default": 60, "description": "Ventana de sesiones para compresión de Bollinger."},
            "bollinger_quantile": {"type": float, "default": 0.20, "description": "Percentil para compresión de Bollinger."},
            "ema_fast": {"type": int, "default": 50, "description": "Período EMA rápida."},
            "ema_slow": {"type": int, "default": 200, "description": "Período EMA lenta."},
            "donchian_period": {"type": int, "default": 20, "description": "Período Donchian."},
            "volume_period": {"type": int, "default": 20, "description": "Período volumen promedio."},
            "volume_factor": {"type": float, "default": 1.5, "description": "Factor multiplicador de volumen."},
            "atr_period": {"type": int, "default": 14, "description": "Período ATR."},
            "stop_atr_mult": {"type": float, "default": 2.5, "description": "Múltiplo ATR stop inicial."},
            "chandelier_mult": {"type": float, "default": 3.0, "description": "Múltiplo ATR Chandelier exit."},
        },
    },
    "SmaCrossStrategy": {
        "cls": SmaCrossStrategy,
        "type": "deterministic",
        "cli_alias": "sma-cross",
        "display_name": SmaCrossStrategy.describe()["display_name"],
        "description": SmaCrossStrategy.describe()["description"],
        "min_bars": 201,
        "params": {
            "fast": {
                "type": int,
                "default": 50,
                "description": "Fast simple moving average period (sessions).",
            },
            "slow": {
                "type": int,
                "default": 200,
                "description": "Slow simple moving average period (sessions).",
            },
        },
    },
    "Momentum12mStrategy": {
        "cls": Momentum12mStrategy,
        "type": "deterministic",
        "cli_alias": "momentum-12m",
        "display_name": Momentum12mStrategy.describe()["display_name"],
        "description": Momentum12mStrategy.describe()["description"],
        "min_bars": 253,
        "params": {
            "lookback": {
                "type": int,
                "default": 252,
                "description": "Sessions of the reference return (12 months).",
            },
        },
    },
    "HalloweenStrategy": {"cls": HalloweenStrategy, "type": "deterministic", "cli_alias": "halloween", "display_name": HalloweenStrategy.describe()["display_name"], "description": HalloweenStrategy.describe()["description"], "min_bars": 1, "params": {}},
    "TurnOfTheMonthStrategy": {"cls": TurnOfTheMonthStrategy, "type": "deterministic", "cli_alias": "turn-of-the-month", "display_name": TurnOfTheMonthStrategy.describe()["display_name"], "description": TurnOfTheMonthStrategy.describe()["description"], "min_bars": 1, "params": {}},
    "SenalExterna": {"cls": SenalExterna, "type": "deterministic", "cli_alias": "senal-externa", "display_name": SenalExterna.describe()["display_name"], "description": SenalExterna.describe()["description"], "min_bars": 1, "params": {"ruta": {"type": str, "default": "", "description": "Ruta al CSV con timestamp, posicion."}}},
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


def effective_params(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the registered defaults of `name` updated with `overrides`."""
    defaults = {k: v["default"] for k, v in STRATEGY_REGISTRY[name]["params"].items()}
    return {**defaults, **(overrides or {})}


def register_strategy(
    cls: type[TradingStrategy],
    *,
    min_bars: int,
    cli_alias: str | None = None,
    params: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Register `cls` under its class name; `cli_alias` makes it available in `backtest run`."""
    meta = cls.describe()
    STRATEGY_REGISTRY[cls.__name__] = {
        "cls": cls,
        "type": meta["type"],
        "display_name": meta["display_name"],
        "description": meta["description"],
        "min_bars": min_bars,
        "params": params or {},
        **({"cli_alias": cli_alias} if cli_alias else {}),
    }


def cli_aliases() -> dict[str, str]:
    """Return {CLI short name: registry name} for every strategy with a `cli_alias`."""
    return {e["cli_alias"]: name for name, e in STRATEGY_REGISTRY.items() if "cli_alias" in e}


register_strategy(RsiRegimenStrategy, min_bars=272, cli_alias="rsi-regimen", params=RsiRegimenStrategy.PARAMS)
register_strategy(CurvaPendienteStrategy, min_bars=1, cli_alias="curva-pendiente", params=CurvaPendienteStrategy.PARAMS)
