from .base import TradingSignal, TradingStrategy
from .fifty_two_week_high_strategy import FiftyTwoWeekHighStrategy
from .llm_agent_strategy import LLMAgentStrategy
from .registry import (
    STRATEGY_REGISTRY,
    build_strategy,
    get_strategy_names,
    get_strategy_registry,
)
from .rsi_strategy import RSIMeanReversionStrategy
from .triple_screen_strategy import TripleScreenStrategy

__all__ = [
    "StrategyAssembler",
    "ResolvedConfig",
    "TradingComponents",
    "TradingStrategy",
    "TradingSignal",
    "FiftyTwoWeekHighStrategy",
    "LLMAgentStrategy",
    "RSIMeanReversionStrategy",
    "STRATEGY_REGISTRY",
    "TripleScreenStrategy",
    "build_strategy",
    "get_strategy_names",
    "get_strategy_registry",
]


def __getattr__(name: str):
    if name in {"ResolvedConfig", "StrategyAssembler", "TradingComponents"}:
        from .assembler import ResolvedConfig, StrategyAssembler, TradingComponents

        exports = {
            "ResolvedConfig": ResolvedConfig,
            "StrategyAssembler": StrategyAssembler,
            "TradingComponents": TradingComponents,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
