"""Thresholds used by the account alarm evaluator."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AlarmConfig:
    drawdown_limit_pct: float = 8.0
    shadow_gap: float = 0.05
    stale_minutes_market_open: int = 30
    stale_minutes_market_closed: int = 1440

    @classmethod
    def from_env(cls):
        defaults = cls()
        return cls(
            drawdown_limit_pct=float(os.getenv("QUANTAGENT_ALARM_DRAWDOWN_LIMIT_PCT", defaults.drawdown_limit_pct)),
            shadow_gap=float(os.getenv("QUANTAGENT_ALARM_SHADOW_GAP", defaults.shadow_gap)),
            stale_minutes_market_open=int(os.getenv("QUANTAGENT_ALARM_STALE_MINUTES_MARKET_OPEN", defaults.stale_minutes_market_open)),
            stale_minutes_market_closed=int(os.getenv("QUANTAGENT_ALARM_STALE_MINUTES_MARKET_CLOSED", defaults.stale_minutes_market_closed)),
        )
