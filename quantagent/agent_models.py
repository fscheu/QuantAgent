"""
Pydantic models for structured agent outputs.

These models define the schema for agent reports, enabling:
- Type validation at state boundaries
- Easy parsing in downstream agents (no string parsing)
- Better testing and mocking
- LLM output validation with structured JSON
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class IndicatorReport(BaseModel):
    """Structured output from indicator_agent.

    Contains computed technical indicators and LLM analysis.
    """

    macd: float = Field(description="MACD value (difference between 26-period and 12-period EMA)")
    macd_signal: float = Field(description="MACD signal line (9-period EMA of MACD)")
    macd_histogram: float = Field(description="MACD histogram (MACD - Signal)")

    rsi: float = Field(ge=0.0, le=100.0, description="RSI value (0-100)")
    rsi_level: str = Field(description="RSI interpretation: 'overbought' (>70), 'oversold' (<30), or 'neutral'")

    roc: float = Field(description="Rate of Change percentage")

    stochastic: float = Field(ge=0.0, le=100.0, description="Stochastic oscillator (0-100)")

    willr: float = Field(ge=-100.0, le=0.0, description="Williams %R (-100 to 0)")

    trend_direction: str = Field(description="Interpreted trend: 'bullish', 'bearish', or 'neutral'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level (0.0-1.0)")

    reasoning: str = Field(description="LLM analysis reasoning for indicator interpretation")


class PatternReport(BaseModel):
    """Structured output from pattern_agent.

    Contains candlestick pattern analysis from vision LLM.
    """

    patterns_detected: List[str] = Field(
        default_factory=list,
        description="List of identified candlestick patterns (e.g., 'double_bottom', 'inverse_head_and_shoulders')"
    )

    primary_pattern: Optional[str] = Field(
        default=None,
        description="Most confident pattern identified, if any"
    )

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Pattern confidence level (0.0-1.0)"
    )

    breakout_probability: float = Field(
        ge=0.0, le=1.0,
        description="Probability of breakout based on pattern (0.0-1.0)"
    )

    reasoning: str = Field(
        description="Vision LLM analysis of candlestick chart patterns"
    )


class TrendReport(BaseModel):
    """Structured output from trend_agent.

    Contains trend analysis with support/resistance levels.
    """

    support_level: float = Field(
        description="Support price level derived from trendline analysis"
    )

    resistance_level: float = Field(
        description="Resistance price level derived from trendline analysis"
    )

    trend_direction: str = Field(
        description="Trend interpretation: 'upward', 'downward', or 'sideways'"
    )

    trend_strength: float = Field(
        ge=0.0, le=1.0,
        description="Trend strength (0.0-1.0), based on trendline slope and K-line proximity"
    )

    reasoning: str = Field(
        description="Trend analysis reasoning based on support/resistance interaction"
    )


class TradingDecision(BaseModel):
    """Structured output from decision_agent.

    Final trading decision synthesizing all agent reports.
    """

    decision: str = Field(
        description="Trading decision: 'LONG', 'SHORT', or 'HOLD'"
    )

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall decision confidence (0.0-1.0)"
    )

    reasoning: str = Field(
        description="Reasoning for the trading decision based on all agent reports"
    )

    risk_level: str = Field(
        description="Risk assessment: 'low', 'medium', or 'high'"
    )

    entry_price: Optional[float] = Field(
        default=None,
        description="Suggested entry price if LONG/SHORT"
    )

    stop_loss: Optional[float] = Field(
        default=None,
        description="Suggested stop loss price"
    )

    take_profit: Optional[float] = Field(
        default=None,
        description="Suggested take profit price"
    )
