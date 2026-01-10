"""LLM Agent Strategy - wrapper around TradingGraph."""

from typing import Dict, List, Optional

from ..models import ActivePosition
from .base import TradingSignal, TradingStrategy


class LLMAgentStrategy(TradingStrategy):
    """
    Trading strategy that wraps TradingGraph (multi-agent LLM system).

    Uses default exit logic (fixed % SL/TP + trailing stop).
    Does not re-evaluate positions once opened.
    """

    def __init__(self, trading_graph):
        """
        Initialize LLM Agent Strategy.

        Args:
            trading_graph: TradingGraph instance with configured LLMs
        """
        self.trading_graph = trading_graph

    def generate_signal(
        self,
        kline_data: List[Dict],
        symbol: str,
        timeframe: str,
        current_price: float,
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal using TradingGraph agents.

        Args:
            kline_data: List of OHLCV candles
            symbol: Asset symbol
            timeframe: Timeframe (e.g., "4h")
            current_price: Current market price

        Returns:
            TradingSignal or None if HOLD
        """
        # Prepare initial state for graph
        initial_state: Dict = {
            "kline_data": kline_data,
            "time_frame": timeframe,
            "stock_name": symbol,
            "messages": [],
        }

        # Invoke graph
        result = self.trading_graph.graph.invoke(initial_state)

        # Extract decision
        trading_decision_raw = result.get("final_trade_decision", "HOLD")

        # Parse decision (may be string like "LONG with 0.75 confidence")
        decision, confidence = self._parse_decision(trading_decision_raw)

        # If HOLD, return None
        if decision == "HOLD":
            return None

        # Extract reasoning
        reasoning = result.get("reasoning", "")
        if not reasoning:
            # Fallback: get from decision agent report
            decision_report = result.get("decision_report", {})
            if isinstance(decision_report, dict):
                reasoning = decision_report.get("reasoning", "LLM Agent analysis")

        # Calculate default SL/TP (2% and 3% from current price)
        if decision == "LONG":
            stop_loss = current_price * 0.98
            take_profit = current_price * 1.03
        else:  # SHORT
            stop_loss = current_price * 1.02
            take_profit = current_price * 0.97

        return TradingSignal(
            decision=decision,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            trailing_stop_pct=0.05,  # 5% trailing stop
        )

    def _parse_decision(self, decision_str: str) -> tuple[str, float]:
        """
        Parse decision string from graph output.

        Examples:
            "LONG" -> ("LONG", 1.0)
            "LONG with 0.75 confidence" -> ("LONG", 0.75)
            "SHORT" -> ("SHORT", 1.0)

        Args:
            decision_str: Raw decision string

        Returns:
            (decision, confidence) tuple
        """
        decision_str = str(decision_str).strip().upper()

        # Default confidence
        confidence = 1.0

        # Extract decision
        if "LONG" in decision_str:
            decision = "LONG"
        elif "SHORT" in decision_str:
            decision = "SHORT"
        else:
            decision = "HOLD"

        # Try to extract confidence if present
        # Look for patterns like "0.75" or ".75"
        import re

        # Match floating point numbers (e.g., 0.75, .85, 0.9)
        float_pattern = r"\b\d*\.?\d+\b"
        matches = re.findall(float_pattern, decision_str)

        for match in matches:
            try:
                val = float(match)
                # Only consider values that look like confidence (between 0 and 1)
                if 0.0 <= val <= 1.0:
                    confidence = val
                    break
            except ValueError:
                continue

        return decision, confidence

    def should_reevaluate(
        self,
        position: ActivePosition,
        current_price: float,
    ) -> bool:
        """
        LLM Agent Strategy does not re-evaluate once position is opened.

        Returns:
            False always
        """
        return False
