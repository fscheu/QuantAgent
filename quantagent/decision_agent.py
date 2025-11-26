"""
Agent for making final trade decisions in high-frequency trading (HFT) context.
Combines indicator, pattern, and trend reports to issue a LONG or SHORT order.

Returns structured TradingDecision using with_structured_output pattern.
"""

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from quantagent.agent_models import IndicatorReport, PatternReport, TrendReport, TradingDecision
from quantagent.agent_utils import invoke_with_retry


def create_final_trade_decider(llm):
    """
    Create a trade decision agent node.

    Consumes structured Pydantic reports (IndicatorReport, PatternReport, TrendReport)
    and outputs a structured TradingDecision with LONG/SHORT/HOLD order.

    Uses with_structured_output() pattern for direct Pydantic instance return.
    Includes centralized retry logic for resilience against transient errors.
    """

    def trade_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
        # Extract structured reports from state
        indicator_report = state["indicator_report"]
        pattern_report = state["pattern_report"]
        trend_report = state["trend_report"]
        time_frame = state["time_frame"]
        stock_name = state["stock_name"]

        # Convert Pydantic models to dict for better readability in prompt
        indicator_dict = indicator_report.model_dump() if hasattr(indicator_report, "model_dump") else indicator_report.__dict__
        pattern_dict = pattern_report.model_dump() if hasattr(pattern_report, "model_dump") else pattern_report.__dict__
        trend_dict = trend_report.model_dump() if hasattr(trend_report, "model_dump") else trend_report.__dict__

        # --- System message ---
        system_message = SystemMessage(
            content=f"You are a high-frequency quantitative trading (HFT) analyst operating on the current {time_frame} K-line chart for {stock_name}. Your task is to issue an **immediate execution order**: **LONG** or **SHORT**. ⚠️ HOLD is only acceptable if signals are conflicting or unclear."
        )

        # --- Human message with decision context and structured data ---
        human_content = f"""Your decision should forecast the market move over the **next N candlesticks**, where:
- For example: TIME_FRAME = 15min, N = 1 → Predict the next 15 minutes.
- TIME_FRAME = 4hour, N = 1 → Predict the next 4 hours.

Base your decision on the combined strength, alignment, and timing of the following three **structured reports**:

---

### 1. Technical Indicator Report (Structured):
- **Confidence**: {indicator_dict.get('confidence', 0.0)}
- **Trend Direction**: {indicator_dict.get('trend_direction', 'neutral')}
- **RSI Level**: {indicator_dict.get('rsi_level', 'neutral')} (RSI = {indicator_dict.get('rsi', 50.0)})
- **MACD**: {indicator_dict.get('macd', 0.0)} (Signal: {indicator_dict.get('macd_signal', 0.0)})
- **ROC**: {indicator_dict.get('roc', 0.0)}
- **Stochastic**: {indicator_dict.get('stochastic', 50.0)}
- **Williams %R**: {indicator_dict.get('willr', -50.0)}
- **Analysis**: {indicator_dict.get('reasoning', 'No analysis')}

Evaluation strategy:
    - Evaluate momentum (e.g., MACD, ROC) and oscillators (e.g., RSI, Stochastic, Williams %R).
    - Give **higher weight to strong directional signals** such as MACD crossovers, RSI divergence, extreme overbought/oversold levels.
    - **Ignore or down-weight neutral or mixed signals** unless they align across multiple indicators.
---

### 2. Pattern Report (Structured):
- **Patterns Detected**: {pattern_dict.get('patterns_detected', [])}
- **Primary Pattern**: {pattern_dict.get('primary_pattern', 'None')}
- **Confidence**: {pattern_dict.get('confidence', 0.0)}
- **Breakout Probability**: {pattern_dict.get('breakout_probability', 0.0)}
- **Analysis**: {pattern_dict.get('reasoning', 'No analysis')}

Evaluation strategy:
- Only act on **clearly recognizable, mostly complete patterns** with **confirmed breakout/breakdown**  or highly probable based on price and momentum (e.g., strong wick, volume spike, engulfing candle).
- **Do NOT act** on early-stage or speculative patterns without indicator confirmation. Do not treat consolidating setups as tradable unless there is **breakout confirmation** from other reports.

---

### 3. Trend Report (Structured):
- **Trend Direction**: {trend_dict.get('trend_direction', 'sideways')}
- **Trend Strength**: {trend_dict.get('trend_strength', 0.0)}
- **Support Level**: {trend_dict.get('support_level', 0.0)}
- **Resistance Level**: {trend_dict.get('resistance_level', 0.0)}
- **Analysis**: {trend_dict.get('reasoning', 'No analysis')}

Evaluation strategy:
- Analyze price interaction with support/resistance trendlines.
- **Upward sloping support** = buying interest. **Downward sloping resistance** = selling pressure.
- Predict breakout **only with confluence** (strong candles + indicator confirmation).
- **Do NOT assume breakout direction** from geometry alone.
---

### ✅ Decision Algorithm

1. **Alignment Score**: Count how many reports agree on direction (0-3).
   - 3 aligned = **Very strong signal** → Confidence ≥ 0.8
   - 2 aligned = **Moderate signal** → Confidence ≥ 0.6
   - 1 or 0 aligned = **Weak/conflicting** → Default to HOLD or weakest evidence

2. **Signal Strength**:
   - **Indicator confidence** × **Pattern confidence** × **Trend strength** = **Overall confidence**
   - Only trade if overall confidence ≥ 0.5

3. **Direction Priority** (if alignment exists):
   - **Bullish**: Indicator bullish + Pattern bullish/neutral + Trend upward/sideways
   - **Bearish**: Indicator bearish + Pattern bearish/neutral + Trend downward/sideways

4. **Risk-Reward Ratio**:
   - Base on **trend strength** and **current volatility indicators** (RSI extremes = wider stops)
   - Suggest ratio between **1.2 and 1.8**

5. **Final Decision**:
   - If all three align → **LONG** or **SHORT** with confidence ≥ 0.75
   - If two align → **LONG** or **SHORT** with confidence 0.55-0.74
   - If one or zero align → **HOLD** (wait for clearer signals)

---

### 📊 Output Format (JSON):

Respond ONLY with valid JSON (no markdown, no explanation):

```json
{{
  "decision": "<LONG|SHORT|HOLD>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<Concise explanation based on alignment and signal strength>",
  "risk_level": "<low|medium|high>",
  "entry_price": <float or null>,
  "stop_loss": <float or null>,
  "take_profit": <float or null>
}}
```
"""

        # Build message list
        # messages = state.get("messages", [])
        # if not messages:
        messages = [system_message, HumanMessage(content=human_content)]

        try:
            # --- Structured LLM call for decision ---
            # Use with_structured_output for direct Pydantic instance return
            structured_llm = llm.with_structured_output(TradingDecision)

            trading_decision = invoke_with_retry(
                structured_llm.invoke,
                messages,
                retries=3,
                wait_sec=2
            )

            # Ensure we got a valid TradingDecision
            if not isinstance(trading_decision, TradingDecision):
                trading_decision = TradingDecision(
                    decision="HOLD",
                    confidence=0.0,
                    reasoning="Output validation failed. Decision agent recommends HOLD until signals clarify.",
                    risk_level="high"
                )
        except Exception as e:
            # Fallback decision if LLM call fails
            trading_decision = TradingDecision(
                decision="HOLD",
                confidence=0.0,
                reasoning=f"Decision analysis failed: {str(e)}. Recommending HOLD until signals clarify.",
                risk_level="high"
            )

        return {
            "final_trade_decision": trading_decision,
            "messages": messages,
        }

    return trade_decision_node
