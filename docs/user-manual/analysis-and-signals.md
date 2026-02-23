# Analysis & Signals Guide

This guide explains how QuantAgent's AI agents analyze markets and generate trading signals.

---

## How Analysis Works

QuantAgent uses **four specialized AI agents** working together:

```
Market Data → 4 AI Agents → Trading Signal → (Optional) Order Execution
```

### The Four Agents

**1. Indicator Agent**
- **Analyzes:** Technical indicators (RSI, MACD, Stochastic, Momentum)
- **Outputs:** Overbought/oversold levels, momentum strength, divergences
- **Example:** "RSI at 28 indicates oversold condition. MACD bullish crossover."

**2. Pattern Agent**
- **Analyzes:** Chart patterns using vision AI
- **Outputs:** Support/resistance levels, candlestick patterns, trendlines
- **Example:** "Bullish engulfing pattern at $50,000 support. Strong bounce."

**3. Trend Agent**
- **Analyzes:** Trend direction and strength
- **Outputs:** Uptrend/downtrend identification, channel analysis
- **Example:** "Clear uptrend since Feb 15. Price above 20-period EMA."

**4. Decision Agent**
- **Analyzes:** All three agent outputs combined
- **Outputs:** Final trading signal (LONG, SHORT, HOLD) with confidence score
- **Example:** "LONG recommendation. 78% confidence. All three agents bullish."

<!-- screenshot: Diagram showing 4 agents feeding into decision agent -->

---

## Signal Types

### LONG Signal

**Meaning:** Buy the asset (or close SHORT position)

**When generated:**
- Bullish indicators (oversold RSI, MACD crossover up)
- Bullish patterns (support bounce, hammer candle)
- Strong uptrend confirmation

**Action taken:**
- If no position: Open LONG position
- If SHORT position: Close SHORT, optionally open LONG (position reversal)
- If already LONG: May add to position (depending on risk limits)

### SHORT Signal

**Meaning:** Sell the asset (or close LONG position)

**When generated:**
- Bearish indicators (overbought RSI, MACD crossover down)
- Bearish patterns (resistance rejection, shooting star candle)
- Strong downtrend confirmation

**Action taken:**
- If no position: Open SHORT position
- If LONG position: Close LONG, optionally open SHORT
- If already SHORT: May add to position

**Note:** Paper trading simulates SHORT positions (tracks P&L as if you sold)

### HOLD Signal

**Meaning:** No action, maintain current position or stay out

**When generated:**
- Conflicting signals from agents
- Low confidence (<50%)
- Unclear market conditions (consolidation, choppy price action)

**Action taken:** None. System waits for clearer signal.

**Why important:** Prevents overtrading and false signals

---

## Viewing Signals

**Location:** Dashboard → Analyses tab

### Analyses Table

Shows all historical signals with columns:

| Column | Description |
|--------|-------------|
| **Timestamp** | When analysis was performed |
| **Symbol** | Asset analyzed (BTC, SPX, etc.) |
| **Timeframe** | Candle period (1h, 4h, 1d) |
| **Signal** | LONG, SHORT, or HOLD |
| **Confidence** | 0-100% (how certain the agents are) |
| **Environment** | backtest or paper |
| **Order ID** | Link to resulting trade (if executed) |

<!-- screenshot: Analyses table with multiple signals -->

### Filtering Analyses

**Left sidebar filters:**

**By Symbol:**
- Select specific asset (e.g., only BTC analyses)
- Multi-select for multiple assets

**By Date Range:**
- Last 24 hours
- Last 7 days
- Last 30 days
- Custom range

**By Signal Type:**
- Show only LONG signals
- Show only SHORT signals
- Show only HOLD (useful for understanding when system stays out)

**By Confidence:**
- Slider to filter (e.g., only show signals with >70% confidence)
- Helps identify high-conviction trades

**By Environment:**
- Backtest - Historical tests
- Paper - Simulated live trading

<!-- screenshot: Filter sidebar with options selected -->

---

## Understanding Confidence Scores

**Confidence = How strongly agents agree**

### High Confidence (>70%)

**Means:**
- All three analysis agents agree
- Clear technical setup
- Strong conviction

**Example:**
```
Indicator Agent: Oversold RSI (20), bullish MACD → LONG
Pattern Agent: Double bottom at support → LONG
Trend Agent: Uptrend continuation → LONG
Decision Agent: LONG, 85% confidence
```

**Trading implication:** Higher probability of success

### Medium Confidence (50-70%)

**Means:**
- Two agents agree, one neutral or disagrees
- Mixed technical picture
- Reasonable setup but not perfect

**Example:**
```
Indicator Agent: Neutral RSI (50), flat MACD → HOLD
Pattern Agent: Bullish engulfing candle → LONG
Trend Agent: Weak uptrend → LONG
Decision Agent: LONG, 62% confidence
```

**Trading implication:** Acceptable for execution, but watch closely

### Low Confidence (<50%)

**Means:**
- Agents disagree significantly
- Unclear market conditions
- Conflicting signals

**Example:**
```
Indicator Agent: Overbought RSI (75) → SHORT
Pattern Agent: Failed breakout → SHORT
Trend Agent: Strong uptrend → LONG
Decision Agent: HOLD, 45% confidence (conflicting data)
```

**Trading implication:** System typically outputs HOLD or skips trade

---

## Signal Provenance

Every signal tracks its full history for auditability.

### Provenance Fields

**Model Information:**
- Provider (openai, anthropic)
- Model name (gpt-4o-mini, claude-haiku)
- Temperature used (0.1, 0.5, etc.)

**Execution Context:**
- Thread ID (LangGraph execution chain)
- Checkpoint ID (resume point for replay)
- Environment (backtest vs paper)

**Outcome Tracking:**
- Order ID (if signal triggered a trade)
- Trade ID (final execution record)
- P&L (profit/loss of resulting trade)

**Why this matters:**
- **Reproducibility** - Replay exact analysis with same model
- **Debugging** - Understand why a trade was made
- **Model comparison** - Test different AI models on same data

<!-- screenshot: Expanded signal showing provenance details -->

---

## Analysis Workflow (Behind the Scenes)

### Step 1: Data Gathering

System fetches:
- OHLC data (Open, High, Low, Close prices)
- Volume
- Historical candles (lookback period)

**Sources:**
- Primary: yfinance API
- Fallback: Cached database (18x faster on repeat runs)

### Step 2: Indicator Calculation

**Indicator Agent calculates:**
- RSI (Relative Strength Index) - 14 periods
- MACD (Moving Average Convergence Divergence) - 12/26/9
- Stochastic Oscillator - 14/3/3
- Momentum indicators

**Output:** JSON with values and interpretations

### Step 3: Pattern Recognition

**Pattern Agent analyzes:**
- Chart image (generated from price data)
- Candlestick patterns (engulfing, doji, hammer, etc.)
- Support/resistance zones
- Trendlines

**Uses:** Vision-enabled LLM (gpt-4o, claude-sonnet with vision)

**Output:** Detected patterns and strength

### Step 4: Trend Analysis

**Trend Agent evaluates:**
- Moving averages (20, 50, 200-period)
- Price channels
- Higher highs / lower lows
- Trend strength

**Output:** Trend direction and confidence

### Step 5: Decision Synthesis

**Decision Agent:**
1. Reads all three agent outputs
2. Identifies agreements and conflicts
3. Applies weighting (recent patterns > old signals)
4. Generates final signal with confidence score
5. Provides reasoning for decision

**Output:** Trading signal + detailed explanation

### Step 6: Risk Check (If Order Triggered)

**Risk Manager validates:**
- Position size within limits
- Daily loss not exceeded
- Not too many open positions
- Slippage assumptions applied

**Output:** Order approved or rejected

### Step 7: Persistence

**Saves to database:**
- Analysis record with all agent outputs
- Signal and confidence
- Provenance fields (model, thread ID, etc.)
- Links to any resulting orders/trades

---

## Interpreting Agent Outputs

### Indicator Agent Examples

**Bullish Setup:**
```
RSI: 25 (oversold, bounce expected)
MACD: Bullish crossover (signal > MACD line)
Stochastic: Oversold and turning up
Momentum: Increasing positive momentum

Interpretation: Strong buy signal, likely bounce
```

**Bearish Setup:**
```
RSI: 78 (overbought, correction likely)
MACD: Bearish crossover (signal < MACD line)
Stochastic: Overbought and turning down
Momentum: Decreasing positive momentum

Interpretation: Strong sell signal, likely pullback
```

**Neutral Setup:**
```
RSI: 52 (neutral range)
MACD: Flat, no clear direction
Stochastic: Mid-range
Momentum: Low

Interpretation: No clear signal, wait for better setup
```

### Pattern Agent Examples

**Bullish Pattern:**
```
Detected: Bullish engulfing candle at $50,000
Context: Price bounced off key support level
Additional: Volume spike on reversal candle

Interpretation: High probability reversal, consider LONG
```

**Bearish Pattern:**
```
Detected: Shooting star at resistance ($65,000)
Context: Failed breakout above prior high
Additional: Declining volume on rally

Interpretation: Likely rejection, consider SHORT
```

**No Clear Pattern:**
```
Detected: Doji candles, indecision
Context: Sideways consolidation
Additional: No clear support or resistance

Interpretation: Wait for breakout direction
```

### Trend Agent Examples

**Strong Uptrend:**
```
Price above 20, 50, 200-day moving averages
Higher highs and higher lows intact
Trend channel: Ascending
Strength: Strong

Interpretation: Favor LONG signals, avoid SHORTs
```

**Strong Downtrend:**
```
Price below all moving averages
Lower highs and lower lows
Trend channel: Descending
Strength: Strong

Interpretation: Favor SHORT signals, avoid LONGs
```

**Trendless/Choppy:**
```
Price crossing moving averages frequently
No clear higher/lower pattern
Range-bound
Strength: Weak

Interpretation: High risk environment, prefer HOLD
```

---

## Common Signal Scenarios

### Scenario 1: All Agents Agree (High Confidence)

**Example:**
```
Indicator Agent: LONG (oversold RSI, bullish MACD)
Pattern Agent: LONG (double bottom at support)
Trend Agent: LONG (uptrend continuation)
→ Decision Agent: LONG, 88% confidence
```

**What happens:** Order likely executed (if risk allows)

**Success rate:** Typically highest (60-70% win rate)

### Scenario 2: Two Agree, One Disagrees (Medium Confidence)

**Example:**
```
Indicator Agent: LONG (oversold conditions)
Pattern Agent: SHORT (resistance rejection)
Trend Agent: LONG (uptrend intact)
→ Decision Agent: LONG, 58% confidence (2 vs 1)
```

**What happens:** May execute, but lower conviction

**Success rate:** Moderate (45-55% win rate)

### Scenario 3: Conflict (Low Confidence → HOLD)

**Example:**
```
Indicator Agent: SHORT (overbought)
Pattern Agent: LONG (bullish pattern)
Trend Agent: HOLD (unclear trend)
→ Decision Agent: HOLD, 40% confidence (no consensus)
```

**What happens:** No order placed, system waits

**Why good:** Avoids risky trades in unclear conditions

### Scenario 4: Position Reversal

**Situation:** System is LONG, receives strong SHORT signal

**Example:**
```
Current position: LONG BTC at $50,000
New signal: SHORT, 82% confidence

Decision Agent reasoning:
"Strong bearish signals across all agents. 
Recommend closing LONG and opening SHORT."

Actions:
1. Close LONG position (book P&L)
2. Open SHORT position (if risk allows)
```

**See:** [Position Reversal Fix](../06_implementation/QuantAgent-g3c-IM-position-reversal-fix.md) - QuantAgent-g3c

---

## Analyzing Past Signals

### Finding Winning Patterns

**Approach:**
1. Go to Analyses tab
2. Filter by high confidence (>75%)
3. Look at signals that resulted in profitable trades
4. Identify common characteristics

**Example findings:**
- "All wins had RSI < 30 and bullish engulfing"
- "Trend continuation signals (uptrend + LONG) win 70% of the time"
- "Counter-trend signals (downtrend + LONG) win only 35%"

### Understanding Losses

**Approach:**
1. Filter signals linked to losing trades
2. Review agent outputs for patterns
3. Identify weaknesses

**Common loss patterns:**
- Agents disagreed (low confidence executed anyway)
- Ignored broader trend (SHORT in strong uptrend)
- Breakout fakeouts (pattern failed)
- External events (news, sudden volatility)

**Action:** Adjust strategy to avoid these setups

---

## Signal Quality Metrics (Backtest Validation)

After running backtests, evaluate signal quality:

### High-Quality Signals (Keep)

**Characteristics:**
- Win rate >50%
- High confidence correlates with wins
- Profit factor >1.5

**Example:**
```
LONG signals with confidence >70%
Backtest results: 42 trades, 65% win rate, 2.1 profit factor
```

### Low-Quality Signals (Refine)

**Characteristics:**
- Win rate <40%
- Confidence doesn't predict outcome
- Losses larger than wins

**Example:**
```
SHORT signals in uptrend markets
Backtest results: 18 trades, 28% win rate, 0.6 profit factor
```

**Action:** Filter these signals out or improve agent logic

---

## Advanced: Checkpointing & Replay

### What Are Checkpoints?

**Checkpoints** save the complete state of an analysis execution:
- All agent outputs
- Intermediate reasoning steps
- Model parameters used

**Stored as:** Thread ID + Checkpoint ID in database

### Why Replay Analysis?

**Use cases:**
1. **Reproduce results** - Run exact same analysis again
2. **Debug issues** - Step through agent decisions
3. **Model comparison** - Replay with different AI model
4. **Audit compliance** - Prove what system decided and when

### How to Replay

**Not exposed in MVP UI, but available via API:**

```python
from quantagent.graph_setup import TradingGraph

graph = TradingGraph()

# Replay from checkpoint
result = graph.analyze_from_checkpoint(
    thread_id="abc-123",
    checkpoint_id="def-456"
)
```

**See:** [Checkpoint Integration](../01_requirements/trading_system_requirements.md#c-checkpoint-integration-for-analyses)

---

## Troubleshooting

### "No signals generated for my backtest"

**Possible causes:**
1. All analyses resulted in HOLD (no clear signals)
2. Risk manager rejected all trades (limits too tight)
3. Date range has no data

**Check:**
- Analyses tab - Do analyses exist? What signals?
- Logs tab - Look for "Order rejected by RiskManager"
- Configuration - Are risk limits reasonable?

### "Signals don't make sense"

**Example:** "Signal says LONG but RSI is overbought"

**Explanation:** Decision agent weighs all factors, not just one indicator

**What to do:**
- Click signal to expand full reasoning
- Check if pattern or trend overrode indicator
- Review confidence score (low confidence = conflicting data)

### "Low confidence signals executing anyway"

**Cause:** No minimum confidence threshold in MVP

**Workaround:** Filter out low-confidence signals in analysis

**Future feature:** Configurable confidence threshold in risk manager

### "Signals differ between runs"

**Possible causes:**
1. Different model temperature (randomness)
2. Model provider changed (OpenAI vs Anthropic)
3. Market data changed (live data vs cached)

**Reproducibility:**
- Use same model preset
- Set temperature to 0.0 (fully deterministic)
- Check config snapshot in backtest run

---

## Related Documentation

- **Position Monitoring System**: [QuantAgent-nu7-DS-active-position-monitoring.md](../03_design/QuantAgent-nu7-DS-active-position-monitoring.md) - QuantAgent-nu7
- **TradingStrategy Abstraction**: [QuantAgent-enn-IM-trading-strategy.md](../06_implementation/QuantAgent-enn-IM-trading-strategy.md) - QuantAgent-enn
- **Position Reversal Logic**: [QuantAgent-g3c-IM-position-reversal-fix.md](../06_implementation/QuantAgent-g3c-IM-position-reversal-fix.md) - QuantAgent-g3c
- **Message State Management**: [QuantAgent-h7d-IM-message-state-validation.md](../06_implementation/QuantAgent-h7d-IM-message-state-validation.md) - QuantAgent-h7d
- **Requirements**: [Trading System Requirements](../01_requirements/trading_system_requirements.md)

---

*Understanding how signals are generated is key to trusting the system and improving strategies.*
