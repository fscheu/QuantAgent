# Backtesting Guide

Backtesting lets you test trading strategies on historical market data to see how they would have performed in the past.

**Why backtest?**
- Validate strategy ideas before risking real money
- Compare different approaches (timeframes, risk levels, assets)
- Build confidence in AI agent decisions
- Identify what works and what doesn't

---

## How Backtesting Works

QuantAgent replays historical price data and simulates what the AI agents would have decided at each moment:

```
Historical Data → AI Analysis → Simulated Trades → Performance Metrics
```

**Step-by-step:**
1. System fetches historical candles (e.g., 4-hour bars for last 90 days)
2. For each candle, the 4 AI agents analyze market conditions
3. Decision agent produces a signal: LONG, SHORT, or HOLD
4. If signal triggers a trade, system simulates execution with realistic slippage
5. Portfolio tracks positions and calculates P&L
6. Final metrics show overall strategy performance

**Important:** Backtest trades are **simulated only**. No real orders are placed.

<!-- screenshot: Backtesting workflow diagram -->

---

## Running a Backtest

You can run backtests two ways:
1. **Dashboard (Recommended for beginners)** - Visual interface
2. **Python Script** - Programmatic control

### Method 1: Dashboard Backtest

**Location:** Streamlit Dashboard → Backtesting tab

#### Step 1: Configure the Backtest

**Profile Selection:**
- Choose a saved strategy profile (see [Configuration Guide](strategy-configuration.md))
- Or leave empty to use default settings

**Assets:**
- Leave blank to use profile's universe
- Or manually select specific symbols (e.g., BTC, SPX)

**Timeframe:**
- `1h` - 1-hour candles (more trades, slower)
- `4h` - 4-hour candles **(recommended for MVP)**
- `1d` - Daily candles (fewer trades, faster)

**Date Range:**
- Last 30 days - Quick validation
- Last 90 days - Standard testing
- Custom range - Pick specific start/end dates

**Model Settings:**
- Select a saved model preset
- Or keep default (OpenAI gpt-4o-mini)

**Backtest Name:**
- Use descriptive names: "BTC 4h Conservative", "Multi-Asset Q1"

<!-- screenshot: Backtesting form filled out ready to run -->

#### Step 2: Execute the Backtest

Click **"Create & Run Backtest"**

**What Happens:**
1. "Fetching data..." - Downloads market data (slow on first run)
2. "Running analysis..." - AI agents analyze each candle
3. "Calculating metrics..." - Computes performance results
4. "Complete!" - Results display below

**Typical Duration:**
- First run: 2-5 minutes (downloads + caches data)
- Subsequent runs: 30-60 seconds (uses cached data)

**Progress Indicator:**
Streamlit shows a spinning animation while running. Do not close the browser tab.

<!-- screenshot: Backtest in progress with spinner -->

#### Step 3: Review Results

Once complete, you'll see three sections:

**A. Metrics Summary Table**

| Metric | Value | What It Means |
|--------|-------|---------------|
| Total Trades | 15 | Number of positions opened |
| Win Rate | 60% | Percentage of profitable trades |
| Profit Factor | 2.1 | Dollars won per dollar lost |
| Sharpe Ratio | 1.4 | Risk-adjusted return quality |
| Max Drawdown | 8.2% | Worst peak-to-trough decline |
| Total P&L | $4,250 | Net profit/loss |
| Total Return | 4.25% | Percentage gain on capital |

**B. Equity Curve Chart**

Line graph showing portfolio value over time.

**What to look for:**
- **Smooth upward trend** = Consistent profits
- **Steep drops** = Large losing streaks
- **Flat periods** = Strategy not finding trades

<!-- screenshot: Equity curve chart showing upward trend with minor drawdowns -->

**C. Trade List**

Table of all executed trades with:
- Entry/exit dates and prices
- P&L per trade
- Position size
- Duration held

Click on trades to see the analysis that triggered them.

<!-- screenshot: Trade list table with P&L column highlighted -->

### Method 2: Python Script

**Location:** `examples/run_backtest.py`

**Run from terminal:**
```bash
python examples/run_backtest.py
```

**Edit the script** to customize:
```python
backtest = Backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    assets=['BTC', 'SPX'],
    timeframe='4h',
    initial_capital=100000.0,
    config={
        'base_position_pct': 0.05,
        'max_daily_loss_pct': 0.05,
        # ... more settings
    }
)

metrics = backtest.run(name="My Custom Backtest")
```

**Advantages:**
- Automate multiple backtests
- Integrate with other Python tools
- Save results to CSV/JSON

**See also:** [Backtesting Engine Technical Docs](../03_design/backtesting_engine.md)

---

## Understanding Metrics

### Win Rate

**Formula:** `Winning Trades ÷ Total Trades`

**Example:** 12 wins out of 20 trades = 60% win rate

**What's Good:**
- **Above 50%** = Excellent (more winners than losers)
- **40-50%** = Acceptable (if profit factor is high)
- **Below 40%** = Poor (needs improvement)

**Reality Check:**
Professional strategies often have 40-55% win rates. High win rates don't guarantee profits if losses are large.

### Profit Factor

**Formula:** `Total Gains ÷ Total Losses`

**Example:**
- Winning trades: +$10,000
- Losing trades: -$5,000
- Profit Factor: 10,000 ÷ 5,000 = 2.0

**What's Good:**
- **Above 2.0** = Strong (wins are 2x losses)
- **1.5-2.0** = Acceptable
- **Below 1.5** = Weak
- **Below 1.0** = Strategy loses money

### Sharpe Ratio

**What it measures:** Return per unit of risk taken

**Formula:** `(Return - Risk-Free Rate) ÷ Volatility`

**What's Good:**
- **Above 2.0** = Excellent
- **1.0-2.0** = Good (MVP target is ≥1.0)
- **0.5-1.0** = Acceptable for high-risk strategies
- **Below 0.5** = Poor risk-adjusted returns

**Why it matters:**
A strategy with 10% return and low volatility is better than 15% return with huge swings.

### Max Drawdown

**What it measures:** Worst peak-to-trough portfolio decline

**Example:**
- Portfolio peaks at $110,000
- Drops to $95,000
- Drawdown: (110k - 95k) ÷ 110k = 13.6%

**What's Good:**
- **Below 10%** = Low risk
- **10-15%** = Moderate risk (MVP target is ≤15%)
- **15-25%** = High risk (uncomfortable for most traders)
- **Above 25%** = Extreme risk (hard to recover from)

**Psychology:**
A 20% drawdown requires a 25% gain to break even. Large drawdowns are emotionally difficult to endure.

### Total Return

**Formula:** `(Final Value - Initial Value) ÷ Initial Value × 100`

**Example:**
- Start with $100,000
- End with $112,000
- Return: 12%

**Context Matters:**
- 12% in 1 month = Amazing
- 12% in 1 year = Good for crypto, mediocre for stocks
- 12% over 5 years = Poor

**Compare to benchmarks:**
- S&P 500 averages ~10% annually
- Bitcoin highly variable (can be +200% or -70% in a year)

---

## Strategy Viability Criteria

QuantAgent considers a strategy **viable for paper trading** if it meets **all three** criteria:

| Metric | Minimum Threshold |
|--------|------------------|
| Win Rate | ≥ 40% |
| Sharpe Ratio | ≥ 1.0 |
| Max Drawdown | ≤ 15% |

**Example Good Strategy:**
```
Win Rate:     48%  ✓
Sharpe Ratio: 1.35 ✓
Max Drawdown: 11%  ✓
→ Ready for paper trading
```

**Example Weak Strategy:**
```
Win Rate:     55%  ✓
Sharpe Ratio: 0.7  ✗
Max Drawdown: 22%  ✗
→ Needs improvement
```

**What to do if criteria aren't met:**
1. Adjust risk settings (reduce position sizes)
2. Try different timeframes (longer = less noise)
3. Filter assets (maybe strategy works for BTC but not SPX)
4. Review losing trades to find patterns

---

## Optimizing Backtest Performance

### First Run is Slow (Data Caching)

**Problem:** First backtest on new symbols takes 3-5 minutes

**Why:** System downloads historical data from yfinance API

**Solution:** Data is automatically cached in PostgreSQL. Second run uses cache and takes ~10 seconds.

**18x Speedup** after initial data fetch!

### Choosing the Right Timeframe

**1-hour candles:**
- **Pros:** More trading opportunities, detailed analysis
- **Cons:** Slow (lots of API calls), noisy signals
- **Use when:** Testing high-frequency strategies, short periods

**4-hour candles (Recommended):**
- **Pros:** Balanced speed and detail, less noise
- **Cons:** Fewer trades than 1h
- **Use when:** General strategy development (MVP default)

**Daily candles:**
- **Pros:** Fast, clear trends
- **Cons:** Very few trades, misses intraday opportunities
- **Use when:** Long-term strategies, quick validation

### Date Range Selection

**Short periods (7-30 days):**
- Quick sanity checks
- Not statistically significant
- Risk of overfitting to recent market conditions

**Medium periods (90 days / 3 months):**
- **Recommended for MVP**
- Includes different market conditions
- Enough trades for meaningful metrics

**Long periods (1+ year):**
- Robust validation
- Slower execution
- Better confidence in results

**Tip:** Start with 30 days for development, expand to 90-180 days for final validation.

---

## Common Backtest Scenarios

### Scenario 1: Testing a New Strategy Idea

**Goal:** See if strategy concept has merit

**Approach:**
1. Run 30-day backtest with default settings
2. Review win rate and P&L quickly
3. If positive, expand to 90 days
4. If negative, adjust or abandon

### Scenario 2: Comparing Risk Profiles

**Goal:** Find optimal position sizing

**Approach:**
1. Create 3 profiles: conservative (2%), balanced (5%), aggressive (10%)
2. Run backtest once to generate analyses
3. Use **Replay** feature to test all 3 profiles
4. Compare metrics side-by-side

**See:** [Replay Feature](dashboard.md#tab-5-replay)

### Scenario 3: Multi-Asset Performance

**Goal:** Understand which assets work best

**Approach:**
1. Run separate backtests for each asset
2. Compare win rates and Sharpe ratios
3. Build portfolio of best performers

**Example Results:**
- BTC: 55% win rate, Sharpe 1.8 → Keep
- SPX: 42% win rate, Sharpe 0.9 → Keep (borderline)
- CL: 30% win rate, Sharpe 0.3 → Drop

### Scenario 4: Finding Optimal Timeframe

**Goal:** Which timeframe gives best risk-adjusted returns

**Approach:**
1. Run backtest on 1h, 4h, and 1d for same period
2. Compare Sharpe ratios (not just P&L)
3. Choose highest Sharpe with acceptable drawdown

**Why Sharpe matters more than P&L:**
Higher returns with lower volatility = sustainable strategy

---

## Interpreting Results

### Red Flags (Strategy May Not Work)

**Low Win Rate + Low Profit Factor:**
```
Win Rate:     28%
Profit Factor: 0.8
→ Loses more than it wins, AND losses are bigger than wins
```

**High Drawdown:**
```
Max Drawdown: 35%
→ Too risky; hard to recover from such losses
```

**Inconsistent Equity Curve:**
```
[Chart shows huge spike, then crashes below starting value]
→ Lucky early, then gave it all back
```

### Encouraging Signs

**Consistent Small Wins:**
```
Win Rate:     52%
Avg Win:      $350
Avg Loss:     $200
→ Steady grind upward
```

**Smooth Equity Curve:**
```
[Chart shows gradual upward slope with small dips]
→ Controlled risk, consistent performance
```

**Good Sharpe + Acceptable Win Rate:**
```
Win Rate:     45%
Sharpe Ratio: 1.6
→ Losses are small and controlled
```

---

## Limitations of Backtesting

**Backtesting is NOT perfect prediction.** Be aware of these limitations:

### Survivorship Bias
**Issue:** Only tests assets that exist today. Doesn't account for delisted stocks or failed cryptocurrencies.

**Impact:** Overstates historical performance.

### Look-Ahead Bias
**Issue:** Using data that wouldn't have been available at decision time.

**QuantAgent Protection:** Checkpoints ensure each analysis only sees past data.

### Slippage Simulation
**Issue:** Backtests assume 1% slippage, but real markets vary.

**Reality:** Volatile markets or large orders may have 2-5% slippage.

### Market Regime Changes
**Issue:** Markets evolve. What worked in 2023 may not work in 2026.

**Mitigation:** Test on multiple time periods. If strategy works across different years, more likely to be robust.

### Overfitting
**Issue:** Strategy fits historical data perfectly but fails on new data.

**Signs:** 90%+ win rate in backtest but 30% in paper trading.

**Prevention:** Keep strategies simple. Avoid excessive parameter tuning.

---

## After the Backtest: Next Steps

### If Strategy Looks Good (Meets Viability Criteria)

1. **Run longer backtest** (6-12 months) to confirm robustness
2. **Paper trade** for 30 days to validate in live market
3. **Monitor carefully** - compare paper results to backtest expectations
4. **Refine if needed** based on live performance

### If Strategy Needs Work

1. **Analyze losing trades** - Find patterns in losses
2. **Adjust risk parameters** - Reduce position sizes
3. **Try different assets** - Maybe strategy works better for BTC than stocks
4. **Consider different timeframes** - More noise in 1h, clearer trends in 4h
5. **Review AI agent outputs** - Are agents making logical decisions?

### Before Going Live with Real Money

**QuantAgent MVP focuses on backtesting and paper trading.** Real broker integration is Phase 2.

**Safety checklist before real trading:**
- [ ] Strategy profitable in 6+ month backtest
- [ ] Win rate ≥ 45%, Sharpe ≥ 1.2, Drawdown ≤ 12%
- [ ] 30+ days successful paper trading
- [ ] Understand why strategy works (not just "it's profitable")
- [ ] Risk only capital you can afford to lose

---

## Troubleshooting

### "No data available for symbol"

**Cause:** yfinance API doesn't recognize symbol or data not available for date range

**Solution:**
- Check symbol spelling (BTC vs BTC-USD)
- Try different date range (some symbols have limited history)
- Check internet connection

### "Backtest taking too long"

**Cause:** Large date range or 1-hour timeframe with many API calls

**Solutions:**
- Reduce date range (test 30 days first)
- Use 4h or 1d timeframe
- Let first run complete to populate cache

### "Zero trades executed"

**Cause:** Risk manager blocking all trades (limits too tight)

**Solution:**
- Check profile: `max_position_pct` might be too low
- Relax `max_daily_loss_pct` for testing (e.g., 20%)
- Review logs tab for rejection reasons

### Metrics show NaN or infinity

**Cause:** Division by zero (e.g., no losing trades so profit factor is infinite)

**Solution:**
- Extend backtest period to get more trades
- This is actually good (no losses!) but not realistic long-term

---

## Related Documentation

- **Backtesting Engine**: [backtesting_engine.md](../03_design/backtesting_engine.md) - Technical architecture
- **Position Monitoring**: [QuantAgent-nu7-DS-active-position-monitoring.md](../03_design/QuantAgent-nu7-DS-active-position-monitoring.md) - QuantAgent-nu7
- **Market Hours Filtering**: [QuantAgent-s92-DS-backtest-market-hours.md](../03_design/QuantAgent-s92-DS-backtest-market-hours.md) - QuantAgent-s92
- **Backtest Isolation**: [QuantAgent-94d-DS-backtest-isolation.md](../03_design/QuantAgent-94d-DS-backtest-isolation.md) - QuantAgent-94d
- **Trade P&L Calculation**: [QuantAgent-r78-DS-trade-pnl-calculation.md](../03_design/QuantAgent-r78-DS-trade-pnl-calculation.md) - QuantAgent-r78
- **Manual Test Cases**: [MVP_MANUAL_TEST_CASES.md](../05_acceptance_tests/MVP_MANUAL_TEST_CASES.md)
- **Backtest Metrics**: [QuantAgent-r6y (Paper Metrics)](../02_planning/phase1_roadmap.md)

---

*Backtesting is a tool, not a guarantee. Always validate with paper trading before risking real capital.*
