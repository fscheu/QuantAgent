# Dashboard Guide

The QuantAgent dashboard is a web-based interface built with Streamlit. It provides visual access to all system features without requiring command-line interaction.

---

## Launching the Dashboard

Start the dashboard from your terminal:

```bash
streamlit run apps/streamlit/app.py
```

**Access URL:** http://localhost:8501

The dashboard automatically connects to your PostgreSQL database to display real-time data.

<!-- screenshot: Dashboard home screen with all tabs visible -->

---

## Dashboard Overview

The interface has **7 main tabs** organized by function:

| Tab | Purpose | When to Use |
|-----|---------|-------------|
| **Dashboard** | View KPIs and recent activity | Daily check-in, performance overview |
| **Configuration** | Create strategy profiles and model settings | Before backtesting, strategy setup |
| **Analyses** | Browse AI agent decisions | Review why trades were made |
| **Backtesting** | Run historical strategy tests | Validate new strategies |
| **Replay** | Re-test with different settings | Compare risk profiles |
| **Paper Trading** | Monitor automated scheduler and view live paper trade status | Daily check when automation is active |
| **Orders & Positions** | View trades and holdings | Track active positions |
| **Logs** | System events and errors | Troubleshooting |

**Environment Selector** (top-right): Switch between `backtest` and `paper` modes to filter data.

---

## Tab 1: Dashboard

**Purpose:** High-level system overview and key performance indicators.

### What You'll See

**Top Section: Quick Stats**
- Total trades executed
- Current win rate
- Active positions count
- Portfolio value (for paper trading)

**Recent Trades Table**
Shows last 10 trades with:
- Symbol (e.g., BTC, SPX)
- Entry and exit prices
- Profit/Loss amount
- Duration (how long position was held)

**System Status Indicators**
- Database connection: Green ✓ or Red ✗
- API keys configured: Check marks for active providers
- Scheduler status: Running or Stopped (future feature)

<!-- screenshot: Dashboard tab showing KPIs and recent trades -->

### How to Use It

**Daily Check-in Routine:**
1. Open Dashboard tab
2. Check win rate trend (improving or declining?)
3. Review recent trades for patterns
4. Verify database connection is healthy

**What Good Performance Looks Like:**
- Win rate: Above 40%
- Profit factor: Above 1.5
- No connection errors in status section

---

## Tab 2: Configuration

**Purpose:** Create and manage strategy profiles and AI model settings.

### Strategy Profiles

Strategy profiles define **what to trade** and **how much risk to take**.

**Profile Types:**
- **Portfolio** - Position sizing and universe (asset list)
- **Risk** - Loss limits and circuit breakers
- **Combined** - Both portfolio and risk in one profile

<!-- screenshot: Configuration tab with empty profile form -->

### Creating a Profile

**Step-by-step:**

1. **Select Profile Kind**: Choose `combined` (recommended for beginners)

2. **Enter Profile Name**: Use descriptive names like `conservative-crypto` or `aggressive-stocks`

3. **Choose Universe**: Multi-select assets to trade
   - BTC - Bitcoin
   - SPX - S&P 500 index
   - CL - Crude Oil
   - *(More symbols supported)*

4. **Edit Profile JSON**: Adjust settings in the text editor

**Example Configuration:**
```json
{
  "universe": ["BTC", "SPX"],
  "base_position_pct": 0.05,
  "max_position_pct": 0.10,
  "max_daily_loss_pct": 0.05,
  "slippage_pct": 0.01
}
```

**What Each Setting Means:**
- `base_position_pct`: Default trade size (5% = $5,000 per trade with $100k capital)
- `max_position_pct`: Maximum allocation per asset (10% cap)
- `max_daily_loss_pct`: Stop trading if daily loss exceeds 5%
- `slippage_pct`: Simulated execution cost (1% = trade executes 1% worse than signal price)

5. **Click "Save profile"**

**Success Message:** `Saved combined profile 'your-profile-name' to database.`

<!-- screenshot: Filled configuration form with JSON editor -->

### Model Presets

Model presets control **which AI model** analyzes the markets.

**Supported Providers:**
- **OpenAI** - gpt-4o-mini, gpt-4o
- **Anthropic** - claude-haiku, claude-sonnet
- **Qwen** - qwen3-max (Chinese provider)

**Creating a Model Preset:**

1. **Provider**: Select from dropdown (e.g., `openai`)
2. **Model Name**: Choose model version (e.g., `gpt-4o-mini`)
3. **Temperature**: Set randomness (0.0-1.0)
   - 0.1 = Conservative, consistent decisions
   - 0.5 = Balanced
   - 0.9 = Creative, varied responses
4. **Save as Name**: Enter preset name (e.g., `fast-gpt4`)
5. **Click "Save preset"**

**Cost Tip:** Use cheaper models like `gpt-4o-mini` or `claude-haiku` for development. Reserve expensive models for final validation.

<!-- screenshot: Model preset form with saved presets table -->

### Managing Profiles

**View Saved Profiles:**
Scroll down to see the **Profiles** table showing:
- Source: `db` (saved) or `session` (temporary)
- Kind: portfolio, risk, or combined
- Name and version

**Delete a Profile:**
*(Not yet implemented in UI - use database tools or recreate with same name)*

---

## Tab 3: Analyses

**Purpose:** View historical AI agent decisions and market analysis results.

### What Are Analyses?

Every time QuantAgent examines a market candle, it generates an **analysis record** containing:
- What each of the 4 AI agents saw
- Trading signal (LONG, SHORT, HOLD)
- Confidence score (0-100%)
- Reasoning and context

<!-- screenshot: Analyses tab with filter sidebar and results table -->

### Filtering Analyses

**Left Sidebar Filters:**
- **Symbol**: Focus on specific asset (e.g., only BTC analyses)
- **Date Range**: View analyses from specific time period
- **Signal Type**: Filter by LONG, SHORT, or HOLD decisions
- **Confidence**: Show only high-confidence signals (e.g., >70%)

**Example Use Case:**
"Show me all LONG signals for BTC with confidence above 80% from last week"

### Analysis Details

Click on any row to expand and see:

**Agent Outputs:**
- **Indicator Agent**: RSI, MACD, momentum metrics
- **Pattern Agent**: Chart patterns detected (support/resistance)
- **Trend Agent**: Trend direction and strength
- **Decision Agent**: Final recommendation synthesis

**Provenance Fields:**
- Model provider and version used
- Thread ID (for checkpoint replay)
- Order ID if trade was executed
- Environment (backtest vs paper)

**Why This Matters:**
You can trace exactly why a trade was made, which model version decided it, and whether it resulted in profit or loss.

<!-- screenshot: Expanded analysis showing all 4 agent outputs -->

### Common Patterns to Look For

**High Win Rate Signals:**
- Look for recurring technical patterns in winning trades
- Note confidence thresholds that correlate with success

**Losing Trades:**
- Check if agents disagreed (conflicting signals)
- Review whether external market events caused unexpected moves

---

## Tab 4: Backtesting

**Purpose:** Test trading strategies on historical market data.

*Detailed guide: See [Backtesting Guide](backtesting.md)*

**Quick Overview:**
1. Select strategy profile
2. Choose timeframe and date range
3. Run backtest
4. View performance metrics and equity curve

<!-- screenshot: Backtesting tab with completed run results -->

---

## Tab 5: Replay

**Purpose:** Re-run stored analyses with different strategy profiles without calling AI models again.

### Why Use Replay?

**Problem:** Backtests are slow because they call expensive AI APIs.

**Solution:** Replay reuses saved analysis results but applies different portfolio/risk rules.

**Example:**
- Run one backtest with AI analysis (generates 100 signals)
- Replay those 100 signals with 5 different risk profiles
- Compare which risk profile performs best
- **Time saved:** 4 additional backtests without API calls

<!-- screenshot: Replay tab showing sequential sweep setup -->

### How to Run a Replay

1. **Select Source Backtest**: Choose a completed backtest run from the dropdown.
2. **Choose Profiles to Test**: Multi-select one or more saved strategy profiles.
3. **Or reuse the source config**: Leave the profile list empty to replay with the original run settings.
4. **Click "Start replay (sequential)"**.

**Execution Mode:**
Runs one profile at a time (sequential) and reuses the stored backtest signals instead of calling the AI agents again.

**When It's Done:**
The tab shows one row per replay with the new run ID, trade count, win rate, profit factor, Sharpe ratio, max drawdown, total P&L, return, and elapsed time.

---

## Tab 6: Orders & Positions

**Purpose:** View trade history and currently open positions.

### Orders Table

Shows all trades with:
- **Symbol**: Asset traded
- **Side**: LONG (buy) or SHORT (sell)
- **Quantity**: Shares/contracts
- **Price**: Entry price
- **Status**: Filled, Pending, Cancelled
- **Timestamp**: When order was placed

**Filtering:**
Use environment selector to show only backtest or paper orders.

<!-- screenshot: Orders table with multiple entries -->

### Positions Table

Shows currently open positions (paper trading only):
- **Symbol**: Asset held
- **Quantity**: Current holdings
- **Entry Price**: Average cost
- **Current Price**: Live market price
- **Unrealized P&L**: Profit/loss if closed now

---

## Tab 7: Logs

**Purpose:** View system events, errors, and debugging information.

### Log Levels

- **INFO**: Normal operations (analysis completed, order placed)
- **WARNING**: Potential issues (API rate limit, missing data)
- **ERROR**: Problems that need attention (database connection lost)
- **DEBUG**: Detailed technical information (for developers)

<!-- screenshot: Logs tab with various log levels -->

### Filtering Logs

**Search by:**
- Time range
- Log level
- Module (e.g., only show `backtesting` logs)
- Symbol (e.g., only BTC-related events)

**Common Use Cases:**
- **Troubleshooting errors**: Filter by ERROR level
- **Audit trail**: Search for specific symbol and date
- **Performance investigation**: Look for WARNING messages about slow operations

### Understanding Log Messages

**Example Log Entry:**
```
[2026-02-23 14:35:12] INFO [quantagent.backtesting] 
Backtest 'Q1 Strategy' completed. 
Trades: 42, Win Rate: 57.1%, P&L: $8,450
{symbol: "BTC", environment: "backtest", thread_id: "abc-123"}
```

**Key Fields:**
- Timestamp: When event occurred
- Level: Severity
- Module: Which system component
- Message: What happened
- Metadata: Additional context (JSON)

---

## Tab 8: Paper Trading

**Purpose:** Monitor the automated paper trading scheduler and inspect its recent execution cycles.

### What You'll See

**Status Card**
- Current scheduler health: 🟢 Active, 🟡 Stale, 🔴 Stopped, or ⏳ Running
- Last successful cycle timestamp (human-readable, e.g. "12m ago")
- Cycle summary: assets processed, errors encountered, duration

**Recent Runs Table**
Shows up to 10 most recent scheduler cycles in descending order:
- Timestamp: when the cycle started
- Duration: how long the full cycle took
- Assets: tickers processed
- Errors: count of assets that failed
- Status: running, completed, or error

> If the table is empty and the scheduler hasn't been started, follow the [Paper Trading Automation guide](paper-trading-automation.md) to launch it.

### How to Use It

**Daily automation check:**
1. Open the Paper Trading tab
2. Confirm status is 🟢 Active or ⏳ Running
3. Review recent runs for error trends (rising error count = check logs)
4. If status is 🟡 Stale or 🔴 Stopped, investigate the terminal running `python apps/paper_trading.py` or check Logs tab → filter module `quantagent.trading.scheduler`

**Related:** [Paper Trading Automation](paper-trading-automation.md) · [Monitoring Guide](monitoring.md#scheduler-status)

<!-- screenshot: Paper Trading tab showing status card and recent runs table -->

---

## Tips for Effective Dashboard Use

### Daily Workflow

**Morning Check:**
1. Open Dashboard → Check overnight performance
2. Logs → Scan for any errors
3. Positions → Review open trades

**Strategy Development:**
1. Configuration → Create new profile
2. Backtesting → Test on historical data
3. Analyses → Review AI decisions
4. Replay → Optimize risk settings

**Troubleshooting:**
1. Logs → Find error timestamp
2. Analyses → Check what happened before error
3. Orders → Verify trades executed correctly

### Performance Tips

**Slow Dashboard?**
- Reduce date ranges in filters
- Close unused browser tabs
- Restart Streamlit if memory grows large

**Data Not Showing?**
- Check database connection (green indicator)
- Verify environment selector matches your data
- Refresh page (F5)

---

## Keyboard Shortcuts

Streamlit provides some navigation shortcuts:

- **R**: Rerun app (refresh data)
- **C**: Clear cache
- **?**: Show keyboard shortcuts

---

## Related Documentation

- **Streamlit Architecture**: [streamlit_app_architecture.md](../03_design/streamlit_app_architecture.md)
- **Configuration Details**: [CONFIGURATION.md](../03_design/CONFIGURATION.md) - QuantAgent-cxu
- **Logging System**: [LOGGING_STRATEGY.md](../03_design/LOGGING_STRATEGY.md) - QuantAgent-yuk
- **Manual Test Cases**: [MVP_MANUAL_TEST_CASES.md](../05_acceptance_tests/MVP_MANUAL_TEST_CASES.md)

---

*The dashboard is under active development. Features and layouts may change as the system evolves.*
