# Monitoring Guide

This guide explains how to track system performance, monitor logs, and ensure QuantAgent is running correctly.

---

## Monitoring Overview

Effective monitoring helps you:
- Catch errors early before they cause problems
- Understand system behavior and performance
- Validate that strategies are executing as expected
- Debug issues when things go wrong
- Track long-term performance trends

**Key monitoring locations:**
- Dashboard tab - High-level KPIs
- Logs tab - Detailed system events
- Orders & Positions tab - Trade execution tracking
- Database queries - Deep dive analysis

---

## Dashboard Monitoring

**Location:** Dashboard → Dashboard tab

### Key Performance Indicators (KPIs)

**Portfolio Value (Paper Trading)**
- Shows current total capital
- Updates with each trade
- Tracks growth over time

**Total Trades**
- Count of all executed trades
- Cumulative across all assets

**Win Rate**
- Percentage of profitable trades
- **Good:** >50%
- **Acceptable:** 40-50%
- **Needs work:** <40%

**Active Positions**
- Number of currently open positions
- Should match your risk limits (usually 1-3 open at once)

**Recent P&L**
- Last 24 hours profit/loss
- Quick gauge of current performance

<!-- screenshot: Dashboard KPIs section with metrics -->

### System Health Indicators

**Database Connection**
- ✅ Green checkmark = Connected
- ❌ Red X = Connection lost

**API Keys Configured**
- Shows which providers are set up (OpenAI, Anthropic, etc.)
- ⚠️ Warning if missing

**Scheduler Status**
- **Running (green):** The `apps/paper_trading.py` scheduler checked in during the last minute. Automation is healthy.
- **Idle (yellow):** Scheduler is installed but `settings.scheduler.enabled` is `False` or the process was paused. No paper trades will execute until you start it manually.
- **Stopped (red):** No heartbeat detected. Check the terminal where you started `python apps/paper_trading.py` or review the Logs tab (filter by `quantagent.trading.scheduler`).
- Use the [Paper Trading Automation guide](paper-trading-automation.md) for startup and recovery steps.

### Recent Trades Table

Shows last 10 trades with:
- Entry/exit times
- P&L amount
- Duration held
- Symbol

**What to watch for:**
- **Large losses** - Review what went wrong
- **Quick wins** - Identify successful patterns
- **Long holds** - Are positions stuck? Why no exit signal?

<!-- screenshot: Recent trades table showing various outcomes -->

---

## Log Monitoring

**Location:** Dashboard → Logs tab

Logs provide detailed system event tracking.

### Log Structure

Each log entry contains:

**Timestamp:** When event occurred (UTC)

**Level:** Severity
- **DEBUG** - Detailed technical info (for developers)
- **INFO** - Normal operations
- **WARNING** - Potential issues, not critical
- **ERROR** - Problems needing attention

**Module:** Which system component generated the log
- `quantagent.backtesting` - Backtest execution
- `quantagent.agents` - AI agent processing
- `quantagent.portfolio` - Position management
- `quantagent.risk` - Risk checks
- `apps.streamlit` - UI interactions

**Message:** Description of what happened

**Metadata:** Additional context (JSON)
- Symbol
- Environment (backtest/paper)
- Thread ID
- User-defined tags

<!-- screenshot: Logs tab with various log levels and filters -->

### Filtering Logs

**By Time Range:**
- Last hour
- Last 24 hours
- Last 7 days
- Custom range

**By Log Level:**
- Show only errors
- Errors + warnings
- All logs (includes INFO and DEBUG)

**By Module:**
- Filter to specific component
- Example: Only show backtesting logs

**By Symbol:**
- See only BTC-related events
- Track specific asset behavior

**By Environment:**
- Backtest logs
- Paper trading logs
- Separate test from production data

### Understanding Common Log Messages

**INFO Level (Normal Operations):**

```
[INFO] Analysis completed for BTC-4h
Metadata: {symbol: "BTC", timeframe: "4h", signal: "LONG", confidence: 0.75}
```
**Meaning:** AI completed analysis, generated signal

```
[INFO] Order executed: LONG BTC @ $51,234
Metadata: {order_id: 123, quantity: 0.2, environment: "paper"}
```
**Meaning:** Trade successfully placed

```
[INFO] Position closed: BTC, P&L: +$1,234.56
Metadata: {position_id: 45, entry_price: 50000, exit_price: 51234}
```
**Meaning:** Trade exited with profit

**WARNING Level (Potential Issues):**

```
[WARNING] API rate limit approaching (80% used)
Metadata: {provider: "openai", remaining_calls: 200}
```
**Meaning:** Slow down requests or risk being throttled
**Action:** Space out analyses or upgrade API plan

```
[WARNING] High slippage detected: 3.2%
Metadata: {symbol: "CL", expected: 0.01, actual: 0.032}
```
**Meaning:** Trade executed worse than expected
**Action:** Check market conditions (volatile?) or adjust slippage assumptions

```
[WARNING] No data available for date range
Metadata: {symbol: "XYZ", start: "2026-01-01", end: "2026-01-31"}
```
**Meaning:** yfinance couldn't fetch data
**Action:** Check symbol spelling or try different date range

**ERROR Level (Problems):**

```
[ERROR] Database connection lost
```
**Meaning:** Can't save/read from PostgreSQL
**Action:** Check Docker (`docker-compose ps`), restart database

```
[ERROR] Invalid API key for provider: openai
```
**Meaning:** Authentication failed
**Action:** Verify key in `.env` file, check expiration

```
[ERROR] RiskManager rejected order: Daily loss limit exceeded
Metadata: {daily_loss: -0.06, limit: -0.05}
```
**Meaning:** Circuit breaker triggered (hit 5% daily loss)
**Action:** Expected behavior, system protecting capital

---

## Order & Position Monitoring

**Location:** Dashboard → Orders & Positions tab

### Orders Table

**Columns:**

**Order ID:** Unique identifier

**Symbol:** Asset traded

**Side:** LONG (buy) or SHORT (sell)

**Type:** Market, Limit (MVP only supports Market)

**Quantity:** Amount traded

**Price:** Entry price (with slippage applied)

**Status:**
- **Filled** - Executed successfully
- **Pending** - Awaiting execution (paper trading doesn't have this)
- **Cancelled** - Order was rejected

**Timestamp:** When order was placed

**Environment:** backtest or paper

<!-- screenshot: Orders table with multiple entries -->

### What to Monitor

**Rejected Orders:**
- Filter by status = Cancelled
- Check logs for rejection reason
- Common causes: Risk limit hit, insufficient capital, duplicate order

**Slippage Patterns:**
- Compare order price to signal price
- Expected: ~1% difference
- If consistently >2%, adjust slippage assumptions in config

**Order Frequency:**
- Overtrading = Many orders, low win rate
- Undertrading = Few orders, missing opportunities
- Balance depends on strategy and timeframe

### Positions Table

**Columns:**

**Position ID:** Unique identifier

**Symbol:** Asset held

**Side:** LONG or SHORT

**Quantity:** Amount held

**Entry Price:** Average cost basis

**Current Price:** Live market price

**Unrealized P&L:** Profit/loss if closed now

**Duration:** How long position has been open

**Status:** Open or Closed

<!-- screenshot: Positions table showing open positions with P&L -->

### What to Monitor

**Open Positions:**
- Should align with current strategy
- Check if exit conditions met but system didn't close (bug?)

**Duration:**
- Long-held losers = Poor exit strategy or stuck trade
- Quick flips = Good timing or overtrading?

**Unrealized P&L:**
- Large unrealized gains = Consider taking profit
- Large unrealized losses = Review what went wrong

**Position Count:**
- Too many open = Risk concentration
- None open for days = Strategy not finding setups

---

## Performance Tracking (Long-Term)

### Weekly Review Checklist

**Every week, review:**

1. **Overall metrics**
   - Win rate trend (improving or declining?)
   - P&L cumulative (growing steadily?)
   - Max drawdown (staying under 15%?)

2. **Trade quality**
   - High-confidence signals performing better?
   - Specific assets outperforming?
   - Losing patterns to avoid?

3. **System health**
   - Any recurring errors?
   - API usage reasonable?
   - Database size growing too fast?

4. **Strategy adjustments**
   - Do risk limits need tightening?
   - Should position sizes change?
   - Add/remove assets from universe?

### Monthly Deep Dive

**Once a month:**

1. **Export backtest results** to CSV
2. **Calculate monthly metrics:**
   - Total P&L
   - Win rate
   - Sharpe ratio
   - Max drawdown

3. **Compare to benchmarks:**
   - How did BTC strategy compare to buying and holding BTC?
   - SPX strategy vs S&P 500 index return?

4. **Review strategy profiles:**
   - Are any profiles underperforming? Delete or refine.
   - Create new profiles based on learnings.

5. **Model performance:**
   - If testing multiple models, which performs best?
   - Cost vs performance trade-off (cheap model good enough?)

---

## Alert Setup (Manual for MVP)

**MVP doesn't have automatic alerts**, but you can set up manual checks:

### Daily Morning Check

**Routine:**
1. Open Dashboard
2. Check overnight performance (if paper trading)
3. Scan logs for errors
4. Verify database connection

**Time:** 5 minutes

### End-of-Day Review

**Routine:**
1. Review today's trades (Orders tab)
2. Check P&L
3. Look for any warnings in logs

**Time:** 10 minutes

### Weekly Detailed Review

**Routine:**
1. Calculate win rate for the week
2. Identify best/worst trades
3. Review AI agent decisions (Analyses tab)
4. Adjust strategies if needed

**Time:** 30 minutes

### Setting Calendar Reminders

**Use your calendar app:**
- Daily reminder at 9 AM: "Check QuantAgent Dashboard"
- Weekly reminder Friday 5 PM: "Review QuantAgent Performance"
- Monthly reminder 1st of month: "QuantAgent Monthly Analysis"

---

## Database Monitoring

### Checking Database Size

**Why:** Logs and checkpoints can grow large over time

**How to check:**

```bash
docker-compose exec db psql -U postgres -d quantagent_dev -c "SELECT pg_size_pretty(pg_database_size('quantagent_dev'));"
```

**Example output:** `245 MB`

**When to worry:** Database >5GB on MVP (indicates possible issue)

### Table Sizes

**Check individual table sizes:**

```sql
SELECT 
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

**Typically largest tables:**
- `logs` - Event tracking
- `market_data` - Cached OHLC data
- `checkpoints` - LangGraph state snapshots

### Cleaning Old Data

**If database too large, archive old logs:**

```sql
-- Delete logs older than 90 days
DELETE FROM logs WHERE timestamp < NOW() - INTERVAL '90 days';
```

**Backup first:** `docker-compose exec db pg_dump -U postgres quantagent_dev > backup.sql`

---

## Troubleshooting Common Issues

### Dashboard Not Loading

**Symptoms:**
- Blank page
- "Connection refused" error

**Check:**
1. Is Streamlit running? (`ps aux | grep streamlit`)
2. Is database running? (`docker-compose ps`)
3. Correct port? (Default: http://localhost:8501)

**Fix:**
```bash
# Restart Streamlit
streamlit run apps/streamlit/app.py

# Restart database
docker-compose restart db
```

### Missing Data in Dashboard

**Symptoms:**
- Tables show "No data"
- KPIs show zeros

**Check:**
1. Database connection indicator (should be green)
2. Environment selector (backtest vs paper)
3. Date range filters (too narrow?)

**Fix:**
1. Refresh page (F5)
2. Clear Streamlit cache (Settings → Clear cache)
3. Run a backtest to generate data

### Logs Growing Too Fast

**Symptoms:**
- Database size growing rapidly
- Slow queries

**Cause:**
- DEBUG logging enabled
- Very frequent analyses

**Fix:**
1. Change log level to INFO in `.env`:
   ```env
   LOG_LEVEL=INFO
   ```
2. Reduce analysis frequency
3. Archive old logs (see Database Monitoring section)

### High API Costs

**Symptoms:**
- Unexpected large bills from OpenAI/Anthropic

**Check:**
1. How many backtests run?
2. Which model used? (gpt-4o is 30x more expensive than gpt-4o-mini)
3. Logs for excessive retries

**Fix:**
1. Switch to cheaper model (gpt-4o-mini, claude-haiku)
2. Use Replay feature (reuses analyses, no new API calls)
3. Reduce backtest frequency
4. Set API usage alerts in provider dashboard

---

## System Health Metrics

### Good Health Indicators

**Database:**
- ✅ Connection stable (no disconnections in logs)
- ✅ Query response times <100ms
- ✅ Size growing predictably (~10MB/week)

**API Usage:**
- ✅ No rate limit warnings
- ✅ Costs within budget
- ✅ Successful responses >95%

**Execution:**
- ✅ Analyses completing in <30 seconds
- ✅ Orders executing without rejections (or expected rejections only)
- ✅ No ERROR logs

**Strategy Performance:**
- ✅ Win rate stable or improving
- ✅ Drawdown within limits
- ✅ P&L trending upward

### Warning Signs

**Database:**
- ⚠️ Frequent connection errors
- ⚠️ Slow queries (>5 seconds)
- ⚠️ Size growing >100MB/week

**API Usage:**
- ⚠️ Rate limit warnings
- ⚠️ Costs increasing unexpectedly
- ⚠️ Many failed requests

**Execution:**
- ⚠️ Analyses timing out
- ⚠️ Many rejected orders
- ⚠️ Frequent ERROR logs

**Strategy Performance:**
- ⚠️ Win rate declining over time
- ⚠️ Drawdown approaching limits
- ⚠️ P&L flat or declining

---

## Maintenance Schedule

### Daily (5 minutes)
- [ ] Check Dashboard KPIs
- [ ] Scan logs for errors
- [ ] Verify database connection

### Weekly (30 minutes)
- [ ] Calculate win rate
- [ ] Review top 3 best/worst trades
- [ ] Check system health metrics
- [ ] Update strategy notes

### Monthly (2 hours)
- [ ] Full performance analysis
- [ ] Benchmark against buy-and-hold
- [ ] Database maintenance (archive old logs)
- [ ] Strategy profile review and updates
- [ ] Model performance comparison (if using multiple)

### Quarterly (Half day)
- [ ] Deep strategy review
- [ ] Test new assets or timeframes
- [ ] Review and update risk limits
- [ ] Clean up unused profiles
- [ ] Database backup and optimization

---

## Related Documentation

- **Logging System**: [LOGGING_STRATEGY.md](../03_design/LOGGING_STRATEGY.md) - QuantAgent-yuk
- **Logging Implementation**: [QuantAgent-yuk-IM-structured-logging.md](../06_implementation/) - QuantAgent-yuk subtasks
- **Logging Acceptance Tests**: [QuantAgent-yuk-AC-structured-logging.md](../05_acceptance_tests/QuantAgent-yuk-AC-structured-logging.md) - QuantAgent-yuk
- **Dashboard Architecture**: [streamlit_app_architecture.md](../03_design/streamlit_app_architecture.md)
- **Database Schema**: [MIGRATIONS.md](../03_design/MIGRATIONS.md)
- **Position Monitoring**: [QuantAgent-nu7-DS-active-position-monitoring.md](../03_design/QuantAgent-nu7-DS-active-position-monitoring.md) - QuantAgent-nu7

---

*Regular monitoring catches issues early and keeps your strategy on track. Set up a routine and stick to it.*
