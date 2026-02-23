# Getting Started with QuantAgent

This guide walks you through installing QuantAgent and running your first backtest.

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.11 or higher** installed
- **Docker Desktop** (for PostgreSQL database)
- **API key** from OpenAI or Anthropic (for AI agents)
- **10GB free disk space** (for historical market data cache)
- **Internet connection** (to fetch market data)

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/fscheu/QuantAgent.git
cd QuantAgent
```

### 2. Create Python Environment

Using Conda (recommended):
```bash
conda create -n quantagent python=3.11
conda activate quantagent
```

Or using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- LangGraph - Multi-agent orchestration
- SQLAlchemy - Database management
- Streamlit - Web dashboard
- yfinance - Market data provider
- pandas, numpy - Data analysis

### 4. Start the Database

QuantAgent uses PostgreSQL to store analysis results, trades, and performance metrics.

```bash
docker-compose up -d db
```

**Verify it's running:**
```bash
docker-compose ps
```

You should see `quantagent-db` with status "Up".

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

**For OpenAI:**
```env
OPENAI_API_KEY=sk-your-key-here
AGENT_LLM_PROVIDER=openai
AGENT_LLM_MODEL=gpt-4o-mini
```

**For Anthropic:**
```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
AGENT_LLM_PROVIDER=anthropic
AGENT_LLM_MODEL=claude-haiku-4-5-20251001
```

**Database connection (default, usually no changes needed):**
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/quantagent_dev
```

<!-- screenshot: .env file configuration example -->

### 6. Initialize the Database

Run database migrations to create required tables:

```bash
python -m alembic upgrade head
```

**Expected output:**
```
INFO  [alembic.runtime.migration] Running upgrade -> abc123, create logs table
INFO  [alembic.runtime.migration] Running upgrade abc123 -> def456, add environment field
```

---

## Verify Installation

### Quick Test: Run Example Backtest

Run the included example script:

```bash
python examples/run_backtest.py
```

**What happens:**
1. Fetches 10 days of Bitcoin and S&P 500 data
2. Runs AI analysis on 4-hour candles
3. Simulates trades based on agent decisions
4. Displays performance metrics

**Expected output:**
```
Running backtest from 2026-02-13 to 2026-02-23
Assets: ['BTC', 'SPX']
Timeframe: 4h
Initial capital: $100,000.00
------------------------------------------------------------

BACKTEST RESULTS
============================================================
Total Trades:      8
Winning Trades:    5
Losing Trades:     3
Win Rate:          62.50%
Profit Factor:     1.85
Sharpe Ratio:      1.23
Max Drawdown:      8.45%
Total P&L:         $4,250.00
Total Return:      4.25%
============================================================
```

**If you see errors:**
- `Connection refused` → Database not running. Check `docker-compose ps`
- `Invalid API key` → Check your `.env` file has correct key
- `No module named 'quantagent'` → Activate your Python environment

---

## Launch the Dashboard

Start the web interface:

```bash
streamlit run apps/streamlit/app.py
```

**Access the dashboard:**
Open your browser to http://localhost:8501

You should see 7 tabs:
- **Dashboard** - Overview and metrics
- **Configuration** - Strategy settings
- **Analyses** - View AI decisions
- **Backtesting** - Run historical tests
- **Replay** - Re-test with different settings
- **Orders & Positions** - Track trades
- **Logs** - System events

<!-- screenshot: Streamlit dashboard home page showing all tabs -->

---

## Your First Backtest (Dashboard Method)

Now that everything is installed, let's run a backtest through the web interface.

### Step 1: Create a Strategy Profile

1. Click the **Configuration** tab
2. Select profile kind: **combined**
3. Enter profile name: **my-first-strategy**
4. Select assets: **BTC** and **SPX**
5. Keep default settings (5% position size, 10% max position)
6. Click **Save profile**

<!-- screenshot: Configuration tab with filled profile form -->

### Step 2: Run the Backtest

1. Click the **Backtesting** tab
2. Select your profile: **my-first-strategy**
3. Choose timeframe: **4h** (4-hour candles)
4. Select date range: **Last 30 days**
5. Enter backtest name: **First Test**
6. Click **Create & Run Backtest**

**Wait time:** 2-5 minutes for first run (downloads market data)

<!-- screenshot: Backtesting tab with run in progress -->

### Step 3: View Results

Once complete, you'll see:
- **Metrics table** - Win rate, profit factor, Sharpe ratio
- **Equity curve chart** - Portfolio value over time
- **Trade list** - All executed trades with P&L

**What to look for:**
- Win rate above 40% is good
- Sharpe ratio above 1.0 indicates positive risk-adjusted returns
- Max drawdown below 15% means controlled risk

<!-- screenshot: Backtest results showing metrics and equity curve -->

---

## Next Steps

You're now ready to:

1. **[Configure strategies](strategy-configuration.md)** - Adjust position sizing and risk limits
2. **[Understand analysis](analysis-and-signals.md)** - Learn how AI agents make decisions
3. **[Run more backtests](backtesting.md)** - Test different timeframes and assets
4. **[Monitor performance](monitoring.md)** - Track system logs and metrics

---

## Troubleshooting Common Issues

### Database Connection Errors

**Problem:** `could not connect to server`

**Solution:**
```bash
# Check if database is running
docker-compose ps

# Restart if needed
docker-compose restart db

# Check logs
docker-compose logs db
```

### API Key Errors

**Problem:** `Invalid API key` or `Authentication failed`

**Solution:**
1. Verify your key in `.env` file
2. Check for extra spaces or line breaks
3. Ensure key is active (check provider's dashboard)
4. Restart the app after changing `.env`

### Market Data Fetch Errors

**Problem:** `Symbol 'BTC' not found` or `No data available`

**Solution:**
- Check internet connection
- Try a different symbol (e.g., 'SPX' instead of 'BTC')
- yfinance may have temporary outages - wait and retry

### "Out of Memory" Errors

**Problem:** System freezes or crashes during backtests

**Solution:**
- Reduce date range (test shorter periods)
- Use fewer assets (start with 1-2 symbols)
- Increase Docker memory limit in Docker Desktop settings

---

## Related Documentation

- **Technical Setup**: [Docker Deployment Guide](../03_design/docker_deployment.md) - QuantAgent-729
- **Database Schema**: [Migrations Guide](../03_design/MIGRATIONS.md)
- **Manual Test Cases**: [MVP Test Cases](../05_acceptance_tests/MVP_MANUAL_TEST_CASES.md)
- **Configuration Options**: [CONFIGURATION.md](../03_design/CONFIGURATION.md) - QuantAgent-cxu

---

*Having trouble? Check the [GitHub Issues](.beads/) or consult the [technical docs](../03_design/).*
