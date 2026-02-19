<div align="center">

# 🤖 QuantAgent

**From Research Paper to Trading System**

*A multi-agent LLM system that analyzes markets, executes paper trades, and validates strategies through backtesting.*

[![Based on Paper](https://img.shields.io/badge/📄_Based_on-arXiv:2509.09995-B31B1B?style=flat-square)](https://arxiv.org/abs/2509.09995)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Built_with-LangGraph-1C3C3C?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## What is this?

Four AI agents analyze the market. One makes the call.

```
📊 Indicator Agent    →  RSI, MACD, Stochastic, momentum signals
🔍 Pattern Agent      →  Chart patterns via vision LLM
📈 Trend Agent        →  Support/resistance, channel analysis
🎯 Decision Agent     →  Synthesizes all → LONG / SHORT / HOLD
```

This fork extends the [original QuantAgent research](https://arxiv.org/abs/2509.09995) into a **functional trading system** with:

- **Backtesting engine** — validate strategies on historical data
- **Paper trading** — simulate execution without real money
- **Risk management** — position sizing, daily limits, circuit breakers
- **Full audit trail** — every decision tracked and reproducible

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/fscheu/QuantAgent.git
cd QuantAgent
conda create -n quantagent python=3.11 && conda activate quantagent
pip install -r requirements.txt

# 2. Start database
docker-compose up -d db

# 3. Configure API key
export OPENAI_API_KEY="sk-..."  # or ANTHROPIC_API_KEY

# 4. Run a backtest
python examples/run_backtest.py
```

**First time?** See the [detailed setup guide](docs/03_design/docker_deployment.md).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         QuantAgent                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Data Layer              Analysis Engine         Execution      │
│   ──────────              ───────────────         ─────────      │
│   • yfinance              • Indicator Agent       • Paper Broker │
│   • Local DB cache        • Pattern Agent         • Order Manager│
│   • 18x faster            • Trend Agent           • Risk Manager │
│     backtests             • Decision Agent        • Portfolio    │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Backtesting             Persistence             UI             │
│   ───────────             ───────────             ──             │
│   • Historical runs       • PostgreSQL            • Streamlit    │
│   • Metrics calc          • LangGraph checkpoints • 7 tabs       │
│   • Config snapshots      • Full provenance       • Real-time    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features at a Glance

| Feature | Status | Description |
|---------|--------|-------------|
| **Multi-Agent Analysis** | ✅ | 4 specialized agents with structured outputs |
| **Backtesting** | ✅ | Run strategies on historical data, get metrics |
| **Paper Trading** | ✅ | Simulated execution with realistic slippage |
| **Risk Management** | ✅ | Position limits, daily loss caps, circuit breaker |
| **Data Caching** | ✅ | Local DB cache makes backtests 18x faster |
| **Checkpointing** | ✅ | Resume long runs, full execution history |
| **Streamlit UI** | ✅ | Dashboard, analyses, backtesting, logs |
| **Auto Scheduler** | 🚧 | Coming soon — hourly automated analysis |
| **Real Broker** | 📋 | Phase 2 — after strategy validation |

---

## Project Status

> **Phase 1: ~75% complete** — Core trading components done, UI functional, scheduler pending.

See the [full audit report](docs/2026-02-19_repository_audit.md) for details.

**Active development tracked in:** [Beads issues](.beads/)

---

## Documentation

| Doc | What's inside |
|-----|---------------|
| [Trading Requirements](docs/01_requirements/trading_system_requirements.md) | Full MVP specification |
| [Phase 1 Roadmap](docs/02_planning/phase1_roadmap.md) | Week-by-week progress |
| [Backtesting Engine](docs/03_design/backtesting_engine.md) | How backtests work |
| [Docker Setup](docs/03_design/docker_deployment.md) | Dev environment guide |
| [Manual Test Cases](docs/05_acceptance_tests/MVP_MANUAL_TEST_CASES.md) | Validate the system |

---

## Usage Examples

### Run a Backtest

```python
from quantagent.backtesting import Backtest
from datetime import datetime, timedelta

backtest = Backtest(
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now(),
    assets=['BTC', 'SPX'],
    timeframe='4h',
    initial_capital=100000.0
)

metrics = backtest.run(name="Q4 Strategy Test")

print(f"Win Rate: {metrics.win_rate:.1%}")
print(f"Sharpe:   {metrics.sharpe_ratio:.2f}")
print(f"P&L:      ${metrics.total_pnl:,.2f}")
```

### Analyze a Single Asset

```python
from quantagent.graph_setup import TradingGraph

graph = TradingGraph()
result = graph.analyze(symbol="BTC", timeframe="4h")

print(result["final_trade_decision"])
# → "LONG with 78% confidence. RSI oversold, bullish engulfing pattern..."
```

### Launch the Dashboard

```bash
streamlit run apps/streamlit/app.py
```

Open http://localhost:8501 → Dashboard, Analyses, Backtesting, Logs.

---

## Configuration

```python
# Example config for backtesting
config = {
    'base_position_pct': 0.05,      # 5% of portfolio per trade
    'max_position_pct': 0.10,       # Max 10% in single position
    'max_daily_loss_pct': 0.05,     # Circuit breaker at 5% daily loss
    'slippage_pct': 0.01,           # 1% simulated slippage
    'agent_llm_provider': 'openai',
    'agent_llm_model': 'gpt-4o-mini',
}
```

See [CONFIGURATION.md](docs/03_design/CONFIGURATION.md) for all options.

---

## Based On

This project builds upon the research paper:

> **QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading**
> Xiong, Zhang, Feng, Sun, You — [arXiv:2509.09995](https://arxiv.org/abs/2509.09995)

The original implementation focused on analysis. This fork adds execution, backtesting, and production infrastructure.

---

## Tech Stack

- **LangGraph** — Multi-agent orchestration with checkpointing
- **PostgreSQL** — Persistence, caching, state management
- **Streamlit** — Dashboard UI
- **yfinance** — Market data
- **SQLAlchemy + Alembic** — ORM and migrations

---

## Contributing

1. Check [open issues](.beads/) for current priorities
2. Fork → Branch → PR
3. Follow existing code patterns
4. Tests required for new features

See [AGENTS.md](AGENTS.md) for development guidelines.

---

## License

MIT — See [LICENSE](LICENSE)

---

<div align="center">

**Questions?** Open an issue or check the [docs](docs/).

*Built with ☕ and AI agents that occasionally disagree with each other.*

</div>
