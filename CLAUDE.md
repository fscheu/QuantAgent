# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation

The documentation lives in docs/ folder.

When generate documentation always divide between analysis or functional documentation and technical specification. Mainly, do not mix both in the same file, unless when a technical detail helps in the functional explanation.

Fallback to update existing documents. Try to not generate new documents all the time. We should have a main document for each one of the types described next and detail documents for specific requirements or technical decissions/implementations. The main type of documents should be:
* Requirements: Functional and UI specifications.
* Planning or task management: details of which are the plans and current status
* Technical specification: Architecture, code, technical configurations.



# ULTRAIMPORTANT
Think carefully and only action the specific task I have given you with the most concise and elegant solution that changes as little code as possible.
Avoid over-engineering. Only make changes that are directly requested or clearly necessary. Keep solutions simple and focused.

Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability.

Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use backwards-compatibility shims when you can just change the code.

Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task. Reuse existing abstractions where possible and follow the DRY principle.

Do not write test with excessive mockups that do not test any solution code at all. 

## Project Overview

**QuantAgent** is a multi-agent trading analysis system that uses vision-capable LLMs (Claude, GPT-4, Qwen) to analyze financial markets in high-frequency trading (HFT) contexts. It combines technical indicators, candlestick pattern recognition, and trend analysis through a LangGraph-orchestrated agent pipeline.

**Reference Paper:** arXiv:2509.09995 - "Price-Driven Multi-Agent LLMs for High-Frequency Trading"

## Repository Structure

### Core Agent Pipeline
- `indicator_agent.py` - Computes 5 technical indicators (RSI, MACD, Stochastic, ROC, Williams %R)
- `pattern_agent.py` - Generates K-line charts and uses vision LLM to identify candlestick patterns
- `trend_agent.py` - Analyzes trendlines, support/resistance with trend-annotated charts
- `decision_agent.py` - Synthesizes all agent reports into LONG/SHORT trading decision

### Orchestration & State Management
- `trading_graph.py` - Main orchestrator that initializes LLMs, manages multi-provider support (OpenAI, Anthropic, Qwen), and invokes the graph
- `graph_setup.py` - LangGraph StateGraph definition and compilation (where agent nodes are wired together)
- `agent_state.py` - StateGraph TypedDict schema containing kline_data, indicator outputs, pattern/trend chart images, and final_trade_decision

### Tools & Utilities
- `graph_util.py` - `TechnicalTools` class wrapping all indicator computation and chart generation methods
- `static_util.py` - Data preparation utilities (pandas operations, OHLCV formatting)
- `color_style.py` - mplfinance chart styling configuration

### Web Interface & Configuration
- `web_interface.py` - Flask app (~1047 lines) with real-time yfinance data fetching, chart generation, and runtime API key management
- `default_config.py` - LLM provider selection, temperature (0.1 default), model names, API key placeholders
- `requirements.txt` - 19 direct dependencies (LangChain, LangGraph, yfinance, TA-Lib, mplfinance, Flask, etc.)

### Data & Documentation
- `benchmark/` - Historical OHLCV data (1h and 4h timeframes) for BTC, CL, DAX, DJI, ES, NQ, QQQ, SPX; useful for testing
- `templates/` - HTML templates for Flask web interface (demo.html, output.html)
- `docs/` - Academic paper PDF

## Architecture Highlights

### Multi-Agent Flow
```
START → Indicator Agent → Pattern Agent → Trend Agent → Decision Agent → END
                ↓              ↓              ↓
        Technical metrics    K-line chart   Trend chart
        & LLM analysis      + vision LLM    + optimization
```

### Key Design Patterns
1. **LangGraph StateGraph** - All agents are nodes that read/write to shared typed state (`IndicatorAgentState`)
2. **Tool Calling** - Indicator Agent uses LangChain `@tool` decorated functions for deterministic computation
3. **Vision Integration** - Pattern and Trend agents send base64-encoded PNG charts to vision-capable LLMs
4. **Provider Abstraction** - Single `_create_llm()` method in `trading_graph.py` handles OpenAI, Anthropic, Qwen
5. **Retry Logic** - Trend agent includes exponential backoff for rate-limited API calls

### State Schema (agent_state.py)
- **Inputs:** `kline_data` (OHLCV dict), `time_frame`, `stock_name`
- **Outputs:** indicator values, `pattern_chart_img` (base64), `trend_chart_img` (base64), `final_trade_decision` (string)
- **Processing:** `messages` (LangChain message history)

## Technology Stack

- **LangChain** (v0.1+) - LLM interactions, prompting, tool bindings
- **LangGraph** - Multi-agent state machine orchestration
- **yfinance** - Real-time market data (stocks, crypto, commodities, indices)
- **TA-Lib** - Technical indicator computation; install via `conda install -c conda-forge ta-lib`
- **mplfinance** - Candlestick chart generation with custom styling
- **Flask** - Web server with Jinja2 templating
- **vision-capable LLMs** - GPT-4o, Claude Haiku 4.5, Qwen3-VL (required for pattern/trend agents)

## Development Setup

### Installation
```bash
conda create -n quantagents python=3.11
conda activate quantagents
pip install -r requirements.txt
conda install -c conda-forge ta-lib
export OPENAI_API_KEY="your_key"  # Or set via web interface
```

### Running
```bash
python web_interface.py
# Access at http://127.0.0.1:5000
```

### Programmatic Usage
```python
from trading_graph import TradingGraph

tg = TradingGraph()
initial_state = {"kline_data": df_dict, "time_frame": "4hour", "stock_name": "BTC", "analysis_results": None, "messages": []}
result = tg.graph.invoke(initial_state)
print(result["final_trade_decision"])
```

## Configuration & Customization

### Model Selection (default_config.py)
- `agent_llm_provider` / `graph_llm_provider` - Choose "openai", "anthropic", or "qwen" per agent
- `temperature` - Set to 0.1 for deterministic outputs (do not increase without justification)
- API keys read from config dict, then fall back to environment variables

### Web Interface
- 12+ assets (stocks, crypto, commodities, indices) via yfinance symbols
- Timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo
- Fetches latest 30 candlesticks for optimal analysis window
- Runtime key updates without restart

## Important Notes

### Vision LLM Requirement
Pattern and Trend agents generate PNG charts and send them as base64-encoded images to the LLM for visual analysis. **Must use vision-capable models** (GPT-4o, Claude Haiku 4.5, Qwen3-VL).

### Temperature Setting
All agents use temperature=0.1 for professional, deterministic outputs. Higher values risk speculative, unreliable trading decisions.

### Data Dependencies
- Minimum ~30 candlesticks for meaningful analysis
- Real-time yfinance data; some symbols may have limited historical availability
- Benchmark data (100 CSV files per asset) can be used for testing without API calls

### Error Handling
- Trend agent has 3-retry logic with 4-second exponential backoff for rate limiting
- Web interface gracefully handles missing data or API failures

## Common Development Tasks

### Add a New Technical Indicator
1. Implement computation in `graph_util.py` as `@tool` decorated function
2. Update `indicator_agent.py` prompt and tool binding
3. Add output field to `IndicatorAgentState` in `agent_state.py`
4. Update decision agent prompt to reference new indicator

### Add a New LLM Provider
1. Extend `_create_llm()` in `trading_graph.py` with new provider case
2. Add API key handling and model name to `default_config.py`
3. Update web interface key input if needed

### Test with Benchmark Data
Use CSV files in `benchmark/` directory to test graph without rate-limited API calls:
```python
import pandas as pd
df = pd.read_csv("benchmark/btc/BTC_4h_1.csv")
# Convert to OHLCV dict format expected by kline_data
```

### Modify Chart Styling
Chart colors, fonts, and volume styling are in `color_style.py`. Pass style dict to mplfinance.plot() in `graph_util.py`.

## Caveats

- **Research/Educational Only** - Not financial advice; for academic/research use
- **HFT Context** - System optimized for 1-4 candlestick predictions, not long-term investing
- **API Costs** - OpenAI/Anthropic usage incurs charges; yfinance is free but may have rate limits
- **Internet Dependent** - Requires real-time yfinance connectivity
- **No Token Limits** - Agents use full context; monitor API cost/latency

## Quick Reference

```bash
# View state schema
grep -A 20 "class IndicatorAgentState" agent_state.py

# Check supported assets
grep "assets\|symbols" web_interface.py

# Run single indicator test
python -c "from graph_util import TechnicalTools; t = TechnicalTools(); print(t.compute_rsi(...))"

# Git operations
git status
git diff HEAD~1  # View recent changes
```
