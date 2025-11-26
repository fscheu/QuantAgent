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

### quantagent/ (Core Package)

#### Agent Pipeline
- `indicator_agent.py` - Computes 5 technical indicators (RSI, MACD, Stochastic, ROC, Williams %R)
- `pattern_agent.py` - Generates K-line charts and uses vision LLM to identify candlestick patterns
- `trend_agent.py` - Analyzes trendlines, support/resistance with trend-annotated charts
- `decision_agent.py` - Synthesizes all agent reports into LONG/SHORT trading decision

#### Orchestration & State Management
- `trading_graph.py` - Main orchestrator that initializes LLMs, manages multi-provider support (OpenAI, Anthropic, Qwen), and invokes the graph
- `graph_setup.py` - LangGraph StateGraph definition and compilation
- `agent_state.py` - StateGraph TypedDict schema for multi-agent state

#### Database Layer (NEW)
- `database.py` - SQLAlchemy engine, session management, Base declarative class
  - Supports PostgreSQL (primary), SQLite (dev), MySQL (alternative)
  - Reads `DATABASE_URL` from environment (`.env`) for secure credential management
- `models.py` - 6 SQLAlchemy ORM models:
  - `Order` - Trading orders with status tracking
  - `Fill` - Partial/complete order executions
  - `Position` - Open positions with P&L tracking
  - `Signal` - Trading signals with technical indicators and confidence
  - `Trade` - Completed trades with entry/exit analysis
  - `MarketData` - Historical OHLCV candle data

#### Tools & Utilities
- `graph_util.py` - `TechnicalTools` class for indicator computation and chart generation
- `static_util.py` - Data preparation utilities (pandas, OHLCV formatting)
- `color_style.py` - mplfinance chart styling configuration
- `migrations_helper.py` - CLI helper for Alembic migrations (upgrade, create, downgrade, current, history)

#### Web Interface & Configuration
- `web_interface.py` - Flask app with real-time yfinance data, chart generation, API key management
- `default_config.py` - LLM provider selection, temperature, model names, API key placeholders

### alembic/ (Database Migrations)
- `env.py` - Alembic environment config; reads `DATABASE_URL` from environment
- `script.py.mako` - Migration file template
- `versions/` - Migration scripts (auto-generated with `alembic revision --autogenerate`)
  - Example: `830b128fd5df_create_initial_schema.py` - Creates all 6 tables with indexes

### docs/ (Project Documentation)
- `MIGRATIONS.md` - **Complete guide for database setup & migrations**
  - Docker setup (recommended for Windows)
  - PostgreSQL configuration options
  - Migration workflow and common tasks
  - Best practices and troubleshooting
- Original architecture & trading documentation

### scripts/ (Utility Scripts)
- `setup_postgres.py` - Interactive PostgreSQL configuration
  - Auto-detects Docker via `docker-compose.yml`
  - Prompts for host, port, user, password, database
  - Generates `.env` with `DATABASE_URL`
  - Provides next steps based on configuration choice

### tests/ (Test Suite)
- `test_migrations.py` - Comprehensive migration test
  - Creates all tables via SQLAlchemy ORM
  - Inserts sample data into all 6 models
  - Tests relationships and constraints
  - Validates data retrieval
  - Cleans up after execution
- Other test files for agents and utilities

### Configuration Files (Root)
- `requirements.txt` - 21 dependencies (added: SQLAlchemy, Alembic, psycopg2-binary)
- `alembic.ini` - Alembic configuration (no hardcoded credentials)
- `.env.example` - Template for environment variables (`.env` is gitignored)
- `docker-compose.yml` - PostgreSQL 15 service with pre-configured database `quantagent_dev`
- `pyproject.toml` - Package metadata and build configuration
- `pytest.ini` - Pytest configuration

### Data & Web Interface
- `benchmark/` - Historical OHLCV data (1h, 4h timeframes) for BTC, CL, DAX, DJI, ES, NQ, QQQ, SPX
- `templates/` - Flask HTML templates (demo.html, output.html)
- `static/` - CSS, JavaScript, images for web interface

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

### AI & Analytics
- **LangChain** (v0.1+) - LLM interactions, prompting, tool bindings
- **LangGraph** - Multi-agent state machine orchestration
- **vision-capable LLMs** - GPT-4o, Claude Haiku 4.5, Qwen3-VL (required for pattern/trend agents)
- **TA-Lib** - Technical indicator computation; install via `conda install -c conda-forge ta-lib`

### Market Data & Visualization
- **yfinance** - Real-time market data (stocks, crypto, commodities, indices)
- **mplfinance** - Candlestick chart generation with custom styling
- **pandas** - Data manipulation and OHLCV formatting

### Web & API
- **Flask** - Web server with Jinja2 templating
- **OpenAI, Anthropic, Qwen SDKs** - LLM provider APIs

### Database & Persistence (NEW)
- **SQLAlchemy** (>=2.0.0) - ORM for database abstraction and model definitions
- **Alembic** (>=1.12.0) - Database migration management and schema versioning
- **psycopg2-binary** (>=2.9.0) - PostgreSQL adapter
- **PostgreSQL** 15 - Primary database (via docker-compose)
  - Supports SQLite for development, MySQL as alternative
  - Auto-creates `quantagent_dev` database with Docker

### Testing & Development
- **pytest** - Test framework with fixtures and plugin support
- **Docker & docker-compose** - Containerized PostgreSQL for easy setup

## Development Setup

### Installation
```bash
conda create -n quantagents python=3.11
conda activate quantagents
pip install -r requirements.txt
conda install -c conda-forge ta-lib
```

### Database Setup (PostgreSQL via Docker)
```bash
# 1. Start PostgreSQL container (creates quantagent_dev database automatically)
docker-compose up -d

# 2. Configure database connection
python setup_postgres.py
# Select: yes (use docker-compose PostgreSQL)
# This creates .env with DATABASE_URL

# 3. Set environment variable in PowerShell
$env:DATABASE_URL = "postgresql://postgres:password@localhost:5432/quantagent_dev"

# 4. Run migrations
python -m alembic upgrade head

# 5. Test migrations
python tests/test_migrations.py
```

**Note:** See `docs/MIGRATIONS.md` for detailed setup options (Docker, manual PostgreSQL, environment variables).

### API Keys Configuration
```bash
# Set environment variables or create .env file
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
# Or configure via web interface at runtime
```

### Running the Web Interface
```bash
python web_interface.py
# Access at http://127.0.0.1:5000
```

### Programmatic Usage
```python
from trading_graph import TradingGraph
from quantagent.database import SessionLocal
from quantagent.models import Signal, Trade

# Run analysis
tg = TradingGraph()
initial_state = {
    "kline_data": df_dict,
    "time_frame": "4hour",
    "stock_name": "BTC",
    "analysis_results": None,
    "messages": []
}
result = tg.graph.invoke(initial_state)
print(result["final_trade_decision"])

# Persist to database
db = SessionLocal()
trade = Trade(
    symbol="BTC",
    entry_price=45000.00,
    quantity=0.5,
    side="BUY"
)
db.add(trade)
db.commit()
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

## Database Management

### Core Concepts
- **Models** (quantagent/models.py) - SQLAlchemy ORM classes define schema
- **Migrations** (alembic/) - Version-controlled database schema changes
- **Sessions** (quantagent/database.py) - Database connections and transactions
- **Environment** (.env) - Credentials stored securely (never hardcoded, never committed)

### Key Files
- `quantagent/models.py` - Define/modify data models here
- `quantagent/database.py` - Database configuration and session factory
- `alembic/env.py` - Migration environment (auto-detects model changes)
- `alembic.ini` - Migration configuration (no credentials here)
- `.env` - Contains `DATABASE_URL` (gitignored)
- `docs/MIGRATIONS.md` - Complete migration documentation

### Common Database Tasks

#### Create a New Model
1. Define class in `quantagent/models.py` inheriting from `Base`
2. Add fields as SQLAlchemy Columns with appropriate types
3. Add relationships if needed (e.g., Order → Fill via foreign key)
4. Create migration: `python -m alembic revision --autogenerate -m "Add new model"`
5. Run migration: `python -m alembic upgrade head`

#### Add a Column to Existing Model
1. Modify model in `quantagent/models.py`
2. Create migration: `python -m alembic revision --autogenerate -m "Add column to table"`
3. Review generated migration in `alembic/versions/`
4. Run migration: `python -m alembic upgrade head`

#### Verify Database State
```bash
# Check current migration version
python -m alembic current

# View all migrations
python -m alembic history

# Check table structure (Docker)
docker-compose exec db psql -U postgres -d quantagent_dev -c "\dt"

# View specific table schema
docker-compose exec db psql -U postgres -d quantagent_dev -c "\d orders"
```

#### Troubleshoot Migration Issues
```bash
# If target database is not up to date
python -m alembic upgrade head

# Downgrade last migration
python -m alembic downgrade -1

# Downgrade to specific revision
python -m alembic downgrade <revision_id>

# View what will be executed
python -m alembic upgrade head --sql
```

### Query Examples
```python
from quantagent.database import SessionLocal
from quantagent.models import Signal, Trade, Order

db = SessionLocal()

# Get signals for BTC
signals = db.query(Signal).filter(Signal.symbol == "BTC").all()

# Get trades within date range
from datetime import datetime, timedelta
trades = db.query(Trade).filter(
    Trade.opened_at >= datetime.now() - timedelta(days=7)
).all()

# Get open orders
from quantagent.models import OrderStatus
open_orders = db.query(Order).filter(
    Order.status == OrderStatus.PENDING
).all()

# Get completed trades with P&L
profitable = db.query(Trade).filter(Trade.pnl > 0).all()

db.close()
```

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

## Project Documentation & Files

### Documentation Location (docs/)
- **MIGRATIONS.md** - Complete database setup, migration workflow, troubleshooting guide
  - Docker setup instructions (recommended for Windows)
  - PostgreSQL configuration options
  - Best practices and index strategy
  - Migration examples and common tasks

### Documentation Standards
When adding/modifying documentation:
1. **Functional** documentation goes in `docs/01_requirements/` (requirements, guides, how-to)
2. **Technical** documentation goes in `docs/03_technical/` (architecture, specs, API details)
3. **Planning** documentation goes in `docs/02_planning/` (Phases, task planning, little TODOs)
4. **Code** comments: Only when logic isn't self-evident
5. Prefer updating existing docs over creating new ones
6. Link between related documents. **DO NOT REPEAT** the same writing in different docs.

### Testing & Quality Assurance (tests/)

**See `docs/03_technical/TESTING_PATTERNS.md` for detailed testing guidelines, patterns, and anti-patterns to follow when writing tests for this project.**

- **test_migrations.py** - Comprehensive database migration testing
  - Creates all 6 ORM tables
  - Inserts sample data into all models
  - Tests relationships and foreign key constraints
  - Validates data retrieval via ORM queries
  - Tests cleanup (drop all tables)
  - **Run with:** `python tests/test_migrations.py`
- Additional test files for agents, utilities, and integration testing
- Run all tests: `python -m pytest tests/ -v`

### Key Project Files for Management
- **docs/02_planning/** - Phase planning, milestones, task tracking, TODOs
- **docs/03_technical/MIGRATIONS.md** - Database setup and migration guide
- **docs/03_technical/TESTING_PATTERNS.md** - Testing standards, patterns, and best practices
- **tests/test_migrations.py** - Database test suite
- **CLAUDE.md** (this file) - Development guidelines and architecture


## Quick Reference

### Agent & Analysis
```bash
# View state schema
grep -A 20 "class IndicatorAgentState" quantagent/agent_state.py

# Check supported assets
grep "assets\|symbols" quantagent/web_interface.py

# Run single indicator test
python -c "from quantagent.graph_util import TechnicalTools; t = TechnicalTools(); print(t.compute_rsi(...))"
```

### Database Management
```bash
# Docker operations
docker-compose up -d       # Start PostgreSQL
docker-compose ps          # Check status
docker-compose down        # Stop PostgreSQL
docker-compose logs db     # View logs

# Database setup
python setup_postgres.py              # Configure connection
python -m alembic upgrade head        # Run migrations
python -m alembic current             # Check version
python -m alembic history             # View all migrations

# Testing
python tests/test_migrations.py       # Test database operations
python -m pytest tests/               # Run all tests
python -m pytest tests/test_migrations.py -v  # Verbose output

# Query database (Docker)
docker-compose exec db psql -U postgres -d quantagent_dev
# In psql:
\dt                 # List tables
\d orders           # Describe orders table
SELECT * FROM signals LIMIT 5;
```

### Python Operations
```bash
# Check database models
python -c "from quantagent.models import *; from quantagent.database import Base; print(Base.metadata.tables.keys())"

# Test database connection
python -c "from quantagent.database import SessionLocal; db = SessionLocal(); print('Connected!'); db.close()"

# Insert test data
python -c "from quantagent.database import SessionLocal; from quantagent.models import Signal; db = SessionLocal(); ..."
```

### Git Operations
```bash
# Project status
git status
git log --oneline -10    # Recent commits
git diff HEAD~1          # View recent changes

# Migration-related
git log --oneline -- alembic/versions/  # Migration history
git show <hash>          # View specific migration
```
