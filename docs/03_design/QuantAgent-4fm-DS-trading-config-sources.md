# QuantAgent-4fm — DS — Trading config sources & precedence

## Goal
Define a minimal, consistent way to source trading configuration (portfolio/risk/backtest runtime parameters) without scattering defaults across modules.

## Affected Components
- `quantagent/settings.py` (env loading + typed access)
- `quantagent/strategy/assembler.py` (defaults + merge logic)
- `quantagent/backtesting/backtest.py` (config ingestion)
- (indirectly) `quantagent/trading/*` components that currently embed defaults in `__init__`

## Configuration Categories
### A) Runtime trading parameters (this issue)
- `initial_cash` / `initial_capital`
- `base_position_pct`
- `max_daily_loss_pct`
- `max_position_pct`
- `slippage_pct`
- `market_hours_filter`

### B) Model/LLM parameters (already env-based)
- Provider/model/temperature in `settings.py`

## Proposed Source of Truth
1. **Environment** (via `quantagent/settings.py`)
   - Provide typed constants for the parameters in category (A).
   - Update `.env.example` accordingly.

2. **Database profiles** (already implemented)
   - `strategy_configs` table via `ConfigManager`.
   - Streamlit configuration UI already persists/loads these profiles.

3. **Call-site explicit overrides**
   - Dicts passed into `Backtest(config=...)`.
   - Dicts passed to `StrategyAssembler.from_profiles(..., overrides=...)`.

## Precedence (must be consistent across modules)
Highest → lowest:
1. Explicit per-run overrides (call-site dict / overrides)
2. Database profile values (loaded dict)
3. Environment defaults (`settings.*`)
4. No additional fallback literals in business modules

## Minimal Implementation Notes (non-code)
- `StrategyAssembler.DEFAULTS` should become settings-backed (either computed at import from `settings`, or retrieved from a small function like `StrategyAssembler.get_defaults()` returning a dict).
- `Backtest.__init__` should avoid `config.get(key, <literal>)` and instead use the assembler/settings-backed defaults.
- Component class defaults (`PositionSizer(...=0.05)`, `RiskManager(...=0.05)`, `PaperBroker(...=0.01)`) should not be relied on as the system defaults; callers should pass resolved values from the assembler.

## Env Keys (proposal)
Names should align with existing style (`AGENT_LLM_*`, `GRAPH_LLM_*`).

- `TRADING_INITIAL_CASH` (float)
- `TRADING_BASE_POSITION_PCT` (float)
- `TRADING_MAX_DAILY_LOSS_PCT` (float)
- `TRADING_MAX_POSITION_PCT` (float)
- `TRADING_SLIPPAGE_PCT` (float)
- `BACKTEST_MARKET_HOURS_FILTER` (bool: true/false)

If naming is contentious, prefer minimal churn: keep short, explicit, and avoid collisions.

### Example (minimal)
```env
TRADING_INITIAL_CASH=100000
TRADING_BASE_POSITION_PCT=0.05
TRADING_MAX_DAILY_LOSS_PCT=0.05
TRADING_MAX_POSITION_PCT=0.10
TRADING_SLIPPAGE_PCT=0.01
BACKTEST_MARKET_HOURS_FILTER=true
```

## Risks / Edge Cases
- Env parsing (bool/float) must be consistent with existing patterns in `settings.py`.
- Some callers may still depend on component constructor defaults; ensure resolved values are always passed from assembler/backtest.
- Multiple schema names in config snapshots (`initial_capital` vs `initial_cash`) already exist; changes should preserve snapshot compatibility.
