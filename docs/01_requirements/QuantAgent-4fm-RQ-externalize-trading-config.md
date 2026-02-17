# QuantAgent-4fm — RQ — Externalize hardcoded trading configuration

## Objective
Remove hardcoded default trading configuration values from runtime modules and make them configurable via:
- environment variables (loaded by `quantagent/settings.py`), and/or
- persisted database profiles (`strategy_configs` / `ConfigManager`).

Targeted modules called out by the issue:
- `quantagent/strategy/assembler.py`
- `quantagent/trading_graph.py`
- `quantagent/backtesting/backtest.py`

## In Scope
1. **Centralize default trading parameters** into `quantagent/settings.py` (env-backed) for:
   - `initial_cash` / `initial_capital`
   - `base_position_pct`
   - `max_daily_loss_pct`
   - `max_position_pct`
   - `slippage_pct`
   - `market_hours_filter`
   - (optional if present today) `use_checkpointing`

2. **Ensure StrategyAssembler defaults are not hardcoded in-module**
   - Replace `StrategyAssembler.DEFAULTS` literals with values sourced from `settings.py` (or a settings-backed helper).

3. **Ensure Backtest uses centralized defaults**
   - Any fallback values currently embedded in `Backtest.__init__` (e.g., `dict.get(..., <literal>)`) should defer to the same centralized defaults.

4. **Database profile override remains supported**
   - If a user provides a `StrategyConfig` / profile dict (via existing Streamlit config UI or `ConfigManager`), those values must override env defaults.

5. **Documentation update**
   - Add/maintain `.env.example` entries and docs describing the supported env keys.

## Out of Scope
- Adding new UI screens or workflows in Streamlit/Flask.
- Introducing new database tables or migrations.
- Changing trading logic, order execution, risk rules, or strategy semantics.
- Refactoring unrelated configuration (LLM providers/models) beyond what is needed to avoid duplication.

## Constraints
- Keep changes minimal and scoped to this issue (no opportunistic refactors).
- Backward compatibility expectation: existing callers that pass explicit config dicts must continue to work.
- Configuration precedence must be explicit and stable (see DS).

## Definition of Done
- No targeted runtime module contains hardcoded default literals for the listed trading parameters.
- Defaults can be changed via env without code changes.
- Profile-based configs in DB continue to override defaults.
- Acceptance criteria in `docs/05_acceptance_tests/QuantAgent-4fm-AC-externalize-trading-config.md` are satisfied.
