# QuantAgent-4fm — PL — Externalize hardcoded trading configuration

## Level of detail
STANDARD (cross-cutting config touchpoints; keep implementation minimal).

## Work Breakdown (0.5–2h tasks)
1. **Inventory hardcoded defaults** (0.5h)
   - Confirm all in-scope defaults in:
     - `quantagent/strategy/assembler.py`
     - `quantagent/backtesting/backtest.py`
     - Component constructors: `position_sizer.py`, `risk_manager.py`, `paper_broker.py`

2. **Add env-backed settings** (1h)
   - Extend `quantagent/settings.py` with typed constants for in-scope parameters.
   - Add keys + comments to `.env.example`.

3. **Wire StrategyAssembler defaults to settings** (1h)
   - Replace `StrategyAssembler.DEFAULTS` literals with settings-backed values.
   - Ensure precedence remains: overrides > profiles > defaults.

4. **Wire Backtest to StrategyAssembler defaults** (1h)
   - Remove per-key literal fallbacks in `Backtest.__init__` and any other callsites in `backtest.py`.
   - Defer all fallback/default behavior to the assembler/settings.

5. **Docs & consistency pass** (0.5h)
   - Update configuration docs if needed (link from existing `docs/03_design/CONFIGURATION.md` or add mention).
   - Update docs folder READMEs to link the new per-issue docs.

## Validation / Checks
- `rg` checks for removed literals in the targeted modules (see AC).
- Run a small backtest with env-only defaults.
- Run a backtest using a saved StrategyConfig profile and confirm profile overrides env.

## Rollout
- No migration.
- Safe as long as default behavior remains the same when env keys are unset (values match previous defaults).
