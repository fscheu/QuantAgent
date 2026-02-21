# QuantAgent-3o8: Replay Execution Mode - Requirements

**Issue**: QuantAgent-3o8  
**Type**: Feature  
**Priority**: 2  
**Status**: open

---

## Objective

Enable replay execution mode that reuses stored analyses from a completed backtest run with different Portfolio/Risk profiles, without making new LLM calls.

This satisfies **Requirement D** in `trading_system_requirements.md`: "Backtest Setup Recording and Replayable Execution".

---

## Context

- BacktestRun model already stores `config_snapshot` and `assets`
- Signal model has `thread_id`, `checkpoint_id`, `state_snapshot` fields for provenance
- UI Replay tab exists (`apps/streamlit/views/replay.py`) but execution is not functional
- Current gap: No mechanism to select a backtest run, choose alternative profiles, and execute using stored analyses

---

## Scope

### In Scope

1. **Selection Interface**
   - Select a completed backtest run (by ID or name)
   - Select one or multiple portfolio/risk profiles to apply
   - View original run configuration vs. new profile configurations

2. **Replay Execution Engine**
   - Load stored analyses (signals) from the selected backtest run
   - Apply selected portfolio/risk profile to re-evaluate position sizing, risk checks, and trade execution
   - Execute sequentially (one profile at a time) per MVP decision
   - Generate new BacktestRun record with `replay_source_run_id` linkage
   - Tag all new orders/trades/positions with `environment=backtest`

3. **Results Storage & Comparison**
   - Store replay results as new BacktestRun with full metrics
   - Link replay run to original run via foreign key
   - UI comparison view showing side-by-side metrics and equity curves

### Out of Scope

- Parallel execution of multiple replay profiles (deferred post-MVP)
- Modifying or regenerating analyses (replay is read-only for analyses)
- Cross-environment replay (only backtest → backtest supported)
- Real-time replay (only completed runs with stored analyses)
- Model variant selection (use analyses from source run as-is)

---

## Constraints

- **No LLM calls**: Replay must use stored analyses exclusively
- **Sequential execution**: One profile execution at a time (MVP)
- **Same universe**: Replay must use same asset universe as source run
- **Immutable analyses**: Cannot modify or re-generate analyses during replay
- **Provenance tracking**: All replay runs must link to source run

---

## User Flows

### Flow 1: Single Profile Replay

1. User navigates to Replay tab
2. System displays list of completed backtest runs
3. User selects source backtest run
4. System displays run details (config, metrics, date range, assets)
5. User selects one portfolio/risk profile (different from original)
6. System shows profile comparison (original vs. selected)
7. User clicks "Run Replay"
8. System executes replay using stored analyses + new profile
9. System displays completion with link to results
10. User views comparison: original run vs. replay run

### Flow 2: Multiple Profile Sweep

1. Steps 1-4 same as Flow 1
2. User selects multiple portfolio/risk profiles (e.g., 3 different risk levels)
3. System shows batch preview with estimated execution time
4. User clicks "Run Sweep"
5. System executes sequentially (profile 1 → profile 2 → profile 3)
6. System displays progress indicator
7. Upon completion, system shows comparison table and charts
8. User views multi-run comparison with metrics matrix and overlaid equity curves

---

## Acceptance Criteria Summary

See `docs/05_acceptance_tests/QuantAgent-3o8-AC-replay-execution.md` for detailed test cases.

**Key criteria**:
- ✅ Can select completed backtest run with stored analyses
- ✅ Can select one or multiple portfolio/risk profiles
- ✅ Replay execution completes without LLM calls
- ✅ Two replay runs with different profiles yield distinct P&L/metrics
- ✅ UI comparison view shows side-by-side results
- ✅ All replay orders/trades tagged with `environment=backtest`
- ✅ Provenance link from replay run to source run maintained

---

## Definition of Done

1. Replay execution functional in backend (`quantagent/backtesting/`)
2. UI Replay tab wired to backend replay executor
3. Can execute single replay run successfully
4. Can execute multi-profile sweep successfully
5. Comparison view displays metrics and equity curves
6. All acceptance tests pass
7. Documentation updated (this doc + design/planning)

---

## References

- **Source Requirement**: `docs/01_requirements/trading_system_requirements.md` (Section D)
- **Related Design**: `docs/03_design/strategy_assembler_architecture.md`
- **Existing Architecture**: `docs/03_design/backtesting_engine.md`
- **UI Design**: `docs/03_design/streamlit_app_architecture.md`
- **Acceptance Tests**: `docs/05_acceptance_tests/QuantAgent-3o8-AC-replay-execution.md`
