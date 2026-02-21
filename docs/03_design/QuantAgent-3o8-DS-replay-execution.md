# QuantAgent-3o8: Replay Execution Mode - Design

**Issue**: QuantAgent-3o8  
**Type**: Feature  
**Status**: open

---

## Overview

Design for replay execution mode that reuses stored analyses from a completed backtest run with different Portfolio/Risk profiles, without making new LLM calls.

---

## Architecture Decision

**Option A**: Extend `Backtest` class with `run_replay()` method  
**Option B**: Create separate `ReplayExecutor` class

**Decision**: **Option A** (extend Backtest)

**Rationale**:
- Reuses existing date iteration, portfolio tracking, and metrics calculation logic
- Minimal code duplication
- Simpler testing (same test infrastructure)
- Clear separation via method name and internal flags

---

## Components Modified

### 1. Database Schema

**New field in `BacktestRun` model**:

```python
# Add to quantagent/models.py
class BacktestRun(Base):
    # ... existing fields ...
    
    # Provenance: link to source run if this is a replay
    replay_source_run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=True)
    replay_source_run = relationship("BacktestRun", remote_side=[id])
```

**Migration**: Add `replay_source_run_id` column (nullable, indexed).

### 2. Backtest Class

**New method signature**:

```python
# In quantagent/backtesting/backtest.py
class Backtest:
    def run_replay(
        self,
        source_run_id: int,
        target_config: dict,
        name: Optional[str] = None
    ) -> BacktestMetrics:
        """
        Execute replay using stored analyses from source_run_id
        with target_config profile settings.
        
        Args:
            source_run_id: ID of completed BacktestRun to replay
            target_config: Portfolio/risk configuration to apply
            name: Optional name for replay run
        
        Returns:
            BacktestMetrics with results from replay execution
        """
```

**Internal flow**:
1. Load source BacktestRun and validate (has analyses, completed)
2. Load all signals from source run (filtered by run_id linkage or date/asset match)
3. Override self.config with target_config
4. Iterate through date range (same as source run)
5. For each date/asset, load stored signal instead of calling TradingGraph
6. Apply position sizing and risk checks with target_config
7. Execute trades via existing OrderManager/Broker logic
8. Track portfolio and calculate metrics
9. Create new BacktestRun with `replay_source_run_id` set

### 3. StrategyAssembler Integration

**Usage pattern**:

```python
# Load source run
source_run = db.query(BacktestRun).get(source_run_id)

# Resolve target profile
resolved = StrategyAssembler.from_profiles(
    portfolio_profile=target_portfolio,
    risk_profile=target_risk,
    model_profile=source_run.config_snapshot['model'],  # Keep original model metadata
    overrides=None
)

# Execute replay
backtest = Backtest(
    start_date=source_run.start_date,
    end_date=source_run.end_date,
    assets=source_run.assets,
    timeframe=source_run.timeframe,
    initial_capital=resolved.initial_cash,
    config=StrategyAssembler.config_snapshot(resolved),
    db_session=db
)

metrics = backtest.run_replay(
    source_run_id=source_run.id,
    target_config=resolved.__dict__,
    name=f"Replay {source_run.name} - Profile {target_profile_name}"
)
```

---

## Data Flow

### Standard Backtest (existing)
```
Date Loop → Fetch Data → TradingGraph (LLM) → Signal → Order → Trade → Metrics
```

### Replay Execution (new)
```
Date Loop → Load Stored Signal → Skip LLM → Order → Trade → Metrics
                  ↑
          (from source_run)
```

**Key differences**:
- No DataProvider calls (data already validated during source run)
- No TradingGraph invocation (signals pre-exist)
- Order/Trade generation uses target_config parameters
- New BacktestRun record with provenance link

---

## Implementation Strategy

### Phase 1: Backend Core
1. Add `replay_source_run_id` field to `BacktestRun` (migration)
2. Implement `Backtest.run_replay()` method
3. Add signal loading logic (query by source run, date, asset)
4. Wire existing position sizing/risk/execution with target config

### Phase 2: Validation & Testing
1. Unit tests for `run_replay()` method
2. Integration test with real source run + multiple target configs
3. Verify provenance links and environment tagging
4. Performance test (confirm < 10% of original time)

### Phase 3: UI Integration
1. Wire Streamlit Replay tab to `Backtest.run_replay()`
2. Add run selection UI (list completed runs with analyses count)
3. Add profile selection UI (single or multiple)
4. Implement comparison view (metrics table + equity curves)
5. Add progress indicator for multi-profile sweeps

---

## Key Design Decisions

### D1: Signal Linkage Strategy

**Option**: Query signals by (date, asset, timeframe) matching source run's range  
**Alternative**: Store explicit run_id in Signal model

**Decision**: Use date/asset/timeframe matching for MVP (existing fields)

**Rationale**:
- Avoids schema change to Signal model
- Signals already have timestamp, symbol, timeframe
- Unique enough for backtest context (controlled environment)
- Post-MVP: consider adding `backtest_run_id` for explicit linkage

### D2: Universe Handling

**Decision**: Always use source run's asset list for replay

**Rationale**:
- Analyses exist only for source run's assets
- Cannot replay assets without stored analyses
- Target profile's universe is ignored (logged as warning)
- Simplifies validation and error handling

### D3: Model Metadata Handling

**Decision**: Preserve source run's model metadata in replay run

**Rationale**:
- Analyses were generated by source model
- Replay doesn't change model (only portfolio/risk)
- Maintains full provenance chain
- Enables "which model generated this?" queries

### D4: Sequential vs. Parallel Execution

**Decision**: Sequential execution for multi-profile sweeps (MVP)

**Rationale**:
- Simpler implementation (no concurrency management)
- Adequate performance (replay is fast: no LLM calls)
- Avoids resource contention (DB, memory)
- Post-MVP: add parallel execution with worker pool if needed

---

## Observability

### Logging Points
- `[REPLAY] Starting replay: source={id}, target_config={...}`
- `[REPLAY] Loaded {N} signals from source run`
- `[REPLAY] Processing date {date}, asset {symbol}`
- `[REPLAY] Using stored signal {id}, skipping LLM call`
- `[REPLAY] Completed: {total_trades} trades, {P&L} P&L`

### Metrics
- Execution time (target: < 10% of original)
- Signal reuse count (should match source run analyses)
- New orders generated (may differ from source due to risk limits)
- Zero LLM API calls (verify via provider request logs)

---

## Error Handling

| Error | Cause | Mitigation |
|-------|-------|-----------|
| Source run not found | Invalid ID | Validate ID before execution, return clear error |
| Source run incomplete | No stored signals | Check `total_trades > 0` or explicit analyses count |
| Date/asset mismatch | Data inconsistency | Log warning, skip missing dates, continue |
| Config validation failure | Invalid target profile | Validate config before execution, return error |

---

## Limitations & Future Work

### MVP Limitations
- Sequential execution only (no parallelism)
- Same universe constraint (cannot add/remove assets)
- No intermediate state persistence (replay must complete in one run)
- No real-time progress API (UI must poll for completion)

### Post-MVP Enhancements
- Parallel multi-profile execution (thread/process pool)
- Partial replay (specific date range within source run)
- Replay with model variant selection (if multiple models analyzed same candles)
- Incremental replay (pause/resume capability)
- Live replay comparison view (side-by-side chart updates)

---

## Testing Strategy

### Unit Tests
- `test_run_replay_single_profile()` - Basic replay execution
- `test_run_replay_metrics_differ()` - Verify distinct P&L with different configs
- `test_run_replay_provenance()` - Verify `replay_source_run_id` linkage
- `test_run_replay_missing_signals()` - Error handling for incomplete data
- `test_run_replay_same_universe()` - Universe constraint validation

### Integration Tests
- `test_replay_full_flow()` - End-to-end: source run → replay → comparison
- `test_replay_multi_profile_sweep()` - Sequential execution of 3 profiles
- `test_replay_performance()` - Verify < 10% execution time vs. original

### UI Tests
- Manual: Run selection, profile selection, comparison view
- Verify charts render correctly with multiple equity curves
- Verify metrics table displays differences clearly

---

## References

- **Requirements**: `docs/01_requirements/QuantAgent-3o8-RQ-replay-execution.md`
- **Acceptance**: `docs/05_acceptance_tests/QuantAgent-3o8-AC-replay-execution.md`
- **Planning**: `docs/02_planning/QuantAgent-3o8-PL-replay-execution.md`
- **Existing Backtest**: `docs/03_design/backtesting_engine.md`
- **StrategyAssembler**: `docs/03_design/strategy_assembler_architecture.md`
