# QuantAgent-3o8: Replay Execution Mode - Planning

**Issue**: QuantAgent-3o8  
**Type**: Feature  
**Status**: open  
**Estimated Effort**: 2-3 days

---

## Task Breakdown

### Phase 1: Database Schema (0.5 day)

#### Task 1.1: Add replay_source_run_id field
- **File**: `quantagent/models.py`
- **Action**: Add `replay_source_run_id` column to `BacktestRun` model
- **Details**:
  - Column: Integer, ForeignKey("backtest_runs.id"), nullable=True, indexed
  - Add relationship: `replay_source_run = relationship("BacktestRun", remote_side=[id])`
- **Testing**: Verify model loads without errors
- **Dependencies**: None
- **Estimated Time**: 1 hour

#### Task 1.2: Create database migration
- **File**: `alembic/versions/XXX_add_replay_source_run_id.py`
- **Action**: Generate Alembic migration for new field
- **Commands**:
  ```bash
  alembic revision --autogenerate -m "Add replay_source_run_id to backtest_runs"
  alembic upgrade head
  ```
- **Testing**: Migration applies cleanly on dev database
- **Dependencies**: Task 1.1
- **Estimated Time**: 0.5 hours

#### Task 1.3: Verify schema changes
- **Action**: Confirm field exists in database, test foreign key constraint
- **Testing**: Insert test BacktestRun with replay_source_run_id, verify relationship
- **Dependencies**: Task 1.2
- **Estimated Time**: 0.5 hours

---

### Phase 2: Backend Core (1 day)

#### Task 2.1: Implement Backtest.run_replay() method
- **File**: `quantagent/backtesting/backtest.py`
- **Action**: Add `run_replay(source_run_id, target_config, name)` method
- **Key Logic**:
  1. Load and validate source BacktestRun (exists, has signals, completed)
  2. Query signals filtered by source run date range + assets
  3. Override internal config with target_config
  4. Iterate date range (reuse existing date loop logic)
  5. For each date/asset: load stored signal, skip TradingGraph call
  6. Execute position sizing/risk checks with target_config
  7. Simulate trades via existing OrderManager/Broker
  8. Track portfolio and calculate metrics (reuse existing logic)
  9. Create new BacktestRun with replay_source_run_id set
- **Testing**: Unit test with mock signals and target config
- **Dependencies**: Task 1.3
- **Estimated Time**: 4 hours

#### Task 2.2: Add signal loading helper
- **File**: `quantagent/backtesting/backtest.py`
- **Action**: Create `_load_signals_for_replay(source_run)` helper method
- **Details**:
  - Query signals by date range, assets, timeframe
  - Return dict keyed by (date, asset) for fast lookup
  - Validate signal count matches expected (log warnings if gaps)
- **Testing**: Unit test with sample signals in DB
- **Dependencies**: Task 2.1
- **Estimated Time**: 1 hour

#### Task 2.3: Add validation logic
- **File**: `quantagent/backtesting/backtest.py`
- **Action**: Create `_validate_replay_source(source_run)` helper method
- **Details**:
  - Check source_run exists and status is completed
  - Check signals exist for source run (count > 0)
  - Check date range is valid
  - Raise descriptive errors if validation fails
- **Testing**: Unit test with valid and invalid source runs
- **Dependencies**: Task 2.1
- **Estimated Time**: 1 hour

#### Task 2.4: Add logging and observability
- **File**: `quantagent/backtesting/backtest.py`
- **Action**: Add structured logging for replay events
- **Log Points**:
  - `[REPLAY] Starting: source={id}, config={summary}`
  - `[REPLAY] Loaded {N} signals`
  - `[REPLAY] Processing {date}/{asset}, using signal {id}`
  - `[REPLAY] Completed: {trades} trades, {pnl} P&L, {time}s`
- **Dependencies**: Task 2.1
- **Estimated Time**: 1 hour

---

### Phase 3: Testing (0.5 day)

#### Task 3.1: Unit tests for run_replay()
- **File**: `tests/test_backtest.py`
- **Action**: Add test cases for replay method
- **Test Cases**:
  - `test_run_replay_single_profile()` - Basic replay execution
  - `test_run_replay_validates_source()` - Error handling for invalid source
  - `test_run_replay_loads_signals()` - Verify signal loading
  - `test_run_replay_provenance()` - Verify replay_source_run_id linkage
  - `test_run_replay_no_llm_calls()` - Confirm TradingGraph not invoked
- **Dependencies**: Task 2.4
- **Estimated Time**: 2 hours

#### Task 3.2: Integration test for replay flow
- **File**: `tests/test_backtest_integration.py`
- **Action**: Add end-to-end replay test
- **Test Flow**:
  1. Create and execute source backtest run
  2. Verify signals stored
  3. Execute replay with different config
  4. Verify new BacktestRun created with provenance link
  5. Verify metrics differ from source run
- **Dependencies**: Task 3.1
- **Estimated Time**: 1 hour

#### Task 3.3: Performance test
- **File**: `tests/test_backtest_integration.py`
- **Action**: Add performance comparison test
- **Test**: Measure replay time vs. original backtest time, assert < 10% ratio
- **Dependencies**: Task 3.2
- **Estimated Time**: 0.5 hours

---

### Phase 4: UI Integration (0.5 day)

#### Task 4.1: Wire Replay tab to backend
- **File**: `apps/streamlit/views/replay.py`
- **Action**: Implement replay execution logic
- **UI Flow**:
  1. Query and display completed BacktestRuns (filter: has analyses)
  2. On run selection: display run details and config
  3. Profile selection: single or multiple profiles (checkboxes)
  4. Config comparison preview
  5. "Run Replay" button → call `Backtest.run_replay()`
  6. Display progress and completion message
  7. Link to results/comparison view
- **Dependencies**: Task 2.4
- **Estimated Time**: 2 hours

#### Task 4.2: Implement comparison view
- **File**: `apps/streamlit/views/replay.py`
- **Action**: Add side-by-side comparison display
- **Components**:
  - Metrics comparison table (original vs. replay)
  - Configuration diff table
  - Overlaid equity curves chart (Plotly)
  - "Best run" indicator (by selected metric)
- **Dependencies**: Task 4.1
- **Estimated Time**: 1.5 hours

---

### Phase 5: Documentation & Polish (0.5 day)

#### Task 5.1: Update README files
- **Files**:
  - `docs/01_requirements/README.md`
  - `docs/02_planning/README.md`
  - `docs/03_design/README.md`
  - `docs/05_acceptance_tests/README.md`
- **Action**: Add links to QuantAgent-3o8 documents
- **Dependencies**: All previous tasks
- **Estimated Time**: 0.5 hours

#### Task 5.2: Add example usage
- **File**: `examples/run_replay.py` (new)
- **Action**: Create example script demonstrating replay execution
- **Content**:
  ```python
  # Example: replay existing backtest with different risk profile
  from quantagent.backtesting.backtest import Backtest
  from quantagent.database import SessionLocal
  
  # Load source run
  db = SessionLocal()
  source_run = db.query(BacktestRun).filter_by(name="Q1 2024 BTC").first()
  
  # Define target config
  target_config = {
      'base_position_pct': 0.10,  # More aggressive
      'max_daily_loss_pct': 0.08,
      # ... other params
  }
  
  # Execute replay
  backtest = Backtest(
      start_date=source_run.start_date,
      end_date=source_run.end_date,
      assets=source_run.assets,
      timeframe=source_run.timeframe,
      initial_capital=100000.0,
      config=target_config,
      db_session=db
  )
  
  metrics = backtest.run_replay(
      source_run_id=source_run.id,
      target_config=target_config,
      name="Replay Aggressive Profile"
  )
  
  print(f"Original P&L: ${source_run.total_pnl}")
  print(f"Replay P&L: ${metrics.total_pnl}")
  ```
- **Dependencies**: Task 5.1
- **Estimated Time**: 1 hour

#### Task 5.3: Manual testing and validation
- **Action**: Execute full replay flow manually via UI and CLI
- **Checklist**:
  - [ ] Source run selection works
  - [ ] Profile selection works (single + multiple)
  - [ ] Replay executes without errors
  - [ ] No LLM calls during replay (check logs)
  - [ ] Provenance links correct
  - [ ] Comparison view displays correctly
  - [ ] Multi-profile sweep executes sequentially
- **Dependencies**: Task 5.2
- **Estimated Time**: 1.5 hours

---

## Dependency Graph

```
Phase 1 (DB)
  Task 1.1 → Task 1.2 → Task 1.3
                          ↓
Phase 2 (Backend)                    
  Task 2.1 ← Task 1.3
    ↓
  Task 2.2, Task 2.3, Task 2.4
    ↓
Phase 3 (Testing)
  Task 3.1 → Task 3.2 → Task 3.3
    ↓
Phase 4 (UI)
  Task 4.1 → Task 4.2
    ↓
Phase 5 (Polish)
  Task 5.1 → Task 5.2 → Task 5.3
```

---

## Checkpoints

### Checkpoint 1: Database Ready (after Phase 1)
- [ ] Migration applied successfully
- [ ] Field exists in schema
- [ ] Foreign key constraint works
- **Deliverable**: Updated `quantagent/models.py` + migration file

### Checkpoint 2: Backend Functional (after Phase 2)
- [ ] `run_replay()` method implemented
- [ ] Signal loading works
- [ ] Validation logic works
- [ ] Logging in place
- **Deliverable**: Updated `quantagent/backtesting/backtest.py`

### Checkpoint 3: Tests Pass (after Phase 3)
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Performance test passes
- **Deliverable**: Test suite with 8+ new tests

### Checkpoint 4: UI Wired (after Phase 4)
- [ ] Replay tab functional
- [ ] Comparison view renders
- [ ] Manual smoke test passes
- **Deliverable**: Updated `apps/streamlit/views/replay.py`

### Checkpoint 5: Complete (after Phase 5)
- [ ] Documentation updated
- [ ] Example script works
- [ ] All acceptance criteria verified
- **Deliverable**: Issue closed, feature live

---

## Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| Signal loading query slow | Medium | Low | Add index on (timestamp, symbol), optimize query |
| Missing signals in source run | High | Medium | Validation step blocks replay, clear error message |
| Config incompatibility | Medium | Low | StrategyAssembler validation catches errors early |
| UI performance with large equity curves | Low | Low | Limit chart points to 1000 max, downsample if needed |

---

## Testing & Rollout Strategy

### Testing Phases
1. **Unit Tests**: Isolated component testing (Phase 3, Task 3.1)
2. **Integration Tests**: Full flow testing (Phase 3, Task 3.2-3.3)
3. **Manual Testing**: UI smoke tests (Phase 5, Task 5.3)
4. **User Acceptance**: Fede validates on real backtest runs

### Rollout
1. Merge to feature branch (`feature/QuantAgent-3o8-replay-execution`)
2. Deploy to dev environment
3. Run full test suite
4. Manual validation by Fede
5. Merge to main (PR with documentation + tests)

---

## Success Criteria

- [ ] All tasks completed
- [ ] All acceptance tests pass (see `QuantAgent-3o8-AC-replay-execution.md`)
- [ ] Performance target met (< 10% of original backtest time)
- [ ] Zero LLM calls during replay verified
- [ ] UI functional and comparison view works
- [ ] Documentation complete and up-to-date

---

## References

- **Requirements**: `docs/01_requirements/QuantAgent-3o8-RQ-replay-execution.md`
- **Design**: `docs/03_design/QuantAgent-3o8-DS-replay-execution.md`
- **Acceptance**: `docs/05_acceptance_tests/QuantAgent-3o8-AC-replay-execution.md`
- **Backtest Architecture**: `docs/03_design/backtesting_engine.md`
- **StrategyAssembler**: `docs/03_design/strategy_assembler_architecture.md`
