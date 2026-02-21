# QuantAgent-3o8: Replay Execution Mode - Acceptance Criteria

**Issue**: QuantAgent-3o8  
**Type**: Feature  
**Status**: open

---

## Test Cases

### TC1: Select Completed Backtest Run

**Given**: At least one backtest run exists with `total_trades > 0` and stored signals  
**When**: User opens Replay tab  
**Then**:
- System displays list of completed runs with name, date range, assets, metrics summary
- Each run shows number of stored analyses available
- Only runs with analyses are selectable
- Runs in progress or failed runs are not selectable

---

### TC2: Load Source Run Details

**Given**: User selects a completed backtest run  
**When**: System loads run details  
**Then**:
- Original configuration snapshot is displayed (portfolio/risk/model params)
- Date range and asset list are shown
- Summary metrics are displayed (total trades, win rate, total P&L)
- Number of stored analyses is shown (should match expected count)

---

### TC3: Select Single Profile for Replay

**Given**: Source run is selected  
**When**: User selects one portfolio/risk profile (different from original)  
**Then**:
- System shows configuration comparison table (original vs. selected)
- Key differences are highlighted (position sizing, risk limits)
- "Run Replay" button becomes enabled
- Estimated execution time is shown (should be fast: no LLM calls)

---

### TC4: Execute Single Replay Run

**Given**: Source run and target profile are selected  
**When**: User clicks "Run Replay"  
**Then**:
- System creates new BacktestRun with `replay_source_run_id` set
- System loads all signals from source run
- System re-executes position sizing and order generation using target profile
- System simulates trades with target profile's slippage/limits
- No new LLM calls are made (verify via logs: no API requests)
- New BacktestRun record is created with metrics calculated
- Execution completes in < 10% of original run time

---

### TC5: Verify Distinct P&L with Different Profiles

**Given**: Same source run, two different risk profiles (e.g., 5% vs. 10% position size)  
**When**: Both replays are executed  
**Then**:
- Two distinct BacktestRun records are created
- Metrics differ between runs (e.g., larger positions → larger P&L variance)
- Same signals are used (verify signal IDs match source run)
- Order quantities differ according to profile position sizing
- Trade P&L values differ according to order quantities

---

### TC6: Select Multiple Profiles for Sweep

**Given**: Source run is selected  
**When**: User selects multiple profiles (e.g., 3 profiles)  
**Then**:
- System shows batch preview with profile names
- Estimated total execution time is shown
- "Run Sweep" button becomes enabled
- System indicates sequential execution order

---

### TC7: Execute Multi-Profile Sweep

**Given**: Source run and 3 target profiles are selected  
**When**: User clicks "Run Sweep"  
**Then**:
- System executes replays sequentially (profile 1 → profile 2 → profile 3)
- Progress indicator shows current profile and completion status
- Three new BacktestRun records are created
- Each run links to source via `replay_source_run_id`
- All runs complete successfully
- Total time is < 30% of running 3 full backtests from scratch

---

### TC8: View Comparison for Single Replay

**Given**: Replay run is completed  
**When**: User views comparison  
**Then**:
- Side-by-side metrics table is displayed (original vs. replay)
- Key metrics compared: total trades, win rate, total P&L, Sharpe, max drawdown
- Equity curves are plotted together (original and replay lines on same chart)
- Configuration differences are highlighted in comparison table

---

### TC9: View Multi-Run Comparison

**Given**: Multiple replay runs are completed (same source, different profiles)  
**When**: User views multi-run comparison  
**Then**:
- Metrics matrix is displayed (rows = runs, cols = metrics)
- All equity curves are overlaid on same chart with distinct colors
- Configuration table shows each profile's key parameters
- Best/worst runs are highlighted (by selected metric: e.g., Sharpe)

---

### TC10: Verify Environment Tagging

**Given**: Replay run is completed  
**When**: System queries orders and trades  
**Then**:
- All orders have `environment = 'backtest'`
- All trades have `environment = 'backtest'`
- Orders are linked to signals from source run via `trigger_signal_id`
- Signals retain original `thread_id`, `checkpoint_id` from source run

---

### TC11: Verify Provenance Links

**Given**: Replay run is completed  
**When**: System queries BacktestRun and related records  
**Then**:
- Replay run has `replay_source_run_id` pointing to original run
- Original run's signals are not duplicated (same signal IDs used)
- New orders reference original signals via `trigger_signal_id`
- New trades reference new orders via `order_id`
- Full chain is traceable: source run → signals → replay orders → replay trades

---

### TC12: Handle Missing Analyses

**Given**: Source run has incomplete or missing analyses  
**When**: User attempts replay  
**Then**:
- System detects missing analyses and displays warning
- Replay is blocked with clear error message
- Error specifies which dates/assets lack stored analyses
- User is directed to re-run original backtest to populate analyses

---

### TC13: Verify No LLM Calls During Replay

**Given**: Replay execution is in progress  
**When**: System monitors API request logs  
**Then**:
- Zero LLM provider API calls are made (OpenAI, Anthropic, etc.)
- No new checkpoints are created (existing checkpoints are referenced only)
- Log confirms: "Replay mode: using stored analyses, no LLM calls"

---

### TC14: Same Universe Constraint

**Given**: Source run used assets `['BTC', 'ETH']`  
**When**: User attempts replay with profile that has different universe `['BTC', 'SPX']`  
**Then**:
- System displays warning: "Universe mismatch detected"
- Replay either: (a) uses source run's universe and ignores profile universe, or (b) blocks execution
- MVP decision: use source run universe, log warning

---

## Performance Oracles

| Metric | Target | Verification |
|--------|--------|-------------|
| Execution time | < 10% of original backtest | Measure start to completion time |
| LLM API calls | 0 | Check API logs, no provider requests |
| Database queries | < 50 per replay | Monitor query logs |
| Memory usage | < 500 MB | Profile during execution |

---

## Negative Cases

### NC1: No Completed Runs
**Given**: No backtest runs exist  
**When**: User opens Replay tab  
**Then**: Message displayed: "No completed runs available. Create a backtest first."

### NC2: Run Without Analyses
**Given**: BacktestRun exists but no signals stored  
**When**: User selects run and attempts replay  
**Then**: Error: "Run {id} has no stored analyses. Cannot replay."

### NC3: Invalid Profile Selection
**Given**: User selects same profile as original run  
**When**: User attempts replay  
**Then**: Warning: "Selected profile is identical to source run. No changes expected."

---

## Integration Points

- **BacktestRun model**: `replay_source_run_id` foreign key
- **Signal model**: Read `thread_id`, `checkpoint_id`, `state_snapshot`
- **StrategyAssembler**: Load profiles and resolve configuration
- **Backtest class**: New method `run_replay(source_run_id, target_config)` or separate `ReplayExecutor`
- **UI Replay tab**: Wire to backend executor, display results

---

## References

- **Requirements**: `docs/01_requirements/QuantAgent-3o8-RQ-replay-execution.md`
- **Design**: `docs/03_design/QuantAgent-3o8-DS-replay-execution.md`
- **Trading System Requirements**: `docs/01_requirements/trading_system_requirements.md` (Requirement D)
